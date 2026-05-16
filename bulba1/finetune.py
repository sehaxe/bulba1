"""
Fine-tuning module for Bulba1 (SFT and DPO training).

Provides unified FineTuner class and dataset implementations for:
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
"""

import json
import os

import torch
import torch.nn.functional as F
import safetensors.torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(data_path) as f:
            for line in f:
                r = json.loads(line)
                msgs = r.get("messages", [])
                if len(msgs) >= 2:
                    ids, weights = tokenizer.encode_chat(msgs)
                    if len(ids) >= 10:
                        self.examples.append((ids, weights))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, weights = self.examples[idx]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            weights = weights[: self.max_len]
        input_ids = torch.tensor(ids, dtype=torch.long)
        target_ids = torch.tensor(ids, dtype=torch.long)
        loss_weights = torch.tensor(weights, dtype=torch.float32)
        return input_ids, target_ids, loss_weights


class DPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        cid = tokenizer.chat_ids
        user_tag = cid.get("<|user|>", 0)
        assistant_tag = cid.get("<|assistant|>", 0)

        with open(data_path) as f:
            for line in f:
                r = json.loads(line)
                msgs = r.get("messages", [])
                if len(msgs) < 2:
                    continue

                assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
                if not assistant_msgs:
                    continue

                prompt_ids = []
                for m in msgs:
                    if m.get("role") != "assistant":
                        tag_id = cid.get(f"<|{m['role']}|>", 0)
                        if tag_id:
                            prompt_ids.append(tag_id)
                        content = m.get("content", "")
                        if content:
                            prompt_ids.extend(tokenizer.encode(content))

                chosen = assistant_msgs[0].get("content", "")
                chosen_ids = [assistant_tag] + tokenizer.encode(chosen) if assistant_tag else tokenizer.encode(chosen)

                rejected = "I don't know. Let me think... Actually, I'm not sure about this. Maybe someone else can help? Sorry, I cannot answer this question properly."
                rejected_ids = [assistant_tag] + tokenizer.encode(rejected) if assistant_tag else tokenizer.encode(rejected)

                full_chosen = prompt_ids + chosen_ids
                full_rejected = prompt_ids + rejected_ids

                if len(full_chosen) < 10 or len(full_rejected) < 10:
                    continue
                self.examples.append((full_chosen[:max_len], full_rejected[:max_len]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        c, r = self.examples[idx]
        return torch.tensor(c, dtype=torch.long), torch.tensor(r, dtype=torch.long)


def sft_collate(batch):
    max_len = max(b[0].size(0) for b in batch)
    pad_id = 0
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    weights = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, (inp, tgt, w) in enumerate(batch):
        n = inp.size(0)
        input_ids[i, :n] = inp
        target_ids[i, :n] = tgt
        weights[i, :n] = w
    return input_ids, target_ids, weights


def dpo_collate(batch):
    max_c = max(b[0].size(0) for b in batch)
    max_r = max(b[1].size(0) for b in batch)
    max_len = max(max_c, max_r)
    pad = 0
    chosen_ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    rejected_ids = torch.full((len(batch), max_len), pad, dtype=torch.long)
    chosen_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    rejected_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, (c, r) in enumerate(batch):
        chosen_ids[i, :c.size(0)] = c
        rejected_ids[i, :r.size(0)] = r
        chosen_mask[i, :c.size(0)] = True
        rejected_mask[i, :r.size(0)] = True
    return chosen_ids, rejected_ids, chosen_mask, rejected_mask


class FineTuner:
    def __init__(self, checkpoint_path: str, config_path: str = "configs/default.yaml"):
        from bulba1.config import load_config
        from bulba1.model.minichat import MiniChat
        from bulba1.tokenizer import FastTokenizer

        self.cfg = load_config(config_path)
        self.device = torch.device("cuda")

        self.tokenizer = FastTokenizer("data/tokenizer_fast.json")
        self.tokenizer.load()
        self.tokenizer.add_chat_tokens()

        self.model = MiniChat(self.cfg).to(self.device).to(torch.bfloat16)
        state = safetensors.torch.load_file(checkpoint_path)
        self.model.load_state_dict(state, strict=False)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)

    def sft(self, data_path: str, output_dir: str, lr: float = 2e-5, epochs: int = 3,
            batch_size: int = 4, grad_accum: int = 4, log_every: int = 10):
        dataset = SFTDataset(data_path, self.tokenizer, max_len=512)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=sft_collate, drop_last=True)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
        total_steps = (len(dataset) // batch_size // grad_accum) * epochs

        os.makedirs(output_dir, exist_ok=True)
        step = 0
        best_loss = float("inf")

        print(f"SFT: {len(dataset)} examples, {total_steps} steps, {epochs} epochs")
        self.model.train()
        for epoch in range(epochs):
            accum_loss = 0.0
            for i, (inp, tgt, w) in enumerate(loader):
                inp, tgt, w = inp.to(self.device), tgt.to(self.device), w.to(self.device)
                with autocast("cuda", dtype=torch.bfloat16):
                    logits, _, _, _ = self.model(inp)
                    shift_logits = logits[:, :-1, :].reshape(-1, self.cfg.vocab_size)
                    shift_tgt = tgt[:, 1:].reshape(-1)
                    shift_w = w[:, 1:].reshape(-1)
                    loss = F.cross_entropy(shift_logits, shift_tgt, reduction="none")
                    loss = (loss * shift_w).sum() / shift_w.sum().clamp_min(1)
                    loss = loss / grad_accum

                loss.backward()
                if (i + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    accum_loss += loss.item() * grad_accum

                    if step % log_every == 0:
                        avg = accum_loss / log_every
                        print(f"  step {step:4d}/{total_steps} | loss={avg:.4f}")
                        if avg < best_loss:
                            best_loss = avg
                            save_path = os.path.join(output_dir, "sft_best.safetensors")
                            safetensors.torch.save_file(self.model.state_dict(), save_path)
                        accum_loss = 0.0

        final_path = os.path.join(output_dir, "sft_final.safetensors")
        safetensors.torch.save_file(self.model.state_dict(), final_path)
        print(f"Done. Saved to {final_path}")
        return final_path


    @torch.no_grad()
    def _get_logprobs(self, input_ids, mask):
        logits, _, _, _ = self.model(input_ids)
        shift_logits = logits[:, :-1, :]
        shift_ids = input_ids[:, 1:]
        shift_mask = mask[:, 1:]
        logprobs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = logprobs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
        return (token_logprobs * shift_mask).sum(dim=-1)


    def dpo(self, data_path: str, output_dir: str, lr: float = 5e-6, epochs: int = 2,
            batch_size: int = 2, grad_accum: int = 8, beta: float = 0.1, log_every: int = 10):
        dataset = DPODataset(data_path, self.tokenizer, max_len=512)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=dpo_collate, drop_last=True)

        reference = MiniChat(self.cfg).to(self.device).to(torch.bfloat16)
        reference.load_state_dict(self.model.state_dict(), strict=False)
        for p in reference.parameters():
            p.requires_grad = False

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
        total_steps = (len(dataset) // batch_size // grad_accum) * epochs

        os.makedirs(output_dir, exist_ok=True)
        step = 0
        print(f"DPO: {len(dataset)} pairs, {total_steps} steps, beta={beta}")

        self.model.train()
        for epoch in range(epochs):
            for i, (chosen_ids, rejected_ids, c_mask, r_mask) in enumerate(loader):
                chosen_ids = chosen_ids.to(self.device)
                rejected_ids = rejected_ids.to(self.device)
                c_mask = c_mask.to(self.device)
                r_mask = r_mask.to(self.device)

                with autocast("cuda", dtype=torch.bfloat16):
                    policy_chosen_lp = self._get_logprobs(chosen_ids, c_mask)
                    policy_rejected_lp = self._get_logprobs(rejected_ids, r_mask)
                    ref_chosen_lp = self._get_logprobs(chosen_ids, c_mask)
                    ref_rejected_lp = self._get_logprobs(rejected_ids, r_mask)

                    policy_ratio = policy_chosen_lp - policy_rejected_lp
                    ref_ratio = ref_chosen_lp - ref_rejected_lp
                    loss = -F.logsigmoid(beta * (policy_ratio - ref_ratio)).mean()
                    loss = loss / grad_accum

                loss.backward()
                if (i + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                    step += 1
                    if step % log_every == 0:
                        print(f"  step {step:4d}/{total_steps} | loss={loss.item() * grad_accum:.4f}")

        final_path = os.path.join(output_dir, "dpo_final.safetensors")
        safetensors.torch.save_file(self.model.state_dict(), final_path)
        print(f"DPO done. Saved to {final_path}")
        return final_path