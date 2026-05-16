#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import safetensors.torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent


def load_cfg():
    import yaml
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        raw = yaml.safe_load(f) or {}
    merged = {}
    merged.update(raw.get("model", {}))
    merged.update(raw.get("training", {}))
    return merged


def find_sft_checkpoint():
    for f in ["checkpoints/sft/sft_final.safetensors", "checkpoints/sft/sft_best.safetensors"]:
        p = PROJECT_ROOT / f
        if p.exists():
            return str(p)
    return None


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


def collate_dpo(batch):
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


@torch.no_grad()
def get_logprobs(model, input_ids, mask):
    logits, _, _, _ = model(input_ids)
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    shift_mask = mask[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * shift_mask).sum(dim=-1)


def train_dpo(sft_path: str, data_path: str, output_dir: str,
              lr: float = 5e-6, epochs: int = 2, batch_size: int = 2,
              grad_accum: int = 8, beta: float = 0.1, log_every: int = 10):
    from bulba1.config import ModelConfig
    from bulba1.model.minichat import MiniChat
    from bulba1.tokenizer import FastTokenizer

    raw_cfg = load_cfg()
    cfg = ModelConfig(**raw_cfg)
    device = torch.device("cuda")

    tokenizer = FastTokenizer("data/tokenizer_fast.json")
    tokenizer.load()
    tokenizer.add_chat_tokens()

    policy = MiniChat(cfg).to(device).to(torch.bfloat16)
    reference = MiniChat(cfg).to(device).to(torch.bfloat16)

    state = safetensors.torch.load_file(sft_path)
    policy.load_state_dict(state, strict=False)
    reference.load_state_dict(state, strict=False)
    policy.resize_token_embeddings(tokenizer.vocab_size)
    reference.resize_token_embeddings(tokenizer.vocab_size)
    for p in reference.parameters():
        p.requires_grad = False

    dataset = DPODataset(data_path, tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_dpo, drop_last=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    total_steps = (len(dataset) // batch_size // grad_accum) * epochs

    os.makedirs(output_dir, exist_ok=True)
    step = 0
    print(f"DPO: {len(dataset)} pairs, {total_steps} steps, beta={beta}")

    policy.train()
    for epoch in range(epochs):
        for i, (chosen_ids, rejected_ids, c_mask, r_mask) in enumerate(loader):
            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)
            c_mask = c_mask.to(device)
            r_mask = r_mask.to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                policy_chosen_lp = get_logprobs(policy, chosen_ids, c_mask)
                policy_rejected_lp = get_logprobs(policy, rejected_ids, r_mask)
                ref_chosen_lp = get_logprobs(reference, chosen_ids, c_mask)
                ref_rejected_lp = get_logprobs(reference, rejected_ids, r_mask)

                policy_ratio = policy_chosen_lp - policy_rejected_lp
                ref_ratio = ref_chosen_lp - ref_rejected_lp
                loss = -F.logsigmoid(beta * (policy_ratio - ref_ratio)).mean()
                loss = loss / grad_accum

            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % log_every == 0:
                    print(f"  step {step:4d}/{total_steps} | loss={loss.item() * grad_accum:.4f}")

    final_path = os.path.join(output_dir, "dpo_final.safetensors")
    safetensors.torch.save_file(policy.state_dict(), final_path)
    print(f"DPO done. Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", help="SFT checkpoint path (default: auto-find)")
    parser.add_argument("--data", default="data/sft/sft_claude_opus47.jsonl")
    parser.add_argument("--output", default="checkpoints/dpo")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    sft_ckpt = args.sft_checkpoint or find_sft_checkpoint()
    if not sft_ckpt:
        print("No SFT checkpoint found. Run SFT first.")
        return
    print(f"SFT checkpoint: {sft_ckpt}")
    train_dpo(sft_ckpt, args.data, args.output, args.lr, args.epochs, args.batch_size, args.grad_accum, args.beta)


if __name__ == "__main__":
    main()
