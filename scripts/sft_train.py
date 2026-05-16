#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import safetensors.torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).parent.parent


def load_cfg():
    with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
        raw = yaml.safe_load(f) or {}
    merged = {}
    merged.update(raw.get("model", {}))
    merged.update(raw.get("training", {}))
    return merged


def find_best_checkpoint():
    cfg = load_cfg()
    ckpt_dir = PROJECT_ROOT / cfg.get("checkpoint_dir", "checkpoints/run_bulba1_67m")
    best = ckpt_dir / "best.safetensors"
    if best.exists():
        return str(best)
    files = sorted(
        ckpt_dir.glob("checkpoint_step_*.safetensors"),
        key=lambda f: int(re.search(r"step_(\d+)", f.name).group(1)),
        reverse=True,
    )
    return str(files[0]) if files else None


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


def collate(batch):
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


def train_sft(checkpoint_path: str, data_path: str, output_dir: str,
              lr: float = 2e-5, epochs: int = 3, batch_size: int = 4,
              grad_accum: int = 4, log_every: int = 10):
    from bulba1.config import ModelConfig
    from bulba1.model.minichat import MiniChat
    from bulba1.tokenizer import FastTokenizer

    raw_cfg = load_cfg()
    cfg = ModelConfig(**raw_cfg)
    device = torch.device("cuda")

    tokenizer = FastTokenizer("data/tokenizer_fast.json")
    tokenizer.load()
    tokenizer.add_chat_tokens()

    model = MiniChat(cfg).to(device).to(torch.bfloat16)
    state = safetensors.torch.load_file(checkpoint_path)
    model.load_state_dict(state, strict=False)
    model.resize_token_embeddings(tokenizer.vocab_size)

    dataset = SFTDataset(data_path, tokenizer, max_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8)
    total_steps = (len(dataset) // batch_size // grad_accum) * epochs

    os.makedirs(output_dir, exist_ok=True)
    step = 0
    best_loss = float("inf")

    print(f"SFT: {len(dataset)} examples, {total_steps} steps, {epochs} epochs")
    model.train()
    for epoch in range(epochs):
        accum_loss = 0.0
        for i, (inp, tgt, w) in enumerate(loader):
            inp, tgt, w = inp.to(device), tgt.to(device), w.to(device)
            with autocast("cuda", dtype=torch.bfloat16):
                logits, _, _, _ = model(inp)
                shift_logits = logits[:, :-1, :].reshape(-1, cfg.vocab_size)
                shift_tgt = tgt[:, 1:].reshape(-1)
                shift_w = w[:, 1:].reshape(-1)
                loss = F.cross_entropy(shift_logits, shift_tgt, reduction="none")
                loss = (loss * shift_w).sum() / shift_w.sum().clamp_min(1)
                loss = loss / grad_accum

            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                        safetensors.torch.save_file(model.state_dict(), save_path)
                    accum_loss = 0.0

    final_path = os.path.join(output_dir, "sft_final.safetensors")
    safetensors.torch.save_file(model.state_dict(), final_path)
    print(f"Done. Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Checkpoint path (default: best)")
    parser.add_argument("--data", default="data/sft/sft_claude_opus47.jsonl")
    parser.add_argument("--output", default="checkpoints/sft")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    ckpt = args.checkpoint or find_best_checkpoint()
    if not ckpt:
        print("No checkpoint found.")
        return
    print(f"Checkpoint: {ckpt}")
    train_sft(ckpt, args.data, args.output, args.lr, args.epochs, args.batch_size, args.grad_accum)


if __name__ == "__main__":
    main()
