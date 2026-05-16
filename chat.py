#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import safetensors.torch

PROJECT_ROOT = Path(__file__).parent


def load_cfg():
    cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def find_checkpoints():
    cfg = load_cfg()
    ckpt_dir = PROJECT_ROOT / cfg.get("training", {}).get("checkpoint_dir", "checkpoints/run_bulba1_67m")
    if not ckpt_dir.exists():
        return []
    files = sorted(
        ckpt_dir.glob("checkpoint_step_*.safetensors"),
        key=lambda f: int(re.search(r"step_(\d+)", f.name).group(1)),
    )
    return [int(re.search(r"step_(\d+)", f.name).group(1)) for f in files]


def load_model(step=None, sft=False, dpo=False):
    from bulba1.config import ModelConfig
    from bulba1.model.minichat import MiniChat
    from bulba1.tokenizer import FastTokenizer

    cfg_raw = load_cfg()
    params = {}
    params.update(cfg_raw.get("model", {}))
    params.update(cfg_raw.get("training", {}))
    cfg = ModelConfig(**params)

    tokenizer = FastTokenizer("data/tokenizer_fast.json")
    tokenizer.load()

    if dpo:
        ckpt_path = "checkpoints/dpo/dpo_final.safetensors"
        if not os.path.exists(ckpt_path):
            ckpt_path = "checkpoints/dpo/dpo_best.safetensors"
        tokenizer.add_chat_tokens()
    elif sft:
        ckpt_path = "checkpoints/sft/sft_final.safetensors"
        if not os.path.exists(ckpt_path):
            ckpt_path = "checkpoints/sft/sft_best.safetensors"
        tokenizer.add_chat_tokens()
    else:
        if step is None:
            ckpts = find_checkpoints()
            step = ckpts[-1] if ckpts else None
        if step is None:
            raise FileNotFoundError("No checkpoints")
        ckpt_dir = params.get("checkpoint_dir", "checkpoints/run_bulba1_67m")
        ckpt_path = f"{ckpt_dir}/checkpoint_step_{step}.safetensors"
        if not os.path.exists(ckpt_path):
            ckpt_path = f"{ckpt_dir}/best.safetensors"

    model = MiniChat(cfg).to("cuda")
    if sft:
        model.resize_token_embeddings(tokenizer.vocab_size)
    state_dict = safetensors.torch.load_file(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_speculative(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=50):
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    accepted = 0
    draft_attempts = 0

    while ids.size(1) < max_tokens + tokenizer.encode(prompt).__len__():
        inp = ids[:, -model.cfg.max_ctx_len:]
        logits, mtp1, mtp2, _ = model(inp)

        t0_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(t0_logits, min(top_k, t0_logits.size(-1)))
            t0_logits[t0_logits < v[:, [-1]]] = float("-inf")
        t0 = torch.multinomial(F.softmax(t0_logits, dim=-1), num_samples=1)

        drafts = [t0]
        for mtp in [mtp1, mtp2]:
            if mtp is None:
                break
            d_logits = mtp[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(d_logits, min(top_k, d_logits.size(-1)))
                d_logits[d_logits < v[:, [-1]]] = float("-inf")
            drafts.append(torch.multinomial(F.softmax(d_logits, dim=-1), num_samples=1))

        if len(drafts) == 1:
            ids = torch.cat([ids, t0], dim=1)
            continue

        draft_tokens = torch.cat(drafts, dim=1)
        verify_inp = torch.cat([ids, draft_tokens], dim=1)[:, -model.cfg.max_ctx_len:]
        verify_logits, _, _, _ = model(verify_inp)

        n_accepted = 0
        for i, d in enumerate(drafts):
            position = ids.size(1) + i
            v_logits = verify_logits[:, position - 1, :] / temperature
            v_token = torch.argmax(v_logits, dim=-1, keepdim=True)
            if v_token.item() == d.item():
                n_accepted += 1
                accepted += 1
            else:
                break
            draft_attempts += 1

        n_total = max(1, n_accepted)
        ids = torch.cat([ids, draft_tokens[:, :n_total]], dim=1)

    return tokenizer.decode(ids[0].tolist()), accepted, draft_attempts


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=50, thinking=False, speculative=False):
    if speculative:
        result, acc, att = generate_speculative(model, tokenizer, prompt, max_tokens, temperature, top_k)
        return result

    device = next(model.parameters()).device
    if thinking:
        cid = getattr(tokenizer, "chat_ids", {})
        system_id = cid.get("<|system|>", 0)
        user_id = cid.get("<|user|>", 0)
        think_id = cid.get("<|thinking|>", 0)
        assistant_id = cid.get("<|assistant|>", 0)
        if system_id and user_id:
            prompt_ids = [system_id, user_id] + tokenizer.encode(prompt) + [think_id]
        else:
            prompt_ids = tokenizer.encode(prompt)
    else:
        prompt_ids = tokenizer.encode(prompt)

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        inp = ids[:, -model.cfg.max_ctx_len:]
        logits, _, _, _ = model(inp)
        next_logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=1)

    return tokenizer.decode(ids[0].tolist())


def interactive(model, tokenizer, thinking=False, speculative=False):
    mode = "Thinking" if thinking else "Standard"
    if speculative:
        mode += " + Speculative"
    print(f"\nBulba Chat [{mode}]")
    print("Type /exit to quit, /clear to reset\n")
    history = []
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in ("/exit", "/quit"):
            break
        if user.lower() == "/clear":
            history = []
            print("Cleared.")
            continue
        if not user:
            continue
        prompt = "\n".join(history[-5:] + [user])
        print("Bot: ", end="", flush=True)
        result = generate(model, tokenizer, prompt, max_tokens=200, thinking=thinking, speculative=speculative)
        response = result.replace(prompt, "").strip()
        print(response)
        history.append(user)
        history.append(response)


def main():
    parser = argparse.ArgumentParser(description="Bulba Chat")
    parser.add_argument("--step", type=int, help="Checkpoint step (default: latest)")
    parser.add_argument("--sft", action="store_true", help="Use SFT thinking model")
    parser.add_argument("--dpo", action="store_true", help="Use DPO model")
    parser.add_argument("--speculative", action="store_true", help="MTP speculative decoding (faster)")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--prompt", type=str, help="One-shot generation")
    parser.add_argument("--list", action="store_true", help="List checkpoints")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    if args.dpo:
        print("Loading DPO model...")
        model, tokenizer = load_model(dpo=True)
        thinking = True
    elif args.sft:
        print("Loading SFT thinking model...")
        model, tokenizer = load_model(sft=True)
        thinking = True
    else:
        ckpts = find_checkpoints()
        if args.list:
            if not ckpts:
                print("No checkpoints.")
                return
            for c in ckpts:
                print(f"  Step {c}")
            return
        step = args.step or (ckpts[-1] if ckpts else None)
        if step is None:
            print("No checkpoints. Train first.")
            return
        print(f"Loading checkpoint {step}...")
        model, tokenizer = load_model(step=step)
        thinking = args.thinking

    if args.prompt:
        result = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, thinking=thinking, speculative=args.speculative)
        print(result)
    else:
        interactive(model, tokenizer, thinking=thinking, speculative=args.speculative)


if __name__ == "__main__":
    main()
