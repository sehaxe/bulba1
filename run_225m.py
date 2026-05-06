import os, sys, torch

os.chdir("/home/sehaxe/bulba1-python")
sys.path.insert(0, os.getcwd())

LOG_FILE = "logs/bulba1_225m.log"

from bulba1.utils.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.data.tokenizer import FastTokenizer, create_dataloader
from bulba1.training.engine import TrainingEngine

cfg = ModelConfig(
    d_model=768,
    n_layers=16,
    n_heads=12,
    num_experts=16,
    expert_hidden=768,
    vocab_size=12000,
    use_mamba=True,
    attn_every_n_layers=4,
    use_bitlinear=True,
    use_f16=True,
    use_rex=True,
    use_gradient_checkpointing=True,
    use_mtp=True,
    use_skip_gram=True,
    depth_scaled_init=True,
    curriculum_warmup_ratio=0.15,
    curriculum_start_seq_len=128,
    seq_len=512,
    batch_size=5,
    grad_accum_steps=2,
    skip_preflight=True,
    learning_rate=2e-4,
    weight_decay=0.1,
    checkpoint_every=100,
    checkpoint_dir="checkpoints/run_bulba1_225m_clean",
    checkpoint_keep_top_k=3,
    max_batch_reductions=3,
    vram_warn_pct=88,
    vram_critical_pct=95,
)

device = torch.device("cuda")
tokenizer = FastTokenizer("data/tokenizer_fast.json")
tokenizer.load()

loader = create_dataloader(
    tokenizer, "data/train", cfg.batch_size, cfg.seq_len, num_workers=4, prefetch_factor=4
)


def infinite_loader():
    while True:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                yield tuple(b.to(device, non_blocking=True) for b in batch)
            else:
                yield batch.to(device, non_blocking=True)


model = MiniChat(cfg).to(device)
engine = TrainingEngine(model, cfg, tokenizer, device=device, tuned_config=None)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params / 1e6:.1f}M params, d_model={cfg.d_model}, n_layers={cfg.n_layers}")

checkpoint_dir = cfg.checkpoint_dir
resume_step = 0
if os.path.exists(checkpoint_dir):
    files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_step_") and f.endswith(".safetensors")
    ]
    if files:
        steps = [int(f.split("_")[-1].replace(".safetensors", "")) for f in files]
        resume_step = max(steps)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{resume_step}.safetensors")
        optimizer_path = checkpoint_path.replace(".safetensors", "_optimizer.pt")

        print(f"Resuming from step {resume_step}...")
        state_dict = {}
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)

        # Verify config match before loading optimizer
        skip_optimizer = False
        json_path = checkpoint_path.replace(".safetensors", ".json")
        if os.path.exists(json_path):
            import json as json_lib

            with open(json_path) as f:
                saved_cfg = json_lib.load(f).get("config", {})
            # Compare key fields
            key_fields = ["d_model", "n_layers", "n_heads", "vocab_size", "seq_len", "batch_size"]
            mismatch = []
            for field in key_fields:
                saved_val = saved_cfg.get(field)
                curr_val = getattr(cfg, field, None)
                if saved_val is not None and curr_val is not None and saved_val != curr_val:
                    mismatch.append(f"{field}: {saved_val} vs {curr_val}")
            if mismatch:
                print(f"Config mismatch: {', '.join(mismatch)}. Skipping optimizer.")
                skip_optimizer = True
        elif os.path.exists(optimizer_path):
            try:
                opt_state = torch.load(optimizer_path, map_location="cpu")
                old_params = sum(
                    v.numel() for v in opt_state.get("state", {}).values() if hasattr(v, "numel")
                )
                new_params = sum(p.numel() for p in model.parameters())
                if abs(old_params - new_params) > 1000:
                    skip_optimizer = True
            except:
                skip_optimizer = True

        if os.path.exists(optimizer_path) and hasattr(engine, "optimizer") and not skip_optimizer:
            engine.optimizer.load_state_dict(torch.load(optimizer_path))
            print(f"Loaded optimizer state from step {resume_step}")
        else:
            print(f"Skipping optimizer state (config mismatch)")

# Clear log only on fresh start
if resume_step == 0 and os.path.exists(LOG_FILE):
    open(LOG_FILE, "w").close()

print(f"Training for 100000 steps...")

# Smart checkpoint interval: more frequent early, less later
total_steps = 100000
if total_steps <= 1000:
    checkpoint_every = 100
elif total_steps <= 10000:
    checkpoint_every = 500
else:
    checkpoint_every = 1000
log_every = 10  # Log status every 10 steps

print(f"Checkpoint every {checkpoint_every} steps, log every {log_every} steps")

model = engine.train(
    infinite_loader(),
    total_steps,
    checkpoint_every=checkpoint_every,
    log_every=log_every,
    eval_every=5000,
    resume_step=resume_step,
)
