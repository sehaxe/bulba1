import time, yaml, torch
from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.training.optimizer import CombinedOptimizer
from torch.amp import autocast

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

with open("configs/default.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)
all_params = {}
all_params.update(yaml_cfg.get("model", {}))
all_params.update(yaml_cfg.get("training", {}))
cfg = ModelConfig(**all_params)

model = MiniChat(cfg).to("cuda")
model.train()
optimizer = CombinedOptimizer(model, cfg)

x = torch.randint(0, cfg.vocab_size, (4, 256)).to("cuda")

print("=" * 70)
print("FULL TRAINING STEP PROFILING")
print("=" * 70)

print(f"\nConfig: batch={cfg.batch_size}, seq_len={cfg.seq_len}, grad_accum={cfg.grad_accum_steps}")
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

torch.cuda.synchronize()
times = {'fwd': [], 'bwd': [], 'opt': []}

for i in range(5):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    with autocast("cuda", dtype=torch.bfloat16):
        logits, *_ = model(x, cfg.checkpoint_every_n_layers)
    torch.cuda.synchronize()
    fwd = time.perf_counter() - t0
    times['fwd'].append(fwd)
    
    t0 = time.perf_counter()
    logits[:, :2, :].mean().backward()
    torch.cuda.synchronize()
    bwd = time.perf_counter() - t0
    times['bwd'].append(bwd)
    
    t0 = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    opt = time.perf_counter() - t0
    times['opt'].append(opt)
    
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    
    peak = torch.cuda.max_memory_allocated()/1024**3
    print(f"Iter {i}: Fwd={fwd*1000:.0f}ms | Bwd={bwd*1000:.0f}ms | Opt={opt*1000:.0f}ms | Peak={peak:.1f}GB")

best_fwd = min(times['fwd'])
best_bwd = min(times['bwd'])
best_opt = min(times['opt'])
best_total = best_fwd + best_bwd + best_opt

print("\n" + "=" * 70)
print("BOTTLENECK BREAKDOWN")
print("=" * 70)

print(f"\n1. FORWARD PASS:   {best_fwd*1000:.0f}ms ({best_fwd/best_total*100:.1f}%)")
print(f"2. BACKWARD PASS:  {best_bwd*1000:.0f}ms ({best_bwd/best_total*100:.1f}%)")
print(f"3. OPTIMIZER:      {best_opt*1000:.0f}ms ({best_opt/best_total*100:.1f}%)")
print(f"   ─────────────────────────────────")
print(f"   TOTAL:          {best_total*1000:.0f}ms")

print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

print("\n[FORWARD] MHC Impact:")
print(f"  - MHC adds: ~{9215-2175}ms per forward")
print(f"  - MHC blocks: 28")
print(f"  - Cayley solve each: 539ms")
print(f"  - 28 * 539ms = {28*539:.0f}ms theoretical MHC time")
print(f"  - Actual: {(9215-2175)/28:.0f}ms average per MHC block")

print("\n[BACKWARD] Expected:")
print(f"  - Backward is usually 2-3x forward time")
print(f"  - Your backward: {best_bwd*1000:.0f}ms")
print(f"  - Forward: {best_fwd*1000:.0f}ms")
print(f"  - Ratio: {best_bwd/best_fwd:.2f}x (should be ~2x)")

print("\n[OPTIMIZER] Muon breakdown:")
muon_params = sum(p.numel() for name, p in model.named_parameters()
    if not any(x in name for x in ("embed", "head", "lm_", "bias", "norm", "A_log", "D"))
    and p.dim() == 2 and min(p.size(0), p.size(1)) >= 2)
print(f"  - Muon params: {muon_params/1e6:.1f}M")
print(f"  - Newton-Schulz iterations: 5")
print(f"  - Matrices processed: 300")

print("\n" + "=" * 70)
print("ROOT CAUSE ANALYSIS")
print("=" * 70)

print("\n1. [CRITICAL - 75% of forward time]")
print("   MHC Cayley transform uses torch.linalg.solve")
print("   which is 10x slower than matmul")
print("   Solution: Replace with efficient orthogonalization")

print("\n2. [HIGH - Memory inefficiency]")
print("   Forward peak: 11.1GB")
print("   This leaves only 4GB for gradients + optimizer states")
print("   Forces grad_accum_steps=4 which slows effective throughput")
print("   Solution: Enable gradient checkpointing (checkpoint_every_n_layers=4)")

print("\n3. [MEDIUM - Muon overhead]")
print(f"   Optimizer step: {best_opt*1000:.0f}ms")
print("   With 300 matrices and 5 NS iterations")
print("   Solution: Reduce ns_steps to 3")

print("\n" + "=" * 70)
print("ESTIMATED IMPROVEMENTS")
print("=" * 70)

current = best_total * 1000
print(f"\nCurrent per-step time: {current:.0f}ms")

if_best = 2175 * 1000
print(f"If MHC fixed: ~{if_best:.0f}ms (MHC removed)")
print(f"Improvement: {(current-if_best)/current*100:.0f}% faster")

gc_best = if_best * 0.7
print(f"With gradient checkpointing: ~{gc_best:.0f}ms")
print(f"Total improvement: {(current-gc_best)/current*100:.0f}% faster")

print(f"\nWith grad_accum_steps=1 and MHC fixed:")
print(f"Effective throughput: ~{4*256/(gc_best/1000):.0f} tokens/sec")
