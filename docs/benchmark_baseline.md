# Performance Baseline

## Configuration
- Model: Bulba1 20.58M params
- Batch size: 2
- Sequence length: 64
- Device: CPU (mock Mamba)
- Precision: bfloat16

## Results

| Metric | Value |
|--------|-------|
| Tokens/sec | 383.9 |
| Time/step | 333.4 ms |
| Steps/day | 259,164 |
| 100K steps | 0.4 days |

## Breakdown

| Component | Percentage |
|-----------|------------|
| Forward | 42.8% |
| Backward | 57.2% |
| Optimizer | ~0% |

## Notes

- Mamba is mocked (not functional) - real GPU performance will differ
- Gradient checkpointing enabled
- MHC enabled: n=4, iterations=4
- VRAM utilization: N/A (CPU mode)

## Test Command

```bash
python tools/deep_profile.py --config configs/default.yaml --batch 2 --seq 64 --warmup 2 --profile 3
```

## Future Improvements

Target: Improve throughput by 2x through:
1. Triton BitLinear kernels
2. Efficient MoE (Megablocks)
3. DDP for multi-GPU
4. torch.compile on critical paths

---

Updated: 2026-05-16