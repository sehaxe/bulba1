#!/usr/bin/env python3
import torch, sys, time, yaml
from bulba1.config import ModelConfig
from bulba1.model.mamba import MambaBlock

def main():
    config_path = "configs/default.yaml"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config" and len(sys.argv) > 2:
            config_path = sys.argv[2]
        else:
            config_path = sys.argv[1]

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)["model"]
    cfg = ModelConfig(**cfg_dict)

    model = MambaBlock(cfg).cuda()
    x = torch.randn(cfg.batch_size or 2, cfg.seq_len or 128, cfg.d_model, device='cuda')

    # прогрев
    for _ in range(5): model(x)
    torch.cuda.synchronize()

    # замер
    start = time.time()
    for _ in range(100): model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Mamba-3: batch={x.shape[0]}, seq_len={x.shape[1]}, d_model={cfg.d_model}")
    print(f"100 проходов за {elapsed:.3f} сек ({elapsed/100*1000:.2f} мс/проход)")
    print(f"VRAM: {torch.cuda.max_memory_allocated()/1024**2:.1f} МБ")

if __name__ == "__main__":
    main()