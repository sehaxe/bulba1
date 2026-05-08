import torch, sys, time, yaml
from bulba1.config import ModelConfig
from bulba1.model.moe import MoELayer

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/150m.yaml"
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)["model"]
    cfg = ModelConfig(**cfg_dict)

    layer = MoELayer(cfg).cuda()
    x = torch.randn(cfg.batch_size or 2, cfg.seq_len or 128, cfg.d_model, device='cuda')

    # Прогрев
    for _ in range(5): layer(x)
    torch.cuda.synchronize()

    # Замер
    start = time.time()
    for _ in range(100): layer(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"MoE: 100 проходов за {elapsed:.3f} сек ({elapsed/100*1000:.2f} мс/проход)")
    print(f"VRAM: {torch.cuda.max_memory_allocated()/1024**2:.1f} МБ")

if __name__ == "__main__":
    main()