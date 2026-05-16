"""
Performance benchmark tests.
Run manually with: pytest tests/test_benchmarks.py -v -m slow
"""

import pytest
import torch
import time

from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat


@pytest.mark.slow
def test_model_forward_benchmark():
    """Benchmark model forward pass - should complete within 1 second."""
    cfg = ModelConfig(
        d_model=256,
        n_layers=4,
        n_heads=4,
        vocab_size=1000,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        batch_size=4,
        seq_len=64,
        total_steps=1,
        checkpoint_dir="/tmp",
        data_dir="/tmp",
        log_dir="/tmp",
    )
    model = MiniChat(cfg)
    model.eval()

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / 10) * 1000
    print(f"Average forward time: {avg_time_ms:.2f} ms")

    # Should complete within 1 second (liberal for CPU)
    assert avg_time_ms < 1000, f"Forward pass too slow: {avg_time_ms} ms"


@pytest.mark.slow
def test_model_backward_benchmark():
    """Benchmark forward + backward pass."""
    cfg = ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=2,
        vocab_size=500,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        batch_size=2,
        seq_len=32,
        total_steps=1,
        checkpoint_dir="/tmp",
        data_dir="/tmp",
        log_dir="/tmp",
    )
    model = MiniChat(cfg)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    targets = x.clone()

    # Warmup
    for _ in range(2):
        model.zero_grad()
        logits, _, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, cfg.vocab_size),
            targets[:, 1:].reshape(-1)
        )
        loss.backward()

    # Benchmark
    start = time.perf_counter()
    for _ in range(5):
        model.zero_grad()
        logits, _, _, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, cfg.vocab_size),
            targets[:, 1:].reshape(-1)
        )
        loss.backward()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / 5) * 1000
    print(f"Average forward+backward time: {avg_time_ms:.2f} ms")

    assert avg_time_ms < 2000, f"Training step too slow: {avg_time_ms} ms"


@pytest.mark.slow
def test_rope_benchmark():
    """Benchmark RoPE operation."""
    from bulba1.model.rope import RoPE

    rope = RoPE(dim=64, max_seq_len=512)
    x = torch.randn(4, 8, 128, 64)

    # Warmup
    for _ in range(3):
        _ = rope(x, seq_len=128)

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        _ = rope(x, seq_len=128)
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / 100) * 1000
    print(f"Average RoPE time: {avg_time_ms:.4f} ms")

    assert avg_time_ms < 10, f"RoPE too slow: {avg_time_ms} ms"


@pytest.mark.slow
def test_moe_benchmark():
    """Benchmark MoE layer."""
    from bulba1.model.moe import MoELayer

    cfg = ModelConfig(
        d_model=128,
        vocab_size=500,
        num_experts=4,
        top_k=2,
        expert_hidden=128,
        use_moe=True,
        use_rex=False,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        batch_size=4,
        seq_len=32,
        total_steps=1,
        checkpoint_dir="/tmp",
        data_dir="/tmp",
        log_dir="/tmp",
    )
    moe = MoELayer(cfg)
    x = torch.randn(4, 32, 128)

    # Warmup
    for _ in range(3):
        _, _ = moe(x)

    # Benchmark
    start = time.perf_counter()
    for _ in range(50):
        _, _ = moe(x)
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / 50) * 1000
    print(f"Average MoE time: {avg_time_ms:.4f} ms")

    assert avg_time_ms < 50, f"MoE too slow: {avg_time_ms} ms"


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_benchmarks.py -v -m slow
    pytest.main([__file__, "-v", "-m", "slow"])