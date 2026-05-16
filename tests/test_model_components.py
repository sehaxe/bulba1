import pytest
import torch

from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.model.rope import RoPE
from bulba1.model.moe import Expert, MoELayer


def test_minichat_forward_shape(tiny_cfg, dummy_batch):
    model = MiniChat(tiny_cfg)
    logits, mtp1, mtp2, aux = model(dummy_batch)
    B, T, V = dummy_batch.shape[0], dummy_batch.shape[1], tiny_cfg.vocab_size
    expected_seq_len = T + tiny_cfg.num_clr_tokens
    assert logits.shape == (B, expected_seq_len, V)


def test_minichat_forward_no_nan(tiny_cfg, dummy_batch):
    model = MiniChat(tiny_cfg)
    model.eval()
    with torch.no_grad():
        logits, mtp1, mtp2, aux = model(dummy_batch)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(aux).any()


def test_minichat_eval_mode(tiny_cfg, dummy_batch):
    model = MiniChat(tiny_cfg)
    model.eval()
    with torch.no_grad():
        logits1, _, _, _ = model(dummy_batch)
        logits2, _, _, _ = model(dummy_batch)
    assert torch.equal(logits1, logits2)


def test_rope_forward_shapes():
    rope = RoPE(dim=64, max_seq_len=128)
    x = torch.randn(1, 4, 8, 64)
    y = rope(x, seq_len=8)
    assert y.shape == x.shape


def test_rope_different_seq_lengths():
    rope = RoPE(dim=32, max_seq_len=16)
    x8 = torch.randn(1, 2, 8, 32)
    x16 = torch.randn(1, 2, 16, 32)
    y8 = rope(x8, seq_len=8)
    y16 = rope(x16, seq_len=16)
    assert y8.shape == (1, 2, 8, 32)
    assert y16.shape == (1, 2, 16, 32)


def test_expert_forward():
    expert = Expert(d_model=64, hidden_dim=128, use_bitlinear=False)
    x = torch.randn(4, 8, 64)
    out = expert(x)
    assert out.shape == (4, 8, 64)


def test_expert_with_relu2():
    expert = Expert(d_model=64, hidden_dim=128, use_bitlinear=False, activation_fn="relu2")
    x = torch.randn(4, 8, 64)
    out = expert(x)
    assert out.shape == (4, 8, 64)


def test_expert_with_silu():
    expert = Expert(d_model=64, hidden_dim=128, use_bitlinear=False, activation_fn="silu")
    x = torch.randn(4, 8, 64)
    out = expert(x)
    assert out.shape == (4, 8, 64)


def test_moe_layer_forward():
    cfg = ModelConfig(
        d_model=64,
        vocab_size=200,
        num_experts=4,
        top_k=2,
        expert_hidden=64,
        use_moe=True,
        use_rex=False,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        batch_size=2,
        seq_len=8,
        total_steps=10,
        checkpoint_dir="/tmp",
        data_dir="/tmp",
        log_dir="/tmp",
    )
    moe = MoELayer(cfg)
    x = torch.randn(2, 8, 64)
    out, aux = moe(x)
    assert out.shape == (2, 8, 64)
    assert isinstance(aux, torch.Tensor)


def test_moe_routing():
    cfg = ModelConfig(
        d_model=64,
        vocab_size=200,
        num_experts=4,
        top_k=2,
        expert_hidden=64,
        use_moe=True,
        use_rex=False,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        batch_size=2,
        seq_len=8,
        total_steps=10,
        checkpoint_dir="/tmp",
        data_dir="/tmp",
        log_dir="/tmp",
    )
    moe = MoELayer(cfg)
    x = torch.randn(2, 8, 64)
    out, aux = moe(x)
    assert aux.item() >= 0


def test_expert_output_no_nan():
    expert = Expert(d_model=32, hidden_dim=64)
    x = torch.randn(2, 4, 32) * 0.1
    out = expert(x)
    assert not torch.isnan(out).any()


def test_minichat_different_vocab_sizes():
    for vocab_size in [100, 500, 1000]:
        cfg = ModelConfig(
            d_model=32,
            n_layers=1,
            n_heads=2,
            vocab_size=vocab_size,
            use_mamba=False,
            use_mhc=False,
            use_mtp=False,
            num_clr_tokens=0,
            use_bitlinear=False,
            batch_size=1,
            seq_len=4,
            total_steps=1,
            checkpoint_dir="/tmp",
            data_dir="/tmp",
            log_dir="/tmp",
        )
        model = MiniChat(cfg)
        x = torch.randint(0, vocab_size, (1, 4))
        logits, _, _, _ = model(x)
        assert logits.shape[-1] == vocab_size