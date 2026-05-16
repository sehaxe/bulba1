import pytest
import torch
import tempfile
import os
from pathlib import Path


@pytest.fixture
def tiny_cfg():
    from bulba1.config import ModelConfig
    return ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        vocab_size=200,
        num_experts=2,
        expert_hidden=64,
        top_k=2,
        use_moe=True,
        use_rex=False,
        num_shared_experts=1,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        use_diff_attn=True,
        use_kda=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        use_f16=False,
        batch_size=2,
        seq_len=8,
        total_steps=10,
        use_gradient_checkpointing=False,
        checkpoint_dir="/tmp/test_checkpoints",
        data_dir="/tmp/test_data",
        log_dir="/tmp/test_logs",
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )


@pytest.fixture
def dummy_batch(tiny_cfg):
    return torch.randint(0, tiny_cfg.vocab_size, (tiny_cfg.batch_size, tiny_cfg.seq_len))


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir