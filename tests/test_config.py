import pytest
import yaml
from pathlib import Path

from bulba1.config import ModelConfig, load_config


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.d_model == 512
    assert cfg.n_layers == 10
    assert cfg.n_heads == 8
    assert cfg.vocab_size == 26000
    assert cfg.use_moe == True
    assert cfg.use_mamba == True


def test_config_field_types():
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg.d_model, int)
    assert isinstance(cfg.use_mamba, bool)
    assert isinstance(cfg.learning_rate, float)
    assert isinstance(cfg.batch_size, int)


def test_config_defaults():
    cfg = ModelConfig()
    assert cfg.d_model == 512
    assert cfg.n_layers == 10
    assert cfg.vocab_size == 26000
    assert cfg.use_moe == True


def test_config_extra_fields():
    cfg = ModelConfig(unknown_field=123, another_field="test")
    assert cfg.unknown_field == 123
    assert cfg.another_field == "test"


def test_config_model_fields():
    cfg = ModelConfig(
        d_model=128,
        n_layers=4,
        n_heads=4,
        vocab_size=1000,
    )
    assert cfg.d_model == 128
    assert cfg.n_layers == 4
    assert cfg.n_heads == 4
    assert cfg.vocab_size == 1000


def test_load_config_merges_model_and_training():
    cfg = load_config("configs/default.yaml")
    assert hasattr(cfg, 'learning_rate')
    assert hasattr(cfg, 'batch_size')
    assert hasattr(cfg, 'seq_len')
    assert hasattr(cfg, 'checkpoint_dir')


def test_config_moe_fields():
    cfg = ModelConfig(
        num_experts=8,
        top_k=4,
        expert_hidden=256,
        use_moe=True,
    )
    assert cfg.num_experts == 8
    assert cfg.top_k == 4
    assert cfg.expert_hidden == 256


def test_config_kda_fields():
    cfg = ModelConfig(
        kda_use_rope=True,
        kda_double_gate=True,
        kda_gate_dim=32,
        kda_use_parallel_scan=False,
    )
    assert cfg.kda_use_rope == True
    assert cfg.kda_double_gate == True
    assert cfg.kda_gate_dim == 32
    assert cfg.kda_use_parallel_scan == False