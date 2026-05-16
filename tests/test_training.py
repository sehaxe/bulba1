import pytest
import torch
import tempfile
import os
from pathlib import Path

from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat


def test_model_train_step(tiny_cfg, dummy_batch):
    model = MiniChat(tiny_cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logits, _, _, aux = model(dummy_batch)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, tiny_cfg.vocab_size),
        dummy_batch[:, 1:].reshape(-1)
    ) + aux
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert loss.item() > 0


def test_model_gradient_checkpointing(tiny_cfg, dummy_batch):
    tiny_cfg.use_gradient_checkpointing = True
    model = MiniChat(tiny_cfg)
    model.train()

    logits, _, _, _ = model(dummy_batch)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, tiny_cfg.vocab_size),
        dummy_batch[:, 1:].reshape(-1)
    )
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad


def test_model_eval_mode(tiny_cfg, dummy_batch):
    model = MiniChat(tiny_cfg)
    model.eval()

    with torch.no_grad():
        logits1, _, _, _ = model(dummy_batch)
        logits2, _, _, _ = model(dummy_batch)

    assert torch.equal(logits1, logits2)


def test_model_device_transfer(tiny_cfg, dummy_batch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = MiniChat(tiny_cfg).cuda()
    batch = dummy_batch.cuda()

    logits, _, _, _ = model(batch)
    assert logits.is_cuda


def test_model_parameters_count(tiny_cfg):
    model = MiniChat(tiny_cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params == total_params


def test_model_gradients_not_nan(tiny_cfg, dummy_batch):
    model = MiniChat(tiny_cfg)
    model.train()

    logits, _, _, aux = model(dummy_batch)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, tiny_cfg.vocab_size),
        dummy_batch[:, 1:].reshape(-1)
    ) + aux
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_model_lr_scaling(tiny_cfg):
    cfg1 = ModelConfig(**tiny_cfg.model_dump())
    cfg2 = ModelConfig(**tiny_cfg.model_dump())

    cfg1.d_model = 64
    cfg2.d_model = 128

    model1 = MiniChat(cfg1)
    model2 = MiniChat(cfg2)

    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())

    assert params2 > params1


def test_model_with_different_batch_sizes(tiny_cfg):
    for batch_size in [1, 2, 4]:
        cfg = ModelConfig(**tiny_cfg.model_dump())
        cfg.batch_size = batch_size
        model = MiniChat(cfg)
        x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.seq_len))
        logits, _, _, _ = model(x)
        assert logits.shape[0] == batch_size