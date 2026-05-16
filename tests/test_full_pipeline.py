import pytest
import torch
import tempfile
import os
from pathlib import Path

from bulba1.config import ModelConfig, load_config
from bulba1.model.minichat import MiniChat
from bulba1.tokenizer import SmartTokenizer


def test_full_pipeline_train_step(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("hello world " * 200)
        f.write("foo bar baz " * 200)

    tok = SmartTokenizer(vocab_size=100, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        vocab_size=tok.get_vocab_size(),
        num_experts=2,
        expert_hidden=64,
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
        total_steps=5,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        checkpoint_dir=temp_dir,
        data_dir=temp_dir,
        log_dir=temp_dir,
        use_gradient_checkpointing=False,
    )

    model = MiniChat(cfg)
    model.train()

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    targets = x.clone()

    logits, mtp1, mtp2, aux = model(x)

    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, cfg.vocab_size),
        targets[:, 1:].reshape(-1)
    ) + aux

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    optimizer.step()
    optimizer.zero_grad()

    assert loss.item() > 0
    assert not torch.isnan(loss).any()


def test_config_to_model_pipeline(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("test data " * 100)

    tok = SmartTokenizer(vocab_size=50, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    cfg = load_config("configs/smoke_test.yaml")
    cfg.checkpoint_dir = temp_dir
    cfg.log_dir = temp_dir
    cfg.use_mamba = False
    cfg.use_mhc = False
    cfg.use_mtp = False

    model = MiniChat(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits, mtp1, mtp2, aux = model(x)

    assert logits.shape[0] == 2
    assert logits.shape[-1] == cfg.vocab_size


def test_tokenizer_model_vocab_consistency(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("hello world " * 100)

    tok = SmartTokenizer(vocab_size=128, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    vocab_size = tok.get_vocab_size()

    cfg = ModelConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        vocab_size=vocab_size,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        batch_size=1,
        seq_len=4,
        total_steps=1,
        checkpoint_dir=temp_dir,
        data_dir=temp_dir,
        log_dir=temp_dir,
    )

    model = MiniChat(cfg)

    x = torch.randint(0, vocab_size, (1, 4))
    logits, _, _, _ = model(x)
    assert logits.shape[-1] == vocab_size


def test_model_input_output_stability(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("test " * 100)

    tok = SmartTokenizer(vocab_size=100, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    cfg = ModelConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        vocab_size=tok.get_vocab_size(),
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        batch_size=2,
        seq_len=8,
        total_steps=1,
        checkpoint_dir=temp_dir,
        data_dir=temp_dir,
        log_dir=temp_dir,
    )

    model = MiniChat(cfg)
    model.eval()

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))

    with torch.no_grad():
        out1 = model(x)[0]
        out2 = model(x)[0]

    assert torch.equal(out1, out2)


def test_model_with_different_sequences(temp_dir):
    data_file = os.path.join(temp_dir, "train.txt")
    with open(data_file, "w") as f:
        f.write("sample text " * 100)

    tok = SmartTokenizer(vocab_size=64, model_path=os.path.join(temp_dir, "tokenizer.json"))
    tok.train([data_file])

    cfg = ModelConfig(
        d_model=32,
        n_layers=1,
        n_heads=1,
        vocab_size=tok.get_vocab_size(),
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        use_bitlinear=False,
        batch_size=1,
        seq_len=16,
        total_steps=1,
        checkpoint_dir=temp_dir,
        data_dir=temp_dir,
        log_dir=temp_dir,
    )

    model = MiniChat(cfg)

    for seq_len in [4, 8, 16]:
        x = torch.randint(0, cfg.vocab_size, (1, seq_len))
        logits, _, _, _ = model(x)
        assert logits.shape[1] == seq_len