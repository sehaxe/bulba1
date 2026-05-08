import os
import sys
import tempfile
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.model.mamba import MambaBlock
from bulba1.tokenizer import HFTokenizer, TextDataset, create_dataloader
from bulba1.training.engine import TrainingEngine
from bulba1.training.checkpoint import CheckpointManager
from bulba1.training.chunked_ce import chunked_cross_entropy
from bulba1.training.eval import compute_perplexity


def _make_dummy_data(
    dir_path: str, text: str = "hello world this is test data ", repeats: int = 500
):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "dummy.txt"), "w", encoding="utf-8") as f:
        f.write(text * repeats)


def test_dataset_target_shift():
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_dummy_data(tmpdir)
        tok = HFTokenizer(vocab_size=100)
        tok.train([os.path.join(tmpdir, "dummy.txt")])
        ds = TextDataset(tok, tmpdir, seq_len=8, return_target=True)
        for input_ids, target_ids in ds:
            assert input_ids.shape == (8,)
            assert target_ids.shape == (8,)
            assert not torch.equal(input_ids, target_ids)
            assert torch.equal(input_ids[1:], target_ids[:-1])
            break
    print("PASS: dataset_target_shift")


def test_chunked_ce_correctness():
    vocab = 1000
    logits = torch.randn(2, 32, vocab, device="cpu")
    targets = torch.randint(0, vocab, (2, 32), device="cpu")
    loss_ref = F.cross_entropy(logits.view(-1, vocab), targets.view(-1))
    loss_chunk = chunked_cross_entropy(logits, targets.view(-1), chunk_size=256)
    diff = abs(loss_ref.item() - loss_chunk.item())
    assert diff < 1e-5, (
        f"Chunked CE mismatch: ref={loss_ref.item():.6f} chunk={loss_chunk.item():.6f} diff={diff:.6f}"
    )
    print("PASS: chunked_ce_correctness")


def test_mamba_bf16():
    cfg = ModelConfig(d_model=64, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2)
    m = MambaBlock(cfg).cuda().bfloat16()
    x = torch.randn(1, 8, 64, device="cuda").bfloat16()
    out = m(x)
    assert out.shape == x.shape


def test_eval_device_string():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=500,
        num_experts=4,
        expert_hidden=64,
        seq_len=16,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_dummy_data(tmpdir)
        tok = HFTokenizer(vocab_size=500)
        tok.train([os.path.join(tmpdir, "dummy.txt")])
        loader = create_dataloader(
            tok, tmpdir, batch_size=2, seq_len=16, num_workers=0, shuffle=False, return_target=False
        )
        model = MiniChat(cfg).cuda().bfloat16()
        try:
            ppl = compute_perplexity(model, loader, device="cuda", max_batches=2)
            assert isinstance(ppl, float)
            print("PASS: eval_device_string")
        except Exception as e:
            raise AssertionError(f"eval crashed: {e}")


def test_checkpoint_optimizer_ema():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_top_k=2)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        optimizer.step()
        saved = mgr.save(model, optimizer, step=1, loss=1.0, ema=None)
        assert saved
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_step_1.safetensors"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoint_step_1_optimizer.pt"))

        new_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mgr.load(model, os.path.join(tmpdir, "checkpoint_step_1.safetensors"), optimizer=new_opt)
        for key in optimizer.state_dict()["state"]:
            assert key in new_opt.state_dict()["state"]
    print("PASS: checkpoint_optimizer_ema")


def test_generate_with_kv_cache():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=16,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg).eval()
    x = torch.randint(0, 100, (1, 5))
    out1 = model.generate(x, max_new_tokens=5)
    assert out1.shape == (1, 10)
    out2 = model.generate(x, max_new_tokens=10)
    assert out2.shape == (1, 15)
    print("PASS: generate_with_kv_cache")


def test_training_loss_decreases():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=200,
        num_experts=4,
        expert_hidden=64,
        seq_len=16,
        batch_size=2,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
        learning_rate=1e-2,
        use_gradient_checkpointing=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_dummy_data(tmpdir, repeats=1000)
        tok = HFTokenizer(vocab_size=200)
        tok.train([os.path.join(tmpdir, "dummy.txt")])
        loader = create_dataloader(tok, tmpdir, cfg.batch_size, cfg.seq_len, num_workers=0)
        model = MiniChat(cfg)
        engine = TrainingEngine(model, cfg, tok, device="cpu")
        # disable EMA for deterministic CPU test
        engine.ema = None

        def inf():
            while True:
                for batch in loader:
                    yield tuple(b for b in batch)

        losses = []
        for step in range(20):
            for accum_step in range(engine.grad_accum_steps):
                batch = next(inf())
                is_last = accum_step == engine.grad_accum_steps - 1
                metrics = engine.train_step(batch, is_accum_last=is_last)
            losses.append(metrics["loss"].item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
        )
        print(f"PASS: training_loss_decreases ({losses[0]:.4f} -> {losses[-1]:.4f})")


def test_checkpoint_find_latest():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_top_k=5)
        for step in [10, 20, 30]:
            mgr.save(model, optimizer, step=step, loss=1.0 / step, ema=None)
        latest = mgr.find_latest()
        assert "step_30" in latest
    print("PASS: checkpoint_find_latest")


def test_oom_recovery_batch_reduction():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=200,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        batch_size=4,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg)
    engine = TrainingEngine(model, cfg, None, device="cpu")
    engine.cfg.batch_size = 4
    engine.grad_accum_steps = 1
    engine._reduce_batch_size()
    assert engine.cfg.batch_size == 2
    assert engine.grad_accum_steps == 2
    print("PASS: oom_recovery_batch_reduction")


def test_resume_from_checkpoint():
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        use_mamba=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_top_k=2)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        optimizer.step()
        mgr.save(model, optimizer, step=42, loss=1.0, ema=None)

        new_model = MiniChat(cfg)
        new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        loaded_step = mgr.load(new_model, "latest", optimizer=new_opt)
        assert loaded_step == 42
    print("PASS: resume_from_checkpoint")


def test_use_rex_without_moe():
    """Regression: use_rex=True + use_moe=False should not crash with AttributeError."""
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        num_experts=4,
        expert_hidden=64,
        seq_len=8,
        use_moe=False,
        use_rex=True,
        use_mamba=False,
        use_mhc=False,
        use_mtp=False,
        num_clr_tokens=0,
    )
    model = MiniChat(cfg).eval()
    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits, _, _, _ = model(x)
    assert logits.shape == (1, 8, 100)
    print("PASS: use_rex_without_moe")


def run_all():
    test_dataset_target_shift()
    test_chunked_ce_correctness()
    test_mamba_bf16()
    test_eval_device_string()
    test_checkpoint_optimizer_ema()
    test_checkpoint_find_latest()
    test_oom_recovery_batch_reduction()
    test_resume_from_checkpoint()
    test_generate_with_kv_cache()
    test_training_loss_decreases()
    test_use_rex_without_moe()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    run_all()
