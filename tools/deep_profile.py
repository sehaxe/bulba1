#!/usr/bin/env python3
"""
Bulba1 Deep Profiler - Comprehensive training performance analysis.
Run with: uv run python tools/deep_profile.py
"""
import os
import sys
import time
import json
import math
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from bulba1.config import ModelConfig
from bulba1.model.minichat import MiniChat
from bulba1.training.optimizer import CombinedOptimizer


class DeepProfiler:
    def __init__(self, cfg_path="configs/default.yaml", batch_size=4, seq_len=256, warmup_steps=2, profile_steps=5):
        self.cfg_path = cfg_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self.results = {"config": {}, "components": {}, "recommendations": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.cfg = None

    def load_config(self):
        with open(self.cfg_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        all_params = {}
        all_params.update(yaml_cfg.get('model', {}))
        all_params.update(yaml_cfg.get('training', {}))
        self.cfg = ModelConfig(**all_params)
        self.results["config"] = {
            "d_model": self.cfg.d_model,
            "n_layers": self.cfg.n_layers,
            "n_heads": self.cfg.n_heads,
            "vocab_size": self.cfg.vocab_size,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "use_gradient_checkpointing": self.cfg.use_gradient_checkpointing,
            "use_mhc": self.cfg.use_mhc,
            "use_moe": self.cfg.use_moe,
            "use_bitnet_a48": self.cfg.use_bitnet_a48,
            "use_bitlinear": self.cfg.use_bitlinear,
            "muon_ns_steps": self.cfg.muon_ns_steps,
        }

    def init_model(self):
        print("  Initializing model...")
        self.model = MiniChat(self.cfg).to(self.device)
        self.model.train()
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.results["model_info"] = {
            "total_params_m": round(param_count / 1e6, 2),
            "trainable_params_m": round(trainable_count / 1e6, 2),
        }
        print(f"    Params: {param_count/1e6:.1f}M")

    def init_optimizer(self):
        print("  Initializing optimizer...")
        self.optimizer = CombinedOptimizer(self.model, self.cfg)
        muon_count = len(self.optimizer.muon.param_groups[0]["params"]) if self.optimizer.muon else 0
        adamw_count = len(self.optimizer.adamw.param_groups[0]["params"]) if self.optimizer.adamw else 0
        self.results["optimizer_info"] = {"muon_params": muon_count, "adamw_params": adamw_count}
        print(f"    Muon: {muon_count}, AdamW: {adamw_count}")

    def profile_full_step(self):
        print("\n🔄 FULL TRAINING STEP PROFILE")
        print("=" * 60)

        times = {"forward": [], "backward": [], "optimizer": [], "total": []}
        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        targets = x.clone()

        for step in range(self.warmup_steps + self.profile_steps):
            try:
                self.model.zero_grad()
                self.optimizer.zero_grad()
                torch.cuda.synchronize()

                t0 = time.perf_counter()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _, _, aux_loss = self.model(x, 1)

                text_logits = logits[:, self.cfg.num_clr_tokens:self.cfg.num_clr_tokens+self.seq_len, :].reshape(-1, self.cfg.vocab_size)
                loss = torch.nn.functional.cross_entropy(text_logits.float(), targets.reshape(-1))
                loss = loss + aux_loss * 0.001

                torch.cuda.synchronize()
                t_fwd = time.perf_counter()

                loss.backward()

                torch.cuda.synchronize()
                t_bwd = time.perf_counter()

                self.optimizer.step()

                torch.cuda.synchronize()
                t_opt = time.perf_counter()

                if step >= self.warmup_steps:
                    times["forward"].append(t_fwd - t0)
                    times["backward"].append(t_bwd - t_fwd)
                    times["optimizer"].append(t_opt - t_bwd)
                    times["total"].append(t_opt - t0)

                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error at step {step}: {e}")
                torch.cuda.empty_cache()
                break

        if not times["total"]:
            print("  Unable to profile - errors encountered")
            return 0

        avg_total = sum(times["total"]) / len(times["total"]) * 1000
        avg_fwd = sum(times["forward"]) / len(times["forward"]) * 1000
        avg_bwd = sum(times["backward"]) / len(times["backward"]) * 1000
        avg_opt = sum(times["optimizer"]) / len(times["optimizer"]) * 1000

        self.results["step_profile"] = {
            "total_ms": round(avg_total, 2),
            "forward_ms": round(avg_fwd, 2),
            "backward_ms": round(avg_bwd, 2),
            "optimizer_ms": round(avg_opt, 2),
            "pct_forward": round(avg_fwd / avg_total * 100, 1),
            "pct_backward": round(avg_bwd / avg_total * 100, 1),
            "pct_optimizer": round(avg_opt / avg_total * 100, 1),
        }

        print(f"  Total:       {avg_total:>8.2f} ms  (100.0%)")
        print(f"  Forward:     {avg_fwd:>8.2f} ms  ({self.results['step_profile']['pct_forward']:>5.1f}%)")
        print(f"  Backward:    {avg_bwd:>8.2f} ms  ({self.results['step_profile']['pct_backward']:>5.1f}%)")
        print(f"  Optimizer:   {avg_opt:>8.2f} ms  ({self.results['step_profile']['pct_optimizer']:>5.1f}%)")

        return avg_total

    def profile_memory(self):
        print("\n📊 MEMORY PROFILE")
        print("=" * 60)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        baseline_mem = torch.cuda.memory_allocated() / 1024**2

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, _ = self.model(x, 1)

            forward_mem = torch.cuda.max_memory_allocated() / 1024**2

            targets = x.clone()
            text_logits = logits[:, self.cfg.num_clr_tokens:self.cfg.num_clr_tokens+self.seq_len, :].reshape(-1, self.cfg.vocab_size)
            loss = torch.nn.functional.cross_entropy(text_logits.float(), targets.reshape(-1))
            loss.backward()

            backward_mem = torch.cuda.max_memory_allocated() / 1024**2

            self.optimizer.step()
            self.optimizer.zero_grad()

            after_opt_mem = torch.cuda.memory_allocated() / 1024**2
        except Exception as e:
            print(f"  Memory test error: {e}")
            forward_mem = backward_mem = after_opt_mem = 0

        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

        self.results["memory"] = {
            "baseline_mb": round(baseline_mem, 1),
            "forward_mb": round(forward_mem, 1),
            "backward_mb": round(backward_mem, 1),
            "after_optimizer_mb": round(after_opt_mem, 1),
            "total_vram_gb": round(total_vram, 1),
            "utilization_pct": round(forward_mem / (total_vram * 1024) * 100, 1),
        }

        print(f"  Baseline:    {baseline_mem:>8.1f} MB")
        print(f"  Forward:     {forward_mem:>8.1f} MB")
        print(f"  Backward:   {backward_mem:>8.1f} MB")
        print(f"  After Opt:   {after_opt_mem:>8.1f} MB")
        print(f"  Peak VRAM:  {forward_mem:>8.1f} MB ({self.results['memory']['utilization_pct']}%)")

    def profile_block_breakdown(self):
        print("\n🔧 BLOCK COMPONENT PROFILING")
        print("=" * 60)

        block_times = []

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        h = self.model.embedding(x)
        if self.cfg.num_clr_tokens > 0:
            clr = self.model.clr_tokens.expand(self.batch_size, -1, -1)
            h = torch.cat([clr, h], dim=1)

        for _ in range(self.warmup_steps):
            self.model.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for block in self.model.blocks:
                    h, _, _ = block(h, None)

        for i, block in enumerate(self.model.blocks):
            self.model.zero_grad()
            h_temp = h.clone()

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    h_out, _, _ = block(h_temp, None)
                torch.cuda.synchronize()
                block_ms = (time.perf_counter() - t0) * 1000

                block_times.append({
                    "layer": i,
                    "is_attn": block.is_attn_block,
                    "ms": round(block_ms, 2),
                })
                print(f"  Block {i:2d} ({'Attn' if block.is_attn_block else 'Mamba'}): {block_ms:>8.2f} ms")

                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Block {i} error: {e}")
                break

        if block_times:
            total_block_time = sum(b["ms"] for b in block_times)
            self.results["blocks"] = {
                "total_ms": round(total_block_time, 2),
                "per_block_avg_ms": round(total_block_time / len(block_times), 2),
                "details": block_times,
            }

    def analyze_activations(self):
        print("\n🔥 ACTIVATION SCALE ANALYSIS")
        print("=" * 60)

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        h = self.model.embedding(x)
        if self.cfg.num_clr_tokens > 0:
            clr = self.model.clr_tokens.expand(self.batch_size, -1, -1)
            h = torch.cat([clr, h], dim=1)

        scales = []

        for i, block in enumerate(self.model.blocks):
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    h, _, _ = block(h, None)

                scale = h.abs().max().item()
                scales.append(scale)

                if i < 3 or i >= self.cfg.n_layers - 2 or i % 5 == 0:
                    status = ""
                    if scale > 10000:
                        status = " ⚠️ EXPLOSION"
                    print(f"  Block {i:2d}: scale={scale:>12.2f}{status}")
            except Exception as e:
                print(f"  Block {i} error: {e}")
                break

        if scales:
            self.results["activations"] = {
                "initial_scale": round(scales[0], 2),
                "max_scale": round(max(scales), 2),
                "final_scale": round(scales[-1], 2),
                "scale_growth": round(max(scales) / scales[0], 2) if scales[0] > 0 else 0,
                "has_explosion": max(scales) > 10000,
            }
            print(f"\n  Initial: {scales[0]:.2f}")
            print(f"  Max:     {max(scales):.2f}")
            print(f"  Final:   {scales[-1]:.2f}")
            print(f"  Growth:  {self.results['activations']['scale_growth']:.1f}x")

            if self.results["activations"]["has_explosion"]:
                print("  ⚠️  Activation explosion detected!")

    def analyze_gradients(self):
        print("\n📈 GRADIENT ANALYSIS")
        print("=" * 60)

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        targets = x.clone()

        self.model.zero_grad()
        self.optimizer.zero_grad()

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, aux_loss = self.model(x, 1)

            text_logits = logits[:, self.cfg.num_clr_tokens:self.cfg.num_clr_tokens+self.seq_len, :].reshape(-1, self.cfg.vocab_size)
            loss = torch.nn.functional.cross_entropy(text_logits.float(), targets.reshape(-1))
            loss = loss + aux_loss * 0.001

            loss.backward()

            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()

            sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:10]

            self.results["gradients"] = {
                "top_grads": [(n, round(g, 4)) for n, g in sorted_grads],
                "has_nan_grads": any(math.isnan(g) for g in grad_norms.values()),
                "total_params_with_grad": len(grad_norms),
            }

            print("  Top 10 gradient norms:")
            for name, grad in sorted_grads:
                short_name = name if len(name) < 50 else "..." + name[-47:]
                print(f"    {short_name:<50} {grad:>10.4f}")

            if self.results["gradients"]["has_nan_grads"]:
                print("  🔴 NaN gradients detected!")

        except Exception as e:
            print(f"  Gradient analysis error: {e}")

    def analyze_loss(self):
        print("\n💰 LOSS ANALYSIS")
        print("=" * 60)

        x = torch.randint(0, self.cfg.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        targets = x.clone()

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, mtp1, mtp2, aux_loss = self.model(x, 1)

            text_logits = logits[:, self.cfg.num_clr_tokens:self.cfg.num_clr_tokens+self.seq_len, :].reshape(-1, self.cfg.vocab_size)
            main_loss = torch.nn.functional.cross_entropy(text_logits.float(), targets.reshape(-1))

            self.results["loss_analysis"] = {
                "main_loss": round(main_loss.item(), 4),
                "aux_loss": round(aux_loss.item(), 4),
                "logits_scale": round(logits.abs().max().item(), 2),
                "has_nan": torch.isnan(logits).any().item(),
            }

            print(f"  Main loss:   {main_loss.item():.4f}")
            print(f"  Aux loss:    {aux_loss.item():.4f}")
            print(f"  Logits max:  {logits.abs().max().item():.2f}")

            if self.results["loss_analysis"]["has_nan"]:
                print("  🔴 NaN in logits!")

        except Exception as e:
            print(f"  Loss analysis error: {e}")

    def compute_throughput(self):
        print("\n⚡ THROUGHPUT ANALYSIS")
        print("=" * 60)

        if "step_profile" not in self.results:
            print("  No step profile data")
            return

        total_ms = self.results["step_profile"]["total_ms"]
        tokens_per_batch = self.batch_size * self.seq_len
        tokens_per_sec = tokens_per_batch / (total_ms / 1000) if total_ms > 0 else 0

        steps_per_day = (24 * 60 * 60) / (total_ms / 1000) if total_ms > 0 else 0
        days_for_100k = 100000 / steps_per_day if steps_per_day > 0 else float('inf')

        self.results["throughput"] = {
            "tokens_per_sec": round(tokens_per_sec, 1),
            "ms_per_step": round(total_ms, 1),
            "steps_per_day": round(steps_per_day, 0),
            "days_for_100k": round(days_for_100k, 1),
        }

        print(f"  Tokens/sec:      {tokens_per_sec:>8.1f}")
        print(f"  Time/step:       {total_ms:>8.1f} ms")
        print(f"  Steps/day:       {steps_per_day:>8.0f}")
        print(f"  100K steps:      {days_for_100k:>8.1f} days")

    def generate_recommendations(self):
        print("\n💡 RECOMMENDATIONS")
        print("=" * 60)

        recs = []

        if self.results.get("memory", {}).get("utilization_pct", 0) > 85:
            recs.append(("CRITICAL", "High VRAM usage. Consider reducing batch size."))

        if self.results.get("activations", {}).get("has_explosion"):
            recs.append(("CRITICAL", "Activation explosion! Check normalization and residual scaling."))

        if self.results.get("gradients", {}).get("has_nan_grads"):
            recs.append(("CRITICAL", "NaN gradients detected. Training is unstable."))

        if self.results.get("loss_analysis", {}).get("has_nan"):
            recs.append(("CRITICAL", "NaN in logits. Check model initialization."))

        step = self.results.get("step_profile", {})
        if step.get("pct_optimizer", 0) > 30:
            recs.append(("HIGH", f"Optimizer is {step['pct_optimizer']}% of step. Consider reducing muon_ns_steps."))

        if step.get("pct_backward", 0) > 50:
            recs.append(("MEDIUM", f"Backward pass is {step['pct_backward']}% of step. Consider gradient checkpointing."))

        tp = self.results.get("throughput", {}).get("tokens_per_sec", 0)
        if tp > 0 and tp < 50:
            recs.append(("MEDIUM", f"Low throughput ({tp:.1f} tok/s). Check for bottlenecks."))

        if self.cfg.use_mhc:
            recs.append(("INFO", f"MHC enabled: n={self.cfg.mhc_n}, iterations={self.cfg.mhc_iterations}"))

        if self.cfg.use_gradient_checkpointing:
            recs.append(("INFO", "Gradient checkpointing is enabled"))

        self.results["recommendations"] = [{"priority": p, "text": t} for p, t in recs]

        for priority, text in recs:
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "INFO": "🔵"}.get(priority, "⚪")
            print(f"  {icon} {priority}: {text}")

    def save_results(self):
        output_path = Path("logs/deep_profile.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)

        model_info = self.results.get("model_info", {})
        mem = self.results.get("memory", {})
        tp = self.results.get("throughput", {})

        print(f"""
  Model:       {model_info.get('total_params_m', 'N/A')}M params
  VRAM:        {mem.get('forward_mb', 'N/A')} MB ({mem.get('utilization_pct', 'N/A')}% utilized)
  Throughput:  {tp.get('tokens_per_sec', 'N/A')} tokens/sec
  Time/step:   {tp.get('ms_per_step', 'N/A')} ms
  100K steps:  {tp.get('days_for_100k', 'N/A')} days
""")

    def run(self):
        print("\n" + "=" * 60)
        print("🚀 BULBA1 DEEP PROFILER")
        print("=" * 60)

        self.load_config()
        self.init_model()
        self.init_optimizer()

        print(f"\n  Config: {self.cfg_path}")
        print(f"  Batch: {self.batch_size}, Seq: {self.seq_len}")
        print(f"  Model: {self.results['model_info']['total_params_m']}M params")
        print(f"  Device: {self.device}")

        try:
            self.profile_memory()
        except Exception as e:
            print(f"  Memory profile error: {e}")

        try:
            self.profile_full_step()
        except Exception as e:
            print(f"  Step profile error: {e}")

        try:
            self.profile_block_breakdown()
        except Exception as e:
            print(f"  Block breakdown error: {e}")

        try:
            self.analyze_activations()
        except Exception as e:
            print(f"  Activation analysis error: {e}")

        try:
            self.analyze_gradients()
        except Exception as e:
            print(f"  Gradient analysis error: {e}")

        try:
            self.analyze_loss()
        except Exception as e:
            print(f"  Loss analysis error: {e}")

        try:
            self.compute_throughput()
        except Exception as e:
            print(f"  Throughput error: {e}")

        self.generate_recommendations()
        self.save_results()
        self.print_summary()

        return self.results


def main():
    parser = argparse.ArgumentParser(description="Bulba1 Deep Profiler")
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--batch", "-b", type=int, default=4)
    parser.add_argument("--seq", "-s", type=int, default=256)
    parser.add_argument("--warmup", "-w", type=int, default=2)
    parser.add_argument("--profile", "-p", type=int, default=5)
    args = parser.parse_args()

    profiler = DeepProfiler(
        cfg_path=args.config,
        batch_size=args.batch,
        seq_len=args.seq,
        warmup_steps=args.warmup,
        profile_steps=args.profile,
    )

    profiler.run()


if __name__ == "__main__":
    main()