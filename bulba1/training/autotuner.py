import os
import glob
import math
import shutil
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict


@dataclass
class TunedConfig:
    batch_size: int
    optimizer_type: str
    use_bf16: bool
    use_gradient_checkpointing: bool
    seq_len: int
    estimated_vram_mb: float
    can_fit: bool


@dataclass
class DatasetReport:
    data_dir: str
    total_files: int = 0
    total_size_gb: float = 0.0
    estimated_tokens: int = 0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    recommendations: Dict[str, str] = field(default_factory=dict)


class HardwareAutotuner:
    def __init__(self, device=None):
        if device is None:
            self.device = self._detect_device()
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.total_vram_mb = 0.0
        self.sm_major = 0
        self.has_bf16 = False
        self.has_8bit_adam = False
        self.system_ram_gb = 0.0
        self._probe_hardware()

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _probe_hardware(self):
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.total_vram_mb = props.total_memory / 1024 / 1024
            self.sm_major = props.major
            self.has_bf16 = torch.cuda.is_bf16_supported()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True

        try:
            from bitsandbytes.optim import Adam8bit

            self.has_8bit_adam = True
        except ImportError:
            self.has_8bit_adam = False

        try:
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            self.system_ram_gb = mem / (1024**3)
        except (AttributeError, ValueError):
            self.system_ram_gb = 16.0

    @staticmethod
    def chinchilla_optimal_tokens(num_params: int, ratio: float = 20.0) -> int:
        return int(num_params * ratio)

    @staticmethod
    def recommend_model_size(
        dataset_tokens: int,
        max_params: Optional[int] = None,
        tokens_per_param: float = 20.0,
    ) -> Tuple[int, int, float]:
        optimal_params = int(dataset_tokens / tokens_per_param)
        if max_params is not None and max_params > 0:
            capped_params = min(optimal_params, max_params)
        else:
            capped_params = optimal_params
        actual_ratio = dataset_tokens / max(capped_params, 1)
        return capped_params, optimal_params, actual_ratio

    def analyze_dataset(self, data_dir: str) -> DatasetReport:
        report = DatasetReport(data_dir=data_dir)

        txt_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
        report.total_files = len(txt_files)

        total_bytes = 0
        for f in txt_files:
            try:
                total_bytes += os.path.getsize(f)
            except OSError:
                continue
        report.total_size_gb = total_bytes / (1024**3)

        avg_bytes_per_token = 4.0
        if total_bytes > 0:
            try:
                sample_files = txt_files[: min(5, len(txt_files))]
                sample_bytes = 0
                sample_chars = 0
                for f in sample_files:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read(100_000)
                    sample_bytes += len(content.encode("utf-8"))
                    sample_chars += len(content)
                if sample_chars > 0:
                    avg_bytes_per_token = sample_bytes / max(sample_chars / 4, 1)
            except Exception:
                pass

        report.estimated_tokens = int(total_bytes / max(avg_bytes_per_token, 1))

        usage = shutil.disk_usage(
            data_dir if os.path.isdir(data_dir) else os.path.dirname(data_dir) or "."
        )
        report.disk_free_gb = usage.free / (1024**3)
        report.disk_total_gb = usage.total / (1024**3)

        report.recommendations = self._make_dataset_recommendations(report)
        return report

    def _make_dataset_recommendations(self, report: DatasetReport) -> Dict[str, str]:
        recs = {}

        if report.total_files == 0:
            recs["data"] = "No text files found. Run --prepare-data first."
            return recs

        recs["data"] = (
            f"{report.total_files} files, "
            f"{report.total_size_gb:.1f} GB, "
            f"~{report.estimated_tokens / 1e9:.1f}B tokens estimated"
        )

        if report.disk_free_gb < report.total_size_gb * 2:
            recs["disk"] = (
                f"Low disk space: {report.disk_free_gb:.1f} GB free, "
                f"need ~{report.total_size_gb * 2:.1f} GB for checkpoints + processed data"
            )
        else:
            recs["disk"] = f"Disk OK: {report.disk_free_gb:.1f} GB free"

        min_tokens = 1_000_000_000
        if report.estimated_tokens < min_tokens:
            recs["tokens"] = (
                f"Dataset small ({report.estimated_tokens / 1e6:.0f}M tokens). "
                f"Consider downloading more data or removing --max-items limit."
            )

        if report.total_size_gb > self.system_ram_gb * 0.8:
            recs["memory"] = (
                f"Dataset ({report.total_size_gb:.1f} GB) exceeds 80% of system RAM "
                f"({self.system_ram_gb:.1f} GB). Tokenizer training may be slow."
            )

        capped_params, optimal_params, actual_ratio = self.recommend_model_size(
            report.estimated_tokens, max_params=None, tokens_per_param=20.0
        )

        def fmt_params(n):
            if n >= 1e9:
                return f"{n / 1e9:.2f}B"
            if n >= 1e6:
                return f"{n / 1e6:.0f}M"
            if n >= 1e3:
                return f"{n / 1e3:.0f}K"
            return str(n)

        recs["chinchilla"] = (
            f"Chinchilla-optimal: {fmt_params(optimal_params)} params for {report.estimated_tokens / 1e9:.1f}B tokens. "
            f"Actual ratio: {actual_ratio:.0f}:1"
        )

        if actual_ratio < 10:
            recs["chinchilla_warning"] = (
                f"CRITICAL: Dataset too small for meaningful training. "
                f"Need {self.chinchilla_optimal_tokens(capped_params) / 1e9:.1f}B tokens for {capped_params / 1e6:.0f}M params, "
                f"have {report.estimated_tokens / 1e9:.1f}B. Model will be severely undertrained."
            )
        elif actual_ratio < 20:
            recs["chinchilla_warning"] = (
                f"WARNING: Dataset smaller than Chinchilla-optimal. "
                f"Consider reducing model size to {max(50, int(report.estimated_tokens / 20 / 1e6))}M params "
                f"or collecting more data."
            )
        elif actual_ratio > 200:
            recs["chinchilla_overtraining"] = (
                f"Data-rich: {actual_ratio:.0f} tokens/param. Overtraining mode recommended. "
                f"Model will be inference-optimal."
            )

        return recs

    def estimate_model_vram(
        self,
        num_params: int,
        batch_size: int,
        seq_len: int,
        num_layers: int,
        d_model: int = 1024,
        optimizer_type: str = "adamw",
        safety_factor: float = None,
        overhead_factor: float = None,
        use_mamba: bool = False,
        use_kda: bool = False,
        attn_every_n_layers: int = 4,
        mamba_expand: int = 2,
        mamba_d_state: int = 64,
        use_gradient_checkpointing: bool = False,
    ) -> float:
        if safety_factor is None:
            safety_factor = 0.75
        if overhead_factor is None:
            overhead_factor = 1.35

        param_bytes = 2 if self.has_bf16 else 4
        model_mb = num_params * param_bytes / 1024 / 1024
        grad_mb = num_params * 4 / 1024 / 1024

        if optimizer_type == "adamw":
            opt_state_mb = num_params * 2 * 4 / 1024 / 1024
        elif optimizer_type == "adam8bit":
            opt_state_mb = num_params * 2 * 1 / 1024 / 1024
        else:
            opt_state_mb = num_params * 4 / 1024 / 1024

        num_attn_layers = num_layers // attn_every_n_layers + (
            1 if num_layers % attn_every_n_layers else 0
        )
        num_mamba_layers = num_layers - num_attn_layers

        if use_mamba:
            d_inner = d_model * mamba_expand
            mamba_scan_mb = (
                20 * batch_size * seq_len * d_inner * mamba_d_state * param_bytes / 1024 / 1024
            )
            mamba_other_mb = batch_size * seq_len * d_model * 4 * param_bytes / 1024 / 1024
            attn_act_mb = batch_size * seq_len * d_model * 4 * param_bytes / 1024 / 1024
            kv_cache_mb = (
                batch_size * seq_len * num_attn_layers * d_model * 2 * param_bytes / 1024 / 1024
            )
            if use_kda:
                kv_cache_mb *= 0.25

            if use_gradient_checkpointing:
                # Backward recompute creates forward + backward temporaries.
                # Empirically the peak is ~2.5x the forward peak of one layer.
                stored_output_mb = (
                    num_layers * batch_size * seq_len * d_model * param_bytes / 1024 / 1024
                )
                activation_mb = (
                    2.5 * mamba_scan_mb
                    + mamba_other_mb
                    + attn_act_mb
                    + kv_cache_mb
                    + stored_output_mb
                )
            else:
                activation_mb = (
                    num_mamba_layers * (mamba_scan_mb + mamba_other_mb)
                    + num_attn_layers * attn_act_mb
                    + kv_cache_mb
                )
        else:
            if use_gradient_checkpointing:
                stored_output_mb = (
                    num_layers * batch_size * seq_len * d_model * param_bytes / 1024 / 1024
                )
                per_layer_mb = batch_size * seq_len * d_model * 4 * param_bytes / 1024 / 1024
                activation_mb = per_layer_mb + stored_output_mb
            else:
                activation_mb = (
                    batch_size * seq_len * num_layers * d_model * 4 * param_bytes / 1024 / 1024
                )

        overhead_mb = 3072
        total = (model_mb + grad_mb + opt_state_mb + activation_mb + overhead_mb) * overhead_factor
        return total

    def autotune(self, cfg, actual_params: int = None) -> TunedConfig:
        if actual_params is not None and actual_params > 0:
            num_params = actual_params
        else:
            num_params = cfg.num_params()
        n_layers = cfg.n_layers
        vram_limit = getattr(cfg, "vram_safety_factor", 0.75)
        overhead = getattr(cfg, "vram_overhead_factor", 1.35)
        overhead_mb = getattr(cfg, "vram_overhead_mb", 3072.0)

        if getattr(cfg, "use_bitlinear", False) and cfg.learning_rate < 1e-3:
            cfg.learning_rate = 2e-3

        best_config = None

        # Test batch sizes starting from user's config, then common values
        candidate_batches = [cfg.batch_size]
        for b in [16, 12, 8, 4, 2, 1]:
            if b not in candidate_batches:
                candidate_batches.append(b)
        candidate_batches = [b for b in candidate_batches if b >= 1]

        for batch_size in candidate_batches:
            for use_bf16 in [True, False] if self.has_bf16 else [False]:
                for use_checkpoint in [True, False]:
                    opt_types = [("muon", 3), ("adamw", 2)]
                    if self.has_8bit_adam:
                        opt_types.append(("adam8bit", 1))

                    for opt_type, score_mult in opt_types:
                        est_vram = self.estimate_model_vram(
                            num_params,
                            batch_size,
                            cfg.seq_len,
                            n_layers,
                            cfg.d_model,
                            optimizer_type=opt_type,
                            safety_factor=vram_limit,
                            overhead_factor=overhead,
                            use_mamba=getattr(cfg, "use_mamba", False),
                            use_kda=getattr(cfg, "use_kda", False),
                            attn_every_n_layers=getattr(cfg, "attn_every_n_layers", 4),
                            mamba_expand=getattr(cfg, "mamba_expand", 2),
                            mamba_d_state=getattr(cfg, "mamba_d_state", 64),
                            use_gradient_checkpointing=use_checkpoint,
                        )
                        vram = est_vram * (0.7 if use_bf16 else 1.0)

                        can_fit = vram < self.total_vram_mb * vram_limit

                        if can_fit:
                            score = batch_size * score_mult * (2 if use_bf16 else 1)
                            if getattr(cfg, "use_mamba", False) and use_checkpoint:
                                score += 0.5
                            if best_config is None or score > best_config.get("score", 0):
                                best_config = {
                                    "batch_size": batch_size,
                                    "optimizer_type": opt_type,
                                    "use_bf16": use_bf16,
                                    "use_gradient_checkpointing": use_checkpoint,
                                    "seq_len": cfg.seq_len,
                                    "estimated_vram_mb": vram,
                                    "can_fit": can_fit,
                                    "score": score,
                                }

        if best_config is None:
            return TunedConfig(
                batch_size=1,
                optimizer_type="muon",
                use_bf16=self.has_bf16,
                use_gradient_checkpointing=True,
                seq_len=max(16, cfg.seq_len // 2),
                estimated_vram_mb=self.total_vram_mb,
                can_fit=False,
            )

        return TunedConfig(**{k: v for k, v in best_config.items() if k != "score"})

    def recommend_seq_len(self, cfg, tuned: TunedConfig) -> int:
        max_vram = self.total_vram_mb * getattr(cfg, "vram_safety_factor", 0.75)
        headroom = max_vram - tuned.estimated_vram_mb if tuned.can_fit else 0

        if headroom > 5000:
            return min(cfg.seq_len * 4, cfg.max_ctx_len)
        elif headroom > 2000:
            return min(cfg.seq_len * 2, cfg.max_ctx_len)
        elif headroom > 500:
            return int(cfg.seq_len * 1.5)
        return cfg.seq_len

    def full_analysis(self, cfg, data_dir: str = None) -> Dict:
        result = {}

        result["hardware"] = {
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "vram_mb": self.total_vram_mb,
            "sm": f"{self.sm_major}.x",
            "bf16": self.has_bf16,
            "adam8bit": self.has_8bit_adam,
            "system_ram_gb": self.system_ram_gb,
        }

        if data_dir and os.path.isdir(data_dir):
            report = self.analyze_dataset(data_dir)
            result["dataset"] = asdict(report)

        tuned = self.autotune(cfg)
        recommended_seq = self.recommend_seq_len(cfg, tuned)

        result["training"] = {
            "batch_size": tuned.batch_size,
            "optimizer": tuned.optimizer_type,
            "precision": "BF16" if tuned.use_bf16 else "FP32",
            "gradient_checkpointing": tuned.use_gradient_checkpointing,
            "seq_len": recommended_seq,
            "estimated_vram_mb": tuned.estimated_vram_mb,
            "vram_pct": tuned.estimated_vram_mb / self.total_vram_mb * 100
            if self.total_vram_mb > 0
            else 0,
            "can_fit": tuned.can_fit,
        }

        if tuned.can_fit:
            est_tok_per_sec = (
                tuned.batch_size * recommended_seq * 120
                if tuned.optimizer_type == "muon"
                else tuned.batch_size * recommended_seq * 100
            )
            result["training"]["est_tokens_per_sec"] = est_tok_per_sec
        else:
            result["training"]["warning"] = (
                "Model cannot fit in VRAM with current config. Reduce model size or use CPU."
            )

        num_params = cfg.num_params()
        if data_dir and os.path.isdir(data_dir):
            report = self.analyze_dataset(data_dir)
            max_params = getattr(cfg, "max_model_params", None)
            capped_params, optimal_params, _ = self.recommend_model_size(
                report.estimated_tokens,
                max_params=max_params,
                tokens_per_param=cfg.tokens_per_param_target,
            )
            actual_ratio = report.estimated_tokens / max(num_params, 1)
            result["scaling"] = {
                "model_params": num_params,
                "dataset_tokens": report.estimated_tokens,
                "chinchilla_optimal_params": optimal_params,
                "capped_params": capped_params,
                "tokens_per_param": actual_ratio,
                "is_undertrained": actual_ratio < 20,
                "is_overtrained": actual_ratio > 200,
                "recommendation": (
                    "reduce_model"
                    if actual_ratio < 10
                    else "collect_more_data"
                    if actual_ratio < 20
                    else "overtrain"
                    if actual_ratio > 200
                    else "optimal"
                ),
            }

        return result

    def report(self, tuned: TunedConfig) -> str:
        opt_name = {
            "adamw": "AdamW (FP32 state)",
            "adam8bit": "AdamW 8-bit",
            "muon": "Muon + AdamW (embed/norms)",
        }.get(tuned.optimizer_type, tuned.optimizer_type)

        if self.total_vram_mb > 0:
            vram_pct = tuned.estimated_vram_mb / self.total_vram_mb * 100
            vram_str = f"{vram_pct:.1f}%"
        else:
            vram_str = "N/A"

        lines = [
            f"GPU: {torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU'}",
            f"VRAM: {self.total_vram_mb:.0f} MB",
            f"System RAM: {self.system_ram_gb:.1f} GB",
            f"SM: {self.sm_major}.x",
            f"BF16: {'Yes' if self.has_bf16 else 'No'}",
            f"8-bit Adam: {'Yes' if self.has_8bit_adam else 'No'}",
            f"",
            f"Autotuned config:",
            f"  Batch size: {tuned.batch_size}",
            f"  Optimizer: {opt_name}",
            f"  Precision: {'BF16' if tuned.use_bf16 else 'FP32'}",
            f"  Gradient checkpointing: {'On' if tuned.use_gradient_checkpointing else 'Off'}",
            f"  Estimated VRAM: {tuned.estimated_vram_mb:.0f} MB ({vram_str})",
            f"  Device: {self.device}",
        ]
        return "\n".join(lines)
