import os
import argparse
import torch
from safetensors.torch import save_file

from bulba1.utils.config import ModelConfig, find_architecture
from bulba1.model.minichat import MiniChat


def export_safetensors(model, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    state_dict = model.state_dict()
    save_file(state_dict, output_path)
    print(f"Exported SafeTensors to {output_path}")
    total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")


def export_llamacpp_investigation():
    print("\n" + "=" * 60)
    print("LLAMA.CPP EXPORT INVESTIGATION")
    print("=" * 60)
    print("""
Findings based on llama.cpp upstream (2026):

1. BITLINEAR (Ternary {-1, 0, +1} weights)
   - llama.cpp GGUF supports: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K,
     IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS,
     IQ3_S, IQ3_M, IQ4_XS, IQ4_NL, MXFP4 (experimental)
   - NO native ternary quantization type exists in the GGUF enum.
   - Workaround: map ternary to Q2_K (2 bits/parameter) or IQ2_XXS.
   - Custom implementation would require adding a new GGML_TYPE and
     corresponding decode kernel in llama.cpp C++.

2. MOE (Mixture of Experts)
   - llama.cpp officially supports MoE (Qwen3.5 MoE path, gate tensors,
     per-layer MoE weights, --n-cpu-moe option).
   - Evidence: Qwen3.5 MoE support commit 39bf692af1c (2026),
     gate_up for all MoE models commit d5278301370,
     n-cpu-moe option commit ec428b02c34.
   - Bulba uses 32 experts, top-2 with BitLinear experts.
   - Workaround: export as dense by materializing active experts, or
     contribute a custom MoE path with top-k routing knobs.

3. DIFFERENTIAL ATTENTION
   - llama.cpp supports standard MHA, GQA, MQA, and RoPE variants
     (MROPE, IMROPE, Vision RoPE via ggml_rope_ext / ggml_rope_multi).
   - NO native differential attention (dual-stream with lambda).
   - Workaround: fuse at export by computing effective attention weights
     as a single stream, or implement custom C++ kernel.

4. MAMBA-2 SSD
   - llama.cpp DOES support Mamba and Mamba-2 officially.
   - Evidence: Mamba initial PR #9126, Mamba-2 commit 5d46babdc2,
     Zamba2 hybrid converter PR #21412.
   - Bulba Mamba is disabled by default; enable in llama.cpp if needed.

5. CLR TOKENS + MTP HEADS
   - Not supported by inference engines.
   - CLR: discard latent prefix tokens for export.
   - MTP: discard auxiliary heads, keep only main lm_head.

RECOMMENDED EXPORT PATH:
1. Export SafeTensors checkpoint (this script does that)
2. Create compatibility config: disable BitLinear (use FP16 weights),
   disable CLR/MTP, keep MoE + DiffAttn + Mamba optional
3. Write custom GGUF converter using gguf-py library
4. For inference in llama.cpp: load GGUF with custom graph patches
5. Alternative: use PyTorch native inference (generate() works today)

CONCRETE NEXT STEPS:
- Implement Bulba->GGUF converter for standard config (FP16, no BitLinear)
- Map: DiffAttn + RoPE + RMSNorm + MoE (dense export) -> LLaMA-like GGUF
- For BitLinear export: investigate Q2_K quantization as proxy for ternary
- For full fidelity: maintain PyTorch inference path alongside any GGUF work
""")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output", type=str, default="export/model.safetensors", help="Output path"
    )
    parser.add_argument("--investigate", action="store_true", help="Print llama.cpp investigation")
    args = parser.parse_args()

    if args.investigate:
        export_llamacpp_investigation()
        return

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    cfg = ModelConfig()
    if "config" in checkpoint:
        for k, v in checkpoint["config"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    model = MiniChat(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Model: {cfg.format_params()} parameters")
    export_safetensors(model, args.output)


if __name__ == "__main__":
    main()
