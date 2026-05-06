# bulba1/model/ — Model Architecture

**Score:** 18 (high complexity)

## OVERVIEW

10 files: Transformer blocks, attention, MoE, Mamba, quantization.

## FILES

| File | Role |
|------|------|
| `minichat.py` | Full model (Bulba1Chat) |
| `block.py` | TransformerBlock |
| `diff_attn.py` | Differential attention + RoPE |
| `moe.py` | MoE + ReX routing |
| `mamba.py` | Mamba-2 SSD |
| `mamba_cuda.py` | CUDA kernel |
| `bit_linear.py` | BitNet ternary quantization |
| `kda.py` | Kimi Delta Attention |
| `mhc.py` | Hyper-connections |
| `__init__.py` | Exports |

## CONVENTIONS

- All torch.nn.Module subclasses
- forward() returns tuple (output, ...) for intermediate states
- Use `F.scaled_dot_product_attention` for flash attention

## NOTES

- Requires CUDA for Mamba CUDA kernel
- BitLinear uses ternarize_weights(): -1, 0, +1

## ANTI-PATTERNS (THIS PROJECT)

- Avoid circular imports between blocks