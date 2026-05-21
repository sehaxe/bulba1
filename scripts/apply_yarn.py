#!/usr/bin/env python3
"""
Apply YaRN context extension to a trained model.
Based on arXiv:2309.00071.

Usage:
    python scripts/apply_yarn.py \\
        --checkpoint checkpoints/bulba1_27m/best.safetensors \\
        --original-len 4096 \\
        --target-len 32768 \\
        --output checkpoints/bulba1_27m_32k/
"""
import argparse
import os
import torch
from bulba1.config import load_config
from bulba1.model.minichat import MiniChat

def main():
    parser = argparse.ArgumentParser(description="Apply YaRN context extension")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/bulba1_27m_optimized.yaml")
    parser.add_argument("--original-len", type=int, default=4096)
    parser.add_argument("--target-len", type=int, default=32768)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=32.0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    print(f"🔧 Loading config from {args.config}...")
    cfg = load_config(args.config)
    
    print(f"🏗️  Initializing model...")
    model = MiniChat(cfg)
    
    print(f"📥 Loading checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    
    print(f"🧶 Applying YaRN: {args.original_len} → {args.target_len} tokens...")
    # Apply YaRN to all RoPE instances in the model
    for module in model.modules():
        if hasattr(module, 'rope') and hasattr(module.rope, 'apply_yarn'):
            module.rope.apply_yarn(
                new_max_len=args.target_len,
                original_len=args.original_len,
                alpha=args.alpha,
                beta=args.beta
            )
    
    # Update config
    cfg.max_ctx_len = args.target_len
    
    print(f"💾 Saving extended model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "best.safetensors")
    torch.save(model.state_dict(), output_path)
    
    # Save updated config
    import yaml
    config_path = os.path.join(args.output, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(cfg.model_dump(), f)
    
    print(f"✅ Done! Model extended to {args.target_len} tokens")
    print(f"   Checkpoint: {output_path}")
    print(f"   Config: {config_path}")

if __name__ == "__main__":
    main()
