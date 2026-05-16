"""
HuggingFace integration for Bulba1.
Allows loading/saving models via transformers API.
"""

from typing import Optional
import torch
import torch.nn as nn
from pathlib import Path


class BulbaConfig:
    """HuggingFace-compatible config for Bulba1."""
    
    def __init__(
        self,
        vocab_size: int = 26000,
        d_model: int = 512,
        n_layers: int = 10,
        n_heads: int = 8,
        use_moe: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = d_model
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.intermediate_size = d_model * 4
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def to_dict(self):
        return self.__dict__.copy()


class BulbaForCausalLM(nn.Module):
    """HuggingFace-compatible model for causal LM."""
    
    def __init__(self, config: BulbaConfig):
        super().__init__()
        self.config = config
        
        from bulba1.config import ModelConfig
        from bulba1.model.minichat import MiniChat
        
        # Convert HF config to Bulba config
        bulba_cfg = ModelConfig(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            use_moe=config.use_moe,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )
        
        self.model = MiniChat(bulba_cfg)
        self.vocab_size = config.vocab_size
    
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_p=0.9):
        """Simple greedy generation."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _, _, _ = self.model(input_ids)
                next_token = logits[:, -1, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        return input_ids
    
    @classmethod
    def from_pretrained(cls, path: str):
        """Load model from checkpoint."""
        import safetensors
        config_path = Path(path) / "config.json"
        weights_path = Path(path) / "model.safetensors"
        
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            config = BulbaConfig(**config_dict)
        else:
            config = BulbaConfig()
        
        model = cls(config)
        
        if weights_path.exists():
            state = safetensors.torch.load_file(str(weights_path))
            model.model.load_state_dict(state, strict=False)
        
        return model
    
    def save_pretrained(self, path: str):
        """Save model to checkpoint."""
        import safetensors
        from transformers import PreTrainedModel
        from huggingface_hub import HfApi
        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save config
        import json
        with open(Path(path) / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f)
        
        # Save weights
        safetensors.torch.save_file(
            self.model.state_dict(),
            Path(path) / "model.safetensors"
        )
        
        # Create model card
        readme = """# Bulba1 Model

Autonomous LLM training framework with MoE, KDA, and efficient attention.

## Usage
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("your-user/bulba1")
```
"""
        with open(Path(path) / "README.md", "w") as f:
            f.write(readme)


def convert_to_hf(checkpoint_path: str, output_path: str):
    """Convert Bulba checkpoint to HuggingFace format."""
    import safetensors
    
    state = safetensors.torch.load_file(checkpoint_path)
    
    config = {
        "vocab_size": 26000,
        "hidden_size": 512,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
    }
    
    import json
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "config.json", "w") as f:
        json.dump(config, f)
    
    safetensors.torch.save_file(state, Path(output_path) / "model.safetensors")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_to_hf(args.checkpoint, args.output)