from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Minimal config – all values must be provided via YAML."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def format_params(self, n: Optional[int] = None) -> str:
        n = n or 0
        if n >= 1_000_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        elif n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        return f"{n / 1_000:.1f}K"