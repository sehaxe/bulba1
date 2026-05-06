import math
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, base_layer, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        base_layer.weight.requires_grad = False
        if hasattr(base_layer, "bias") and base_layer.bias is not None:
            base_layer.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out


def inject_lora(model, target_modules=None, r=8, lora_alpha=16, lora_dropout=0.0):
    if target_modules is None:
        target_modules = ["moe", "w1", "w2", "w3", "gate"]

    lora_layers = []
    for name, module in model.named_modules():
        if not any(t in name for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)

        lora = LoRALayer(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        setattr(parent, child_name, lora)
        lora_layers.append(lora)

    return lora_layers


def merge_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)

            base = module.base_layer
            delta = (module.lora_A @ module.lora_B * module.scaling).T
            base.weight.data += delta
            setattr(parent, child_name, base)
