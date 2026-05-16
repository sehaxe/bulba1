from typing import Any

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup_cache: dict[str, torch.Tensor] = {}
        self._param_names = []

        # Однократно получаем все параметры и их имена
        self._param_dict = dict(model.named_parameters())
        for name, param in self._param_dict.items():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                self._param_names.append(name)

    def update(self, model: nn.Module) -> None:
        """Обновляет EMA-тени, используя кэшированный dict."""
        params = []
        shadows = []
        for name in self._param_names:
            param = self._param_dict[name]  # dict уже создан, O(1) доступ
            shadows.append(self.shadow[name])
            params.append(param.data)

        if not params:
            return

        torch._foreach_mul_(shadows, self.decay)
        torch._foreach_add_(shadows, params, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        for name in self._param_names:
            param = self._param_dict[name]
            if name not in self._backup_cache:
                self._backup_cache[name] = torch.empty_like(param.data)
            self._backup_cache[name].copy_(param.data)
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name in self._param_names:
            param = self._param_dict[name]
            if name in self._backup_cache:
                param.data.copy_(self._backup_cache[name])

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


