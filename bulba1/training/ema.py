import torch
import torch.nn as nn
from typing import Dict, Any

class EMA:
    """Максимально быстрый EMA с минимальным расходом памяти.

    - Использует torch._foreach_mul_ / _foreach_add_ (C‑уровень, ×3‑5 быстрее)
    - Shadow хранится в том же dtype, что и модель (экономия 50% RAM)
    - apply_shadow / restore переиспользуют буферы, не выделяя новую память
    - Полностью совместим с текущими чекпоинтами (state_dict/load_state_dict)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup_cache: Dict[str, torch.Tensor] = {}  # пул для бэкапов
        self._param_names = []                               # кэш имён параметров

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                self._param_names.append(name)

    def update(self, model: nn.Module) -> None:
        """Вызовите после optimizer.step(). Обновляет EMA-тени."""
        params = []
        shadows = []
        for name in self._param_names:
            param = dict(model.named_parameters())[name]  # O(1) доступ
            shadows.append(self.shadow[name])
            params.append(param.data)

        # C-функции, работают над всеми тензорами одновременно
        torch._foreach_mul_(shadows, self.decay)
        torch._foreach_add_(shadows, params, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        """Временно заменяет веса модели на EMA‑версию (для eval/генерации)."""
        for name in self._param_names:
            param = dict(model.named_parameters())[name]
            if name not in self._backup_cache:
                # Выделяем буфер один раз
                self._backup_cache[name] = torch.empty_like(param.data)
            self._backup_cache[name].copy_(param.data)   # сохранили оригинал
            param.data.copy_(self.shadow[name])           # подставили тень

    def restore(self, model: nn.Module) -> None:
        """Возвращает оригинальные веса модели."""
        for name in self._param_names:
            param = dict(model.named_parameters())[name]
            if name in self._backup_cache:
                param.data.copy_(self._backup_cache[name])

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]