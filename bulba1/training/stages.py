from enum import Enum
from typing import List


class TrainingStage(Enum):
    Warmstart = 0
    DensitySwitch = 1
    BinaryInvasion = 2
    Distillation = 3

    def name(self) -> str:
        names = ["Warmstart", "DensitySwitch", "BinaryInvasion", "Distillation"]
        return names[self.value]

    def lr_multiplier(self, multipliers: List[float] = None) -> float:
        if multipliers is None:
            multipliers = [1.0, 3.33, 1.0, 0.5]
        return multipliers[self.value]


def stage_for_step(step: int, total: int, boundaries: List[float] = None) -> TrainingStage:
    if total == 0:
        return TrainingStage.Warmstart
    if boundaries is None:
        boundaries = [0.25, 0.50, 0.75]
    p = step / total
    if p < boundaries[0]:
        return TrainingStage.Warmstart
    elif p < boundaries[1]:
        return TrainingStage.DensitySwitch
    elif p < boundaries[2]:
        return TrainingStage.BinaryInvasion
    else:
        return TrainingStage.Distillation


def compute_curriculum_seq_len(
    step: int,
    total: int,
    seq_lens: List[int] = None,
    boundaries: List[float] = None,
    target_seq_len: int = 1024,
) -> int:
    if seq_lens is None or len(seq_lens) == 0:
        return target_seq_len
    if boundaries is None:
        boundaries = [0.15, 0.35, 0.60]
    if len(boundaries) >= len(seq_lens):
        boundaries = boundaries[: len(seq_lens) - 1]
    p = step / total if total > 0 else 0
    for i, bound in enumerate(boundaries):
        if p < bound:
            return seq_lens[i]
    return seq_lens[-1]
