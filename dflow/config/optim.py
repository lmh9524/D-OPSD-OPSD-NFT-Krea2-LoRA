"""Optimizer and LR-schedule configuration.

Pure declaration. This module must not import anything else from ``dflow``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class OptimizerConfig:
    type: Literal["adamw"] = "adamw"
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    fused: bool = True


@dataclass(kw_only=True, slots=True)
class LRSchedulerConfig:
    type: Literal["constant", "linear", "cosine"] = "constant"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0


__all__ = ["LRSchedulerConfig", "OptimizerConfig"]
