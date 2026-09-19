"""Optimizer and LR schedule construction."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from dflow.config import LRSchedulerConfig, OptimizerConfig


def build_optimizer(
    config: OptimizerConfig, parameters: Iterable[torch.nn.Parameter]
) -> Optimizer:
    """Build the optimizer over *trainable* parameters only.

    Filtering is the caller's job, because "which parameters train" is a LoRA-versus-full
    decision and passing a frozen parameter here would allocate moment buffers for it —
    twice the parameter's size in fp32, for nothing.
    """
    parameters = [p for p in parameters]
    if not parameters:
        raise ValueError("no parameters to optimise; check that LoRA injection ran")
    if config.type != "adamw":
        raise ValueError(f"unsupported optimizer {config.type!r}")

    # fused AdamW is CUDA-only; fall back silently rather than failing on CPU test runs.
    fused = config.fused and parameters[0].is_cuda
    return torch.optim.AdamW(
        parameters,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=fused,
    )


def build_lr_scheduler(
    config: LRSchedulerConfig, optimizer: Optimizer, *, total_steps: int
) -> LRScheduler:
    """Warmup then constant/linear/cosine decay, as a plain ``LambdaLR``.

    ``LambdaLR`` keeps ``state_dict()`` trivially serialisable, which matters because the
    schedule position is one of the six things a checkpoint must round-trip: a silently reset
    schedule changes the effective learning rate without any error.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= config.warmup_steps <= total_steps:
        raise ValueError(
            f"warmup_steps={config.warmup_steps} must be between 0 and total_steps={total_steps}"
        )
    if not 0.0 <= config.min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    def lr_lambda(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return float(step + 1) / config.warmup_steps
        if config.type == "constant":
            return 1.0

        decay_steps = max(1, total_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, (step - config.warmup_steps) / decay_steps))
        if config.type == "linear":
            factor = 1.0 - progress
        elif config.type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"unsupported lr scheduler {config.type!r}")
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * factor

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


__all__ = ["build_lr_scheduler", "build_optimizer"]
