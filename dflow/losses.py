"""Losses shared across tasks.

Only genuinely cross-task ones belong here. Task-specific weighting — a mask emphasis for inpainting,
say — lives with the task, because it is part of what that task *means*.

``flow_mse`` qualifies: text-to-image and multi-reference compute exactly the same thing. The reason
it is worth a function rather than two inline lines is the weighting. Forgetting to apply it, or
broadcasting it along the wrong axis, changes the effective loss without raising.
"""

from __future__ import annotations

import torch


def flow_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted mean squared error between a predicted and true velocity.

    Args:
        prediction: ``(B, seq, C)`` model output, already sliced to the span under supervision.
        target: the same shape — ``noise - clean`` from the scheduler.
        weights: ``(B,)`` per-sample weights from ``FlowMatchScheduler.weighting``. Broadcast over the
            trailing dimensions here rather than at the call site, since getting that reshape wrong is
            silent.

    Computed in fp32 regardless of the autocast dtype: the squared error of bf16 residuals loses
    precision exactly where the loss is smallest.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction {tuple(prediction.shape)} and target {tuple(target.shape)} must match. "
            f"Slice the prediction to the supervised span before calling."
        )

    squared = (prediction.float() - target.float()) ** 2
    if weights is None:
        return squared.mean()

    if weights.ndim != 1 or weights.shape[0] != prediction.shape[0]:
        raise ValueError(
            f"weights must be (B,) with B={prediction.shape[0]}, got {tuple(weights.shape)}"
        )
    shaped = weights.float().reshape(-1, *([1] * (prediction.ndim - 1)))
    return (shaped * squared).mean()


__all__ = ["flow_mse"]
