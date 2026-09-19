"""Shared vocabulary for tasks.

A task owns *semantics*: which fields to read, how to pair references, how a batch becomes model
kwargs, and how loss is weighted. Mechanism — samplers, bucketing, retry, collation — lives in
``data/`` and is shared.

There is deliberately no base class to inherit. A task is a folder of functions and a schema that
``experiments/`` imports explicitly; nothing calls back into it. The only thing shared here is the
batch-validation helper, so every task's schema test says the same thing the same way.
"""

from __future__ import annotations

from typing import Any

import torch


def validate_batch(
    batch: dict[str, Any],
    *,
    tensors: dict[str, tuple[int, ...]],
    lists: tuple[str, ...] = (),
    value_range: tuple[float, float] | None = None,
) -> None:
    """Assert a batch matches its schema.

    Called from tests, not from the training loop: per-batch validation costs real time at 1024x1024,
    and a schema violation is a code bug rather than a data condition.

    ``tensors`` maps a key to its expected shape, with ``-1`` for any dimension that may vary.
    """
    missing = (set(tensors) | set(lists)) - batch.keys()
    if missing:
        raise AssertionError(f"batch is missing keys: {sorted(missing)}")

    for key, expected in tensors.items():
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise AssertionError(f"{key!r} should be a tensor, got {type(value).__name__}")
        if len(value.shape) != len(expected):
            raise AssertionError(
                f"{key!r} has {len(value.shape)} dims, expected {len(expected)}: {expected}"
            )
        for axis, (actual, want) in enumerate(zip(value.shape, expected, strict=True)):
            if want != -1 and actual != want:
                raise AssertionError(
                    f"{key!r} axis {axis} is {actual}, expected {want} "
                    f"(got {tuple(value.shape)}, expected {expected})"
                )
        if value_range is not None:
            low, high = value_range
            if float(value.min()) < low - 1e-4 or float(value.max()) > high + 1e-4:
                raise AssertionError(
                    f"{key!r} spans [{float(value.min()):.3f}, {float(value.max()):.3f}], "
                    f"expected [{low}, {high}]"
                )

    for key in lists:
        if not isinstance(batch[key], list):
            raise AssertionError(f"{key!r} should be a list, got {type(batch[key]).__name__}")


__all__ = ["validate_batch"]
