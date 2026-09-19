"""Batch collation.

Split out because the natural defaults are wrong here in two ways.

``default_collate`` stacks strings into a list-of-characters when they differ in length, so prompts
must stay a list of strings.

And a field holding a *list of tensors per sample* — multi-reference images, whose extents differ
between slots because aspect ratio is preserved — has to be stacked **per slot**, giving one tensor
per slot rather than one tensor for the field. Stacking naively would demand that every slot in a
sample share a shape, which is exactly what preserving aspect ratio gives up.
"""

from __future__ import annotations

from typing import Any

import torch


def _stack(values: list[torch.Tensor], key: str) -> torch.Tensor:
    shapes = {tuple(v.shape) for v in values}
    if len(shapes) != 1:
        raise ValueError(
            f"cannot stack {key!r}: mixed shapes {sorted(shapes)}. Aspect ratio is preserved, so "
            f"samples differ in extent — use batch_size=1, or bucket by shape (data/bucket.py)."
        )
    return torch.stack(values)


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors, stack lists-of-tensors per slot, keep everything else as a list."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = set(samples[0])
    for sample in samples[1:]:
        if set(sample) != keys:
            raise ValueError(f"inconsistent batch keys: {sorted(keys)} vs {sorted(sample)}")

    batch: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        first = values[0]

        if isinstance(first, torch.Tensor):
            batch[key] = _stack(values, key)
        elif isinstance(first, list) and first and isinstance(first[0], torch.Tensor):
            slots = len(first)
            if any(len(value) != slots for value in values):
                raise ValueError(
                    f"{key!r} has {slots} slots in one sample and "
                    f"{[len(v) for v in values]} across the batch; the count must be fixed"
                )
            batch[key] = [
                _stack([value[slot] for value in values], f"{key}[{slot}]")
                for slot in range(slots)
            ]
        else:
            batch[key] = values
    return batch


__all__ = ["collate"]
