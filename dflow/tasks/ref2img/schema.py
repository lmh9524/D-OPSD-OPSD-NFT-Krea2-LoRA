"""Batch contract for multi-image-reference generation.

Both the output contract of ``dataset.py`` and the input contract of ``conditioning.py``. Keeping it in
the same folder as both is what stops the two drifting: a field added to the dataset without a
consumer, or consumed without being produced, shows up in one file.

``references`` is a **list of per-slot tensors**, not one stacked tensor. Aspect ratio is preserved, so
two slots in the same sample can have different extents — and each contributes its own token count to
the concatenated sequence, which is why ``_prepare_image_ids`` takes a list upstream too.

The slot *count* is fixed even though the shapes are not, so the slot -> RoPE-offset mapping is stable.
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch

MULTIPLE_OF = 16


class Ref2ImgBatch(TypedDict):
    """What ``Ref2ImgDataset`` emits after collation."""

    target: torch.Tensor  # (B, 3, H, W) in [-1, 1]
    references: list[torch.Tensor]  # N x (B, 3, Hi, Wi) in [-1, 1]
    prompt: list[str]


def validate(
    batch: dict[str, Any],
    *,
    num_references: int,
    target_max_area: int,
    reference_max_area: int,
) -> None:
    """Assert the batch matches the contract. Test-time only.

    Shapes are checked as *properties* — multiples of 16, within the area cap — rather than against
    fixed numbers, because that is what preserving aspect ratio means.
    """
    missing = {"target", "references", "prompt"} - batch.keys()
    if missing:
        raise AssertionError(f"batch is missing keys: {sorted(missing)}")

    target = batch["target"]
    if not isinstance(target, torch.Tensor) or target.ndim != 4 or target.shape[1] != 3:
        raise AssertionError(f"target should be (B, 3, H, W), got {getattr(target, 'shape', None)}")
    _check_geometry("target", target, target_max_area)

    references = batch["references"]
    if not isinstance(references, list):
        raise AssertionError(f"references should be a list of tensors, got {type(references).__name__}")
    if len(references) != num_references:
        raise AssertionError(f"expected {num_references} reference slots, got {len(references)}")
    for slot, reference in enumerate(references):
        if reference.ndim != 4 or reference.shape[1] != 3:
            raise AssertionError(
                f"references[{slot}] should be (B, 3, H, W), got {tuple(reference.shape)}"
            )
        if reference.shape[0] != target.shape[0]:
            raise AssertionError(
                f"references[{slot}] has batch {reference.shape[0]}, target has {target.shape[0]}"
            )
        _check_geometry(f"references[{slot}]", reference, reference_max_area)

    if len(batch["prompt"]) != target.shape[0]:
        raise AssertionError(f"{len(batch['prompt'])} prompts for {target.shape[0]} targets")


def _check_geometry(name: str, tensor: torch.Tensor, max_area: int) -> None:
    height, width = tensor.shape[-2:]
    if height % MULTIPLE_OF or width % MULTIPLE_OF:
        raise AssertionError(f"{name} is {height}x{width}; both sides must be multiples of 16")
    if height * width > max_area:
        raise AssertionError(f"{name} is {height}x{width} = {height * width} px, over cap {max_area}")
    low, high = float(tensor.min()), float(tensor.max())
    if low < -1.0 - 1e-4 or high > 1.0 + 1e-4:
        raise AssertionError(f"{name} spans [{low:.3f}, {high:.3f}], expected [-1, 1]")


__all__ = ["MULTIPLE_OF", "Ref2ImgBatch", "validate"]
