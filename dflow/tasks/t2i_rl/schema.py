"""Batch contract for text-to-image RL. No target images.

Both the output contract of ``dataset.py`` and the input contract of every reward, which is why the
OCR target string lives here: ``rewards/ocr.py`` reads it out of ``metadata`` by the key
``OCRRewardConfig.target_key`` names, and this file is the other end of that agreement. Same folder
as the dataset that produces it, so the two cannot drift.

The shape of an RL batch differs from SFT's in one structural way: **there is no supervision
target.** A prompt and whatever a reward needs to judge the result is the whole sample. That is why
this is a task of its own rather than ``ImageFolderDataset`` with the images ignored.
"""

from __future__ import annotations

from typing import Any, TypedDict


class T2IRLBatch(TypedDict):
    """What ``PromptDataset`` emits after collation."""

    prompt: list[str]
    #: Per-sample extras the rewards read. Empty dicts for a purely aesthetic run.
    metadata: list[dict[str, Any]]


def validate(batch: dict[str, Any], *, required_keys: tuple[str, ...] = ()) -> None:
    """Assert the batch matches the contract. Test-time only, per ``SKILL.md``.

    ``required_keys`` is what the enabled rewards need — pass ``("text",)`` for an OCR run. Checked
    here rather than in the reward because a missing field discovered mid-run has already cost a
    rollout, and because a reward that defaulted it would score zero and look like a bad image.
    """
    missing = {"prompt", "metadata"} - batch.keys()
    if missing:
        raise AssertionError(f"batch is missing keys: {sorted(missing)}")

    prompts, metadata = batch["prompt"], batch["metadata"]
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise AssertionError("prompt should be a list of strings")
    if not isinstance(metadata, list) or len(metadata) != len(prompts):
        raise AssertionError(
            f"metadata should be one dict per prompt, got {len(metadata)} for {len(prompts)}"
        )
    for index, entry in enumerate(metadata):
        if not isinstance(entry, dict):
            raise AssertionError(f"metadata[{index}] should be a dict, got {type(entry).__name__}")
        absent = set(required_keys) - entry.keys()
        if absent:
            raise AssertionError(f"metadata[{index}] is missing {sorted(absent)}")


__all__ = ["T2IRLBatch", "validate"]
