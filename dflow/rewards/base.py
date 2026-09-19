"""What a reward is, and how several of them combine.

L3, beside ``encoders/``, under the same rule and for the same reason: **never re-implement
scoring.** A reward model's preprocessing is the archetypal thing that is wrong without raising —
drop the L2 normalisation in front of the aesthetic head and it still emits numbers in roughly the
right range that do not track aesthetics. So ``rewards/`` owns what a training run needs and an
inference script does not (whether to load a scorer at all, placement, batching, lifecycle around a
memory-constrained step) and delegates every computation it can.

## The pixel contract

Every scorer takes **uint8 pixels in [0, 255]**, shape ``(B, C, H, W)``. One explicit contract at
the boundary, because the alternative is the classic silent failure: a scorer handed [0, 1] floats
when it expected [0, 255], or [-1, 1] when it expected [0, 1], produces a valid-looking score from
a black image. The VAE decode and the conversion are the caller's job, which is also where the
model's output range is known.

That is the same convention verl-omni's agent loop uses for its rollout output, so a reward written
against one works against the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Reward(Protocol):
    """A scorer. One scalar per image, higher is better."""

    #: How this component appears in the logs. Distinct within a composite.
    name: str

    def score(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        """``(B,)`` float scores for ``(B, C, H, W)`` uint8 images in [0, 255].

        ``metadata`` carries whatever the task's schema attached per sample — an OCR target
        string, say. A scorer that needs a field it did not get must raise rather than score zero:
        a silently zero reward is indistinguishable from a bad image.
        """
        ...


@dataclass(frozen=True)
class RewardBreakdown:
    """The total, and every component that went into it.

    Components are returned as tensors rather than pre-reduced means so the caller can compute
    group statistics per component. Reducing here would throw away exactly what makes the
    breakdown useful.
    """

    total: torch.Tensor
    components: dict[str, torch.Tensor]

    def means(self) -> dict[str, float]:
        """Per-component means plus the total, ready to log."""
        out = {f"reward/{name}": float(value.mean()) for name, value in self.components.items()}
        out["reward/total"] = float(self.total.mean())
        return out


def validate_pixels(images: torch.Tensor, count: int) -> None:
    """Enforce the pixel contract, loudly.

    ``dtype`` is checked rather than the value range because a range check passes for a float
    tensor that happens to hold small integers — the exact case that needs catching.
    """
    if images.ndim != 4:
        raise ValueError(f"images must be (B, C, H, W), got {tuple(images.shape)}")
    if images.dtype != torch.uint8:
        raise ValueError(
            f"images must be uint8 in [0, 255], got {images.dtype}. Convert at the call site, "
            f"where the model's output range is known — a float tensor silently scores as a "
            f"near-black image."
        )
    if images.shape[0] != count:
        raise ValueError(f"got {images.shape[0]} images for {count} prompts")


class CompositeReward:
    """Several rewards, summed by weight, each reported separately.

    Two rewards pulling against each other is the normal case — aesthetics and text fidelity
    routinely disagree — so an aggregate that moved tells you nothing about which one moved. That
    is why this returns a :class:`RewardBreakdown` rather than a tensor.

    Weights are applied here and nowhere else. A component that scaled itself would make its own
    weight a lie, and the breakdown would no longer sum to the total.
    """

    def __init__(self, components: Sequence[tuple[Reward, float]]) -> None:
        if not components:
            raise ValueError(
                "a composite needs at least one component; RL without a reward optimises nothing"
            )
        names = [reward.name for reward, _ in components]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(
                f"duplicate component names {sorted(duplicates)}: the breakdown is keyed by name, "
                f"so one would overwrite the other in the logs"
            )
        self.components = list(components)

    @property
    def name(self) -> str:
        return "+".join(name for name, _ in ((r.name, w) for r, w in self.components))

    def score(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, Any]],
    ) -> RewardBreakdown:
        validate_pixels(images, len(prompts))
        if len(metadata) != len(prompts):
            raise ValueError(f"got {len(metadata)} metadata entries for {len(prompts)} prompts")

        breakdown: dict[str, torch.Tensor] = {}
        total = torch.zeros(len(prompts), dtype=torch.float32, device=images.device)
        for reward, weight in self.components:
            raw = reward.score(images, prompts, metadata)
            if raw.shape != (len(prompts),):
                raise ValueError(
                    f"{reward.name} returned {tuple(raw.shape)}, expected ({len(prompts)},)"
                )
            raw = raw.float().to(total.device)
            breakdown[reward.name] = raw
            total = total + weight * raw
        return RewardBreakdown(total=total, components=breakdown)


__all__ = ["CompositeReward", "Reward", "RewardBreakdown", "validate_pixels"]
