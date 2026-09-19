"""L3: reward scorers. Lifecycle, not scoring.

Beside ``encoders/`` and under the same rule, for the same reason: **never re-implement scoring.**
A reward model's preprocessing is the archetypal silent failure — drop the L2 normalisation in
front of the aesthetic head and it emits plausible numbers uncorrelated with aesthetics. So this
package owns what a training run needs and an inference script does not (whether to load a scorer
at all, placement, batching, lifecycle around a memory-constrained step) and delegates the rest.

Every scorer takes **uint8 pixels in [0, 255]**, shape ``(B, C, H, W)``. See ``base.py`` for why
that contract is explicit rather than inferred.

``build_reward`` is the one entry point an experiment needs; the individual classes are exported
for tests and for a run that wants to inject an engine.
"""

from __future__ import annotations

import torch

from dflow.config.rl import RewardConfig
from dflow.rewards.aesthetic import AestheticReward
from dflow.rewards.base import CompositeReward, Reward, RewardBreakdown, validate_pixels
from dflow.rewards.http import HTTPReward
from dflow.rewards.ocr import EasyOCREngine, OCREngine, OCRReward
from dflow.rewards.reference_fidelity import ReferenceFidelityReward


def build_reward(
    config: RewardConfig,
    *,
    device: torch.device,
    ocr_engine: OCREngine | None = None,
    extra: list[tuple[Reward, float]] | None = None,
) -> CompositeReward:
    """Assemble the enabled components into one composite.

    ``extra`` is how an :class:`HTTPReward` joins in — anything too large to share the card is
    constructed by the experiment, which is where its URL lives, and passed here so it still gets
    the weighted sum and the per-component logging.

    ``RewardConfig.__post_init__`` already refuses a config with nothing enabled, so the only way
    to reach an empty composite is to pass ``extra=None`` alongside it, which ``CompositeReward``
    refuses in turn.
    """
    components: list[tuple[Reward, float]] = []

    aesthetic = AestheticReward.load(config.aesthetic, device=device)
    if aesthetic is not None:
        components.append((aesthetic, config.aesthetic.weight))

    ocr = OCRReward.load(config.ocr, device=device, engine=ocr_engine)
    if ocr is not None:
        components.append((ocr, config.ocr.weight))

    reference_fidelity = ReferenceFidelityReward.load(config.reference_fidelity, device=device)
    if reference_fidelity is not None:
        components.append((reference_fidelity, config.reference_fidelity.weight))

    components.extend(extra or [])
    return CompositeReward(components)


__all__ = [
    "AestheticReward",
    "CompositeReward",
    "EasyOCREngine",
    "HTTPReward",
    "OCREngine",
    "OCRReward",
    "ReferenceFidelityReward",
    "Reward",
    "RewardBreakdown",
    "build_reward",
    "validate_pixels",
]
