"""Text-rendering reward: ``1 - normalised edit distance(detected, target)``.

The flow_grpo OCR task. A prompt asks for an image containing a given string, the image is read
back, and the reward is how close the reading is to what was asked for.

## The engine is injected; the normalisation is ours

Detection and recognition are delegated, through the :class:`OCREngine` protocol. That is not
indecision about which engine to use — it is where the memory budget forces the split. A VLM judge
of the kind verl-omni uses (``genrm_ocr.py``, a Qwen3-VL) is ~15 GB and cannot sit alongside a
52 GiB training step, so anything that large belongs behind ``rewards/http.py``. A dedicated
detector-plus-recogniser is a few hundred megabytes and fits; :class:`EasyOCREngine` is one, in the
optional ``[ocr]`` extra so the dependency is not forced on runs that do not score text.

**What is genuinely wrong-without-raising here is the normalisation, not the model.** Case folding,
whitespace handling, and how multiple detected boxes are joined each move the reward substantially,
and every choice produces a valid score. So they are explicit config fields, :func:`normalise` is
one tested function, and ``tests/rewards/test_ocr.py`` asserts specific
``(detected, target) -> score`` pairs rather than checking that a number comes out.

## Why the edit distance is not hand-rolled

Normalised edit distance is a dozen lines of dynamic programming, and an off-by-one in the
recurrence shifts every reward by a constant without failing. ``Levenshtein`` is a tested
implementation; ``difflib`` is stdlib but computes a matching-block ratio, which is a *different
quantity* that happens to look similar. So the dependency is real, lazily imported, and in the
``[ocr]`` extra.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import torch

from dflow.config.rl import OCRRewardConfig
from dflow.rewards.base import validate_pixels


@runtime_checkable
class OCREngine(Protocol):
    """Reads text out of one image. Reading order is the engine's business."""

    def read(self, image: torch.Tensor) -> list[str]:
        """Detected text boxes for a single ``(C, H, W)`` uint8 image, in reading order."""
        ...


def normalise(text: str, config: OCRRewardConfig) -> str:
    """Apply the configured folding. The one place either side of the comparison is transformed.

    Both the detection and the target go through this, which is the property that matters: folding
    one and not the other makes every score wrong in a way that looks like a weak model.
    """
    if not config.case_sensitive:
        text = text.casefold()
    if config.collapse_whitespace:
        text = " ".join(text.split())
    return text


def similarity(detected: str, target: str, config: OCRRewardConfig) -> float:
    """``1 - normalised edit distance`` in [0, 1], after folding both sides.

    An empty target is a configuration error, not a free point: it would score 1.0 for every image
    and quietly turn the reward off.
    """
    from Levenshtein import distance

    left = normalise(detected, config)
    right = normalise(target, config)
    if not right:
        raise ValueError(
            "the OCR target is empty after normalisation, which would score 1.0 for every image "
            f"and silently disable the reward. Check the {config.target_key!r} metadata field."
        )
    if not left:
        return 0.0
    return 1.0 - distance(left, right) / max(len(left), len(right))


class OCRReward:
    """Scores how faithfully an image renders the string its prompt asked for."""

    name = "ocr"

    def __init__(self, engine: OCREngine, config: OCRRewardConfig) -> None:
        self.engine = engine
        self.config = config

    @classmethod
    def load(
        cls,
        config: OCRRewardConfig,
        *,
        device: torch.device,
        engine: OCREngine | None = None,
    ) -> OCRReward | None:
        """Load the reward, or return ``None`` when disabled.

        ``engine`` overrides the default, which is what tests and an HTTP-backed setup use.
        """
        if not config.enabled:
            return None
        if engine is None:
            engine = EasyOCREngine.load(config, device=device)
        return cls(engine, config)

    def score(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        validate_pixels(images, len(prompts))
        key = self.config.target_key
        scores: list[float] = []
        for index, entry in enumerate(metadata):
            if key not in entry:
                raise KeyError(
                    f"sample {index} has no {key!r} field, so there is nothing to compare the "
                    f"reading against. A scorer that returned 0 here would be indistinguishable "
                    f"from an image with no text in it."
                )
            boxes = self.engine.read(images[index])
            scores.append(similarity(self.config.join.join(boxes), str(entry[key]), self.config))
        return torch.tensor(scores, dtype=torch.float32, device=images.device)


class EasyOCREngine:
    """``easyocr``, in the optional ``[ocr]`` extra.

    A detector plus a recogniser at a few hundred megabytes, so it can share a card with training —
    which is the whole reason it is the default rather than something more accurate.
    """

    def __init__(self, reader: Any) -> None:
        self.reader = reader

    @classmethod
    def load(cls, config: OCRRewardConfig, *, device: torch.device) -> EasyOCREngine:
        try:
            import easyocr
        except ImportError as error:
            raise ImportError(
                "the OCR reward needs an engine. Either install the extra "
                "(`pip install -e '.[ocr]'`) or pass `engine=` — an HTTP-backed scorer, say, for "
                "anything too large to share the card."
            ) from error

        reader = easyocr.Reader(list(config.languages), gpu=device.type == "cuda")
        return cls(reader)

    def read(self, image: torch.Tensor) -> list[str]:
        array = image.permute(1, 2, 0).cpu().numpy()
        return [text for _, text, _ in self.reader.readtext(array)]


__all__ = ["EasyOCREngine", "OCREngine", "OCRReward", "normalise", "similarity"]
