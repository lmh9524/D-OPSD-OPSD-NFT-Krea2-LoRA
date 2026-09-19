"""Reward scorers: the pixel contract, the composite, and the OCR normalisation.

What is checked here without downloading 1.7 GB of CLIP:

* **the pixel contract.** A float tensor where uint8 was expected scores a near-black image and
  says nothing, so ``validate_pixels`` rejects it on dtype rather than on value range — a range
  check passes for floats that happen to hold small integers.
* **the aesthetic head's architecture**, against the shapes read from the published checkpoint. The
  ``nn.Sequential`` indices are part of the state-dict contract: dropouts at 1, 3 and 5 hold no
  parameters but occupy positions.
* **the OCR normalisation**, which is the part that is ours. Case folding, whitespace and box
  joining each move the reward and none of them raise, so specific pairs are asserted.
* **the composite**, whose weighted sum must equal the reported breakdown.

The real CLIP weights are exercised by ``test_aesthetic_real.py``, marked ``slow``.
"""

from __future__ import annotations

import pytest
import torch

# Guarded like every other dflow-importing test file. ``dflow/__init__.py`` is the public API
# surface and re-exports from ``models/``, so importing anything under ``dflow.`` pulls in
# diffusers — even here, where nothing under test needs it. Without this the CI fast job, which
# installs no diffusers and is the gate that must stay green, errors at collection instead of
# skipping.
pytest.importorskip("diffusers")

from dflow.config.rl import AestheticRewardConfig, OCRRewardConfig, RewardConfig  # noqa: E402
from dflow.rewards.aesthetic import EMBED_DIM, HEAD_SHAPES, AestheticReward, build_head  # noqa: E402
from dflow.rewards.base import CompositeReward, Reward, validate_pixels  # noqa: E402
from dflow.rewards.ocr import OCRReward, normalise, similarity  # noqa: E402

pytest.importorskip("Levenshtein", reason="the OCR reward's edit distance is an optional extra")


def pixels(count: int = 3, size: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (count, 3, size, size), dtype=torch.uint8)


class ConstantReward:
    """A ``Reward`` that scores what it was told to."""

    def __init__(self, name: str, values: list[float]) -> None:
        self.name = name
        self.values = values

    def score(self, images, prompts, metadata) -> torch.Tensor:
        return torch.tensor(self.values, dtype=torch.float32)


class FakeOCREngine:
    """An ``OCREngine`` returning a scripted reading per image."""

    def __init__(self, readings: list[list[str]]) -> None:
        self.readings = readings
        self.calls = 0

    def read(self, image: torch.Tensor) -> list[str]:
        reading = self.readings[self.calls]
        self.calls += 1
        return reading


# ------------------------------------------------------------------------- the pixel contract


def test_float_pixels_are_refused_on_dtype():
    """A float tensor in [0, 1] would score as a near-black image, silently."""
    with pytest.raises(ValueError, match="uint8"):
        validate_pixels(pixels().float() / 255.0, 3)


def test_a_wrong_rank_is_refused():
    with pytest.raises(ValueError, match=r"\(B, C, H, W\)"):
        validate_pixels(pixels()[0], 1)


def test_a_count_mismatch_is_refused():
    with pytest.raises(ValueError, match="for 5 prompts"):
        validate_pixels(pixels(3), 5)


def test_uint8_pixels_pass():
    validate_pixels(pixels(2), 2)


# ------------------------------------------------------------------------------ the composite


def test_the_total_is_the_weighted_sum_of_the_breakdown():
    composite = CompositeReward(
        [(ConstantReward("a", [1.0, 2.0, 3.0]), 2.0), (ConstantReward("b", [10.0, 0.0, -1.0]), 0.5)]
    )
    breakdown = composite.score(pixels(3), ["x", "y", "z"], [{}, {}, {}])
    torch.testing.assert_close(breakdown.components["a"], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(breakdown.components["b"], torch.tensor([10.0, 0.0, -1.0]))
    torch.testing.assert_close(breakdown.total, torch.tensor([7.0, 4.0, 5.5]))


def test_every_component_is_reported_separately():
    """Two rewards pulling against each other is the normal case; an aggregate hides which moved."""
    composite = CompositeReward(
        [(ConstantReward("aesthetic", [5.0]), 1.0), (ConstantReward("ocr", [0.25]), 1.0)]
    )
    means = composite.score(pixels(1), ["x"], [{}]).means()
    assert means == pytest.approx(
        {"reward/aesthetic": 5.0, "reward/ocr": 0.25, "reward/total": 5.25}
    )


def test_duplicate_component_names_are_refused():
    """The breakdown is keyed by name, so one would overwrite the other in the logs."""
    with pytest.raises(ValueError, match="duplicate component names"):
        CompositeReward([(ConstantReward("a", [1.0]), 1.0), (ConstantReward("a", [2.0]), 1.0)])


def test_an_empty_composite_is_refused():
    with pytest.raises(ValueError, match="at least one component"):
        CompositeReward([])


def test_a_component_returning_the_wrong_shape_is_refused():
    composite = CompositeReward([(ConstantReward("a", [1.0, 2.0]), 1.0)])
    with pytest.raises(ValueError, match="expected"):
        composite.score(pixels(3), ["x", "y", "z"], [{}, {}, {}])


def test_a_metadata_length_mismatch_is_refused():
    composite = CompositeReward([(ConstantReward("a", [1.0]), 1.0)])
    with pytest.raises(ValueError, match="metadata entries"):
        composite.score(pixels(1), ["x"], [])


def test_a_config_with_nothing_enabled_is_refused():
    with pytest.raises(ValueError, match="no reward is enabled"):
        RewardConfig()


def test_the_constant_reward_satisfies_the_protocol():
    assert isinstance(ConstantReward("a", [1.0]), Reward)


# ----------------------------------------------------------------------- the aesthetic head


def test_the_head_matches_the_published_checkpoint_exactly():
    """Shapes *and* ``nn.Sequential`` indices, read from the real ``aesthetic-model.pth``.

    Build the right layer widths in the wrong positions and ``load_state_dict`` succeeds against a
    renumbered module, which is the failure this pins.
    """
    state = build_head().state_dict()
    assert tuple((k, tuple(v.shape)) for k, v in state.items()) == HEAD_SHAPES


def test_the_head_consumes_clip_l14_projection_width():
    assert EMBED_DIM == 768
    assert build_head().layers[0].in_features == EMBED_DIM


def test_the_head_maps_a_batch_of_embeddings_to_one_scalar_each():
    head = build_head().eval()
    assert head(torch.zeros(5, EMBED_DIM)).shape == (5, 1)


def test_a_disabled_aesthetic_reward_loads_as_none():
    """``None`` rather than a stub, so the not-loaded case stays visible at the call site."""
    assert AestheticReward.load(AestheticRewardConfig(), device=torch.device("cpu")) is None


def test_an_invalid_aesthetic_batch_size_is_refused():
    with pytest.raises(ValueError, match="batch_size"):
        AestheticRewardConfig(batch_size=0)


# ------------------------------------------------------------------------ the OCR normalisation


@pytest.mark.parametrize(
    ("detected", "target", "expected"),
    [
        ("hello", "hello", 1.0),
        ("HELLO", "hello", 1.0),  # case folded by default
        ("  hello   world ", "hello world", 1.0),  # whitespace collapsed
        ("hella", "hello", 0.8),  # one substitution in five
        ("hell", "hello", 0.8),  # one deletion, normalised by the longer string
        ("", "hello", 0.0),
        ("xxxxx", "hello", 0.0),
    ],
)
def test_similarity_matches_specific_pairs(detected, target, expected):
    config = OCRRewardConfig(enabled=True)
    assert similarity(detected, target, config) == pytest.approx(expected, abs=1e-6)


def test_case_sensitivity_is_a_real_switch():
    cased = OCRRewardConfig(enabled=True, case_sensitive=True)
    assert similarity("HELLO", "hello", cased) < 1.0
    assert similarity("HELLO", "hello", OCRRewardConfig(enabled=True)) == 1.0


def test_whitespace_collapsing_is_a_real_switch():
    kept = OCRRewardConfig(enabled=True, collapse_whitespace=False)
    assert similarity("hello  world", "hello world", kept) < 1.0
    assert similarity("hello  world", "hello world", OCRRewardConfig(enabled=True)) == 1.0


def test_both_sides_go_through_the_same_folding():
    """Folding one side only makes every score wrong in a way that looks like a weak model."""
    config = OCRRewardConfig(enabled=True)
    assert normalise("  HELLO   World ", config) == "hello world"
    assert similarity("hello world", "  HELLO   World ", config) == 1.0


def test_an_empty_target_is_refused_rather_than_scoring_one():
    """It would score 1.0 for every image and silently disable the reward."""
    with pytest.raises(ValueError, match="silently disable"):
        similarity("anything", "   ", OCRRewardConfig(enabled=True))


def test_the_ocr_reward_joins_detected_boxes_with_the_configured_separator():
    config = OCRRewardConfig(enabled=True, join="")
    reward = OCRReward(FakeOCREngine([["HEL", "LO"]]), config)
    scores = reward.score(pixels(1), ["x"], [{"text": "hello"}])
    assert float(scores[0]) == pytest.approx(1.0)

    spaced = OCRReward(FakeOCREngine([["HEL", "LO"]]), OCRRewardConfig(enabled=True))
    assert float(spaced.score(pixels(1), ["x"], [{"text": "hello"}])[0]) < 1.0


def test_the_ocr_reward_scores_one_image_at_a_time():
    reward = OCRReward(
        FakeOCREngine([["cat"], ["dog"], ["cat"]]), OCRRewardConfig(enabled=True)
    )
    scores = reward.score(pixels(3), ["a", "b", "c"], [{"text": "cat"}] * 3)
    assert scores.shape == (3,)
    torch.testing.assert_close(scores[0], torch.tensor(1.0))
    assert float(scores[1]) < 1.0
    torch.testing.assert_close(scores[2], torch.tensor(1.0))


def test_a_missing_target_field_raises_rather_than_scoring_zero():
    """Zero is a real score, indistinguishable from an image with no text in it."""
    reward = OCRReward(FakeOCREngine([["cat"]]), OCRRewardConfig(enabled=True))
    with pytest.raises(KeyError, match="nothing to compare"):
        reward.score(pixels(1), ["a"], [{"caption": "cat"}])


def test_a_disabled_ocr_reward_loads_as_none():
    assert OCRReward.load(OCRRewardConfig(), device=torch.device("cpu")) is None


def test_an_injected_engine_bypasses_the_optional_dependency():
    """How an HTTP-backed or test engine gets in without installing the extra."""
    engine = FakeOCREngine([["x"]])
    reward = OCRReward.load(
        OCRRewardConfig(enabled=True), device=torch.device("cpu"), engine=engine
    )
    assert reward is not None and reward.engine is engine


def test_an_empty_target_key_is_refused():
    with pytest.raises(ValueError, match="target_key"):
        OCRRewardConfig(target_key="")
