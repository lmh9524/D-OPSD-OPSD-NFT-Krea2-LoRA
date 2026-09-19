"""How a sampler decides what conditioning to use.

The most expensive mistakes in this work have all been the same one: sampling a checkpoint under
conditions it was not trained under. It produces plausible images and meaningless numbers, and
nothing raises. `resolve()` is where that is now decided, so it is worth pinning down.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pytest

# `_sampler` imports `dflow`, whose package init re-exports from `models/` and so needs diffusers.
pytest.importorskip("diffusers")

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools" / "krea2"
sys.path.insert(0, str(TOOLS))

from _sampler import FALLBACK, resolve  # noqa: E402

TRAINED = {
    "reference_registration": "disjoint",
    "max_grounded_references": 0,
    "fast_patch_embed": True,
    "ground_references": True,
    "reference_fit_target": True,
    "grounding_max_px": 384,
    "max_length": 1024,
}


def args(**overrides) -> argparse.Namespace:
    unset = dict(
        reference_registration=None, max_grounded_references=None, grounding_max_px=None,
        max_length=None, ground_references=False, fast_patch_embed=False,
        reference_fit_target=False, reference_max_area=None, processor_path=None,
        steps=None, guidance=None, lora_scale=None, negative_prompt=None,
    )
    return argparse.Namespace(**{**unset, **overrides})


@pytest.fixture
def checkpoint(tmp_path):
    def write(record: dict) -> str:
        directory = tmp_path / f"lora{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        (directory / "conditioning.json").write_text(json.dumps(record))
        return str(directory)

    return write


def quiet(*_args, **_kwargs) -> None:
    pass


def test_unset_flags_take_the_training_values(checkpoint) -> None:
    """The whole point: a flag nobody passed must not fall back to a parser default.

    A checkpoint trained with all nine references grounded was evaluated with the parser's default
    of one, and every number and image from that run measured the gap rather than the model.
    """
    settings = resolve(args(), lora=checkpoint(TRAINED), log=quiet)

    assert settings.reference_registration == "disjoint"
    assert settings.max_grounded_references == 0
    assert settings.fast_patch_embed is True
    assert settings.ground_references is True
    assert settings.reference_fit_target is True


def test_an_explicit_flag_still_wins(checkpoint) -> None:
    """Deliberately sampling off-recipe stays possible — it just cannot happen by accident."""
    settings = resolve(
        args(reference_registration="center", max_grounded_references=3),
        lora=checkpoint(TRAINED), log=quiet,
    )

    assert settings.reference_registration == "center"
    assert settings.max_grounded_references == 3
    assert settings.fast_patch_embed is True, "untouched settings still come from the checkpoint"


def test_an_explicit_zero_is_not_mistaken_for_unset(checkpoint) -> None:
    """``max_grounded_references=0`` means *all of them* and is also falsy.

    A truthiness check here would silently replace "ground everything" with whatever the checkpoint
    happened to record — the same class of error as the default that caused the original mismatch,
    only harder to spot.
    """
    settings = resolve(
        args(max_grounded_references=0),
        lora=checkpoint({"max_grounded_references": 5}), log=quiet,
    )

    assert settings.max_grounded_references == 0


def test_a_checkpoint_without_a_record_falls_back_and_stays_conservative(tmp_path) -> None:
    """Checkpoints predating `conditioning.json` must still load, with the defaults they used."""
    settings = resolve(args(), lora=str(tmp_path / "no-such-directory"), log=quiet)

    assert settings.reference_registration == FALLBACK["reference_registration"]
    assert settings.max_grounded_references == FALLBACK["max_grounded_references"]
    assert (settings.ground_references, settings.fast_patch_embed) == (False, False)


def test_no_lora_at_all_is_the_base_model(tmp_path) -> None:
    settings = resolve(args(), lora=None, log=quiet)

    assert settings.reference_registration == "center"
    assert settings.ground_references is False


def test_a_flag_can_enable_what_the_checkpoint_recorded_as_off(checkpoint) -> None:
    settings = resolve(
        args(fast_patch_embed=True), lora=checkpoint({"fast_patch_embed": False}), log=quiet
    )

    assert settings.fast_patch_embed is True
