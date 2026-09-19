"""The FLUX.2 family adapter, against the real transformer.

These tests pin the facts that would otherwise be silent to get wrong: the id layout, the
timestep convention, which span the output covers, and that the parallel spec is derived
from upstream declarations rather than hardcoded.
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import build_ids, build_text_ids

pytest.importorskip("diffusers")

from dflow.models.family import (  # noqa: E402
    Flux2Family,
    find_block_module_names,
    find_keep_fp32_patterns,
)
from dflow.models.family.base import ParallelSpec  # noqa: E402

TARGET_GRID = (2, 3)
REF_GRIDS = ((2, 2), (2, 2))
TARGET_LEN = TARGET_GRID[0] * TARGET_GRID[1]
REF_LEN = sum(h * w for h, w in REF_GRIDS)
TEXT_LEN = 5


@pytest.fixture
def batch(tiny_flux2_config):
    channels = tiny_flux2_config["in_channels"]
    text_dim = tiny_flux2_config["joint_attention_dim"]
    torch.manual_seed(1)
    return {
        "tokens": torch.randn(1, TARGET_LEN + REF_LEN, channels),
        "token_ids": build_ids(target_grid=TARGET_GRID, ref_grids=REF_GRIDS),
        "text_embeds": torch.randn(1, TEXT_LEN, text_dim),
        "text_ids": build_text_ids(TEXT_LEN),
        "timestep": torch.tensor([0.3]),
    }


# ------------------------------------------------------------------ derived metadata


def test_block_names_derived_from_upstream_declaration(tiny_flux2):
    """FLUX.2 has two block lists; both must be found, in forward order."""
    assert find_block_module_names(tiny_flux2) == (
        "transformer_blocks",
        "single_transformer_blocks",
    )


def test_keep_fp32_patterns_are_taken_from_the_model(tiny_flux2):
    """vflow cast blocks wholesale; these are the modules that must not be."""
    assert find_keep_fp32_patterns(tiny_flux2) == ("pos_embed", "norm")


def test_parallel_spec_reports_native_cp_plan(tiny_flux2):
    spec = Flux2Family().parallel_spec(tiny_flux2)
    assert spec.block_module_names == ("transformer_blocks", "single_transformer_blocks")
    assert spec.sequence_dim == 1
    assert spec.has_native_cp_plan, "diffusers ships _cp_plan; we must not hand-write Ulysses"


def test_parallel_spec_rejects_empty_blocks():
    with pytest.raises(ValueError, match="at least one block module"):
        ParallelSpec(block_module_names=(), keep_fp32_patterns=())


def test_block_discovery_fails_loudly_on_a_bare_module():
    import torch.nn as nn

    with pytest.raises(ValueError, match="declares no _no_split_modules"):
        find_block_module_names(nn.Linear(2, 2))


# ---------------------------------------------------------------------- forward pass


def test_output_covers_target_plus_references_with_text_stripped(tiny_flux2, batch):
    """The model removes text tokens itself, so index 0 is the first target token."""
    output = tiny_flux2(**Flux2Family().prepare_inputs(**batch))[0]
    assert output.shape[1] == TARGET_LEN + REF_LEN
    assert output.shape[1] != TEXT_LEN + TARGET_LEN + REF_LEN


def test_prepare_inputs_matches_a_hand_written_call(tiny_flux2, batch):
    family = Flux2Family()
    through_family = tiny_flux2(**family.prepare_inputs(**batch))[0]
    by_hand = tiny_flux2(
        hidden_states=batch["tokens"],
        encoder_hidden_states=batch["text_embeds"],
        timestep=batch["timestep"],
        img_ids=batch["token_ids"],
        txt_ids=batch["text_ids"],
        guidance=None,
        return_dict=False,
    )[0]
    torch.testing.assert_close(through_family, by_hand)


def test_take_target_span(tiny_flux2, batch):
    output = tiny_flux2(**Flux2Family().prepare_inputs(**batch))[0]
    target = Flux2Family.take_target_span(output, TARGET_LEN)
    assert target.shape == (1, TARGET_LEN, batch["tokens"].shape[-1])
    torch.testing.assert_close(target, output[:, :TARGET_LEN])


def test_take_target_span_rejects_overlong_request(tiny_flux2, batch):
    output = tiny_flux2(**Flux2Family().prepare_inputs(**batch))[0]
    with pytest.raises(ValueError, match="exceeds output sequence"):
        Flux2Family.take_target_span(output, output.shape[1] + 1)


def test_reference_count_changes_only_the_sequence_length(tiny_flux2, tiny_flux2_config):
    """Varying reference count is a sequence-length change, not a weight-shape change.

    This is why multi-reference needs no channel surgery and works with plain LoRA.
    """
    family = Flux2Family()
    channels = tiny_flux2_config["in_channels"]
    outputs = []
    for count in (0, 1, 3):
        refs = ((2, 2),) * count
        length = TARGET_LEN + 4 * count
        output = tiny_flux2(
            **family.prepare_inputs(
                tokens=torch.randn(1, length, channels),
                token_ids=build_ids(target_grid=TARGET_GRID, ref_grids=refs),
                text_embeds=torch.randn(1, TEXT_LEN, tiny_flux2_config["joint_attention_dim"]),
                text_ids=build_text_ids(TEXT_LEN),
                timestep=torch.tensor([0.5]),
            )
        )[0]
        outputs.append(output.shape[1])
        assert family.take_target_span(output, TARGET_LEN).shape[1] == TARGET_LEN
    assert outputs == [TARGET_LEN, TARGET_LEN + 4, TARGET_LEN + 12]


# ------------------------------------------------------------------------ validation


def test_timestep_must_be_normalised(batch):
    """FLUX.2's forward multiplies by 1000 itself; 0..1000 in would shift the schedule."""
    family = Flux2Family()
    with pytest.raises(ValueError, match="normalised to \\[0, 1\\]"):
        family.prepare_inputs(**{**batch, "timestep": torch.tensor([300.0])})


def test_ids_must_be_four_dimensional(batch):
    family = Flux2Family()
    three_axis = batch["token_ids"][..., :3]
    with pytest.raises(ValueError, match="4-D"):
        family.prepare_inputs(**{**batch, "token_ids": three_axis})


def test_ids_must_align_with_tokens(batch):
    family = Flux2Family()
    with pytest.raises(ValueError, match="must align with tokens"):
        family.prepare_inputs(**{**batch, "token_ids": batch["token_ids"][:, :-1]})


def test_text_ids_must_align_with_text_embeds(batch):
    family = Flux2Family()
    with pytest.raises(ValueError, match="must align with text_embeds"):
        family.prepare_inputs(**{**batch, "text_ids": batch["text_ids"][:, :-1]})


def test_timestep_must_be_per_sample(batch):
    family = Flux2Family()
    with pytest.raises(ValueError, match="timestep must be"):
        family.prepare_inputs(**{**batch, "timestep": torch.tensor(0.3)})


def test_guidance_requirement_is_read_from_the_config():
    """Passing guidance=None to a guidance-embedded model is accepted but changes conditioning."""
    assert not Flux2Family.requires_guidance({"guidance_embeds": False})
    assert Flux2Family.requires_guidance({"guidance_embeds": True})
    assert not Flux2Family.requires_guidance({})


# --------------------------------------------------------------------- noise schedule


def test_mu_comes_from_the_family_not_the_scheduler():
    """How the shift is derived is a model property, so it lives here.

    Verified against pipeline_flux2_klein.py:815: image_seq_len is latents.shape[1], i.e. the
    target only. Reference latents are a separate variable, concatenated inside the loop.
    """
    from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

    family = Flux2Family()
    for tokens, steps in [(4096, 50), (1024, 4), (4096, 200)]:
        assert family.noise_shift_mu(
            image_tokens=tokens, inference_steps=steps
        ) == compute_empirical_mu(image_seq_len=tokens, num_steps=steps)


def test_default_mu_is_resolution_only_and_equals_flux1_calculate_shift():
    """The step-independent term of FLUX.2's fit *is* FLUX.1's calculate_shift.

    Slope (1.15 - 0.5) / (4096 - 256) == 1.6927e-4 == a2; intercept 0.5 - a2 * 256 == b2. So the
    default is the canonical resolution schedule these models train at (shift 3.16 at 4096
    tokens), not an approximation of it. The step term is inference discretisation.
    """
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    family = Flux2Family()
    for tokens in (256, 1024, 4096):
        assert family.noise_shift_mu(image_tokens=tokens) == pytest.approx(
            calculate_shift(tokens), abs=1e-4
        )


def test_explicit_step_count_pushes_mu_up():
    """Few-step schedules need a higher shift, which is why it is opt-in rather than default.

    Training at the 50-step value would concentrate samples at far higher noise (shift 7.6)
    than FLUX.1 or SD3 use (3.16).
    """
    family = Flux2Family()
    resolution_only = family.noise_shift_mu(image_tokens=4096)
    fifty = family.noise_shift_mu(image_tokens=4096, inference_steps=50)
    four = family.noise_shift_mu(image_tokens=4096, inference_steps=4)

    assert resolution_only == pytest.approx(1.1500, abs=1e-3)
    assert fifty == pytest.approx(2.0234, abs=1e-3)
    assert four == pytest.approx(2.2912, abs=1e-3)
    assert resolution_only < fifty < four


def test_the_default_fixed_shift_is_the_dynamic_value_at_a_1024px_target():
    """The constant is not a taste call -- it is what dynamic produces at 4096 tokens.

    That identity is the whole argument for it: a full-size target trains identically under
    either mode, so ``--flow-match.shift None`` changes nothing for those samples and only
    lowers the shift for smaller ones. If the default drifts off this value, the reasoning
    in ``config/diffusion.py`` stops holding and the flag becomes a bare preference.
    """
    import math

    from dflow.config import FlowMatchConfig

    family = Flux2Family()
    default = FlowMatchConfig().shift
    assert default is not None
    assert math.log(default) == pytest.approx(family.noise_shift_mu(image_tokens=4096), abs=1e-4)


def test_mu_jumps_past_the_empirical_branch():
    """Above ~4300 tokens the fit changes branch and stops depending on step count.

    A 1024x1024 target is 4096 tokens and safe; 1152x1152 would be 5184 and would not be.
    """
    family = Flux2Family()
    below = family.noise_shift_mu(image_tokens=4296, inference_steps=50)
    above = family.noise_shift_mu(image_tokens=4356, inference_steps=50)
    assert abs(below - above) > 0.5
    assert family.noise_shift_mu(image_tokens=4356, inference_steps=4) == above
