"""Krea 2 family adapter, checked against a real (tiny) transformer.

Same approach as ``test_family_flux2.py``: a genuinely small ``Krea2Transformer2DModel`` rather than
a mock, so the calling convention is exercised for real — unbatched 3-axis position ids, text
prepended into the sequence and stripped from the output, the internal x1000 timestep scaling.

The reference-conditioning tests here assert *mechanics*, not learned behaviour. No Krea 2 checkpoint
has been trained with a non-zero T coordinate or a second image span, so what these pin is that the
sequence we build is the one we intend — the model's ability to use it is what training decides.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from dflow.models.family import Krea2Family  # noqa: E402
from dflow.models.family.krea2 import DEFAULT_LORA_TARGETS  # noqa: E402
from dflow.tasks.ref2img.conditioning import (  # noqa: E402
    build_krea2_latent_ids,
    build_krea2_sequence,
    build_krea2_text_ids,
)

TEXT_LEN = 4
TEXT_LAYERS = 3
TEXT_DIM = 16
CHANNELS = 16


def _latents(height: int, width: int, *, batch: int = 1) -> torch.Tensor:
    torch.manual_seed(height * 100 + width)
    return torch.randn(batch, CHANNELS, height, width)


def _text(batch: int = 1) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(batch, TEXT_LEN, TEXT_LAYERS, TEXT_DIM)


@pytest.fixture
def family():
    return Krea2Family()


# ------------------------------------------------------------------ derived metadata


def test_block_names_derived_from_upstream_declaration(tiny_krea2, family):
    """Only ``transformer_blocks`` is a ModuleList child; the text fusion blocks are nested."""
    spec = family.parallel_spec(tiny_krea2)
    assert spec.block_module_names == ("transformer_blocks",)


def test_keep_fp32_patterns_are_taken_from_the_model(tiny_krea2, family):
    patterns = family.parallel_spec(tiny_krea2).keep_fp32_patterns
    # Union of _keep_in_fp32_modules and _skip_layerwise_casting_patterns.
    assert {"norm", "norm1", "norm2", "norm_q", "norm_k", "time_embed"} <= set(patterns)


def test_krea2_ships_no_context_parallel_plan(tiny_krea2, family):
    """FLUX.2 ships ``_cp_plan``; Krea 2 does not, so CP cannot simply be switched on."""
    assert family.parallel_spec(tiny_krea2).has_native_cp_plan is False


def test_lora_targets_include_the_attention_gate():
    """``to_gate`` has no FLUX.2 counterpart and is the knob reference conditioning needs."""
    assert "attn.to_gate" in DEFAULT_LORA_TARGETS


def test_lora_targets_match_real_module_names(tiny_krea2):
    names = {name for name, _ in tiny_krea2.named_modules()}
    for target in DEFAULT_LORA_TARGETS:
        assert any(name.endswith(target) for name in names), target


# ---------------------------------------------------------------------- forward pass


def test_prepare_inputs_matches_a_hand_written_call(tiny_krea2, family):
    """The adapter must produce exactly the call a reader would write by hand."""
    latents = _latents(2, 2)
    sequence = build_krea2_sequence(target_latents=latents, reference_latents=[])
    text = _text()
    text_ids = build_krea2_text_ids(TEXT_LEN, device=latents.device)
    timestep = torch.tensor([0.5])

    kwargs = family.prepare_inputs(
        tokens=sequence.tokens,
        token_ids=sequence.ids,
        text_embeds=text,
        text_ids=text_ids,
        timestep=timestep,
    )
    with torch.no_grad():
        through_adapter = tiny_krea2(**kwargs)[0]
        by_hand = tiny_krea2(
            hidden_states=sequence.tokens,
            encoder_hidden_states=text,
            timestep=timestep,
            position_ids=torch.cat([text_ids, sequence.ids], dim=0),
            encoder_attention_mask=None,
            return_dict=False,
        )[0]
    torch.testing.assert_close(through_adapter, by_hand)


def test_output_covers_the_image_span_with_text_stripped(tiny_krea2, family):
    """The model removes its own text span, so the output starts at the first target token."""
    latents = _latents(2, 2)
    references = [_latents(1, 2)]
    sequence = build_krea2_sequence(target_latents=latents, reference_latents=references)
    assert sequence.target_len == 4
    assert sequence.tokens.shape[1] == 4 + 2

    with torch.no_grad():
        output = tiny_krea2(
            **family.prepare_inputs(
                tokens=sequence.tokens,
                token_ids=sequence.ids,
                text_embeds=_text(),
                text_ids=build_krea2_text_ids(TEXT_LEN, device=latents.device),
                timestep=torch.tensor([0.5]),
            )
        )[0]
    assert output.shape[1] == sequence.tokens.shape[1]
    assert family.take_target_span(output, sequence.target_len).shape[1] == 4


def test_reference_count_changes_only_the_sequence_length(tiny_krea2, family):
    """References enter by concatenation, so no weight shape depends on how many there are."""
    target = _latents(2, 2)
    lengths = []
    for count in (0, 1, 2):
        sequence = build_krea2_sequence(
            target_latents=target, reference_latents=[_latents(1, 2)] * count
        )
        with torch.no_grad():
            output = tiny_krea2(
                **family.prepare_inputs(
                    tokens=sequence.tokens,
                    token_ids=sequence.ids,
                    text_embeds=_text(),
                    text_ids=build_krea2_text_ids(TEXT_LEN, device=target.device),
                    timestep=torch.tensor([0.5]),
                )
            )[0]
        lengths.append(output.shape[1])
        assert family.take_target_span(output, sequence.target_len).shape[1] == 4
    assert lengths == [4, 6, 8]


def test_encoder_attention_mask_reaches_the_model(tiny_krea2, family):
    """Krea 2 pads in the middle of its template, so masking must change the result."""
    latents = _latents(2, 2)
    sequence = build_krea2_sequence(target_latents=latents, reference_latents=[])
    text = _text()
    text_ids = build_krea2_text_ids(TEXT_LEN, device=latents.device)
    mask = torch.ones(1, TEXT_LEN, dtype=torch.bool)
    mask[:, -1] = False

    common = dict(
        tokens=sequence.tokens,
        token_ids=sequence.ids,
        text_embeds=text,
        text_ids=text_ids,
        timestep=torch.tensor([0.5]),
    )
    with torch.no_grad():
        unmasked = tiny_krea2(**family.prepare_inputs(**common))[0]
        masked = tiny_krea2(**family.prepare_inputs(**common, text_mask=mask))[0]
    assert not torch.allclose(unmasked, masked)


# ---------------------------------------------------------------------- id layout


def test_target_sits_at_the_origin_of_the_t_axis():
    ids = build_krea2_latent_ids(_latents(2, 3), frame=0)
    assert ids.shape == (6, 3)
    assert torch.equal(ids[:, 0], torch.zeros(6))
    # Row-major (h, w), matching the pipeline's grid construction.
    assert torch.equal(ids[:, 1], torch.tensor([0.0, 0, 0, 1, 1, 1]))
    assert torch.equal(ids[:, 2], torch.tensor([0.0, 1, 2, 0, 1, 2]))


def test_references_use_frames_1_and_2_not_flux2_spacing():
    """Krea 2's T axis has only ever seen 0, and its fastest rotary component turns 1 rad/unit.

    FLUX.2's spacing of 10 per reference is safe there because klein was trained with it; here it
    lands past the wrap point (T=7) at an arbitrary phase. Frames 1 and 2 are a small extrapolation
    from the trained point, and are what the working community edit LoRA uses.
    """
    sequence = build_krea2_sequence(
        target_latents=_latents(2, 2), reference_latents=[_latents(1, 1), _latents(1, 1)]
    )
    t_axis = sequence.ids[:, 0]
    assert float(t_axis[0]) == 1.0  # reference 0
    assert float(t_axis[1]) == 2.0  # reference 1
    assert torch.equal(t_axis[2:], torch.zeros(4))  # target, last


def test_the_target_span_comes_last():
    """`[refs | target]`, matching the layout the edit LoRAs were trained on.

    ``take_target_span`` must then slice the tail; taking the head would return reference tokens as
    the prediction and train against the wrong pixels without raising.
    """
    references = [_latents(1, 1), _latents(1, 1)]
    sequence = build_krea2_sequence(target_latents=_latents(2, 2), reference_latents=references)
    assert sequence.target_len == 4
    assert sequence.target_offset == 2
    assert sequence.tokens.shape[1] == 6

    output = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    span = Krea2Family().take_target_span(output, 4, target_offset=sequence.target_offset)
    assert span.flatten().tolist() == [2.0, 3.0, 4.0, 5.0]


def test_a_smaller_reference_is_centred_in_the_target_grid():
    """Reference H/W is registered to the target, not restarted at (0, 0).

    A reference token at (h, w) then shares its spatial coordinate with the target token at (h, w),
    which is what makes "keep this region" cheap to learn. The offset is fractional because RoPE is
    continuous — an integer floor puts an odd-sized reference half a token off centre.
    """
    ids = build_krea2_latent_ids(_latents(2, 2), frame=1, target_grid=(4, 5))
    # (4-2)/2 = 1.0 exactly; (5-2)/2 = 1.5, the fractional half.
    assert torch.equal(ids[:, 1], torch.tensor([1.0, 1.0, 2.0, 2.0]))
    assert torch.equal(ids[:, 2], torch.tensor([1.5, 2.5, 1.5, 2.5]))


def test_an_oversized_reference_is_not_centred():
    """Centring a reference larger than the target would push coordinates negative."""
    ids = build_krea2_latent_ids(_latents(4, 4), frame=1, target_grid=(2, 2))
    assert float(ids[:, 1].min()) == 0.0
    assert float(ids[:, 2].min()) == 0.0


def test_ids_are_unbatched():
    """The model raises unless ``position_ids.ndim == 2``, so the builder must not add a batch."""
    sequence = build_krea2_sequence(
        target_latents=_latents(2, 2, batch=1), reference_latents=[_latents(1, 1, batch=1)]
    )
    assert sequence.ids.ndim == 2
    assert sequence.ids.shape == (5, 3)


def test_text_ids_are_all_zero():
    ids = build_krea2_text_ids(4, device=torch.device("cpu"))
    assert ids.shape == (4, 3)
    assert torch.equal(ids, torch.zeros(4, 3))


# ---------------------------------------------------------------------- validation


def test_batched_ids_that_disagree_are_rejected(family):
    """Silently taking element 0 would train the rest of the batch on the wrong geometry."""
    ids = torch.zeros(2, 4, 3)
    ids[1, :, 1] = 5.0
    with pytest.raises(ValueError, match="differs across the batch"):
        family.prepare_inputs(
            tokens=torch.randn(2, 4, CHANNELS),
            token_ids=ids,
            text_embeds=_text(batch=2),
            text_ids=build_krea2_text_ids(TEXT_LEN, device=torch.device("cpu")),
            timestep=torch.tensor([0.5, 0.5]),
        )


def test_flattened_text_embeds_are_rejected(family):
    """FLUX.2's (B, L, 3H) is a different layout, and would otherwise fail deep inside the model."""
    latents = _latents(2, 2)
    sequence = build_krea2_sequence(target_latents=latents, reference_latents=[])
    with pytest.raises(ValueError, match="stacked text hidden states"):
        family.prepare_inputs(
            tokens=sequence.tokens,
            token_ids=sequence.ids,
            text_embeds=torch.randn(1, TEXT_LEN, TEXT_LAYERS * TEXT_DIM),
            text_ids=build_krea2_text_ids(TEXT_LEN, device=latents.device),
            timestep=torch.tensor([0.5]),
        )


def test_guidance_is_refused(family):
    """Krea 2 has no guidance embedder; accepting the argument would drop it silently."""
    latents = _latents(2, 2)
    sequence = build_krea2_sequence(target_latents=latents, reference_latents=[])
    with pytest.raises(ValueError, match="no guidance argument"):
        family.prepare_inputs(
            tokens=sequence.tokens,
            token_ids=sequence.ids,
            text_embeds=_text(),
            text_ids=build_krea2_text_ids(TEXT_LEN, device=latents.device),
            timestep=torch.tensor([0.5]),
            guidance=torch.tensor([3.5]),
        )


def test_timestep_must_be_normalised(family):
    latents = _latents(2, 2)
    sequence = build_krea2_sequence(target_latents=latents, reference_latents=[])
    with pytest.raises(ValueError, match="normalised to \\[0, 1\\]"):
        family.prepare_inputs(
            tokens=sequence.tokens,
            token_ids=sequence.ids,
            text_embeds=_text(),
            text_ids=build_krea2_text_ids(TEXT_LEN, device=latents.device),
            timestep=torch.tensor([500.0]),
        )


# ------------------------------------------------------------------- noise schedule


def test_mu_matches_the_pipeline_for_the_undistilled_checkpoint():
    from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift

    family = Krea2Family(distilled=False)
    for tokens in (256, 1024, 4096, 6400):
        assert family.noise_shift_mu(image_tokens=tokens) == pytest.approx(
            calculate_shift(tokens, 256, 6400, 0.5, 1.15), abs=1e-6
        )


def test_the_distilled_checkpoint_pins_mu_at_every_resolution():
    """Turbo's pipeline hardcodes ``mu = 1.15``, so resolution must not move it."""
    family = Krea2Family(distilled=True)
    values = {family.noise_shift_mu(image_tokens=n) for n in (256, 1024, 4096, 6400)}
    assert values == {1.15}


def test_the_default_fixed_shift_equals_turbo_s_own_mu():
    """``FlowMatchConfig.shift``'s default is ``exp(1.15)`` — Turbo's schedule exactly.

    Training Turbo under the default fixed shift therefore matches its inference schedule at every
    resolution, which is the one case where fixed is unambiguously right.
    """
    import math

    from dflow.config import FlowMatchConfig

    default = FlowMatchConfig().shift
    assert default is not None
    turbo_mu = Krea2Family(distilled=True).noise_shift_mu(image_tokens=1)
    assert math.log(default) == pytest.approx(turbo_mu, abs=1e-4)


def test_step_count_is_ignored_rather_than_pretended_to_matter():
    """Krea 2's shift has no step term at all, unlike FLUX.2's empirical fit."""
    family = Krea2Family(distilled=False)
    assert family.noise_shift_mu(image_tokens=4096) == family.noise_shift_mu(
        image_tokens=4096, inference_steps=4
    )


# ---------------------------------------------------------------------- registry


def test_both_checkpoints_are_registered_with_the_right_schedule():
    from dflow.models.registry import resolve

    assert resolve("krea2-turbo").family.distilled is True
    assert resolve("krea2-raw").family.distilled is False
    assert resolve("krea2-turbo").default_repo == "krea/Krea-2-Turbo"


def test_krea2_taps_twelve_text_layers():
    """Twelve, not FLUX.2's three — and stacked on their own axis rather than flattened."""
    from dflow.models.registry import resolve

    assert len(resolve("krea2-turbo").text_out_layers) == 12
    assert resolve("krea2-turbo").text_out_layers == resolve("krea2-raw").text_out_layers


def test_only_one_place_integrates_the_denoise_loop() -> None:
    """No tool may carry its own copy of the reference-conditioned denoise loop.

    One did. `tools/krea2/sample_krea2.py` kept a private loop written for an earlier
    `[target | references]` token layout, and under the current `[references | target]` layout it
    took `tokens[:, :target_len]` — the reference *head* — as the span to denoise, never touched the
    noise it had sampled, and fed token order that disagreed with the position ids so every token
    drew another token's RoPE phase. Nothing raised: `_unpack_latents` reshapes whatever it is
    handed and eight Euler steps push the result back onto the model's image manifold, so it wrote
    plausible photographs with nothing to do with the references. Three rounds of evaluation
    reported that the model ignored them.

    A source-level assertion rather than a behavioural one, because the failure mode *is* drift
    between two implementations: the durable fix is that the second one does not exist. Checked
    across the whole directory rather than one file, since the copy can reappear anywhere.
    """
    import pathlib

    tools = pathlib.Path(__file__).resolve().parents[2] / "tools" / "krea2"
    integrates = {
        path.name for path in tools.glob("*.py") if "scheduler.step(" in path.read_text()
    }
    assert integrates == set(), (
        f"{sorted(integrates)} step a scheduler directly; go through "
        "dflow.tasks.ref2img.sampling.denoise"
    )

    imports_denoise = {
        path.name
        for path in tools.glob("*.py")
        if "from dflow.tasks.ref2img.sampling import denoise" in path.read_text()
    }
    assert imports_denoise == {"_sampler.py"}, (
        f"denoise should be reached through _sampler.py alone, but {sorted(imports_denoise)} "
        "import it"
    )
    # The head slice is the specific mistake that layout change introduced. It must not come back.
    for path in tools.glob("*.py"):
        assert "[:, : sequence.target_len]" not in path.read_text(), (
            f"{path.name} slices the head; the target comes last, use target_offset"
        )



def test_sampling_defaults_differ_between_raw_and_turbo() -> None:
    """The two checkpoints are not interchangeable, and the wrong pair hides a working adapter.

    krea-ai/krea-2's README gives them explicitly: Raw is ``--steps 52 --cfg 3.5``, Turbo is
    ``--steps 8 --cfg 0.0``. A LoRA is trained on Raw and applied on Turbo, so both are routinely
    in play in one run. Rendering Raw at Turbo's settings produces a generic, condition-weak image
    whatever the adapter learned -- which is exactly how an in-training preview misreported this
    run's progress for a full day.
    """
    from dflow.models.family.krea2 import Krea2Family

    raw = Krea2Family(distilled=False).sampling_defaults()
    turbo = Krea2Family(distilled=True).sampling_defaults()
    assert (raw["steps"], raw["guidance"]) == (52, 3.5)
    assert (turbo["steps"], turbo["guidance"]) == (8, 0.0)
    assert Krea2Family(distilled=False).sampling_defaults() is not raw, "must return a fresh dict"


def test_reference_registration_modes_place_references_differently() -> None:
    """``disjoint`` must leave no reference token sharing an H row with the target.

    ``center`` was the only mode, copied from a recipe whose references are the target's own size
    and shape. For product cutouts it makes every garment's H coordinate claim "torso height":
    correct for a top, twelve rows wrong for trousers, and nowhere near the feet for shoes. That is
    a positional signal contradicting the caption, and it is why lower-body items transferred worst.
    """
    import torch

    from dflow.tasks.ref2img.conditioning import build_krea2_latent_ids

    reference = torch.zeros(1, 64, 24, 10)  # a tall narrow cutout, like trousers
    target_grid = (42, 24)

    centred = build_krea2_latent_ids(reference, frame=1, target_grid=target_grid)
    origin = build_krea2_latent_ids(reference, frame=1, target_grid=target_grid, registration="origin")
    disjoint = build_krea2_latent_ids(
        reference, frame=1, target_grid=target_grid, registration="disjoint"
    )

    assert (centred[:, 1].min(), centred[:, 1].max()) == (9.0, 32.0)
    assert (origin[:, 1].min(), origin[:, 1].max()) == (0.0, 23.0)
    # Past the target's last row (41), so no H coordinate is shared with any target token.
    assert disjoint[:, 1].min() == 42.0
    assert disjoint[:, 1].max() == 65.0
    # W stays centred in every mode that registers at all: left-right does carry meaning.
    assert disjoint[:, 2].min() == centred[:, 2].min() == 7.0
    assert (disjoint[:, 0] == 1).all(), "the T axis still separates references"


def test_lora_export_records_the_conditioning_a_sampler_must_reproduce(tmp_path, monkeypatch) -> None:
    """A checkpoint has to carry how it was conditioned, because remembering has failed twice.

    First a LoRA trained with image-grounded conditioning was sampled text-only, and three rounds of
    evaluation measured the mismatch rather than the model. Then a run trained with nine grounded
    references was evaluated with the sampler's default of one, voiding its numbers and its images.
    Neither raised: sampling under different conditioning produces perfectly plausible pictures that
    simply say nothing about the checkpoint.

    The weight export is stubbed out — this is about the sidecar, and building a real PEFT adapter
    would test diffusers rather than the thing that keeps being got wrong.
    """
    from torch import nn

    from dflow.checkpoint import lora_io
    from dflow.checkpoint.lora_io import CONDITIONING_NAME, load_conditioning, save_lora

    monkeypatch.setattr(lora_io, "adapter_state_dict", lambda model, adapter_name="default": {})
    import diffusers.loaders.lora_pipeline as pipeline

    monkeypatch.setattr(
        pipeline.Flux2LoraLoaderMixin, "save_lora_weights",
        staticmethod(lambda **kwargs: None),
    )

    settings = {
        "reference_registration": "disjoint",
        "max_grounded_references": 0,
        "fast_patch_embed": True,
    }
    save_lora(nn.Linear(4, 4), tmp_path, conditioning=settings)

    assert (tmp_path / CONDITIONING_NAME).is_file()
    assert load_conditioning(tmp_path) == settings
    # A checkpoint written before this existed reads as empty rather than raising, so directories
    # from earlier runs stay loadable.
    assert load_conditioning(tmp_path / "does-not-exist") == {}

    # Omitting it writes no file at all, rather than an empty one a sampler would trust.
    other = tmp_path / "no-record"
    save_lora(nn.Linear(4, 4), other)
    assert not (other / CONDITIONING_NAME).exists()


def test_t_scale_spaces_references_on_the_frame_axis() -> None:
    """The T axis is what tells one reference from another, and it is learned from scratch.

    Krea 2 has only ever seen T at zero, so nothing pretrained constrains how far apart the values
    sit. Adjacent integers are the smallest separation available; wider spacing gives each reference
    a more distinct rotary phase and pushes the last one further outside the range the model knows.
    """
    import torch

    from dflow.tasks.ref2img.conditioning import build_krea2_sequence

    def frames(t_scale: int) -> list[float]:
        sequence = build_krea2_sequence(
            target_latents=torch.zeros(1, 64, 8, 6),
            reference_latents=[torch.zeros(1, 64, 4, 4)] * 3,
            registration="disjoint", t_scale=t_scale,
        )
        return sorted(sequence.ids[:, 0].unique().tolist())

    assert frames(1) == [0.0, 1.0, 2.0, 3.0]
    assert frames(10) == [0.0, 10.0, 20.0, 30.0]
    # The target stays at frame 0 whatever the spacing: it is the only frame pretraining has seen.
    assert frames(10)[0] == 0.0
