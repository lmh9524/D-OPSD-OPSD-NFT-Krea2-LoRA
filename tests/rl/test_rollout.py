"""The recording denoise loop: what it records, and what it must leave alone.

The failure modes here are alignment failures, and none of them raise. Recording the post-step
state as a step's entry shifts every action one place. Evaluating the velocity at the exit sigma
instead of the entry sigma trains against a field the rollout never sampled. Taking ``latents[-1]``
as the clean output silently scores a partially denoised image whenever the window ends early.

So the tests below pin the *correspondence* between recorded arrays, not their shapes alone — and
the shape checks in ``RolloutGroup.__post_init__`` are tested too, because they are what turns an
off-by-one in the recording into an exception rather than a wrong number.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

# Guarded like every other dflow-importing test file. ``dflow/__init__.py`` is the public API
# surface and re-exports from ``models/``, so importing anything under ``dflow.`` pulls in
# diffusers — even here, where nothing under test needs it. Without this the CI fast job, which
# installs no diffusers and is the gate that must stay green, errors at collection instead of
# skipping.
pytest.importorskip("diffusers")

from dflow.config.rl import SDEConfig  # noqa: E402
from dflow.rl.protocol import SequenceLike  # noqa: E402
from dflow.rl.rollout import sample_group, sample_window  # noqa: E402
from dflow.rl.trajectory import RolloutGroup  # noqa: E402
from dflow.schedulers.flow_sde import build_sde_schedule, ode_step  # noqa: E402
from tests.rl.conftest import (
    CHANNELS,
    GROUP,
    MU,
    NOISE,
    STEPS,
    TARGET_LEN,
    FakeSequence,
    analytic_velocity,
    counting_velocity,
)  # noqa: E402

# --------------------------------------------------------------------- the module is decoupled


def test_rl_imports_neither_models_nor_tasks():
    """``rl/`` sits at L4 beside ``tasks/`` and must not reach into it or into ``models/``.

    ``tools/checks/check_layering.py`` enforces the layer ordering, but same-layer imports are
    legal there — so the decoupling that lets a task supply the sequence structurally is checked
    here instead.
    """
    import inspect

    from dflow.rl import protocol, rollout, trajectory

    for module in (protocol, rollout, trajectory):
        for line in inspect.getsource(module).splitlines():
            statement = line.strip()
            if statement.startswith(("import ", "from ")):
                assert "dflow.models" not in statement, (module.__name__, statement)
                assert "dflow.tasks" not in statement, (module.__name__, statement)
                assert "diffusers" not in statement, (module.__name__, statement)


def test_the_real_task_sequence_satisfies_the_protocol():
    """``ReferenceSequence`` conforms structurally, with no base class and no import either way.

    Nothing else would notice if that broke: the rollout would keep type-checking against the
    protocol and fail at runtime on the first ``replace_target``.
    """
    from dflow.tasks.ref2img.conditioning import ReferenceSequence

    instance = ReferenceSequence(
        tokens=torch.zeros(1, 4, 8), ids=torch.zeros(1, 4, 4), target_len=2
    )
    assert isinstance(instance, SequenceLike)
    swapped = instance.replace_target(torch.ones(1, 2, 8))
    assert isinstance(swapped, SequenceLike)


# ------------------------------------------------------------------------------ what is recorded


def test_recorded_shapes_describe_the_window_not_the_schedule(group):
    """``W + 1`` latents, not ``K + 1``: steps outside the window record nothing."""
    assert group.window_size == 3
    assert group.group_size == GROUP
    assert group.latents.shape == (4, GROUP, TARGET_LEN, CHANNELS)
    assert group.sigmas.shape == (4,)
    assert group.logprobs.shape == (3, GROUP)
    assert group.final_latents.shape == (GROUP, TARGET_LEN, CHANNELS)
    assert group.window_start == 1
    assert group.prompt_index == 3


def test_recorded_sigmas_are_the_schedule_s_own(group, schedule):
    """Recorded, not rebuilt. A replay that re-derived these could differ and nothing would say so."""
    torch.testing.assert_close(group.sigmas, schedule.sigmas[1:5], rtol=0, atol=0)
    torch.testing.assert_close(group.sigma_max, schedule.sigma_max, rtol=0, atol=0)
    assert group.noise_level == NOISE


def test_the_velocity_is_evaluated_at_each_step_s_entry_sigma(sequence, schedule, generator):
    """Once per schedule step, at the entry sigma — the exit sigma would be a different field."""
    calls: list[torch.Tensor] = []
    sample_group(
        sequence=sequence,
        velocity_fn=counting_velocity(TARGET_LEN, calls),
        schedule=schedule,
        window=(1, 4),
        noise_level=NOISE,
        generator=generator,
    )
    assert len(calls) == STEPS
    for index, sigma in enumerate(calls):
        assert sigma.shape == (GROUP,)
        torch.testing.assert_close(
            sigma, schedule.sigmas[index].expand(GROUP), rtol=0, atol=0
        )


def test_the_final_latent_is_past_the_window(group, schedule):
    """The reward scores the clean output, which is six deterministic steps past this window."""
    assert not torch.allclose(group.final_latents, group.latents[-1])


def test_the_final_latent_is_the_window_exit_when_the_window_ends_the_schedule(
    sequence, schedule, generator
):
    rolled = sample_group(
        sequence=sequence,
        velocity_fn=analytic_velocity(TARGET_LEN),
        schedule=schedule,
        window=(STEPS - 2, STEPS),
        noise_level=NOISE,
        generator=generator,
    )
    torch.testing.assert_close(rolled.final_latents, rolled.latents[-1], rtol=0, atol=0)


def test_steps_outside_the_window_are_deterministic_ode_steps(sequence, schedule, generator):
    """Reproduced independently: the pre-window steps must be plain Euler, with no noise drawn.

    If a step outside the window drew noise, it would both waste exploration where no gradient
    lands and desynchronise the generator from the replay.
    """
    rolled = sample_group(
        sequence=sequence,
        velocity_fn=analytic_velocity(TARGET_LEN),
        schedule=schedule,
        window=(3, 5),
        noise_level=NOISE,
        generator=torch.Generator(device=torch.device("cpu")).manual_seed(0),
    )

    # Replay the deterministic prefix by hand from the same initial noise.
    fresh = torch.Generator(device=torch.device("cpu")).manual_seed(0)
    latents = torch.randn(
        (GROUP, TARGET_LEN, CHANNELS), generator=fresh, device=sequence.tokens.device
    )
    velocity_fn = analytic_velocity(TARGET_LEN)
    for index in range(3):
        sigma, sigma_next = schedule.step(index)
        velocity = velocity_fn(sequence.replace_target(latents).tokens, sigma.expand(GROUP))
        latents = ode_step(
            sample=latents, velocity=velocity, sigma=sigma, sigma_next=sigma_next
        )
    torch.testing.assert_close(rolled.latents[0], latents, rtol=0, atol=0)


def test_the_trajectory_starts_from_pure_noise(schedule, generator):
    """``sigma == 1`` means ``x_t`` is the noise, so the sequence's target contents are ignored."""
    torch.manual_seed(2)
    references = torch.randn(GROUP, 3, CHANNELS)
    kwargs = dict(
        velocity_fn=analytic_velocity(TARGET_LEN),
        schedule=schedule,
        window=(0, 2),
        noise_level=NOISE,
    )
    first = sample_group(
        sequence=FakeSequence(
            tokens=torch.cat([torch.zeros(GROUP, TARGET_LEN, CHANNELS), references], dim=1),
            target_len=TARGET_LEN,
        ),
        generator=torch.Generator(device=torch.device("cpu")).manual_seed(5),
        **kwargs,
    )
    second = sample_group(
        sequence=FakeSequence(
            tokens=torch.cat([torch.full((GROUP, TARGET_LEN, CHANNELS), 9.0), references], dim=1),
            target_len=TARGET_LEN,
        ),
        generator=torch.Generator(device=torch.device("cpu")).manual_seed(5),
        **kwargs,
    )
    torch.testing.assert_close(first.latents, second.latents, rtol=0, atol=0)


def test_references_stay_clean_for_every_step(sequence, schedule, generator):
    """References are read-only context. Noising them would teach the model to denoise them."""
    seen: list[torch.Tensor] = []
    inner = analytic_velocity(TARGET_LEN)

    def velocity_fn(tokens, sigma):
        seen.append(tokens[:, TARGET_LEN:].clone())
        return inner(tokens, sigma)

    sample_group(
        sequence=sequence,
        velocity_fn=velocity_fn,
        schedule=schedule,
        window=(1, 4),
        noise_level=NOISE,
        generator=generator,
    )
    assert len(seen) == STEPS
    for references in seen:
        torch.testing.assert_close(references, sequence.references, rtol=0, atol=0)


def test_rollout_records_no_gradient(group):
    """It runs under ``no_grad``: these are recorded constants, not part of any graph."""
    for field in (group.latents, group.logprobs, group.final_latents, group.sigmas):
        assert not field.requires_grad
        assert field.grad_fn is None


def test_rollout_is_reproducible_from_a_generator(sequence, schedule):
    """CP ranks share a sequence and must roll out the same trajectory."""
    kwargs = dict(
        sequence=sequence,
        velocity_fn=analytic_velocity(TARGET_LEN),
        schedule=schedule,
        window=(1, 4),
        noise_level=NOISE,
    )
    first = sample_group(
        generator=torch.Generator(device=torch.device("cpu")).manual_seed(11), **kwargs
    )
    second = sample_group(
        generator=torch.Generator(device=torch.device("cpu")).manual_seed(11), **kwargs
    )
    torch.testing.assert_close(first.latents, second.latents, rtol=0, atol=0)
    torch.testing.assert_close(first.logprobs, second.logprobs, rtol=0, atol=0)


def test_group_members_explore_differently(group):
    """One prompt, one window, ``G`` different noises — otherwise the baseline has nothing to
    compare."""
    for row in range(1, GROUP):
        assert not torch.allclose(group.final_latents[0], group.final_latents[row])


# ------------------------------------------------------------------------------------ rejections


def test_rollout_rejects_a_window_outside_the_schedule(sequence, schedule, generator):
    for window in [(0, STEPS + 1), (-1, 3), (4, 4)]:
        with pytest.raises(ValueError, match="window"):
            sample_group(
                sequence=sequence,
                velocity_fn=analytic_velocity(TARGET_LEN),
                schedule=schedule,
                window=window,
                noise_level=NOISE,
                generator=generator,
            )


def test_rollout_rejects_a_velocity_that_is_not_the_target_span(sequence, schedule, generator):
    """The closure must slice, because where the target sits is family-specific."""

    def whole_sequence(tokens, sigma):
        return tokens

    with pytest.raises(ValueError, match="target span only"):
        sample_group(
            sequence=sequence,
            velocity_fn=whole_sequence,
            schedule=schedule,
            window=(1, 3),
            noise_level=NOISE,
            generator=generator,
        )


# -------------------------------------------------------------------------------- RolloutGroup


def test_step_pairs_each_latent_with_its_own_sigma(group, schedule):
    """The accessor exists so an off-by-one is impossible at the call site."""
    for index in range(group.window_size):
        sample, action, sigma, sigma_next = group.step(index)
        torch.testing.assert_close(sample, group.latents[index], rtol=0, atol=0)
        torch.testing.assert_close(action, group.latents[index + 1], rtol=0, atol=0)
        assert sigma.item() == schedule.sigmas[group.window_start + index].item()
        assert sigma_next.item() == schedule.sigmas[group.window_start + index + 1].item()


def test_step_is_bounds_checked(group):
    with pytest.raises(IndexError):
        group.step(group.window_size)
    with pytest.raises(IndexError):
        group.step(-1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("logprobs", torch.zeros(2, GROUP), "logprobs must be"),
        ("sigmas", torch.tensor([0.9, 0.5]), "sigmas must be"),
        ("final_latents", torch.zeros(GROUP, 1, CHANNELS), "final_latents must be"),
        ("noise_level", 0.0, "noise_level must be positive"),
        ("window_start", -1, "window_start"),
    ],
)
def test_inconsistent_records_are_refused(group, field, value, message):
    """An off-by-one in the recording loop must surface here rather than as a wrong log-prob."""
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(group, **{field: value})


def test_ascending_sigmas_are_refused(group):
    with pytest.raises(ValueError, match="strictly descending"):
        dataclasses.replace(group, sigmas=group.sigmas.flip(0))


# ------------------------------------------------------------------------------ window sampling


def test_the_window_always_fits_inside_its_range():
    """Covered steps stay inside ``window_range`` for every draw, whatever the size."""
    config = SDEConfig(inference_steps=STEPS, window_size=2, window_range=(0, 5))
    generator = torch.Generator(device=torch.device("cpu")).manual_seed(0)
    seen = set()
    for _ in range(200):
        start, stop = sample_window(config, generator=generator)
        assert start >= 0 and stop <= 5
        assert stop - start == 2
        seen.add(start)
    # Every admissible start is reachable, so over a run every step in the range gets gradient.
    assert seen == {0, 1, 2, 3}


def test_a_window_size_of_none_takes_the_whole_trajectory_bar_the_last_step():
    """``sigma_next == 0`` at the final step is where the log-prob is most delicate."""
    config = SDEConfig(inference_steps=STEPS, window_size=None)
    assert sample_window(config) == (0, STEPS - 1)
    assert config.trained_steps == STEPS - 1


def test_window_sampling_is_reproducible():
    config = SDEConfig(inference_steps=STEPS, window_size=2, window_range=(0, 5))
    first = [
        sample_window(config, generator=torch.Generator(device=torch.device("cpu")).manual_seed(7))
        for _ in range(3)
    ]
    assert len(set(first)) == 1


def test_a_window_that_exactly_fills_its_range_is_deterministic():
    """``high == low + 1``, which ``randint`` must still accept — config validation guarantees it."""
    config = SDEConfig(inference_steps=STEPS, window_size=5, window_range=(0, 5))
    assert sample_window(config) == (0, 5)


def test_rolling_out_with_a_sampled_window_stays_consistent(sequence, schedule, generator):
    config = SDEConfig(inference_steps=STEPS, window_size=2, window_range=(0, 5))
    window = sample_window(config, generator=generator)
    rolled = sample_group(
        sequence=sequence,
        velocity_fn=analytic_velocity(TARGET_LEN),
        schedule=schedule,
        window=window,
        noise_level=NOISE,
        generator=generator,
    )
    assert rolled.window_size == config.trained_steps == 2
    assert rolled.window_start == window[0]


def test_the_schedule_can_be_shorter_than_the_default():
    """Nothing in the rollout assumes ten steps."""
    schedule = build_sde_schedule(3, mu=MU)
    sequence = FakeSequence(tokens=torch.zeros(2, TARGET_LEN + 1, CHANNELS), target_len=TARGET_LEN)
    rolled = sample_group(
        sequence=sequence,
        velocity_fn=analytic_velocity(TARGET_LEN),
        schedule=schedule,
        window=(0, 1),
        noise_level=NOISE,
        generator=torch.Generator(device=torch.device("cpu")).manual_seed(0),
    )
    assert isinstance(rolled, RolloutGroup)
    assert rolled.window_size == 1
    assert rolled.logprobs.shape == (1, 2)
