"""The RL loop against a **real transformer**. The gap every other test leaves open.

Everything else in ``tests/rl/`` drives the rollout with an analytic velocity field. That is what
makes the replay comparison exact, and it is deliberate — but it means nothing so far has run
``sample_group`` -> reward -> advantage -> ``replay_logprobs`` -> ``ppo_clip_loss`` -> ``backward``
through an actual ``Flux2Transformer2DModel``, with the family's real calling convention, real
4-D position ids, and a real LoRA attached.

A tiny config is enough, and is the same trick ``tests/conftest.py`` already uses for the family
tests: the architecture's *contract* is what is under test, not its capacity. This keeps the test in
the fast suite instead of behind a multi-gigabyte download.

## The claim this settles

``test_rollout_replay.py`` says, in its own docstring, that end-to-end exactness additionally needs
the model's forward to be reproducible, and defers the check. This is that check:
:func:`test_the_ratio_is_one_on_the_first_inner_epoch_with_a_real_model` runs a real forward twice —
once recording, once replaying — and asserts the importance ratio comes back to 1.

That is the whole basis for having no rollout correction, so it is worth knowing whether it
survives contact with a transformer rather than a closed-form function.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from dflow.config.rl import GroupConfig, PPOConfig, SDEConfig  # noqa: E402
from dflow.models.family.flux2 import Flux2Family  # noqa: E402
from dflow.rl.advantage import group_advantage  # noqa: E402
from dflow.rl.objective import ppo_clip_loss  # noqa: E402
from dflow.rl.rollout import replay_logprobs, sample_group, sample_window  # noqa: E402
from dflow.schedulers.flow_sde import build_sde_schedule  # noqa: E402
from dflow.tasks.ref2img import build_sequence  # noqa: E402
from tests.conftest import build_text_ids  # noqa: E402

GROUP = 4
GRID = (2, 3)  # 6 target tokens
TEXT_LEN = 5
STEPS = 6
WINDOW = (1, 3)
NOISE = 0.7
MU = 1.15


@pytest.fixture
def family():
    return Flux2Family()


@pytest.fixture
def pieces(tiny_flux2, tiny_flux2_config):
    """A group's conditioning against the tiny model, shaped exactly as the experiment shapes it."""
    torch.manual_seed(0)
    channels = tiny_flux2_config["in_channels"]
    height, width = GRID

    # The rollout starts from pure noise, so only this tensor's geometry matters.
    blank = torch.zeros(GROUP, channels, height, width)
    sequence = build_sequence(target_latents=blank, reference_latents=[])
    # `text_ids` carries the group in its batch dim, matching what `TextEncoder.encode` returns:
    # its ids come from `_prepare_text_ids(embeds)`, so they are already `(B, L, 4)`. The family
    # checks the alignment, and `conftest.build_text_ids` builds a single row.
    conditioning = {
        "embeds": torch.randn(GROUP, TEXT_LEN, tiny_flux2_config["joint_attention_dim"]),
        "text_ids": build_text_ids(TEXT_LEN).expand(GROUP, -1, -1),
        "ids": sequence.ids,
    }
    return sequence, conditioning


def make_velocity_fn(model, family, *, conditioning, target_len):
    """The experiment's ``make_velocity_fn``, without guidance. One definition, two call sites."""

    def velocity_fn(tokens: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        output = model(
            **family.prepare_inputs(
                tokens=tokens.to(model.dtype),
                token_ids=conditioning["ids"],
                text_embeds=conditioning["embeds"].to(model.dtype),
                text_ids=conditioning["text_ids"],
                timestep=sigma.to(model.dtype),
            )
        )[0]
        return family.take_target_span(output, target_len).float()

    return velocity_fn


@pytest.fixture
def rolled(tiny_flux2, family, pieces):
    sequence, conditioning = pieces
    schedule = build_sde_schedule(STEPS, mu=MU)
    return schedule, sample_group(
        sequence=sequence,
        velocity_fn=make_velocity_fn(
            tiny_flux2, family, conditioning=conditioning, target_len=sequence.target_len
        ),
        schedule=schedule,
        window=WINDOW,
        noise_level=NOISE,
        generator=torch.Generator().manual_seed(7),
        prompt_index=0,
    )


# ----------------------------------------------------------------- the rollout drives the model


def test_a_real_forward_produces_a_well_formed_group(rolled, pieces):
    """The family's calling convention accepts what the rollout builds, and the record lines up."""
    schedule, group = rolled
    sequence, _ = pieces
    window_size = WINDOW[1] - WINDOW[0]

    assert group.group_size == GROUP
    assert group.window_size == window_size
    assert group.latents.shape[0] == window_size + 1
    assert group.logprobs.shape == (window_size, GROUP)
    assert group.final_latents.shape == (GROUP, sequence.target_len, group.latents.shape[-1])
    torch.testing.assert_close(group.sigmas, schedule.sigmas[WINDOW[0] : WINDOW[1] + 1], rtol=0, atol=0)
    assert torch.isfinite(group.logprobs).all()
    assert torch.isfinite(group.final_latents).all()


def test_group_members_diverge_under_a_real_model(rolled):
    """Same prompt, same weights, different noise — otherwise the baseline compares nothing."""
    _, group = rolled
    for row in range(1, GROUP):
        assert not torch.allclose(group.final_latents[0], group.final_latents[row])


# ------------------------------------------------------------------------- the deferred check


def test_the_ratio_is_one_on_the_first_inner_epoch_with_a_real_model(
    rolled, pieces, tiny_flux2, family
):
    """**The claim the whole design rests on**, now against a transformer.

    ``test_rollout_replay.py`` establishes it for the scheduler half, with the model removed by an
    analytic velocity field, and explicitly defers this. Here the same weights that produced the
    trajectory score it again: if the forward is reproducible, the replayed log-probs equal the
    recorded ones and the ratio is exactly 1 — which is what makes a separate ``old_log_prob`` pass
    and rollout correction unnecessary.

    Asserted at ``rtol=0, atol=0``. If a future change to autocast or batching makes a real forward
    non-reproducible, this is where it surfaces, rather than as ratios that drift for no visible
    reason during a run.
    """
    _, group = rolled
    sequence, conditioning = pieces

    replayed = replay_logprobs(
        group=group,
        sequence=sequence,
        velocity_fn=make_velocity_fn(
            tiny_flux2, family, conditioning=conditioning, target_len=sequence.target_len
        ),
    )
    torch.testing.assert_close(replayed, group.logprobs, rtol=0, atol=0)

    ratio = torch.exp(replayed - group.logprobs)
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)


def test_changing_the_weights_moves_the_ratio_off_one(rolled, pieces, tiny_flux2, family):
    """So the test above is not passing because the replay ignores the model."""
    _, group = rolled
    sequence, conditioning = pieces

    with torch.no_grad():
        for parameter in tiny_flux2.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)

    replayed = replay_logprobs(
        group=group,
        sequence=sequence,
        velocity_fn=make_velocity_fn(
            tiny_flux2, family, conditioning=conditioning, target_len=sequence.target_len
        ),
    )
    assert not torch.allclose(replayed, group.logprobs)


# ------------------------------------------------------------------- reward to gradient, whole


def test_the_full_path_produces_a_gradient_on_a_lora_adapter(pieces, tiny_flux2, family):
    """rollout -> reward -> advantage -> replay -> clipped loss -> backward, on a real model.

    LoRA rather than the full parameter set, because that is what an RL run actually trains, and
    because it checks the thing a whole-model gradient would hide: that the adapter is reached at
    all.

    **At step 0 the gradient reaches ``B``, not ``A``.** The adapter is ``(alpha/r) * B @ A`` with
    ``B`` zero-initialised so it starts as an identity, and that makes ``dL/dA`` proportional to
    ``B`` — exactly zero on the first step — while ``dL/dB`` is proportional to ``A``, which is not.
    Measured, not assumed: asserting on ``A`` here fails against a correctly wired model, which is
    what this comment exists to stop the next reader from "fixing".
    """
    from dflow.config import LoRAConfig
    from dflow.models.adapter import apply_lora, reset_lora_parameters

    sequence, conditioning = pieces
    apply_lora(tiny_flux2, LoRAConfig(enabled=True, rank=4, alpha=4.0), targets=["to_q", "to_v"])
    reset_lora_parameters(tiny_flux2)
    trainable = [p for name, p in tiny_flux2.named_parameters() if "lora_" in name]
    assert trainable, "no LoRA parameters were attached"

    sde = SDEConfig(inference_steps=STEPS, window_size=2, window_range=(0, 4))
    schedule = build_sde_schedule(sde.inference_steps, mu=MU)
    velocity_fn = make_velocity_fn(
        tiny_flux2, family, conditioning=conditioning, target_len=sequence.target_len
    )

    generator = torch.Generator().manual_seed(3)
    group = sample_group(
        sequence=sequence,
        velocity_fn=velocity_fn,
        schedule=schedule,
        window=sample_window(sde, generator=generator),
        noise_level=sde.noise_level,
        generator=generator,
        prompt_index=0,
    )

    # A stand-in reward: the mean of the clean latents. Real scorers decode and score pixels; what
    # matters here is that a per-trajectory scalar reaches the loss.
    rewards = group.final_latents.flatten(1).mean(dim=1).reshape(1, GROUP)
    advantages, stats = group_advantage(rewards, GroupConfig(size=GROUP))
    assert torch.isfinite(advantages).all()
    assert abs(float(advantages.mean())) < 1e-5, "a group's advantages must centre on zero"

    replayed = replay_logprobs(group=group, sequence=sequence, velocity_fn=velocity_fn)
    loss, ppo = ppo_clip_loss(
        logprobs=replayed,
        old_logprobs=group.logprobs,
        advantages=advantages[0],
        config=PPOConfig(),
    )
    assert torch.isfinite(loss)
    # First inner epoch: the ratio is 1, so nothing is clipped and the loss is -mean(advantage).
    assert ppo.clipfrac == 0.0
    assert ppo.ratio_mean == pytest.approx(1.0, abs=1e-6)

    loss.backward()

    def total_grad(kind: str) -> float:
        return sum(
            float(p.grad.abs().sum())
            for name, p in tiny_flux2.named_parameters()
            if kind in name and p.grad is not None
        )

    assert total_grad("lora_B") > 0, (
        "no LoRA B matrix received gradient: the objective is not connected to the parameters the "
        "optimizer will move"
    )
    assert total_grad("lora_A") == 0.0, (
        "lora_A received gradient at step 0, which means B is not zero-initialised — the adapter "
        "does not start as an identity and the run begins from a perturbed model"
    )
    assert stats.degenerate_fraction == 0.0


# ------------------------------------------------------------- the objective points uphill


def test_one_step_makes_above_average_trajectories_more_likely(pieces, tiny_flux2, family):
    """The direction check: after one update, log-prob change must correlate with advantage.

    Every other test on the objective examines it in isolation — the sign of the loss, which
    branch the clip takes, where gradient flows. None of them answers the question that actually
    matters, which is whether the composition of rollout, advantage, replay and optimizer step
    moves the policy *towards* reward rather than away from it. A sign error anywhere along that
    chain produces a run that trains smoothly downhill, and the only symptom is a reward curve
    that drifts the wrong way over hundreds of steps.

    The reward here is synthetic and monotone in the trajectory index, so the expected outcome is
    unambiguous: the correlation should be strongly positive.
    """
    from dflow.config import LoRAConfig
    from dflow.models.adapter import apply_lora, reset_lora_parameters

    sequence, conditioning = pieces
    apply_lora(tiny_flux2, LoRAConfig(enabled=True, rank=8, alpha=8.0), targets=["to_q", "to_v"])
    reset_lora_parameters(tiny_flux2)
    velocity_fn = make_velocity_fn(
        tiny_flux2, family, conditioning=conditioning, target_len=sequence.target_len
    )

    group = sample_group(
        sequence=sequence,
        velocity_fn=velocity_fn,
        schedule=build_sde_schedule(STEPS, mu=MU),
        window=WINDOW,
        noise_level=NOISE,
        generator=torch.Generator().manual_seed(5),
        prompt_index=0,
    )

    # Trajectory j scores j, so the ranking is known and the advantages span both signs.
    rewards = torch.arange(GROUP, dtype=torch.float32).reshape(1, GROUP)
    advantages, _ = group_advantage(rewards, GroupConfig(size=GROUP))
    assert float(advantages[0, 0]) < 0 < float(advantages[0, -1])

    optimizer = torch.optim.AdamW(
        [p for p in tiny_flux2.parameters() if p.requires_grad], lr=1e-3
    )
    before = replay_logprobs(group=group, sequence=sequence, velocity_fn=velocity_fn).detach()
    loss, _ = ppo_clip_loss(
        logprobs=replay_logprobs(group=group, sequence=sequence, velocity_fn=velocity_fn),
        old_logprobs=group.logprobs,
        advantages=advantages[0],
        config=PPOConfig(),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = replay_logprobs(group=group, sequence=sequence, velocity_fn=velocity_fn).detach()

    delta = (after - before).mean(dim=0)
    correlation = float(torch.corrcoef(torch.stack([advantages[0], delta]))[0, 1])
    assert correlation > 0.5, (
        f"corr(advantage, delta log-prob) = {correlation:+.3f}. The update moved the policy away "
        f"from the better trajectories, which is a sign error somewhere in the chain."
    )
