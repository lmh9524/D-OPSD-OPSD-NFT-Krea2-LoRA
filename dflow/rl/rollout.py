"""The recording denoise loop, and the replay that scores it under current weights.

Two functions that must agree exactly, which is why they live in one file and share every helper:

* :func:`sample_group` walks the schedule under ``no_grad``, drawing an action inside the SDE
  window and stepping deterministically outside it, and records what a replay will need.
* :func:`replay_logprobs` walks the recorded window again *with* gradient and scores the recorded
  actions against the Gaussians current weights imply.

At the first inner epoch these produce the same numbers, because the policy that took the actions is
the policy being scored. That is the whole reason this design needs no rollout correction, and
``tests/rl/test_rollout_replay.py`` is where the claim is checked rather than asserted in prose.

## Neither function knows what a model is

Both take a ``velocity_fn`` (``rl/protocol.py``) and a sequence that satisfies ``SequenceLike``.
That keeps ``rl/`` at L4 with no import of ``models/`` or ``tasks/``, and it has a practical payoff:
the tests below drive the whole loop with an analytic velocity field, so they are exact, fast, and
need neither diffusers nor a checkpoint.

## What is easy to get wrong here

* **Which latent is the action.** ``latents[j + 1]`` is the action taken at step ``j``; recording
  the post-step state as the entry shifts every pair by one and scores each action against the next
  step's Gaussian.
* **The window bounds.** ``[start, stop)``, and the entry latent of the *first* window step has to
  be recorded before that step runs. Miss it and the array is a step short in a way the shape check
  in ``RolloutGroup`` catches — which is why that check exists.
* **The final latent.** The reward scores the clean output, which is past the window whenever the
  window does not reach the end of the schedule.
* **The sigma the velocity is evaluated at.** It is the *entry* sigma of the step, not the exit.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from dflow.config.rl import SDEConfig
from dflow.rl.protocol import SequenceLike, VelocityFn
from dflow.rl.trajectory import RolloutGroup
from dflow.schedulers.flow_sde import SDESchedule, ode_step, sde_step, step_logprob


def sample_window(config: SDEConfig, *, generator: torch.Generator | None = None) -> tuple[int, int]:
    """Draw one ``[start, stop)`` SDE window for a group.

    The start is uniform over the positions at which a window of ``config.window_size`` still fits
    inside ``config.window_range``, so the covered steps always stay inside that span. Drawing per
    group rather than fixing it means every step in the range receives gradient over a run, while
    any single group is compared against itself on identical steps.

    ``window_size=None`` makes the whole trajectory stochastic, stopping one step short of the end
    — matching verl-omni, and for the reason its default hints at: the final step lands on
    ``sigma_next == 0``, where the log-prob is at its most delicate.
    """
    if config.window_size is None:
        return (0, config.inference_steps - 1)

    low, stop = config.window_range
    high = stop - config.window_size + 1  # exclusive; config validation guarantees high > low
    device = generator.device if generator is not None else torch.device("cpu")
    start = int(torch.randint(low, high, (1,), generator=generator, device=device).item())
    return (start, start + config.window_size)


@torch.no_grad()
def sample_group(
    *,
    sequence: SequenceLike,
    velocity_fn: VelocityFn,
    schedule: SDESchedule,
    window: tuple[int, int],
    noise_level: float,
    generator: torch.Generator | None = None,
    prompt_index: int = 0,
) -> RolloutGroup:
    """Roll out one prompt's group and record what the training phase needs.

    ``sequence.tokens`` carries the group in its leading dimension: the caller expands one prompt's
    conditioning to ``G`` rows, so a group is one forward per step rather than ``G``.

    Args:
        sequence: the conditioning, with its target span standing in for the latent under
            denoising. Its contents at the target positions are ignored — the rollout starts from
            pure noise, which is what ``sigma == 1`` means.
        velocity_fn: ``(tokens, sigma) -> velocity`` over the target span. See ``rl/protocol.py``.
        schedule: the sigma grid, from ``build_sde_schedule``.
        window: ``[start, stop)`` step indices that are stochastic, trained and recorded.
        noise_level: the SDE noise scale.
        generator: draws the initial noise and every step's noise. Pass the run's generator, which
            ``runtime/seed.py`` seeded from ``dp_rank`` — so CP ranks sharing one sequence roll out
            the *same* trajectory while DP ranks roll out different ones. Do not build one here.
        prompt_index: the grouping key carried into ``RolloutGroup``.
    """
    start, stop = window
    if not 0 <= start < stop <= schedule.steps:
        raise ValueError(
            f"window {window} must satisfy 0 <= start < stop <= steps={schedule.steps}"
        )

    tokens = sequence.tokens
    group_size, _, channels = tokens.shape
    # sigma[0] == 1, and x_t = (1 - sigma) * x_0 + sigma * noise, so the trajectory starts as pure
    # noise. fp32 because that is where sde_step computes; velocity_fn casts to the model's dtype.
    latents = torch.randn(
        (group_size, sequence.target_len, channels),
        generator=generator,
        device=tokens.device,
        dtype=torch.float32,
    )

    recorded_latents: list[torch.Tensor] = []
    recorded_logprobs: list[torch.Tensor] = []

    for index in range(schedule.steps):
        sigma, sigma_next = schedule.step(index)
        velocity = velocity_fn(
            sequence.replace_target(latents).tokens, sigma.expand(group_size)
        )
        if velocity.shape != latents.shape:
            raise ValueError(
                f"velocity_fn returned {tuple(velocity.shape)} but the target span is "
                f"{tuple(latents.shape)}. It must return the target span only — slicing is "
                f"family-specific, so it belongs in the closure."
            )

        if start <= index < stop:
            if index == start:
                recorded_latents.append(latents)
            taken = sde_step(
                sample=latents,
                velocity=velocity,
                sigma=sigma,
                sigma_next=sigma_next,
                noise_level=noise_level,
                sigma_max=schedule.sigma_max,
                generator=generator,
            )
            recorded_logprobs.append(
                step_logprob(action=taken.action, mean=taken.mean, std=taken.std)
            )
            recorded_latents.append(taken.action)
            latents = taken.action
        else:
            latents = ode_step(
                sample=latents, velocity=velocity, sigma=sigma, sigma_next=sigma_next
            )

    return RolloutGroup(
        prompt_index=prompt_index,
        final_latents=latents,
        latents=torch.stack(recorded_latents),
        sigmas=schedule.sigmas[start : stop + 1].clone(),
        logprobs=torch.stack(recorded_logprobs),
        window_start=start,
        sigma_max=schedule.sigma_max.clone(),
        noise_level=noise_level,
    )


def replay_logprobs(
    *,
    group: RolloutGroup,
    sequence: SequenceLike,
    velocity_fn: VelocityFn,
    steps: Sequence[int] | None = None,
) -> torch.Tensor:
    """Score the recorded actions under current weights. ``(len(steps), G)``, with gradient.

    Everything the Gaussian depends on comes from ``group`` — the sigmas, ``sigma_max`` and
    ``noise_level`` — never from a config or a rebuilt schedule. That is what makes the ratio
    ``exp(replay - group.logprobs)`` exactly 1 on the first inner epoch instead of merely close, and
    it is why this function's signature has no schedule in it.

    Args:
        group: the recorded rollout.
        sequence: the *same* conditioning the rollout used, expanded to the same ``G`` rows. A
            different sequence silently scores against a different velocity field.
        velocity_fn: as in :func:`sample_group`, but now under gradient.
        steps: which window steps to replay, default all of them.
    """
    indices = range(group.window_size) if steps is None else steps
    if sequence.tokens.shape[0] != group.group_size:
        raise ValueError(
            f"sequence has {sequence.tokens.shape[0]} rows but the group has "
            f"{group.group_size}; the conditioning must be expanded to match"
        )

    scored: list[torch.Tensor] = []
    for index in indices:
        sample, action, sigma, sigma_next = group.step(int(index))
        velocity = velocity_fn(
            sequence.replace_target(sample).tokens, sigma.expand(group.group_size)
        )
        replayed = sde_step(
            sample=sample,
            velocity=velocity,
            sigma=sigma,
            sigma_next=sigma_next,
            noise_level=group.noise_level,
            sigma_max=group.sigma_max,
            action=action,
        )
        scored.append(
            step_logprob(action=replayed.action, mean=replayed.mean, std=replayed.std)
        )
    return torch.stack(scored)


__all__ = ["replay_logprobs", "sample_group", "sample_window"]
