"""What a rollout records, and why each field is recorded rather than recomputed.

A :class:`RolloutGroup` is **one prompt's group of trajectories**: the unit the group-relative
baseline is computed over, and the unit that shares conditioning. Naming it after the group rather
than the trajectory is not cosmetic — the advantage is meaningless for a single trajectory, so the
group is the smallest thing the RL loop can do anything with.

## Everything the replay needs travels with the group

``build_sde_schedule`` is a pure function of ``(steps, mu)``, so the sigmas *could* be rebuilt in
the training phase from the same inputs. They must not be, and neither may ``sigma_max`` or
``noise_level``. This is the abstraction of what verl-omni pays for with rollout correction: if the
rollout and the recompute disagree about the schedule, the noise scale, or which sigma stands in at
the first step, then every importance ratio is noise and nothing raises — the shapes match, the loss
is finite, and the run trains on garbage.

So the rule is: **if the log-prob depends on it, it is a field here.** That is why ``sigma_max`` is
stored despite being a schedule constant (it sets the first step's noise scale, where ``1 - sigma``
is zero) and why ``noise_level`` is stored despite being a config value (it scales every standard
deviation). ``replay_logprobs`` then needs nothing but the group and a ``velocity_fn``.

## The window collapses two things into one

``latents`` holds ``W + 1`` entries, not ``K + 1``. Steps outside the SDE window are deterministic
ODE steps: no noise, no log-prob, no stored latent, no gradient. So the window is simultaneously
where the trajectory is stochastic and where it is trained, and there is no separate
``trained_steps`` field because the window *is* the trained steps — carrying both would let them
disagree. See ``docs/rl-design.md``.

``final_latents`` is therefore separate and load-bearing: with a window of ``(0, 2)`` on a ten-step
schedule the clean output is eight deterministic steps past the end of ``latents``, and it is the
clean output the reward scores.

## One window per group, shared

Every trajectory in a group is trained on the same steps. That is a variance argument, not a
convenience one: the advantage compares group members against each other, so comparing them on
different steps of the trajectory adds a difficulty difference to the reward difference the baseline
is trying to isolate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RolloutGroup:
    """One prompt, ``G`` trajectories, one shared SDE window."""

    #: Which prompt this group came from. The grouping key, and an index into whatever per-prompt
    #: context the caller kept — an integer rather than the prompt text, so grouping cannot be
    #: defeated by whitespace.
    prompt_index: int

    #: ``(G, target_len, C)`` — the clean target latents at the end of the full schedule. What the
    #: reward scores. Not ``latents[-1]`` unless the window happens to end the trajectory.
    final_latents: torch.Tensor

    #: ``(W + 1, G, target_len, C)`` — the window's latents: the entry of each trained step, plus
    #: the exit of the last. ``latents[j]`` is step ``j``'s state and ``latents[j + 1]`` is the
    #: action taken in it.
    latents: torch.Tensor

    #: ``(W + 1,)`` — the sigmas of the window's steps, recorded. ``sigmas[j]`` and
    #: ``sigmas[j + 1]`` bracket step ``j``.
    sigmas: torch.Tensor

    #: ``(W, G)`` — the log-probability of each action under the policy that took it. These *are*
    #: the old log-probs; there is no separate recompute pass and no correction term.
    logprobs: torch.Tensor

    #: Which step of the full schedule the window opened at. Informational for logging and for
    #: checking a replay against the schedule; the sigmas above are what the maths uses.
    window_start: int

    #: The schedule's second sigma, which stands in for ``sigma`` in the ``1 - sigma`` denominator
    #: at the first step. Recorded because it sets that step's noise scale.
    sigma_max: torch.Tensor

    #: The SDE noise scale in force during the rollout. Recorded because it scales every standard
    #: deviation, so a replay under a different value would score against a different Gaussian.
    noise_level: float

    def __post_init__(self) -> None:
        if self.latents.ndim != 4:
            raise ValueError(
                f"latents must be (W+1, G, target_len, C), got {tuple(self.latents.shape)}"
            )
        steps_plus_one, group, target_len, channels = self.latents.shape
        if steps_plus_one < 2:
            raise ValueError(
                f"latents needs at least two entries — one step's entry and exit — got "
                f"{steps_plus_one}"
            )
        if self.sigmas.shape != (steps_plus_one,):
            raise ValueError(
                f"sigmas must be ({steps_plus_one},) to bracket {steps_plus_one - 1} steps, got "
                f"{tuple(self.sigmas.shape)}"
            )
        if self.logprobs.shape != (steps_plus_one - 1, group):
            raise ValueError(
                f"logprobs must be ({steps_plus_one - 1}, {group}) — one per step per trajectory "
                f"— got {tuple(self.logprobs.shape)}"
            )
        if self.final_latents.shape != (group, target_len, channels):
            raise ValueError(
                f"final_latents must be ({group}, {target_len}, {channels}), got "
                f"{tuple(self.final_latents.shape)}"
            )
        if not torch.all(self.sigmas[:-1] > self.sigmas[1:]):
            raise ValueError(
                f"sigmas must be strictly descending — the rollout integrates from noise toward "
                f"data — got {self.sigmas.tolist()}"
            )
        if self.window_start < 0:
            raise ValueError(f"window_start must be non-negative, got {self.window_start}")
        if self.noise_level <= 0.0:
            raise ValueError(
                f"noise_level must be positive, got {self.noise_level}. A zero-variance step has "
                f"no log-prob, so a group recorded at zero cannot be replayed."
            )

    @property
    def group_size(self) -> int:
        return int(self.latents.shape[1])

    @property
    def window_size(self) -> int:
        """Trained steps per trajectory."""
        return int(self.latents.shape[0]) - 1

    def step(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(sample, action, sigma, sigma_next)`` for window step ``index``.

        One accessor rather than four index expressions at the call site: an off-by-one between the
        latent pair and the sigma pair scores the action against the wrong Gaussian, and looks like
        nothing at all.
        """
        if not 0 <= index < self.window_size:
            raise IndexError(f"window step {index} outside [0, {self.window_size})")
        return (
            self.latents[index],
            self.latents[index + 1],
            self.sigmas[index],
            self.sigmas[index + 1],
        )


@dataclass(frozen=True, slots=True)
class NFTRolloutGroup:
    """One prompt's group of **clean** rollout latents, for DiffusionNFT.

    DiffusionNFT is forward-process and likelihood-free: unlike Flow-GRPO it does not need an SDE
    window, per-step log-probs, or the stochastic-step machinery that :class:`RolloutGroup` records.
    It needs only the endpoint each trajectory reached — the clean target latents the frozen ``old``
    policy produced by a deterministic few-step rollout — because the update re-noises *that* latent
    at a freshly sampled ``t`` and regresses the trainable policy toward it.

    So this is a deliberately minimal carrier, not a lean :class:`RolloutGroup`: making it a subset
    of that class would invite reading fields (sigmas, logprobs, a window) that DiffusionNFT never
    records and cannot fill. It holds exactly what the reward phase and the update phase read, and
    the experiment's closure attaches whatever conditioning it needs to replay the forward pass by
    keying off :attr:`prompt_index`, the same pattern the Flow-GRPO closures use.
    """

    #: Which prompt this group came from — the grouping key for the advantage baseline, and an index
    #: into whatever per-prompt conditioning the closure kept. An integer, not the prompt text, so
    #: grouping cannot be defeated by whitespace.
    prompt_index: int

    #: ``(G, target_len, C)`` — the clean target latents at the end of the few-step rollout, one per
    #: trajectory. What the reward scores and what the update re-noises. Detached: DiffusionNFT does
    #: not back-propagate through sampling.
    final_latents: torch.Tensor

    def __post_init__(self) -> None:
        if self.final_latents.ndim != 3:
            raise ValueError(
                f"final_latents must be (G, target_len, C), got {tuple(self.final_latents.shape)}"
            )

    @property
    def group_size(self) -> int:
        return int(self.final_latents.shape[0])


__all__ = ["NFTRolloutGroup", "RolloutGroup"]
