"""Rollout-side flow matching: the ODE-to-SDE conversion and its per-step log-probability.

Model- and task-agnostic, exactly like ``flow_matching.py`` and for the same reason: ``mu`` is a
property of the family, and nothing here needs to know which architecture produced the velocity it
is handed. A test asserts this module imports neither ``diffusers`` nor ``dflow.models``.

## Why an SDE at all

Rectified flow sampling is deterministic — ``x_next = x + (sigma_next - sigma) * v`` — so there is
no action to take a gradient through and no policy to improve. Flow-GRPO's contribution is an SDE
with the same marginals at every sigma, which turns each denoising step into a Gaussian draw:

    std_t  = sqrt(sigma / (1 - sigma)) * noise_level
    mean   = x * (1 + std_t^2 / (2 sigma) * dt)
             + v * (1 + std_t^2 (1 - sigma) / (2 sigma)) * dt        dt = sigma_next - sigma < 0
    x_next ~ N(mean, (std_t * sqrt(-dt))^2)

At ``noise_level = 0`` both correction terms vanish and ``mean`` collapses to ``x + v * dt``, the
Euler ODE step. That degeneracy is the cheapest available check that the algebra is right, so it is
a test (``test_flow_sde.py``) rather than a comment — and it is why ``SDEConfig`` still refuses
``noise_level = 0``: a zero-variance step has no log-prob.

## The two things here that are wrong without raising

**The reduction is a mean, not a sum.** ``step_logprob`` averages over the token and channel
dimensions, following flow_grpo and verl-omni. A log-prob is therefore a per-element average, the
ratio ``exp(new - old)`` sits very close to 1, and published clip thresholds are calibrated to that
scale — verl-omni's Qwen-Image recipe uses ``clip_ratio=1e-5``. Summing instead would scale every
log-ratio by the element count (order ``4096 * 128`` for a 1024px target) and make any inherited
threshold meaningless. See ``PPOConfig.clip_ratio``.

**The action is detached.** Gradient must reach the policy through ``mean`` only. The action is a
recorded constant — it was drawn by the rollout, under weights that no longer exist. Letting
gradient into it optimises a different objective, and nothing about the shapes or the loss curve
says so.

## Why the sigmas travel with the trajectory

``build_sde_schedule`` is a pure function of ``(steps, mu)``, so it *could* be recomputed in the
training phase from the same inputs. It must not be. This is the abstraction of what verl-omni pays
for with rollout correction: if the rollout and the recompute disagree about the schedule even
slightly, every ratio is noise and no error is raised. Recording removes the possibility instead of
correcting for it, which is why ``Trajectory`` carries its sigmas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dflow.schedulers.flow_matching import apply_time_shift

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _broadcast(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    """A sigma as fp32, reshaped to broadcast over ``like``'s trailing dims.

    Accepts a python float, a 0-d tensor or a ``(B,)`` per-sample tensor. Per-sample matters
    because the training phase draws a different window per trajectory, so one micro-batch mixes
    steps and a single scalar would apply the wrong sigma to most of it.
    """
    tensor = torch.as_tensor(value, dtype=torch.float32, device=like.device).float()
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim != 1 or tensor.shape[0] != like.shape[0]:
        raise ValueError(
            f"expected one sigma per sample — a scalar or shape ({like.shape[0]},) — "
            f"got {tuple(tensor.shape)}"
        )
    return tensor.reshape(-1, *([1] * (like.ndim - 1)))


# --------------------------------------------------------------------------------- the schedule


@dataclass(frozen=True)
class SDESchedule:
    """The sigma grid a rollout walks: ``steps + 1`` values from 1.0 down to 0.0.

    Built to match what the pipeline does at inference — ``linspace(1, 1/steps, steps)`` put
    through the exponential time shift, with a terminal zero appended — so a rollout visits the
    sigmas the sampler visits.
    """

    sigmas: torch.Tensor  # (steps + 1,), descending, sigmas[0] == 1.0, sigmas[-1] == 0.0

    def __post_init__(self) -> None:
        if self.sigmas.ndim != 1 or self.sigmas.shape[0] < 3:
            raise ValueError(
                f"sigmas must be 1-D with at least three entries (two steps plus the terminal "
                f"zero), got {tuple(self.sigmas.shape)}"
            )

    @property
    def steps(self) -> int:
        return int(self.sigmas.shape[0]) - 1

    @property
    def sigma_max(self) -> torch.Tensor:
        """``sigmas[1]``: what stands in for ``sigma`` in the ``1 - sigma`` denominator at the
        first step, where ``sigma`` is exactly 1 and the ratio would divide by zero.

        The second grid point rather than an arbitrary epsilon, matching flow_grpo and verl-omni,
        so the first step's noise scale is set by the schedule's own resolution.
        """
        return self.sigmas[1]

    def step(self, index: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sigma, sigma_next)`` for step ``index``, which may be per-sample.

        Per-sample because the training phase draws a different window per trajectory, so one
        micro-batch mixes steps. The rollout passes a plain ``int``.
        """
        if isinstance(index, int):
            if not 0 <= index < self.steps:
                raise IndexError(f"step {index} outside [0, {self.steps})")
            return self.sigmas[index], self.sigmas[index + 1]
        if index.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"step index must be integral, got {index.dtype}")
        if int(index.min()) < 0 or int(index.max()) >= self.steps:
            raise IndexError(f"step indices {index.tolist()} outside [0, {self.steps})")
        return self.sigmas[index], self.sigmas[index + 1]


def build_sde_schedule(
    steps: int,
    *,
    mu: float,
    device: torch.device | str = "cpu",
) -> SDESchedule:
    """The shifted sigma grid for a ``steps``-step rollout.

    ``mu`` comes from the family (``family.noise_shift_mu``), never from here — the same split
    ``flow_matching.py`` keeps, and for the same reason.

    Always fp32. The grid is compared against exactly (``sigma >= 1`` selects the first step) and
    divided by, and a bf16 grid would put two adjacent sigmas on the same value at high step counts.
    """
    if steps < 2:
        raise ValueError(f"steps must be >= 2, got {steps}")
    raw = torch.linspace(1.0, 1.0 / steps, steps, device=device, dtype=torch.float32)
    shifted = apply_time_shift(mu, raw)
    terminal = torch.zeros(1, device=device, dtype=torch.float32)
    return SDESchedule(sigmas=torch.cat([shifted, terminal]))


# ------------------------------------------------------------------------------------ the step


@dataclass(frozen=True)
class SDEStep:
    """One step's Gaussian, and the action taken in it.

    ``std`` already includes ``sqrt(-dt)``: it is the standard deviation of the step actually
    taken, not the ``std_t`` of the SDE's diffusion coefficient. Keeping the two apart is worth the
    field name — feeding ``std_t`` to ``step_logprob`` scales every log-prob by ``1/sqrt(-dt)`` and
    produces a plausible number.
    """

    mean: torch.Tensor
    std: torch.Tensor
    action: torch.Tensor


def sde_step(
    *,
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    noise_level: float,
    sigma_max: torch.Tensor,
    generator: torch.Generator | None = None,
    action: torch.Tensor | None = None,
) -> SDEStep:
    """Take one SDE step, drawing an action or scoring a given one.

    Both call sites go through here so there is one place the mean and variance are defined: the
    rollout leaves ``action`` unset and gets a draw; the training recompute passes the recorded
    action and gets the ``mean`` and ``std`` to score it against, under current weights.

    Args:
        sample: ``(B, ...)`` the current latent ``x_t``.
        velocity: ``(B, ...)`` the model's prediction at ``sigma``, same shape as ``sample``.
        sigma: ``(B,)`` or scalar — where the step starts.
        sigma_next: ``(B,)`` or scalar — where it lands. Below ``sigma``.
        noise_level: the SDE noise scale. ``0`` recovers the Euler ODE step exactly, with
            ``std == 0``; useful as a check, unusable as a policy.
        sigma_max: the schedule's second sigma, standing in for ``sigma`` in the ``1 - sigma``
            denominator at the first step. ``SDESchedule.sigma_max``.
        generator: draws the action. Pass the run's generator, which ``runtime/seed.py`` has
            already seeded from ``dp_rank`` — so CP ranks sharing one sequence draw the same
            trajectory, and DP ranks draw different ones. Do not construct one here.
        action: the recorded ``x_next`` to score instead of drawing.

    Everything is computed in fp32 whatever the caller's autocast dtype. The log-prob's squared
    residual loses precision in bf16 exactly where it is smallest — the same argument
    ``flow_mse`` makes — and the ``1 - sigma`` denominator is worse: at ``sigma = 0.996`` bf16
    leaves three significant bits.
    """
    if sample.shape != velocity.shape:
        raise ValueError(
            f"sample {tuple(sample.shape)} and velocity {tuple(velocity.shape)} must match. "
            f"Slice the model output to the supervised span before stepping."
        )
    if action is not None and action.shape != sample.shape:
        raise ValueError(
            f"action {tuple(action.shape)} must match sample {tuple(sample.shape)}"
        )
    if noise_level < 0.0:
        raise ValueError(f"noise_level must be non-negative, got {noise_level}")

    x = sample.float()
    v = velocity.float()
    s = _broadcast(sigma, x)
    s_next = _broadcast(sigma_next, x)
    dt = s_next - s  # negative: sigma decreases toward the clean end
    if bool((dt >= 0).any()):
        raise ValueError(
            "sigma_next must be strictly below sigma — the rollout integrates from noise toward "
            "data. Got a non-decreasing pair, which would make sqrt(-dt) imaginary."
        )

    # At the first step sigma is exactly 1, so 1 - sigma is zero. Substituting the schedule's
    # second sigma in the *denominator only* (the numerator keeps the true sigma) is flow_grpo's
    # and verl-omni's treatment; `>=` rather than `== 1` so a schedule built elsewhere cannot
    # slip a value past the guard.
    ceiling = torch.as_tensor(sigma_max, dtype=torch.float32, device=x.device)
    guarded = torch.where(s >= 1.0, ceiling, s)
    std_t = torch.sqrt(s / (1.0 - guarded)) * noise_level

    mean = x * (1.0 + std_t**2 / (2.0 * s) * dt) + v * (
        1.0 + std_t**2 * (1.0 - s) / (2.0 * s)
    ) * dt
    std = std_t * torch.sqrt(-dt)

    if action is None:
        noise = torch.randn(
            mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
        )
        action = mean + std * noise
    else:
        action = action.float()

    return SDEStep(mean=mean, std=std.expand_as(mean), action=action)


def step_logprob(*, action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """``(B,)`` Gaussian log-density of ``action``, averaged over every non-batch dimension.

    The **mean** reduction is load-bearing, not stylistic: it makes a log-prob a per-element
    average, independent of token count, which is the scale published diffusion clip thresholds are
    calibrated to. ``PPOConfig.clip_ratio`` and this module's docstring say more.

    ``action`` is detached here rather than at the call site. It was drawn during rollout under
    weights that no longer exist, so gradient must reach the policy through ``mean`` alone — and a
    gradient that also flowed into the action would optimise a different objective without changing
    a single shape.
    """
    if action.shape != mean.shape:
        raise ValueError(
            f"action {tuple(action.shape)} and mean {tuple(mean.shape)} must match"
        )
    if not bool((std > 0).all()):
        raise ValueError(
            "std must be strictly positive. A zero-variance step is the deterministic ODE, whose "
            "log-density is a delta — check that noise_level is not zero."
        )

    residual = action.detach().float() - mean.float()
    variance = std.float() ** 2
    density = -(residual**2) / (2.0 * variance) - torch.log(std.float()) - _LOG_SQRT_2PI
    return density.mean(dim=tuple(range(1, density.ndim)))


def ode_step(
    *,
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    """``x + (sigma_next - sigma) * v`` — the deterministic Euler step.

    Used for the steps outside the SDE window, which carry no noise, no log-prob and no gradient.
    Written out rather than routed through ``sde_step(noise_level=0)`` so the cheap path stays
    cheap and the degeneracy test has two independent implementations to compare.
    """
    x = sample.float()
    return x + (_broadcast(sigma_next, x) - _broadcast(sigma, x)) * velocity.float()


__all__ = [
    "SDESchedule",
    "SDEStep",
    "build_sde_schedule",
    "ode_step",
    "sde_step",
    "step_logprob",
]
