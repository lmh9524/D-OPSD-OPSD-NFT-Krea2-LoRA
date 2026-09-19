"""Training-side flow matching. Model-agnostic.

Rectified flow, in the convention FLUX.2 uses:

    x_t     = (1 - t) * x_0 + t * noise          t == sigma, 0 = clean, 1 = noise
    target  = noise - x_0                        the velocity the model predicts

The subtle part is not the interpolation, it is **which t values training sees**. Inference
does not walk a uniform schedule: it builds ``sigmas = linspace(1, 1/steps, steps)`` and then
applies an exponential time shift parameterised by ``mu``. Training that samples t uniformly
spends its effort in a different part of the trajectory than the sampler ever visits.

So this module draws raw timesteps and applies the same shift — but it takes ``mu`` as an
**argument** rather than computing it. How the shift is derived is a property of the model:
FLUX.2 fits it empirically from sequence length and step count, Wan uses a constant, FLUX.1
interpolates between configured bounds. Keeping that in ``models/family/`` is what stops this
file needing a branch per architecture. Ask the family:

    mu = family.noise_shift_mu(image_tokens=n_target, inference_steps=cfg.inference_steps)
    sigmas = scheduler.sample_timesteps(batch, mu=mu, generator=rng)

One algebraic note, so the numbers are legible: the exponential shift
``exp(mu) / (exp(mu) + (1/t - 1))`` equals the classic ``shift * t / (1 + (shift - 1) * t)``
with ``shift = exp(mu)``. ``shift_from_mu`` converts, for comparison with codebases that
report a shift.
"""

from __future__ import annotations

import math

import torch

from dflow.config import FlowMatchConfig


def shift_from_mu(mu: float) -> float:
    """``exp(mu)``: the same quantity other codebases call ``shift``."""
    return math.exp(mu)


def apply_time_shift(mu: float, sigmas: torch.Tensor) -> torch.Tensor:
    """Exponential time shift, matching ``FlowMatchEulerDiscreteScheduler.time_shift``."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigmas - 1.0))


class FlowMatchScheduler:
    """Draws training timesteps and builds noisy inputs and targets.

    Holds no mutable schedule state: sampling is a pure function of the config, ``mu`` and the
    generator. Inference-time stepping is deliberately absent — diffusers' pipelines do that,
    and ``rollout/`` will when RL and distillation need it in-loop.
    """

    def __init__(self, config: FlowMatchConfig, *, device: torch.device) -> None:
        self.config = config
        self.device = device

    # ------------------------------------------------------------------------ sampling

    def _raw_sigmas(self, batch_size: int, generator: torch.Generator | None) -> torch.Tensor:
        config = self.config
        if config.distribution == "uniform":
            return torch.rand(batch_size, device=self.device, generator=generator)
        if config.distribution == "logit_normal":
            normal = torch.normal(
                mean=config.logit_mean,
                std=config.logit_std,
                size=(batch_size,),
                device=self.device,
                generator=generator,
            )
            return torch.sigmoid(normal)
        # "mode": pushes samples toward the trajectory ends (SD3 paper, eq. 20).
        uniform = torch.rand(batch_size, device=self.device, generator=generator)
        return 1.0 - uniform - config.mode_scale * (
            torch.cos(math.pi * uniform / 2.0) ** 2 - 1.0 + uniform
        )

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        mu: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample ``(batch_size,)`` sigmas in (0, 1), shifted by ``mu``.

        The returned values are what the transformer expects as ``timestep``: normalised to
        [0, 1], because FLUX.2's forward multiplies by 1000 itself.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        epsilon = self.config.epsilon
        sigmas = self._raw_sigmas(batch_size, generator).clamp(epsilon, 1.0 - epsilon)
        return apply_time_shift(mu, sigmas).clamp(epsilon, 1.0 - epsilon)

    # ------------------------------------------------------------------- noise & target

    @staticmethod
    def add_noise(clean: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        """``(1 - sigma) * clean + sigma * noise``, broadcasting sigma over trailing dims."""
        if clean.shape != noise.shape:
            raise ValueError(
                f"shape mismatch: clean {tuple(clean.shape)} vs noise {tuple(noise.shape)}"
            )
        if sigmas.shape[0] != clean.shape[0]:
            raise ValueError(f"expected one sigma per sample, got {tuple(sigmas.shape)}")
        shaped = sigmas.reshape(-1, *([1] * (clean.ndim - 1))).to(clean.dtype)
        return (1.0 - shaped) * clean + shaped * noise

    @staticmethod
    def target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """The velocity ``noise - clean``, which is what the model regresses onto."""
        return noise - clean

    def weighting(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weight. ``none`` returns ones, i.e. plain flow matching."""
        scheme = self.config.weighting
        if scheme == "none":
            return torch.ones_like(sigmas)
        if scheme == "sigma_sqrt":
            return (sigmas**-2.0).clamp(max=1e4)
        # "cosmap"
        denominator = 1.0 - 2.0 * sigmas + 2.0 * sigmas**2
        return 2.0 / (math.pi * denominator)


def build_scheduler(config: FlowMatchConfig, *, device: torch.device) -> FlowMatchScheduler:
    return FlowMatchScheduler(config, device=device)


__all__ = [
    "FlowMatchScheduler",
    "apply_time_shift",
    "build_scheduler",
    "shift_from_mu",
]
