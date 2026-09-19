"""Fixtures for the RL tests.

``rl/`` takes its sequence and its policy as structural types, which pays off here: the whole
rollout loop can be driven by an **analytic velocity field** over a fake sequence. So these tests
are exact rather than approximate, run in milliseconds, and need neither diffusers nor a
checkpoint — the things that would otherwise make a replay comparison a tolerance argument.

A local conftest rather than the root one: these helpers are meaningless outside ``tests/rl/``, and
the root conftest's fixtures still apply here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

GROUP = 4
TARGET_LEN = 6
REFERENCE_LEN = 3
CHANNELS = 8
STEPS = 10
MU = 1.15
NOISE = 0.7


@dataclass(frozen=True)
class FakeSequence:
    """``SequenceLike`` with FLUX.2's layout: ``[target | references]``.

    Deliberately not a subclass of anything and deliberately not imported from ``tasks/`` — if
    ``rl/`` can only be tested with a real task's sequence, the protocol is not doing its job.
    """

    tokens: torch.Tensor
    target_len: int

    def replace_target(self, target_tokens: torch.Tensor) -> FakeSequence:
        if target_tokens.shape[1] != self.target_len:
            raise ValueError(f"target span is {self.target_len}, got {target_tokens.shape[1]}")
        return FakeSequence(
            tokens=torch.cat([target_tokens, self.tokens[:, self.target_len :]], dim=1),
            target_len=self.target_len,
        )

    @property
    def references(self) -> torch.Tensor:
        return self.tokens[:, self.target_len :]


def analytic_velocity(target_len: int, *, scale: float = 0.5, bias: float = 0.25):
    """A deterministic velocity field that genuinely depends on the latent it is given.

    Linear in ``x`` and affine in ``sigma``, so a replay that re-derived it from the wrong latent
    or the wrong sigma produces a different answer — which is what makes the replay test
    non-vacuous. Reads the target span at the front, as FLUX.2's layout puts it.
    """

    def velocity_fn(tokens: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        span = tokens[:, :target_len]
        return scale * span + bias * sigma.reshape(-1, *([1] * (span.ndim - 1)))

    return velocity_fn


def counting_velocity(target_len: int, calls: list[torch.Tensor], **kwargs):
    """``analytic_velocity``, recording the sigma of every call."""
    inner = analytic_velocity(target_len, **kwargs)

    def velocity_fn(tokens: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        calls.append(sigma.clone())
        return inner(tokens, sigma)

    return velocity_fn


@pytest.fixture
def generator() -> torch.Generator:
    return torch.Generator(device=torch.device("cpu")).manual_seed(0)


@pytest.fixture
def sequence() -> FakeSequence:
    """A group's worth of conditioning, target span first.

    The target positions hold arbitrary values: a rollout starts from pure noise, so whatever is
    there must be ignored. One of the tests checks exactly that.
    """
    torch.manual_seed(1)
    tokens = torch.randn(GROUP, TARGET_LEN + REFERENCE_LEN, CHANNELS)
    return FakeSequence(tokens=tokens, target_len=TARGET_LEN)


@pytest.fixture
def schedule():
    from dflow.schedulers.flow_sde import build_sde_schedule

    return build_sde_schedule(STEPS, mu=MU)


@pytest.fixture
def group(sequence, schedule, generator):
    """One rolled-out group, window ``(1, 4)`` — three trained steps, ending before the schedule."""
    from dflow.rl.rollout import sample_group

    return sample_group(
        sequence=sequence,
        velocity_fn=analytic_velocity(TARGET_LEN),
        schedule=schedule,
        window=(1, 4),
        noise_level=NOISE,
        generator=generator,
        prompt_index=3,
    )
