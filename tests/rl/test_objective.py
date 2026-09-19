"""The clipped policy objective. Every mistake here produces a loss curve that falls.

The sign, the clip direction, the detach and the advantage broadcast are all invisible to a shape
check and all invisible to "the loss went down". So each gets a test against a closed form.
"""

from __future__ import annotations

import math

import pytest
import torch

# Guarded like every other dflow-importing test file. ``dflow/__init__.py`` is the public API
# surface and re-exports from ``models/``, so importing anything under ``dflow.`` pulls in
# diffusers — even here, where nothing under test needs it. Without this the CI fast job, which
# installs no diffusers and is the gate that must stay green, errors at collection instead of
# skipping.
pytest.importorskip("diffusers")

from dflow.config.rl import PPOConfig  # noqa: E402
from dflow.rl.objective import ppo_clip_loss  # noqa: E402

WINDOW = 2
GROUP = 4


@pytest.fixture
def config():
    # Wide enough that the clip only fires where a test makes it.
    return PPOConfig(clip_ratio=0.1, adv_clip_max=5.0)


def _flat(value: float) -> torch.Tensor:
    return torch.full((WINDOW, GROUP), value)


# --------------------------------------------------------------------------- the closed forms


def test_at_ratio_one_the_loss_is_minus_the_advantage(config):
    """The first inner epoch: the policy being scored is the policy that acted."""
    logprobs = _flat(-1.5)
    advantages = torch.tensor([1.0, -2.0, 0.5, 3.0])
    loss, stats = ppo_clip_loss(
        logprobs=logprobs, old_logprobs=logprobs.clone(), advantages=advantages, config=config
    )
    assert float(loss) == pytest.approx(-float(advantages.mean()), rel=1e-6)
    assert stats.ratio_mean == pytest.approx(1.0, rel=1e-6)
    assert stats.clipfrac == 0.0
    assert stats.ppo_kl == pytest.approx(0.0, abs=1e-7)


def test_the_sign_rewards_a_positive_advantage(config):
    """Raising the log-prob of a better-than-average action must lower the loss."""
    old = _flat(-1.0)
    advantages = torch.ones(GROUP)
    more_likely, _ = ppo_clip_loss(
        logprobs=old + 0.01, old_logprobs=old, advantages=advantages, config=config
    )
    less_likely, _ = ppo_clip_loss(
        logprobs=old - 0.01, old_logprobs=old, advantages=advantages, config=config
    )
    assert float(more_likely) < float(less_likely)


def test_the_sign_flips_for_a_negative_advantage(config):
    old = _flat(-1.0)
    advantages = -torch.ones(GROUP)
    more_likely, _ = ppo_clip_loss(
        logprobs=old + 0.01, old_logprobs=old, advantages=advantages, config=config
    )
    less_likely, _ = ppo_clip_loss(
        logprobs=old - 0.01, old_logprobs=old, advantages=advantages, config=config
    )
    assert float(more_likely) > float(less_likely)


def test_a_zero_advantage_gives_no_gradient(config):
    logprobs = _flat(-1.0).requires_grad_(True)
    loss, _ = ppo_clip_loss(
        logprobs=logprobs,
        old_logprobs=_flat(-1.2),
        advantages=torch.zeros(GROUP),
        config=config,
    )
    loss.backward()
    assert float(logprobs.grad.abs().sum()) == 0.0


# ----------------------------------------------------------------------------------- clipping


def test_the_clip_is_pessimistic(config):
    """``max`` of unclipped and clipped, not ``min``.

    With a positive advantage and a ratio far above the ceiling, the clipped term is the *larger*
    loss, so it wins. ``min`` would take the unclipped one and keep rewarding a move already too
    large — training happily in the wrong direction.
    """
    old = _flat(0.0)
    ratio = 2.0
    loss, stats = ppo_clip_loss(
        logprobs=_flat(math.log(ratio)),
        old_logprobs=old,
        advantages=torch.ones(GROUP),
        config=config,
    )
    assert float(loss) == pytest.approx(-(1.0 + config.clip_ratio), rel=1e-6)
    assert stats.clipfrac == 1.0
    assert stats.clipfrac_high == 1.0
    assert stats.clipfrac_low == 0.0


def test_the_lower_clip_fires_and_is_reported_separately(config):
    loss, stats = ppo_clip_loss(
        logprobs=_flat(math.log(0.5)),
        old_logprobs=_flat(0.0),
        advantages=-torch.ones(GROUP),
        config=config,
    )
    assert stats.clipfrac_low == 1.0
    assert stats.clipfrac_high == 0.0
    assert float(loss) == pytest.approx(1.0 - config.clip_ratio, rel=1e-6)


def test_a_ratio_inside_the_band_is_not_clipped(config):
    ratio = 1.0 + config.clip_ratio / 2
    loss, stats = ppo_clip_loss(
        logprobs=_flat(math.log(ratio)),
        old_logprobs=_flat(0.0),
        advantages=torch.ones(GROUP),
        config=config,
    )
    assert stats.clipfrac == 0.0
    assert float(loss) == pytest.approx(-ratio, rel=1e-6)


def test_the_advantage_clamp_bounds_one_outlier(config):
    """A single extreme reward would otherwise dominate its group."""
    advantages = torch.tensor([1.0, 1.0, 1.0, 1000.0])
    loss, _ = ppo_clip_loss(
        logprobs=_flat(0.0), old_logprobs=_flat(0.0), advantages=advantages, config=config
    )
    bounded = advantages.clamp(-config.adv_clip_max, config.adv_clip_max)
    assert float(loss) == pytest.approx(-float(bounded.mean()), rel=1e-6)


# ------------------------------------------------------------------------------------ gradient


def test_gradient_reaches_the_current_policy_only(config):
    """``old_logprobs`` are recorded constants; gradient through them flips half the objective."""
    # A ratio inside the clip band: outside it the clamp legitimately has zero gradient, which
    # would make this test pass for the wrong reason.
    logprobs = _flat(-1.0).requires_grad_(True)
    old = _flat(-1.02).requires_grad_(True)
    loss, _ = ppo_clip_loss(
        logprobs=logprobs, old_logprobs=old, advantages=torch.ones(GROUP), config=config
    )
    loss.backward()
    assert logprobs.grad is not None and float(logprobs.grad.abs().sum()) > 0
    assert old.grad is None or float(old.grad.abs().sum()) == 0.0


def test_the_advantage_receives_no_gradient(config):
    advantages = torch.ones(GROUP, requires_grad=True)
    loss, _ = ppo_clip_loss(
        logprobs=_flat(-1.0).requires_grad_(True),
        old_logprobs=_flat(-1.0),
        advantages=advantages,
        config=config,
    )
    loss.backward()
    assert advantages.grad is None or float(advantages.grad.abs().sum()) == 0.0


# --------------------------------------------------------------------------------- broadcasting


def test_a_per_trajectory_advantage_is_broadcast_across_the_window(config):
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0])
    per_trajectory, _ = ppo_clip_loss(
        logprobs=_flat(0.0), old_logprobs=_flat(0.0), advantages=advantages, config=config
    )
    expanded, _ = ppo_clip_loss(
        logprobs=_flat(0.0),
        old_logprobs=_flat(0.0),
        advantages=advantages.expand(WINDOW, GROUP),
        config=config,
    )
    assert float(per_trajectory) == pytest.approx(float(expanded), rel=1e-9)


def test_an_advantage_of_the_window_length_is_refused(config):
    """``(W,)`` broadcasts against ``(W, G)`` when G == W, applying the wrong one to every step."""
    with pytest.raises(ValueError, match="one advantage per trajectory|advantages has"):
        ppo_clip_loss(
            logprobs=torch.zeros(GROUP, GROUP),
            old_logprobs=torch.zeros(GROUP, GROUP),
            advantages=torch.ones(GROUP - 1),
            config=config,
        )


def test_a_merely_broadcastable_advantage_is_refused(config):
    with pytest.raises(ValueError, match="not something that merely broadcasts"):
        ppo_clip_loss(
            logprobs=_flat(0.0),
            old_logprobs=_flat(0.0),
            advantages=torch.ones(1, GROUP),
            config=config,
        )


def test_mismatched_logprob_shapes_are_refused(config):
    with pytest.raises(ValueError, match="must match"):
        ppo_clip_loss(
            logprobs=_flat(0.0),
            old_logprobs=torch.zeros(WINDOW + 1, GROUP),
            advantages=torch.ones(GROUP),
            config=config,
        )


def test_a_one_dimensional_logprob_tensor_is_refused(config):
    """The window axis must be explicit, or the reduction silently averages over the wrong thing."""
    with pytest.raises(ValueError, match=r"\(window, group\)"):
        ppo_clip_loss(
            logprobs=torch.zeros(GROUP),
            old_logprobs=torch.zeros(GROUP),
            advantages=torch.ones(GROUP),
            config=config,
        )


# ------------------------------------------------------------------------------------- config


def test_the_clip_ratio_default_is_on_the_averaged_logprob_scale():
    """A threshold from LLM RL, where log-probs are summed over tokens, would never fire.

    ``step_logprob`` reduces with a mean, so ratios sit within a hair of 1. verl-omni's Qwen-Image
    recipe uses 1e-5; anything of order 0.2 is the LLM scale and is the mistake this default
    guards against.
    """
    assert PPOConfig().clip_ratio <= 1e-3


@pytest.mark.parametrize(
    "kwargs", [{"clip_ratio": 0.0}, {"adv_clip_max": -1.0}, {"inner_epochs": 0}]
)
def test_invalid_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        PPOConfig(**kwargs)
