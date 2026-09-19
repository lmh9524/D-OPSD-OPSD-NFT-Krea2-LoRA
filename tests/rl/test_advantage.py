"""Group-relative advantage. The baseline is the quantity most likely to be quietly wrong.

Every failure here yields finite advantages of a plausible magnitude:

* centring on a global mean instead of a per-group one leaves prompt difficulty in the advantage,
  which is the one thing the group exists to remove;
* an all-reduced *mean of standard deviations* is not the standard deviation of the concatenation,
  and differs by a few percent — enough to change the effective step size, not enough to notice;
* the biased estimator differs from the unbiased one by ``sqrt(G/(G-1))``, so a recipe's learning
  rate stops meaning what it meant.

So these tests check the numbers against closed forms, not that a tensor of the right shape
appears.
"""

from __future__ import annotations

import pytest
import torch

# Guarded like every other dflow-importing test file. ``dflow/__init__.py`` is the public API
# surface and re-exports from ``models/``, so importing anything under ``dflow.`` pulls in
# diffusers — even here, where nothing under test needs it. Without this the CI fast job, which
# installs no diffusers and is the gate that must stay green, errors at collection instead of
# skipping.
pytest.importorskip("diffusers")

from dflow.config.rl import GroupConfig  # noqa: E402
from dflow.rl.advantage import group_advantage  # noqa: E402

GROUPS = 3
SIZE = 4


@pytest.fixture
def config():
    return GroupConfig(size=SIZE)


@pytest.fixture
def rewards():
    torch.manual_seed(0)
    # Deliberately offset per group, so a global-mean baseline would leave that offset behind.
    return torch.randn(GROUPS, SIZE) + torch.tensor([[0.0], [10.0], [-5.0]])


# ------------------------------------------------------------------- the baseline is per-group


def test_each_group_is_centred_on_its_own_mean(rewards, config):
    advantages, _ = group_advantage(rewards, config)
    torch.testing.assert_close(
        advantages.mean(dim=1), torch.zeros(GROUPS), rtol=0, atol=1e-6
    )


def test_a_constant_offset_per_group_is_removed(config):
    """The point of the group: prompt difficulty must not reach the advantage."""
    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    plain, _ = group_advantage(base.repeat(2, 1), config)
    offset, _ = group_advantage(
        base.repeat(2, 1) + torch.tensor([[0.0], [100.0]]), config
    )
    torch.testing.assert_close(plain, offset, rtol=1e-6, atol=1e-6)


def test_within_group_std_is_the_unbiased_estimator(rewards, config):
    """Matched to ``torch.std``'s default and to verl-omni, because the two differ by 7% at G=8."""
    advantages, _ = group_advantage(rewards, config)
    expected = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        rewards.std(dim=1, keepdim=True) + config.epsilon
    )
    torch.testing.assert_close(advantages, expected, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------------- global_std


def test_global_std_divides_every_group_by_one_number(rewards):
    """And that number is the standard deviation of the concatenation, from reduced moments."""
    config = GroupConfig(size=SIZE, global_std=True)
    advantages, _ = group_advantage(rewards, config)
    expected = (rewards - rewards.mean(dim=1, keepdim=True)) / (
        rewards.std() + config.epsilon
    )
    torch.testing.assert_close(advantages, expected, rtol=1e-6, atol=1e-6)


def test_global_std_still_centres_within_each_group(rewards):
    """Only the divisor is global. A global *mean* would not be GRPO."""
    advantages, _ = group_advantage(rewards, GroupConfig(size=SIZE, global_std=True))
    torch.testing.assert_close(
        advantages.mean(dim=1), torch.zeros(GROUPS), rtol=0, atol=1e-5
    )


def test_the_global_std_is_not_the_mean_of_per_group_stds(rewards):
    """Pins the distinction the obvious implementation gets wrong.

    If ``_global_std`` averaged per-group standard deviations, this data — whose groups have very
    different spreads — would divide by a visibly different number.
    """
    spread = rewards * torch.tensor([[1.0], [5.0], [0.2]])
    global_advantages, _ = group_advantage(
        spread, GroupConfig(size=SIZE, global_std=True)
    )
    naive = (spread - spread.mean(dim=1, keepdim=True)) / (
        spread.std(dim=1, keepdim=True).mean() + 1e-4
    )
    assert not torch.allclose(global_advantages, naive, rtol=1e-3)


# ------------------------------------------------------------------------------- degeneracies


def test_an_all_equal_group_gives_zero_advantage_without_dividing_by_zero(config):
    """Common early, and whenever a reward saturates."""
    advantages, stats = group_advantage(torch.full((2, SIZE), 3.0), config)
    assert torch.all(advantages == 0)
    assert torch.isfinite(advantages).all()
    assert stats.degenerate_fraction == 1.0


def test_degenerate_fraction_reports_partial_saturation(config):
    rewards = torch.cat([torch.full((1, SIZE), 2.0), torch.arange(SIZE).float().reshape(1, SIZE)])
    _, stats = group_advantage(rewards, config)
    assert stats.degenerate_fraction == pytest.approx(0.5)


def test_dr_grpo_skips_the_division_entirely(rewards):
    """``normalise_by_std=False``: the centring is what removes the baseline."""
    config = GroupConfig(size=SIZE, normalise_by_std=False)
    advantages, _ = group_advantage(rewards, config)
    torch.testing.assert_close(
        advantages, rewards - rewards.mean(dim=1, keepdim=True), rtol=0, atol=0
    )


# ---------------------------------------------------------------------------------- rejections


def test_a_non_finite_reward_stops_the_step(config):
    """One NaN would otherwise poison its whole group's baseline and keep going."""
    rewards = torch.zeros(2, SIZE)
    rewards[1, 2] = float("nan")
    with pytest.raises(FloatingPointError, match="not finite"):
        group_advantage(rewards, config)


def test_a_flat_reward_vector_is_refused(config):
    """``(P*G,)`` still has the right element count, and would centre on the wrong axis."""
    with pytest.raises(ValueError, match="prompts, group_size"):
        group_advantage(torch.zeros(GROUPS * SIZE), config)


def test_a_group_size_mismatch_is_refused(config):
    with pytest.raises(ValueError, match="config.size"):
        group_advantage(torch.zeros(GROUPS, SIZE + 1), config)


def test_a_group_of_one_is_refused_at_config_time():
    with pytest.raises(ValueError, match="no-op"):
        GroupConfig(size=1)


# ---------------------------------------------------------------------------------- statistics


def test_stats_describe_the_reward_distribution(rewards, config):
    _, stats = group_advantage(rewards, config)
    assert stats.reward_mean == pytest.approx(float(rewards.mean()), rel=1e-6)
    assert stats.reward_std == pytest.approx(float(rewards.std()), rel=1e-6)
    assert set(stats.as_dict()) == {
        "reward/mean",
        "reward/std",
        "advantage/absmax",
        "advantage/degenerate_groups",
    }
