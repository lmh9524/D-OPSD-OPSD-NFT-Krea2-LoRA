"""Group-relative advantage: the baseline that replaces a critic.

Each prompt is sampled ``G`` times and the rewards within that group are centred on their own mean,
so the baseline costs nothing to train and the variance it removes is the part shared by the whole
group — the prompt's difficulty.

## The baseline is always within-group. Only the divisor is a choice.

Centring on a *global* mean would not be GRPO: it would leave prompt difficulty in the advantage,
which is precisely what the group is for. So the mean is per-group unconditionally, and
``global_std`` changes the divisor alone. That matches verl-omni, whose ``id2mean`` is per-group
while its ``batch_std`` is not, and the asymmetry is worth stating because a config field named
"global" reads as though it moved both.

## Why the global standard deviation needs sums, not an averaged standard deviation

The obvious implementation is to compute each rank's standard deviation and all-reduce the mean of
them. That is wrong: the standard deviation of a concatenation is not the mean of the parts'
standard deviations. It is also wrong *quietly* — the number has the right magnitude, so the
advantage scale drifts by a few percent and nothing anywhere raises. So :func:`group_advantage`
reduces ``(sum, sum of squares, count)`` and reconstructs the variance from those, which is exact
for any partition across any number of ranks.

Under the whole-group layout the within-group path needs no collective at all, which is the reason
that layout was chosen: see ``docs/rl-design.md``.

## Bessel's correction

``torch.std`` defaults to the unbiased estimator, and verl-omni's group standard deviation inherits
that default. It is matched here rather than "corrected", because the two differ by
``sqrt(G / (G - 1))`` — 7% at ``G = 8`` — and an advantage scale that silently differs from every
published recipe is worse than a debatable estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from dflow.config.rl import GroupConfig


@dataclass(frozen=True)
class AdvantageStats:
    """What the loop logs about the reward distribution. Already reduced where it needs to be."""

    reward_mean: float
    reward_std: float
    advantage_absmax: float
    #: Fraction of groups whose members all scored within ``epsilon`` of each other. These
    #: contribute no gradient, so a value drifting toward 1 means the reward has saturated and the
    #: run has quietly stopped learning — the failure mode that looks like "it converged".
    degenerate_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "reward/mean": self.reward_mean,
            "reward/std": self.reward_std,
            "advantage/absmax": self.advantage_absmax,
            "advantage/degenerate_groups": self.degenerate_fraction,
        }


def _global_std(rewards: torch.Tensor, process_group, epsilon: float) -> torch.Tensor:
    """The standard deviation over every reward on every rank, from reduced moments.

    Not an all-reduced mean of per-rank standard deviations, which would be close enough to look
    right and wrong enough to change the advantage scale. See the module docstring.
    """
    stats = torch.stack(
        [
            rewards.sum(),
            (rewards**2).sum(),
            torch.tensor(float(rewards.numel()), dtype=rewards.dtype, device=rewards.device),
        ]
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=process_group)
    total, total_squared, count = stats
    if count < 2:
        return torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    mean = total / count
    # Bessel-corrected, matching torch.std's default.
    variance = (total_squared - count * mean**2) / (count - 1)
    return variance.clamp_min(0.0).sqrt()


def group_advantage(
    rewards: torch.Tensor,
    config: GroupConfig,
    *,
    process_group=None,
) -> tuple[torch.Tensor, AdvantageStats]:
    """Centre each group's rewards on its own mean. ``(P, G) -> (P, G)``.

    Args:
        rewards: ``(P, G)`` — ``P`` prompts on this rank, ``G`` trajectories each. Whole groups,
            never a group split across ranks; that is what keeps the within-group path exact
            without communication.
        config: the group settings.
        process_group: the data-parallel group, used **only** when ``config.global_std`` is set.
            Passing it in the default mode is harmless and does nothing.

    Returns the advantages and the statistics worth logging. The statistics are computed here
    rather than in the loop because ``degenerate_fraction`` needs the per-group spread, which is
    gone by the time the advantages come back.
    """
    if rewards.ndim != 2:
        raise ValueError(
            f"rewards must be (prompts, group_size), got {tuple(rewards.shape)}. Whole groups: "
            f"the baseline is only meaningful over a complete group."
        )
    prompts, group_size = rewards.shape
    if group_size != config.size:
        raise ValueError(
            f"rewards has {group_size} trajectories per prompt but config.size is {config.size}"
        )
    if not torch.isfinite(rewards).all():
        raise FloatingPointError(
            "a reward is not finite. Stopping rather than letting it reach the advantage, where "
            "one NaN poisons its whole group's baseline."
        )

    scores = rewards.float()
    # The mean is per-group unconditionally — that is what makes this GRPO rather than plain
    # reward centring.
    mean = scores.mean(dim=1, keepdim=True)
    centred = scores - mean

    within = scores.std(dim=1, keepdim=True) if group_size > 1 else torch.zeros_like(mean)
    if not config.normalise_by_std:
        advantages = centred  # Dr.GRPO: centring alone removes the baseline
    elif config.global_std:
        advantages = centred / (_global_std(scores, process_group, config.epsilon) + config.epsilon)
    else:
        advantages = centred / (within + config.epsilon)

    degenerate = float((within.reshape(-1) <= config.epsilon).float().mean()) if prompts else 0.0
    stats = AdvantageStats(
        reward_mean=float(scores.mean()),
        reward_std=float(scores.std()) if scores.numel() > 1 else 0.0,
        advantage_absmax=float(advantages.abs().max()) if advantages.numel() else 0.0,
        degenerate_fraction=degenerate,
    )
    return advantages, stats


__all__ = ["AdvantageStats", "group_advantage"]
