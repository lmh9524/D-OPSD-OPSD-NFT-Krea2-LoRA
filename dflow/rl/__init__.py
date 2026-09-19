"""L4: the RL algorithm — rollout, advantage, objective.

Sits beside ``tasks/`` rather than inside it, and imports neither it nor ``models/``: a rollout
walks a sequence that satisfies ``SequenceLike`` and calls a ``VelocityFn``, both structural types
declared in ``protocol.py``. *The consumer owns the interface* — the same rule that puts
``ParallelSpec`` in ``runtime/``.

``trainer/rl.py`` (L5) drives these; it owns the phase ordering and everything that fails silently
across ranks, exactly as ``fit()`` does for SFT. See ``docs/rl-design.md``.
"""

from dflow.rl.advantage import AdvantageStats, group_advantage
from dflow.rl.objective import (
    NFTStats,
    ObjectiveStats,
    diffusion_nft_loss,
    nft_reward_prob,
    ppo_clip_loss,
)
from dflow.rl.protocol import SequenceLike, VelocityFn
from dflow.rl.rollout import replay_logprobs, sample_group, sample_window
from dflow.rl.trajectory import NFTRolloutGroup, RolloutGroup

__all__ = [
    "AdvantageStats",
    "NFTRolloutGroup",
    "NFTStats",
    "ObjectiveStats",
    "RolloutGroup",
    "SequenceLike",
    "VelocityFn",
    "diffusion_nft_loss",
    "group_advantage",
    "nft_reward_prob",
    "ppo_clip_loss",
    "replay_logprobs",
    "sample_group",
    "sample_window",
]
