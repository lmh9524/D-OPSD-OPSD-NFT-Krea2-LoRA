"""Seeding and per-rank RNG.

Two rules, both of which fail silently when broken.

**Data-parallel ranks must draw different noise; context-parallel ranks must draw the
same.** CP ranks process different slices of the *same* sample, so if their noise differs
the slices no longer belong to one coherent latent. Seeding from ``dp_rank`` — which
excludes the CP dimension by construction — gets both cases right at once.

**RNG state is per rank and must round-trip through the checkpoint.** Restoring one shared
state, or restoring rank 0's state everywhere, makes every rank draw identical noise after
a resume. Training continues and the loss curve looks plausible; the effective batch just
collapses to one distinct sample.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from dflow.runtime.context import MeshBundle


def seed_everything(seed: int, *, mesh: MeshBundle) -> torch.Generator:
    """Seed global RNGs and return the per-rank generator used for training randomness.

    Global seeding covers library code we do not control (dataloader workers, augmentation).
    The returned generator is what ``step_fn`` should use for noise and timesteps, because it
    is the only source whose state is checkpointed.
    """
    # Global state is offset by dp_rank so worker-side randomness differs across data
    # shards, while staying identical across a CP group.
    global_seed = seed + mesh.dp_rank
    random.seed(global_seed)
    np.random.seed(global_seed % (2**32))
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)

    generator = torch.Generator(device=mesh.device)
    generator.manual_seed(global_seed)
    return generator


def matmul_precision(setting: str) -> None:
    """TF32 policy for fp32 matmuls. ``high`` is the usual training choice."""
    torch.set_float32_matmul_precision(setting)


def gather_rng_states(generator: torch.Generator, *, mesh: MeshBundle) -> list[torch.Tensor]:
    """Collect every rank's generator state, ordered by global rank.

    Gathering rather than saving locally keeps a checkpoint a single object that any rank can
    write, at the cost of one small all-gather per save.
    """
    import torch.distributed as dist

    state = generator.get_state()
    if not dist.is_initialized():
        return [state]

    states: list[torch.Tensor | None] = [None] * mesh.world_size
    dist.all_gather_object(states, state)
    return [s for s in states if s is not None]


def restore_rng_state(
    generator: torch.Generator, states: list[torch.Tensor], *, mesh: MeshBundle
) -> None:
    """Restore this rank's slice of a gathered RNG state.

    Refuses a world-size change rather than silently reassigning states to different ranks,
    which would give two ranks the same stream.
    """
    if len(states) != mesh.world_size:
        raise ValueError(
            f"checkpoint holds {len(states)} RNG states but the world size is "
            f"{mesh.world_size}. Resuming would reassign streams across ranks; re-launch "
            f"with the original world size, or start a fresh run."
        )
    generator.set_state(states[mesh.rank])


__all__ = [
    "gather_rng_states",
    "matmul_precision",
    "restore_rng_state",
    "seed_everything",
]
