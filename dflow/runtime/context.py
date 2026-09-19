"""Device mesh and process-group setup.

The mesh is **always three-dimensional** — ``(dp_replicate, dp_shard, cp)`` — even
while ``cp == 1``. This is deliberate: adding context parallelism later must not change
what ``dp_rank`` means. ``dp_rank`` selects which shard of the dataset a process reads
and is baked into every checkpoint through the per-rank RNG state, so changing it
invalidates resume for existing runs. Building the third dimension now costs nothing
(with ``cp == 1`` every value is numerically identical to a 2-D mesh) and makes CP a
local change later.

Two invariants worth stating because breaking them is silent:

* ``dp_rank`` / ``dp_size`` **exclude** the CP dimension. Ranks inside a CP group
  process different slices of the *same* sample, so they must receive the same batch and
  the same noise. Only ``dp`` ranks get different data.
* Parameters are sharded over ``dp_shard * cp``. That folds CP into the FSDP dimension,
  which is why gradients need an explicit correction: FSDP's reduce-scatter *averages*,
  while CP semantics require a *sum* over the CP dimension. See
  ``dflow.runtime.cp.reduce_cp_gradients``.

torch notes:

* ``DeviceMesh._unflatten`` does not exist in torch 2.9, so the mesh is built at full rank and a
  slice is flattened instead.
* ``DeviceMesh._flatten(name)`` registers its result in a **global cache that survives
  ``destroy_process_group()``**. A process therefore cannot tear the mesh down and build a second
  one under the same flattened name: the cached mesh still points at the dead group, and the next
  collective fails with "Could not resolve the process group registered under the name N".
  Training builds the mesh once, so this is only a constraint on tests and multi-stage scripts —
  share one process group rather than re-initialising.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from dflow.config import DistributedConfig


@dataclass(frozen=True, slots=True)
class MeshBundle:
    """Resolved topology for one process."""

    device: torch.device
    rank: int
    local_rank: int
    world_size: int

    # Data-parallel identity, excluding CP. This is what the dataloader must use.
    dp_rank: int
    dp_size: int

    dp_replicate_size: int
    dp_shard_size: int
    cp_size: int

    # None in single-process runs, where sharding is a no-op.
    world_mesh: DeviceMesh | None
    fsdp_mesh: DeviceMesh | None
    cp_mesh: DeviceMesh | None

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_mesh is not None

    @property
    def fsdp_size(self) -> int:
        """Number of ranks a parameter is sharded across."""
        return self.dp_shard_size * self.cp_size

    def describe(self) -> str:
        return (
            f"world={self.world_size} "
            f"dp={self.dp_size} (replicate={self.dp_replicate_size}, shard={self.dp_shard_size}) "
            f"cp={self.cp_size} fsdp={self.fsdp_size} "
            f"rank={self.rank} dp_rank={self.dp_rank} device={self.device}"
        )


def _resolve_device() -> tuple[str, torch.device, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        return "cuda", device, local_rank
    return "cpu", torch.device("cpu"), local_rank


def build_meshes(
    *,
    device_type: str,
    dp_replicate: int,
    dp_shard: int,
    cp: int,
) -> tuple[DeviceMesh, DeviceMesh, DeviceMesh | None]:
    """Build ``(world_mesh, fsdp_mesh, cp_mesh)``.

    ``fsdp_mesh`` is the ``dp_shard x cp`` slice flattened to one dimension, because
    FSDP2 shards over a single logical axis.
    """
    world_mesh = init_device_mesh(
        device_type,
        mesh_shape=(dp_replicate, dp_shard, cp),
        mesh_dim_names=("dp_replicate", "dp_shard", "cp"),
    )
    fsdp_mesh = world_mesh["dp_shard", "cp"]._flatten("fsdp")
    cp_mesh = world_mesh["cp"] if cp > 1 else None
    return world_mesh, fsdp_mesh, cp_mesh


def compute_dp_rank(world_mesh: DeviceMesh, *, dp_shard: int) -> int:
    """Row-major index over ``(dp_replicate, dp_shard)``, ignoring CP."""
    return world_mesh.get_local_rank("dp_replicate") * dp_shard + world_mesh.get_local_rank("dp_shard")


def init_distributed(config: DistributedConfig) -> MeshBundle:
    """Initialise the process group (if needed) and resolve the mesh.

    A single-process run does not initialise a process group at all, so plain
    ``python experiments/...`` works without ``torchrun``.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device_type, device, local_rank = _resolve_device()
    dp_shard = config.resolve_dp_shard(world_size)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device_type == "cuda" else "gloo")

    if not dist.is_initialized():
        # Reaching here implies world_size == 1, and resolve_dp_shard has already
        # rejected any cp > 1 for that case (the degrees cannot multiply to 1).
        return MeshBundle(
            device=device,
            rank=0,
            local_rank=local_rank,
            world_size=1,
            dp_rank=0,
            dp_size=1,
            dp_replicate_size=config.dp_replicate,
            dp_shard_size=dp_shard,
            cp_size=config.cp,
            world_mesh=None,
            fsdp_mesh=None,
            cp_mesh=None,
        )

    world_mesh, fsdp_mesh, cp_mesh = build_meshes(
        device_type=device_type,
        dp_replicate=config.dp_replicate,
        dp_shard=dp_shard,
        cp=config.cp,
    )
    return MeshBundle(
        device=device,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        dp_rank=compute_dp_rank(world_mesh, dp_shard=dp_shard),
        dp_size=config.dp_replicate * dp_shard,
        dp_replicate_size=config.dp_replicate,
        dp_shard_size=dp_shard,
        cp_size=config.cp,
        world_mesh=world_mesh,
        fsdp_mesh=fsdp_mesh,
        cp_mesh=cp_mesh,
    )


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


__all__ = [
    "MeshBundle",
    "build_meshes",
    "compute_dp_rank",
    "destroy_distributed",
    "init_distributed",
]
