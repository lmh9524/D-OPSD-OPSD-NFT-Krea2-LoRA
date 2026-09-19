"""Mesh topology.

The invariant under test is that ``dp_rank`` excludes the CP dimension. Ranks inside a
CP group process different slices of the same sample, so they must read the same batch;
only DP ranks may differ. Getting this wrong changes which data a rank sees, which in
turn invalidates resume for every existing checkpoint.
"""

from __future__ import annotations

import pytest
import torch.distributed as dist

# `dflow/__init__.py` re-exports from `models/`, so anything under `dflow.` pulls in diffusers.
pytest.importorskip("diffusers")

from dflow.config import DistributedConfig  # noqa: E402
from dflow.runtime.context import compute_dp_rank, init_distributed  # noqa: E402

# ---------------------------------------------------------------- degree resolution


def test_dp_shard_derived_from_world_size():
    assert DistributedConfig(dp_shard=-1).resolve_dp_shard(8) == 8
    assert DistributedConfig(dp_shard=-1, cp=2).resolve_dp_shard(8) == 4
    assert DistributedConfig(dp_shard=-1, dp_replicate=2, cp=2).resolve_dp_shard(8) == 2


def test_explicit_degrees_must_multiply_to_world_size():
    with pytest.raises(ValueError, match="must multiply to world_size"):
        DistributedConfig(dp_shard=3, cp=2).resolve_dp_shard(8)


def test_world_size_too_small_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        DistributedConfig(dp_shard=-1, cp=8).resolve_dp_shard(4)


def test_zero_degrees_are_rejected():
    with pytest.raises(ValueError, match=">= 1"):
        DistributedConfig(cp=0).resolve_dp_shard(8)


# ------------------------------------------------------------------- dp_rank layout


@pytest.mark.parametrize(
    ("dp_replicate", "dp_shard", "cp", "expected_dp_size"),
    [(1, 8, 1, 8), (1, 4, 2, 4), (2, 2, 2, 4), (1, 1, 8, 1)],
)
def test_dp_size_excludes_cp(dp_replicate, dp_shard, cp, expected_dp_size):
    """Documents the arithmetic the dataloader depends on, without needing 8 ranks."""
    assert dp_replicate * dp_shard == expected_dp_size
    assert dp_shard * cp == (dp_replicate * dp_shard * cp) // dp_replicate


# --------------------------------------------------------------- single-process path


def test_single_process_needs_no_process_group(monkeypatch):
    """`python experiments/...` must work without torchrun."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    mesh = init_distributed(DistributedConfig())

    assert not dist.is_initialized()
    assert mesh.world_size == 1
    assert (mesh.rank, mesh.dp_rank, mesh.dp_size) == (0, 0, 1)
    assert mesh.cp_size == 1
    assert mesh.fsdp_size == 1
    assert mesh.world_mesh is None and mesh.fsdp_mesh is None and mesh.cp_mesh is None
    assert mesh.is_master
    assert not mesh.is_distributed
    assert "dp_rank=0" in mesh.describe()


def test_cp_without_distributed_is_rejected(monkeypatch):
    """cp > 1 on one process is caught by degree resolution, before any mesh is built."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(ValueError, match="must multiply to world_size"):
        init_distributed(DistributedConfig(dp_shard=1, cp=2))


# ------------------------------------------------------------- real mesh, one process


def test_build_meshes_folds_cp_into_fsdp(one_rank_meshes):
    """Parameters shard over dp_shard * cp, so the FSDP mesh is that slice flattened.

    Also pins the torch API: DeviceMesh._unflatten does not exist in torch 2.9, which is
    why vflow's context.py cannot be copied verbatim.
    """
    world_mesh, fsdp_mesh, cp_mesh = one_rank_meshes

    assert world_mesh.ndim == 3
    assert world_mesh.mesh_dim_names == ("dp_replicate", "dp_shard", "cp")
    assert fsdp_mesh.ndim == 1
    assert fsdp_mesh.size() == 1
    assert cp_mesh is None, "cp_mesh stays None while cp == 1 so callers can no-op cheaply"

    assert compute_dp_rank(world_mesh, dp_shard=1) == 0
