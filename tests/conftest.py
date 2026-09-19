"""Shared fixtures.

The FLUX.2 tests run against a deliberately tiny model built from the real
``Flux2Transformer2DModel``, not a mock. That exercises the actual calling convention —
4-D position ids, text tokens stripped from the output, the timestep scaling — without
needing the gated 9B checkpoint or a GPU.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_flux2_config() -> dict:
    """Smallest coherent FLUX.2 config.

    ``axes_dims_rope`` must sum to ``attention_head_dim`` — the head dimension is split
    four ways, one axis per id component (reference index, height, width, text position).
    """
    return dict(
        patch_size=1,
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=24,
        timestep_guidance_channels=32,
        mlp_ratio=2.0,
        axes_dims_rope=(4, 4, 4, 4),
        rope_theta=2000,
        eps=1e-6,
        guidance_embeds=False,
    )


@pytest.fixture
def tiny_flux2(tiny_flux2_config):
    pytest.importorskip("diffusers")
    from dflow.vendor.flux2 import Flux2Transformer2DModel

    torch.manual_seed(0)
    return Flux2Transformer2DModel.from_config(tiny_flux2_config).eval()


def build_ids(
    *,
    target_grid: tuple[int, int],
    ref_grids: tuple[tuple[int, int], ...] = (),
    t_scale: int = 10,
) -> torch.Tensor:
    """Build ``(1, seq, 4)`` image ids the way the klein pipeline does.

    Target tokens sit at ``T=0``; reference *i* sits at ``T = t_scale * (i + 1)``. The T
    axis is what lets the model tell references apart from the target and from each other.
    Provisional: ``dflow/tasks/ref2img/conditioning.py`` will own this once the task lands,
    at which point these tests should call that instead.
    """
    product = torch.cartesian_prod
    zero = torch.arange(1)

    height, width = target_grid
    spans = [product(zero, torch.arange(height), torch.arange(width), zero)]
    for index, (height, width) in enumerate(ref_grids):
        coordinate = torch.tensor([t_scale + t_scale * index])
        spans.append(product(coordinate, torch.arange(height), torch.arange(width), zero))
    return torch.cat(spans, dim=0).unsqueeze(0).float()


def build_text_ids(length: int) -> torch.Tensor:
    """``(1, length, 4)``: text uses the fourth axis for position, the rest are zero."""
    zero = torch.arange(1)
    return torch.cartesian_prod(zero, zero, zero, torch.arange(length)).unsqueeze(0).float()


@pytest.fixture(scope="session")
def tiny_krea2_config() -> dict:
    """Smallest coherent Krea 2 config.

    ``axes_dims_rope`` must sum to ``attention_head_dim``, split **three** ways here — Krea 2 has no
    text position axis, so its ids are ``(T, H, W)`` where FLUX.2's are ``(T, H, W, L)``.
    """
    return dict(
        in_channels=16,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        timestep_embed_dim=8,
        text_hidden_dim=16,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=1,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(4, 2, 2),
        rope_theta=1000.0,
        norm_eps=1e-5,
    )


@pytest.fixture
def tiny_krea2(tiny_krea2_config):
    pytest.importorskip("diffusers")
    from dflow.vendor.krea2 import Krea2Transformer2DModel

    torch.manual_seed(0)
    return Krea2Transformer2DModel.from_config(tiny_krea2_config).eval()


@pytest.fixture(scope="session")
def distributed_session():
    """One process group for the whole test session.

    Session-scoped rather than per-test because ``DeviceMesh._flatten`` registers its result in a
    global cache that **survives** ``destroy_process_group()``. Tearing a group down and building a
    second mesh with the same flattened name returns a mesh bound to the dead group, which fails
    with "Could not resolve the process group registered under the name N". Production builds the
    mesh once, so this fixture matches production rather than working around it.

    gloo on CPU: the sharding logic, DTensor parameters and the meta -> shard -> materialise order
    are backend-independent, and CPU keeps these tests runnable in CI and on hosts whose NCCL
    cannot initialise a network plugin. Cross-rank collective behaviour needs the cluster anyway.
    """
    import os

    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already initialised")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group(backend="gloo")
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="session")
def one_rank_meshes(distributed_session):
    """``(world_mesh, fsdp_mesh, cp_mesh)`` for a single rank on CPU."""
    from dflow.runtime.context import build_meshes

    return build_meshes(device_type="cpu", dp_replicate=1, dp_shard=1, cp=1)
