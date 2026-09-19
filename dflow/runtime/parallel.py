"""Activation checkpointing, compilation and FSDP2 — applied in one fixed order.

The order is shared-infrastructure item 1, and it is written once here because every way of
getting it wrong is quiet rather than loud:

    build on meta  ->  activation checkpointing  ->  compile  ->  FSDP2  ->  to_empty  ->  load

* **Meta before FSDP.** FSDP2 decides the shard layout first, so only each rank's shard is ever
  allocated. Loading first and sharding afterwards needs the whole model in host memory on every
  rank — 18 GB x 8 for klein-9B.
* **Checkpointing before compile.** Compiling first bakes the uncheckpointed graph, so the
  wrapper never takes effect and memory silently stays high.
* **Compile before FSDP.** Compiling the sharded module means tracing through the all-gather
  hooks; compiling the block first lets Dynamo see plain module code.
* **`to_empty` after FSDP, not before.** Materialising first defeats meta init.

Activation checkpointing and compilation both delegate to diffusers rather than reimplementing
generic versions. ``enable_gradient_checkpointing`` respects the model's own block structure, and
``compile_repeated_blocks`` compiles one instance per *distinct* block class rather than all 32
separately — for FLUX.2 that is two compilations instead of thirty-two.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from dflow.config import ActivationCheckpointConfig, CompileConfig, FSDPConfig
from dflow.runtime.precision import resolve_dtype
from dflow.runtime.spec import ParallelSpec


def apply_activation_checkpointing(
    model: nn.Module, config: ActivationCheckpointConfig
) -> nn.Module:
    """Delegate to the model's own gradient checkpointing.

    Not ``checkpoint_wrapper`` (vflow's approach): the model already knows which units to
    recompute and threads ``_gradient_checkpointing_func`` through its forward, so wrapping from
    outside would duplicate that and can miss blocks the model treats specially.
    """
    if not config.enabled:
        return model
    if not getattr(model, "_supports_gradient_checkpointing", False):
        raise ValueError(
            f"{type(model).__name__} does not support gradient checkpointing, but "
            f"ActivationCheckpointConfig.enabled is True. Multi-reference sequences do not fit "
            f"without it; either use a smaller resolution or a model that supports it."
        )
    model.enable_gradient_checkpointing()
    return model


def apply_compile(model: nn.Module, config: CompileConfig) -> nn.Module:
    """Compile the repeated blocks, not the whole model.

    ``compile_repeated_blocks`` discovers units via ``_repeated_blocks`` and compiles one per
    distinct class, so FLUX.2 pays two compilations rather than thirty-two.
    """
    if not config.enabled:
        return model
    if not getattr(model, "_repeated_blocks", None):
        raise ValueError(
            f"{type(model).__name__} declares no _repeated_blocks, so per-block compilation "
            f"has nothing to target. Disable CompileConfig or compile the model yourself."
        )
    model.compile_repeated_blocks(backend=config.backend, fullgraph=config.fullgraph)
    return model


def apply_fsdp(
    model: nn.Module,
    spec: ParallelSpec,
    config: FSDPConfig,
    *,
    mesh: DeviceMesh | None,
) -> nn.Module:
    """Fully shard each transformer block, then the root.

    A no-op when ``mesh`` is None, which is the single-process case — so the same code path runs
    under ``python experiments/...`` and under ``torchrun``.

    One ``MixedPrecisionPolicy`` covers the whole model. ``spec.keep_fp32_patterns`` is
    deliberately not consulted: see ``runtime/precision.py`` for why those patterns belong to
    layerwise casting rather than to sharding.
    """
    if not config.enabled or mesh is None:
        return model

    options: dict[str, object] = {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=resolve_dtype(config.param_dtype),
            reduce_dtype=resolve_dtype(config.reduce_dtype),
        ),
        "reshard_after_forward": config.reshard_after_forward,
    }
    if config.cpu_offload:
        options["offload_policy"] = CPUOffloadPolicy()

    names = config.module_names or spec.block_module_names
    for name in names:
        module = model.get_submodule(name)
        blocks = module if isinstance(module, nn.ModuleList) else [module]
        for block in blocks:
            fully_shard(block, **options)
    # The root call covers everything the block calls did not (embedders, output projection).
    fully_shard(model, **options)
    return model


def prepare_model(
    model: nn.Module,
    spec: ParallelSpec,
    *,
    activation_checkpoint: ActivationCheckpointConfig,
    compile_config: CompileConfig,
    fsdp: FSDPConfig,
    mesh: DeviceMesh | None,
) -> nn.Module:
    """Apply the three transforms in the one order that works.

    Call this on a meta-device model, then ``to_empty(device)`` and load weights. Exposed as a
    single function precisely so callers cannot reorder the steps.
    """
    model = apply_activation_checkpointing(model, activation_checkpoint)
    model = apply_compile(model, compile_config)
    model = apply_fsdp(model, spec, fsdp, mesh=mesh)
    return model


def materialize(model: nn.Module, device: torch.device) -> nn.Module:
    """Allocate real storage for a meta-device model, after parallelism is in place."""
    model.to_empty(device=device)
    return model


__all__ = [
    "apply_activation_checkpointing",
    "apply_compile",
    "apply_fsdp",
    "materialize",
    "prepare_model",
]
