"""Build a backbone on meta, then fill it.

The order is the first of the six shared-infrastructure items, and it is the reason this
is a shared module rather than something each experiment does:

    load_config  ->  build_meta  ->  (parallelism applied by runtime/)  ->  to_empty
                 ->  load_weights

Meta init means a 9B or 32B model is never materialised whole on one device: FSDP2 decides
the shard layout first, and only each rank's shard is ever allocated. Doing it the other
way round — load, then shard — needs the full model in host memory on every rank.

Weight reading happens on rank 0 only and is broadcast, for the same reason. With eight
ranks on one node, every rank reading the same 18 GB of safetensors is 144 GB of host
memory and eight times the I/O.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from dflow.common.hub import localize
from dflow.config import ModelConfig
from dflow.models.family.base import Family
from dflow.models.registry import resolve


def resolve_path(config: ModelConfig) -> str:
    """Explicit path wins; otherwise fall back to the registry's default repo."""
    return config.path or resolve(config.family).default_repo


def load_architecture_config(config: ModelConfig, family: Family) -> dict[str, Any]:
    """Read ``config.json`` without instantiating anything.

    Nothing about the architecture is hardcoded anywhere in dflow. The diffusers class
    defaults (``in_channels=128``, ``num_layers=8``, ``joint_attention_dim=15360``)
    describe FLUX.2 dev-32B; klein's real values differ, so they must come from the
    checkpoint.
    """
    return family.load_config(
        resolve_path(config), subfolder=config.subfolder, revision=config.revision
    )


def build_meta(config: ModelConfig, family: Family) -> tuple[nn.Module, dict[str, Any]]:
    """Instantiate on the meta device. Returns the model and the config it was built from."""
    architecture = load_architecture_config(config, family)
    model = family.build_meta(architecture)
    return model, architecture


def read_state_dict(path: str | pathlib.Path, *, subfolder: str = "") -> dict[str, torch.Tensor]:
    """Read a diffusers-format safetensors checkpoint, sharded or single-file."""
    from safetensors.torch import load_file

    root = localize(path, subfolder=subfolder)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        weight_map = json.loads(indexes[0].read_text(encoding="utf-8"))["weight_map"]
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(load_file(root / shard, device="cpu"))
        return state_dict

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors found in {root}")
    if len(shards) > 1:
        raise ValueError(
            f"{len(shards)} safetensors files in {root} but no index.json — refusing to "
            f"guess the shard order"
        )
    return load_file(shards[0], device="cpu")


def lora_wrapped_modules(model: nn.Module) -> set[str]:
    """Paths of modules PEFT has wrapped, detected by their ``base_layer`` child.

    Detected structurally rather than by importing ``peft.tuners.lora.LoraLayer``: any tuner that
    follows the same wrap-and-delegate convention needs the same key remap, and this keeps the
    loader free of a tuner-specific import.
    """
    return {
        name
        for name, module in model.named_modules()
        if name and hasattr(module, "base_layer")
    }


def remap_lora_base_keys(
    state_dict: dict[str, torch.Tensor], wrapped: set[str]
) -> dict[str, torch.Tensor]:
    """Rewrite ``<path>.<param>`` to ``<path>.base_layer.<param>`` for wrapped modules.

    **Why this exists.** ``add_adapter`` keeps every module *path* intact, which is what makes the
    parallel spec and checkpoint keys stay valid — but the targeted module itself is replaced by a
    ``lora.Linear`` that holds the original under ``.base_layer``. So the checkpoint's
    ``...attn.to_q.weight`` has nowhere to land: the model now expects
    ``...attn.to_q.base_layer.weight``.

    With ``strict=False`` — which LoRA runs require, since the checkpoint carries no adapter
    tensors — that mismatch is **silent**. Every LoRA-targeted projection keeps the zeros left by
    ``to_empty()``, attention and the feed-forward output nothing, and training proceeds against a
    hollowed-out backbone. The loss still falls, so nothing looks wrong; it just starts far too
    high, because the model being fine-tuned is not the model that was downloaded.
    """
    if not wrapped:
        return state_dict
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        module_path, _, leaf = key.rpartition(".")
        if module_path in wrapped:
            remapped[f"{module_path}.base_layer.{leaf}"] = value
        else:
            remapped[key] = value
    return remapped


def load_weights(
    model: nn.Module,
    config: ModelConfig,
    *,
    is_master: bool,
    strict: bool = True,
) -> None:
    """Load base weights into a materialised (post-``to_empty``) model.

    Reads on rank 0 and broadcasts. ``set_model_state_dict`` handles both a plain module
    and an FSDP2-sharded one, so there is a single path for single-GPU and multi-GPU runs.

    ``strict=False`` is correct when LoRA adapters are already injected: the base
    checkpoint has no adapter tensors, and those get initialised separately by
    ``adapter.reset_lora_parameters``. Because that also suppresses *missing*-key errors, the
    checkpoint keys are remapped onto the adapter-wrapped paths first — see
    :func:`remap_lora_base_keys` for what goes wrong otherwise.
    """
    state_dict: dict[str, torch.Tensor] | None = None
    if is_master:
        state_dict = read_state_dict(resolve_path(config), subfolder=config.subfolder)

    wrapped = lora_wrapped_modules(model)
    if state_dict is not None:
        state_dict = remap_lora_base_keys(state_dict, wrapped)

    if dist.is_initialized():
        dist.barrier()

    set_model_state_dict(
        model,
        state_dict or {},
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=dist.is_initialized(),
            strict=strict,
        ),
    )

    if wrapped:
        _assert_base_weights_landed(model, wrapped)


def _assert_base_weights_landed(model: nn.Module, wrapped: set[str]) -> None:
    """Fail loudly if a wrapped module's base weight is still all zeros after loading.

    ``to_empty()`` leaves uninitialised memory that is in practice zero, so an unmatched key and a
    genuinely zero tensor look identical. This is cheap and turns the one failure mode that
    produces plausible-looking training into an error at setup.
    """
    parameters = dict(model.named_parameters())
    for path in sorted(wrapped):
        weight = parameters.get(f"{path}.base_layer.weight")
        if weight is None:
            continue
        local = weight.to_local() if hasattr(weight, "to_local") else weight
        if local.numel() and not bool(local.any()):
            raise ValueError(
                f"{path}.base_layer.weight is all zeros after loading. The checkpoint key "
                f"{path}.weight did not reach the adapter-wrapped module, so this LoRA run would "
                f"fine-tune a hollowed-out backbone."
            )
        return


__all__ = [
    "build_meta",
    "localize",
    "load_architecture_config",
    "load_weights",
    "lora_wrapped_modules",
    "remap_lora_base_keys",
    "read_state_dict",
    "resolve_path",
]
