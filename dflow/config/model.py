"""Trainable-backbone configuration.

Pure declaration. This module must not import anything else from ``dflow``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .runtime import DType


@dataclass(kw_only=True, slots=True)
class ModelConfig:
    """The trainable transformer.

    Architecture hyperparameters are deliberately absent: they are read from the
    checkpoint's ``config.json`` at load time. The class defaults in diffusers describe
    FLUX.2 dev-32B, not klein, so hardcoding them here would be wrong for our target.
    """

    # Key into dflow.models.registry, e.g. "flux2-klein-base-9b".
    family: str = "flux2-klein"
    # Local path or HF repo id.
    path: str = ""
    subfolder: str = "transformer"
    revision: str | None = None
    dtype: DType = "bfloat16"


@dataclass(kw_only=True, slots=True)
class LoRAConfig:
    """LoRA via ``PeftAdapterMixin.add_adapter``, not ``get_peft_model``.

    ``add_adapter`` injects adapters in place, so module paths stay identical to the
    base model. That keeps the parallel spec, FSDP wrap units and checkpoint keys valid.
    ``get_peft_model`` wraps the model and rewrites every path to
    ``base_model.model.*.base_layer``, which is what forces the string surgery seen in
    vflow's checkpoint code.
    """

    enabled: bool = False
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    adapter_name: str = "default"
    init_weights: bool | str = True
    # None: use the family's default target modules. For FLUX.2 those must include
    # `attn.to_qkv_mlp_proj`, since single-stream blocks fuse QKV *and* the FFN.
    target_modules: tuple[str, ...] | None = None


@dataclass(kw_only=True, slots=True)
class EMAConfig:
    enabled: bool = False
    decay: float = 0.9999
    update_every: int = 1


@dataclass(kw_only=True, slots=True)
class BackboneConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)

    @property
    def trainable(self) -> Literal["lora", "full"]:
        return "lora" if self.lora.enabled else "full"


__all__ = ["BackboneConfig", "EMAConfig", "LoRAConfig", "ModelConfig"]
