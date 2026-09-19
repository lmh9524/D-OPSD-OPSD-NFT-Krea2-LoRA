"""L3: the trainable backbone — construction, parallel metadata, adapters.

Everything architecture-specific is confined to ``family/``. Layers above this one never
learn which backbone is loaded.
"""

from dflow.models.adapter import (
    ParameterSummary,
    apply_dual_lora,
    apply_lora,
    copy_adapter,
    freeze_base,
    parameter_summary,
    reset_lora_parameters,
)
from dflow.models.family import Family, Flux2Family, Krea2Family, ParallelSpec
from dflow.models.loader import (
    build_meta,
    load_architecture_config,
    load_weights,
    localize,
    read_state_dict,
    resolve_path,
)
from dflow.models.registry import REGISTRY, RegistryEntry, get_family, resolve

__all__ = [
    "REGISTRY",
    "Family",
    "Flux2Family",
    "Krea2Family",
    "ParallelSpec",
    "ParameterSummary",
    "RegistryEntry",
    "apply_dual_lora",
    "apply_lora",
    "build_meta",
    "copy_adapter",
    "freeze_base",
    "get_family",
    "load_architecture_config",
    "load_weights",
    "localize",
    "parameter_summary",
    "read_state_dict",
    "reset_lora_parameters",
    "resolve",
    "resolve_path",
]
