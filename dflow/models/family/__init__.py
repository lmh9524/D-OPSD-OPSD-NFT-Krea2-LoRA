"""Per-architecture adapters.

One module per architecture. Adding one must not require touching ``runtime/``,
``trainer/`` or ``data/`` — see ``base.py``.
"""

from dflow.models.family.base import (
    Family,
    LatentLayout,
    find_block_module_names,
    find_keep_fp32_patterns,
)
from dflow.models.family.flux2 import DEFAULT_LORA_TARGETS, Flux2Family
from dflow.models.family.krea2 import Krea2Family
from dflow.models.family.minimax_h3 import MiniMaxH3Family
from dflow.runtime.spec import ParallelSpec

__all__ = [
    "DEFAULT_LORA_TARGETS",
    "Family",
    "Flux2Family",
    "Krea2Family",
    "MiniMaxH3Family",
    "LatentLayout",
    "ParallelSpec",
    "find_block_module_names",
    "find_keep_fp32_patterns",
]
