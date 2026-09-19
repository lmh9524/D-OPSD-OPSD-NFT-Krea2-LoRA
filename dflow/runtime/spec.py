"""What ``runtime/`` needs to know about a model, and nothing more.

``ParallelSpec`` lives here rather than in ``models/family/`` on purpose: ``runtime/`` (L2)
*consumes* it and ``models/`` (L3) *produces* it, so defining it in ``models/`` would make L2
depend on L3 and invert the layering. The consumer owns the interface; the producer implements
it. ``tools/checks/check_layering.py`` catches the mistake if it creeps back.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParallelSpec:
    """Model metadata needed to shard, checkpoint and compile it."""

    #: Attribute names of the block ``ModuleList``s, in forward order. FLUX.2 has two
    #: (``transformer_blocks``, ``single_transformer_blocks``); Wan has one.
    block_module_names: tuple[str, ...]
    #: Module name fragments to exclude from a low-precision cast. Carried for diffusers'
    #: layerwise-casting feature; **not** an input to FSDP's mixed-precision policy — see
    #: ``runtime/precision.py``.
    keep_fp32_patterns: tuple[str, ...]
    #: Which dimension context parallelism shards.
    sequence_dim: int = 1
    #: Whether the model ships a ``_cp_plan``, i.e. whether diffusers' native context
    #: parallelism can be used instead of hand-written collectives.
    has_native_cp_plan: bool = False

    def __post_init__(self) -> None:
        if not self.block_module_names:
            raise ValueError("ParallelSpec needs at least one block module")


__all__ = ["ParallelSpec"]
