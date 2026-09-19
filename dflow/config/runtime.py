"""Runtime configuration: parallelism, precision, compilation.

Pure declaration. This module must not import anything else from ``dflow``.

Note on ``module_names`` fields: they default to ``None``, meaning "derive from the
model". diffusers models already declare their wrap/repeat units via
``_no_split_modules`` and ``_repeated_blocks``, so hardcoding block paths per
architecture is both redundant and a place for them to drift out of sync. Set these
only to override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DType = Literal["float32", "bfloat16", "float16"]


@dataclass(kw_only=True, slots=True)
class DistributedConfig:
    """Logical parallel degrees.

    The mesh is always three-dimensional, even when ``cp == 1``. Adding context
    parallelism later must not change the meaning of ``dp_rank``, because ``dp_rank``
    determines which shard of the dataset a process reads, and it is recorded (via
    per-rank RNG state) in every checkpoint.
    """

    dp_replicate: int = 1
    dp_shard: int = -1  # -1: derive from world_size
    cp: int = 1

    def resolve_dp_shard(self, world_size: int) -> int:
        """Resolve ``dp_shard=-1`` to ``world_size // (dp_replicate * cp)``."""
        if self.dp_replicate < 1 or self.cp < 1:
            raise ValueError("dp_replicate and cp must be >= 1")
        denominator = self.dp_replicate * self.cp
        resolved = self.dp_shard if self.dp_shard != -1 else world_size // denominator
        if resolved < 1:
            raise ValueError(
                f"cannot resolve dp_shard: world_size={world_size} is too small for "
                f"dp_replicate={self.dp_replicate} * cp={self.cp}"
            )
        if denominator * resolved != world_size:
            raise ValueError(
                f"parallel degrees must multiply to world_size: "
                f"dp_replicate={self.dp_replicate} * dp_shard={resolved} * cp={self.cp} "
                f"!= {world_size}"
            )
        return resolved


@dataclass(kw_only=True, slots=True)
class FSDPConfig:
    enabled: bool = True
    param_dtype: DType = "bfloat16"
    reduce_dtype: DType = "float32"
    reshard_after_forward: bool = True
    cpu_offload: bool = False
    module_names: tuple[str, ...] | None = None
    # Modules to exclude from the low-precision cast. None: use the model's
    # _keep_in_fp32_modules / _skip_layerwise_casting_patterns.
    keep_fp32_modules: tuple[str, ...] | None = None


@dataclass(kw_only=True, slots=True)
class ActivationCheckpointConfig:
    # Long sequences (multi-reference at 1024^2 runs to several thousand tokens across
    # 56 blocks) do not fit without this.
    enabled: bool = True
    preserve_rng_state: bool = True
    module_names: tuple[str, ...] | None = None


@dataclass(kw_only=True, slots=True)
class CompileConfig:
    enabled: bool = False
    backend: str = "inductor"
    fullgraph: bool = True
    module_names: tuple[str, ...] | None = None


@dataclass(kw_only=True, slots=True)
class CPConfig:
    """Context parallelism.

    Not implemented yet: degrees stay at 1 and ``reduce_cp_gradients()`` is a no-op.
    The config exists now so recipes do not change shape when CP lands.
    """

    ulysses_degree: int = 1
    ring_degree: int = 1
    # Supports arbitrary sequence lengths (requires ring_degree == 1). Without it, the
    # sharded dimension must be divisible by the CP size.
    ulysses_anything: bool = False

    @property
    def degree(self) -> int:
        return self.ulysses_degree * self.ring_degree

    @property
    def enabled(self) -> bool:
        return self.degree > 1


@dataclass(kw_only=True, slots=True)
class RuntimeConfig:
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    compile: CompileConfig = field(default_factory=CompileConfig)
    cp: CPConfig = field(default_factory=CPConfig)

    matmul_precision: Literal["highest", "high", "medium"] = "high"
    # Must be a context-parallel-capable backend, otherwise enabling CP later fails.
    # diffusers accepts: native, _native_cudnn, _native_flash, flash, _flash_3, sage.
    attention_backend: str = "native"


__all__ = [
    "ActivationCheckpointConfig",
    "CPConfig",
    "CompileConfig",
    "DType",
    "DistributedConfig",
    "FSDPConfig",
    "RuntimeConfig",
]
