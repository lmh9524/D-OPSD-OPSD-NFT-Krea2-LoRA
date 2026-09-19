"""L2: mesh, sharding, precision, compilation, seeding.

Model-agnostic by construction — nothing here imports ``dflow.models``, ``dflow.tasks`` or
``dflow.encoders``. ``ParallelSpec`` is defined here, not in ``models/``, because this layer
consumes it: the consumer owns the interface.
"""

from dflow.runtime.context import (
    MeshBundle,
    build_meshes,
    compute_dp_rank,
    destroy_distributed,
    init_distributed,
)
from dflow.runtime.cp import reduce_cp_gradients
from dflow.runtime.parallel import (
    apply_activation_checkpointing,
    apply_compile,
    apply_fsdp,
    materialize,
    prepare_model,
)
from dflow.runtime.precision import autocast, resolve_dtype
from dflow.runtime.seed import (
    gather_rng_states,
    matmul_precision,
    restore_rng_state,
    seed_everything,
)
from dflow.runtime.spec import ParallelSpec

__all__ = [
    "MeshBundle",
    "ParallelSpec",
    "apply_activation_checkpointing",
    "apply_compile",
    "apply_fsdp",
    "autocast",
    "build_meshes",
    "compute_dp_rank",
    "destroy_distributed",
    "gather_rng_states",
    "init_distributed",
    "materialize",
    "matmul_precision",
    "prepare_model",
    "reduce_cp_gradients",
    "resolve_dtype",
    "restore_rng_state",
    "seed_everything",
]
