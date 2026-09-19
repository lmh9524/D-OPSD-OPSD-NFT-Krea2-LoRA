"""Context parallelism: seam only, no implementation yet.

CP is not needed at phase-1 sequence lengths, but three things are cheap now and
expensive to retrofit, so they exist already:

1. the mesh is 3-D with ``cp == 1`` (``dflow.runtime.context``);
2. ``CPConfig`` exists with degree 1 (``dflow.config.runtime``);
3. ``reduce_cp_gradients()`` is called from the loop in the right place.

Point 3 is the reason this module exists as a no-op rather than not existing. The call
**must** sit between ``backward()`` and ``clip_grad_norm_()``:

    backward()  ->  reduce_cp_gradients()  ->  clip_grad_norm_()  ->  optimizer.step()

If it lands after clipping, the norm is computed on gradients that are still a factor of
``cp_size`` off, so the clip threshold means nothing. That does not raise — it just
trains slightly wrong. Fixing the position later means editing the middle of a loop
that by then has other concerns in it; fixing it now costs one no-op call.

When CP does land, use diffusers' native plan rather than hand-written collectives.
``Flux2Transformer2DModel._cp_plan`` already declares the split/gather, and
``ModelMixin.enable_parallelism()`` accepts a custom mesh so it composes with FSDP2.
What diffusers does **not** do is rescale gradients:

    # diffusers/hooks/context_parallel.py:263
    def backward(ctx, grad_output):
        grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
        return grad_chunks[ctx.rank], None, None

Its semantics are "gradients must be summed across the CP dimension". Because we fold
CP into the FSDP sharding dimension (``dp_shard * cp``), FSDP's reduce-scatter
*averages* over that combined axis instead, leaving gradients a factor of ``cp_size``
too small. Correcting that is this function's whole job.
"""

from __future__ import annotations

import torch.nn as nn

from dflow.runtime.context import MeshBundle


def reduce_cp_gradients(mesh: MeshBundle, model: nn.Module) -> None:
    """Correct gradients for the context-parallel dimension.

    No-op while ``cp_size == 1``, which is every phase-1 run.

    Must be called after ``backward()`` and before ``clip_grad_norm_()``.
    """
    if mesh.cp_size == 1:
        return
    raise NotImplementedError(
        "context parallelism is not implemented yet. Enabling it requires: "
        "(a) model.enable_parallelism(config=ContextParallelConfig(..., mesh=mesh.cp_mesh)); "
        "(b) scaling gradients by cp_size here, because FSDP averages over dp_shard*cp "
        "while CP semantics require a sum over cp; "
        "(c) bucket tables built with seq_multiple=cp_size, since diffusers asserts the "
        "sharded dimension is divisible by the CP size."
    )


__all__ = ["reduce_cp_gradients"]
