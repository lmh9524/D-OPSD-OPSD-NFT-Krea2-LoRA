"""Autocast and mixed precision.

A note on FLUX.2 and fp32 modules, because the obvious reading is wrong.

``Flux2Transformer2DModel`` declares ``_skip_layerwise_casting_patterns = ["pos_embed",
"norm"]``. That belongs to diffusers' *layerwise casting* feature — an inference memory
optimisation that stores weights in low precision and upcasts per module — and is **not** an
input to FSDP's mixed-precision policy. Building an FSDP wrapping scheme around it would mean
putting every LayerNorm in its own communication group, trading a real throughput cost for a
precision problem the model already handles: ``forward`` upcasts where it matters, e.g.

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(...)

Under ``MixedPrecisionPolicy`` the *master* parameters stay in ``reduce_dtype`` (fp32 here);
``param_dtype`` only affects the dtype tensors are all-gathered into for compute. So a single
policy for the whole model is correct, and ``ParallelSpec.keep_fp32_patterns`` is carried for
layerwise casting, not for sharding.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)}") from None


def autocast(device: torch.device, dtype: str | None) -> AbstractContextManager:
    """Autocast context, or a no-op when ``dtype`` is None (full fp32 training)."""
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=resolve_dtype(dtype))


__all__ = ["autocast", "resolve_dtype"]
