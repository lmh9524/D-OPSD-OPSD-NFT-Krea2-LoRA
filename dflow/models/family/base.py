"""The per-architecture seam.

A ``Family`` owns everything that differs between backbones: how a batch of latents
becomes model kwargs, where the sequence dimension is, which modules FSDP wraps, which
must stay fp32, and what LoRA should target. Nothing above this layer — not
``runtime/``, not ``trainer/``, not ``data/`` — knows which architecture is loaded.

That is the whole reason this layer exists. "Must also support video" reduces to adding
one module here, because the differences between an image and a video backbone are
exactly the fields below:

* ``latent_layout`` — ``"BCHW"`` vs ``"BCFHW"``
* patchification inside the model (Wan's ``Conv3d``) vs outside it (FLUX's packed tokens)
* one text encoder vs several, pooled branch or not
* joint attention over ``[text; image]`` vs text through a separate cross-attention

If adding an architecture requires editing ``runtime/`` or ``trainer/``, the protocol is
wrong: fix the protocol rather than leaking the architecture upward.

Wan contract (not implemented; recorded so FLUX assumptions do not silently become the
protocol):

* ``latent_layout = "BCFHW"``; latents carry a frame dimension
* ``patchify_outside = False``; ``patch_embedding`` is a ``Conv3d`` inside the model
* conditioning by **channel** concatenation, so ``expand_input_dim`` grows a ``Conv3d``,
  where FLUX grows a ``Linear``
* text goes through cross-attention and is *not* part of the attention sequence, so under
  context parallelism it must stay replicated rather than being sharded
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

from dflow.runtime.spec import ParallelSpec

LatentLayout = str  # "BCHW" (image) | "BCFHW" (video)


def find_block_module_names(model: nn.Module) -> tuple[str, ...]:
    """Derive block paths from the model's own declarations.

    diffusers already states its FSDP/compile unit in ``_no_split_modules`` (a list of
    *class* names), so hardcoding attribute paths per architecture would duplicate that
    and let the two drift apart. This resolves class names to the attribute names of the
    ``ModuleList``s holding them.
    """
    declared = set(getattr(model, "_no_split_modules", None) or ())
    if not declared:
        raise ValueError(
            f"{type(model).__name__} declares no _no_split_modules; pass block module "
            f"names explicitly in the family's parallel_spec()"
        )

    names: list[str] = []
    for name, module in model.named_children():
        if (
            isinstance(module, nn.ModuleList)
            and len(module) > 0
            and type(module[0]).__name__ in declared
        ):
            names.append(name)
    if not names:
        raise ValueError(
            f"none of {sorted(declared)} were found as children of "
            f"{type(model).__name__}; the upstream layout may have changed"
        )
    return tuple(names)


def find_keep_fp32_patterns(model: nn.Module) -> tuple[str, ...]:
    """Modules that must not be cast to bf16/fp16.

    Both upstream attributes matter and they are not the same list, so take the union.
    vflow ignored this entirely and cast every block wholesale, which puts timestep
    embeddings and norms in bf16 — numerically the weakest place to lose precision.
    """
    patterns: list[str] = []
    for attribute in ("_keep_in_fp32_modules", "_skip_layerwise_casting_patterns"):
        for name in getattr(model, attribute, None) or ():
            if name not in patterns:
                patterns.append(name)
    return tuple(patterns)


@runtime_checkable
class Family(Protocol):
    """Per-architecture adapter. Pure data and pure functions — no control inversion.

    Callers invoke these explicitly from ``step_fn``; the framework never calls back into
    a family on its own.
    """

    name: str
    latent_layout: LatentLayout
    #: True when latents are packed into tokens outside the model (FLUX), False when the
    #: model patchifies internally (Wan).
    patchify_outside: bool

    def load_config(self, path: str, *, subfolder: str, revision: str | None = None) -> dict[str, Any]:
        """Read the architecture config without instantiating or downloading weights."""
        ...

    def build_meta(self, config: dict[str, Any]) -> nn.Module:
        """Instantiate on the meta device, so no host memory is touched."""
        ...

    def parallel_spec(self, model: nn.Module) -> ParallelSpec: ...

    def default_lora_targets(self) -> tuple[str, ...]: ...

    def noise_shift_mu(self, *, image_tokens: int, inference_steps: int | None = None) -> float:
        """The ``mu`` that shifts this family's noise schedule.

        Lives here, not in ``schedulers/``, because how the shift is derived is a property of
        the model. FLUX.2 fits it empirically from sequence length *and* step count; Wan uses
        a constant; FLUX.1 interpolates between two configured bounds. A scheduler that knew
        any of that would need a branch per family.

        ``schedulers/flow_matching.py`` therefore takes ``mu`` as an argument and stays
        model-agnostic. ``inference_steps=None`` asks for the schedule a family trains at,
        which is not necessarily the one it samples with.
        """
        ...

    def prepare_inputs(
        self,
        *,
        tokens: torch.Tensor,
        token_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Map normalised arguments onto this architecture's calling convention.

        The arguments are the *task's* view of a step — a token sequence and its ids, a text
        stream and its ids, a timestep — and each family rearranges them into whatever its
        `forward` actually accepts. FLUX.2 keeps the two id arrays separate and batched; Krea 2
        concatenates them into one unbatched `(seq, 3)` array. Neither convention leaks upward.

        `guidance` is None for architectures without a guidance embedder, and `text_mask` is
        None for those whose text stream has no padding to mask. A family that cannot honour an
        argument raises rather than ignoring it: silently dropping conditioning is the failure
        mode this layer exists to prevent.
        """
        ...


__all__ = [
    "Family",
    "LatentLayout",
    "ParallelSpec",
    "find_block_module_names",
    "find_keep_fp32_patterns",
]
