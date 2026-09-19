"""MiniMax-H3 family adapter.

Covers `MiniMaxAI/MiniMax-H3`, a joint **video + audio** generator. This is the first non-image
backbone in the repo, so several protocol fields take values no FLUX-derived family has used.

Facts this module encodes, read from `transformer_minimax_h3.py` and the modular pipeline in a
`diffusers` `main` checkout (`a949d3dd9`, 2026-08-25). The transformer is *vendored* because
MiniMax-H3 does not exist in the pinned 0.39.0 release — see `dflow/vendor/UPSTREAM.md`.

* **One packed sequence, no cross-attention.** A single block stack runs full self-attention over
  one 1-D sequence holding text rows, conditioning video rows, audio rows and target video rows.
  There are no per-modality block weights: modality-specific behaviour comes only from the two
  input patch projections, a per-row AdaLN modality tag, and the two output heads.
* **The caller builds the layout.** `forward` does not patchify, order or pad anything. It takes
  the already-patchified rows plus the index tensors that say where each modality sits in the
  packed sequence. `patchify_outside = True` for that reason, as with FLUX and Krea 2.
* **Position ids are 3-D `(t, h, w)` and unbatched: `(seq_len, 3)`** — the same convention as
  Krea 2, and `forward` raises unless `ndim == 2`. The batch axis is a pure replication axis: one
  packed layout is shared by every item in the batch.
* **Timesteps are `[0, 1]` and unscaled**, and there are *several per forward*. `timestep` is the
  vector of **distinct** noise levels present in the sequence and `timestep_indices` maps every row
  onto one of them, because target video, target audio and conditioning rows sit at different
  levels in the same call. No other family in this repo has a per-row timestep.
* **No guidance embedder.** `forward` takes no `guidance` argument; the pipeline uses real CFG.
* **Native context-parallel plan.** Unlike Krea 2 this model ships `_cp_plan`, so
  `parallel_spec` reports `has_native_cp_plan=True`. The plan splits at the *first block* rather
  than on `forward`'s inputs, because the per-modality rows are scattered into the packed buffer
  with sequence-wide indices. The packed sequence carries no padding, so its length must be
  divisible by the CP region size — otherwise use `ulysses_anything` / `ring_anything`.
* **Mixed-precision checkpoint.** `_keep_in_fp32_modules` covers the two patch projections, the
  timestep MLP, the two output heads and the `rope` buffer; the block stack is bfloat16.

### Not yet wired for training

`prepare_inputs` deliberately raises. The protocol's signature carries a single token stream, a
single text stream and one timestep — H3 needs the audio stream, the modality tags, the three
index tensors and the per-row timestep map as well, and there is no honest way to synthesise them
from the image-shaped arguments. Per the protocol's own rule, a family that cannot honour its
arguments raises rather than silently dropping conditioning. Building that layout is a *task's*
job (the `video` equivalent of `tasks/ref2img/conditioning.py`) and lands with the task, not here.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from dflow.models.family.base import (
    ParallelSpec,
    find_block_module_names,
    find_keep_fp32_patterns,
)
from dflow.vendor.minimax_h3 import MiniMaxH3Transformer3DModel

#: Every projection inside a `MiniMaxH3TransformerBlock`.
#:
#: `ff` is a diffusers `FeedForward` with `activation_fn="swiglu"`, whose submodules are
#: `net.0.proj` (the fused gate/up projection) and `net.2` (down). Those names were read off an
#: instantiated model rather than assumed.
#:
#: PEFT matches these as **suffixes**, and `MiniMaxH3TokenRefinerBlock` uses the same attribute
#: names, so the two refiner blocks that condition the text stream are adapted too — the same
#: trade-off `Krea2Family` makes with its text fusion blocks. To restrict to the main stack, pass
#: explicit `transformer_blocks.N....` paths via `--backbone.lora.target-modules`.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "net.0.proj",
    "net.2",
)

#: The two flow-matching shifts MiniMax-H3 samples with, from the schedulers its repository ships
#: (`shift = 12.0` for video, `shift = 3.0` for audio). They are constants: unlike FLUX.2 and
#: Krea 2 there is no resolution or step-count term, so token count does not enter.
#:
#: `dflow.schedulers.flow_matching` is parameterised by ``mu`` where ``shift == exp(mu)``, so these
#: are converted rather than passed through.
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


class MiniMaxH3Family:
    """Adapter for ``MiniMaxH3Transformer3DModel``."""

    name = "minimax-h3"
    #: Video latents carry a frame dimension. The task patchifies them into rows with the model's
    #: `(t, h, w)` `patch_size` before they reach `forward`.
    latent_layout = "BCFHW"
    #: `proj_in` is a Linear over already-patchified channels; the model contains no Conv3d
    #: patch embedding.
    patchify_outside = True

    def load_config(
        self, path: str, *, subfolder: str = "transformer", revision: str | None = None
    ) -> dict[str, Any]:
        config = MiniMaxH3Transformer3DModel.load_config(
            path, subfolder=subfolder, revision=revision
        )
        return dict(config)

    def build_meta(self, config: dict[str, Any]) -> nn.Module:
        with torch.device("meta"):
            return MiniMaxH3Transformer3DModel.from_config(config)

    def parallel_spec(self, model: nn.Module) -> ParallelSpec:
        return ParallelSpec(
            block_module_names=find_block_module_names(model),
            keep_fp32_patterns=find_keep_fp32_patterns(model),
            #: The packed sequence is dim 1 of `(batch, seq_len, hidden)`.
            sequence_dim=1,
            has_native_cp_plan=getattr(model, "_cp_plan", None) is not None,
        )

    def default_lora_targets(self) -> tuple[str, ...]:
        return DEFAULT_LORA_TARGETS

    def noise_shift_mu(
        self, *, image_tokens: int, inference_steps: int | None = None, audio: bool = False
    ) -> float:
        """MiniMax-H3's shift, imported so training cannot drift from inference.

        Both arguments the protocol declares are accepted and **ignored**: H3's schedule is a
        constant shift per modality, with no resolution or step-count term. They are not dropped
        silently — this docstring is the record that they genuinely do not apply here.

        ``audio`` selects the audio schedule. It is an extension beyond the protocol because H3
        denoises two streams at two different shifts in one forward, which no image family does;
        a caller that only wants the video schedule can ignore it.
        """
        return math.log(AUDIO_SHIFT if audio else VIDEO_SHIFT)

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
        """Not implemented: the protocol's arguments cannot describe an H3 forward.

        `MiniMaxH3Transformer3DModel.forward` additionally requires `audio_hidden_states`,
        `token_tags`, `timestep_indices`, `video_indices`, `audio_indices` and `text_indices` —
        the packed-sequence layout. None of them can be derived from a single token stream and a
        single timestep, so synthesising them here would mean inventing conditioning.

        Raising is the protocol's documented contract for exactly this case. The layout belongs to
        a video task module, which will call `forward` directly the way `tasks/ref2img` does.
        """
        raise NotImplementedError(
            "MiniMaxH3Family.prepare_inputs: H3 forward needs the packed-sequence layout "
            "(audio_hidden_states, token_tags, timestep_indices, video_indices, audio_indices, "
            "text_indices), which this protocol signature does not carry. Build the layout in the "
            "video task and call the transformer directly."
        )


__all__ = ["MiniMaxH3Family", "DEFAULT_LORA_TARGETS", "VIDEO_SHIFT", "AUDIO_SHIFT"]
