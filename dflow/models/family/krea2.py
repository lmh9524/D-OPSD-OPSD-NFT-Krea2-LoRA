"""Krea 2 family adapter.

Covers `krea/Krea-2-Raw` and `krea/Krea-2-Turbo`, which share one architecture class and one
`transformer/config.json`; only `model_index.json`'s `is_distilled` differs, exactly as with klein.
That flag changes the noise schedule, so it is carried by the *registry entry* rather than read from
the transformer config, which does not contain it.

Facts this module encodes, read from diffusers 0.39.0 (the pinned release; `transformer_krea2.py` is
byte-identical to the 0.40.0.dev0 checkout):

* `forward` takes `timestep` **normalised to [0, 1]** and multiplies by 1000 itself
  (`Krea2TimestepEmbedding.forward`: `timestep.float() * 1e3`). Passing 0..1000 silently shifts the
  whole schedule, exactly as with FLUX.2.
* **Position ids are 3-D `(T, H, W)` and *unbatched*: `(sequence_length, 3)`**, covering the
  concatenated `[text; image]` sequence. `forward` raises unless `ndim == 2`. Every sample in a batch
  therefore shares one id array — another reason `batch_size=1` is the operating point.
* **Text comes first.** `forward` does `cat([encoder_hidden_states, hidden_states], dim=1)`, the
  reverse of nothing in FLUX.2 — there text is a separate stream. It then strips the text span from
  its own output (`hidden_states[:, text_seq_len:]`), so the returned sequence is the image span
  and the caller slices the target out of it. Krea 2's layout puts the target **last** —
  `[text | refs | target]` — so that slice is the tail, not the head; see `conditioning.py`.
* `encoder_hidden_states` is **4-D**: `(B, text_len, num_text_layers, text_hidden_dim)` — a *stack* of
  tapped decoder layers, not FLUX.2's flattened `(B, L, 3H)`. The text fusion stage collapses the
  layer axis inside the model.
* `encoder_attention_mask` is a real input, not an optimisation: Krea 2 pads **in the middle** of its
  chat template (`[prefix | prompt | PAD | suffix]`), so padded positions sit between real ones and
  must be masked as attention keys.
* **There is no guidance embedding.** `forward` takes no `guidance` argument at all; the pipeline uses
  real CFG with a negative prompt, so caption dropout during training is what keeps the
  unconditional branch alive — same reasoning as klein.
* **No `_cp_plan`.** Unlike FLUX.2, this model ships no native context-parallel plan, so
  `parallel_spec` reports `has_native_cp_plan=False` and CP would need one written before it can run.

### Multi-image reference is *not* a pretrained capability here

`Krea2Pipeline` is text-to-image only. `prepare_position_ids` emits `(0, h, w)` for image tokens
— **the T axis is always zero**, and no checkpoint has ever seen a non-zero value on it, nor a
second image span in the sequence.

`dflow/tasks/ref2img/conditioning.py` builds reference spans by sequence concatenation, in the
layout a working community edit LoRA uses: frames 1..N, target last, reference H/W registered to
the target grid. That is structurally sound, but it is still teaching a **new conditioning
modality from scratch** rather than adapting an existing one. See `docs/krea2-reference.md`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dflow.models.family.base import (
    ParallelSpec,
    find_block_module_names,
    find_keep_fp32_patterns,
)
from dflow.vendor.krea2 import Krea2Transformer2DModel

#: Every projection inside a `Krea2TransformerBlock`, plus the SwiGLU feed-forward.
#:
#: `attn.to_gate` matters and has no FLUX.2 counterpart: `Krea2AttnProcessor` multiplies the attention
#: output by `sigmoid(to_gate(x))`, so a LoRA that skips it cannot change how much attention each
#: channel admits — which is precisely the knob reference conditioning needs.
#:
#: PEFT matches these as **suffixes**, and `Krea2TextFusionBlock` uses the same attribute names, so
#: the text fusion stage is adapted too. That is deliberate: it is four small blocks next to 28 large
#: ones, and letting the model re-read its text stream helps when captions describe the references.
#: To restrict to the main stack, pass explicit `transformer_blocks.N....` paths via
#: `--backbone.lora.target-modules`.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_gate",
    "attn.to_out.0",
    "ff.gate",
    "ff.up",
    "ff.down",
)

#: `calculate_shift` bounds, read from `pipeline_krea2.py`. Note `max_image_seq_len=6400`, not
#: FLUX.1's 4096 — the pipeline overrides the copied helper's defaults from `scheduler_config.json`.
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 6400
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


class Krea2Family:
    """Adapter for ``Krea2Transformer2DModel``."""

    name = "krea2"
    latent_layout = "BCHW"
    # Latents are packed into tokens by the task; `img_in` is a Linear over already-packed channels.
    patchify_outside = True

    def __init__(self, *, distilled: bool = False) -> None:
        #: Turbo is the few-step distilled checkpoint and pins `mu = 1.15` regardless of resolution;
        #: Raw derives it from the token count. Lives here because `transformer/config.json` is
        #: identical between the two — only `model_index.json` distinguishes them.
        self.distilled = distilled

    def load_config(
        self, path: str, *, subfolder: str = "transformer", revision: str | None = None
    ) -> dict[str, Any]:
        config = Krea2Transformer2DModel.load_config(path, subfolder=subfolder, revision=revision)
        return dict(config)

    def build_meta(self, config: dict[str, Any]) -> nn.Module:
        with torch.device("meta"):
            return Krea2Transformer2DModel.from_config(config)

    def parallel_spec(self, model: nn.Module) -> ParallelSpec:
        return ParallelSpec(
            block_module_names=find_block_module_names(model),
            keep_fp32_patterns=find_keep_fp32_patterns(model),
            sequence_dim=1,
            has_native_cp_plan=getattr(model, "_cp_plan", None) is not None,
        )

    def default_lora_targets(self) -> tuple[str, ...]:
        return DEFAULT_LORA_TARGETS

    #: Sampling settings the model's own repository recommends, keyed by whether it is distilled.
    #:
    #: Verbatim from krea-ai/krea-2's README::
    #:
    #:     # Raw   -- the base undistilled model, "use the full sampler with classifier-free guidance"
    #:     uv run inference.py "..." --checkpoint oss_raw   --steps 52 --cfg 3.5
    #:     # Turbo -- "run with 8 steps and CFG disabled"
    #:     uv run inference.py "..." --checkpoint oss_turbo --steps 8  --cfg 0.0 --mu 1.15
    #:
    #: These are not interchangeable and the difference is not subtle: Raw at Turbo's 8 unguided
    #: steps produces a generic image that follows its conditioning weakly, which reads exactly like
    #: an adapter that learned nothing. LoRAs are trained on Raw and applied on Turbo, so a run
    #: routinely has both in play and the right numbers depend on which one is loaded.
    SAMPLING_DEFAULTS = {
        True: {"steps": 8, "guidance": 0.0},
        False: {"steps": 52, "guidance": 3.5},
    }

    def sampling_defaults(self) -> dict[str, float]:
        """Recommended ``steps`` and ``guidance`` for whichever checkpoint this adapter wraps."""
        return dict(self.SAMPLING_DEFAULTS[self.distilled])

    def noise_shift_mu(self, *, image_tokens: int, inference_steps: int | None = None) -> float:
        """Krea 2's shift, imported so training cannot drift from inference.

        Pass the **target** token count, for the same reason as FLUX.2: references are read-only
        context and must not move the noise schedule.

        Unlike FLUX.2 there is no step-count term at all — `pipeline_krea2.py` branches on
        `is_distilled` and otherwise calls `calculate_shift(image_seq_len, 256, 6400, 0.5, 1.15)`,
        which is FLUX.1's linear resolution shift with wider bounds. `inference_steps` is therefore
        accepted (the protocol declares it) and ignored, rather than silently pretending to matter.

        For the distilled Turbo checkpoint the pipeline pins `mu = 1.15` — shift 3.158 — at every
        resolution. That happens to equal `FlowMatchConfig.shift`'s default, so training Turbo under
        the default fixed schedule matches its own inference schedule exactly.
        """
        if self.distilled:
            return MAX_SHIFT
        from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift

        return float(
            calculate_shift(
                image_tokens, BASE_IMAGE_SEQ_LEN, MAX_IMAGE_SEQ_LEN, BASE_SHIFT, MAX_SHIFT
            )
        )

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
        """Build the kwargs for one forward pass.

        Takes the same normalised arguments every family does and maps them onto Krea 2's
        convention, which differs in two ways worth stating plainly:

        * `text_ids` and `token_ids` are **concatenated into one unbatched `(seq, 3)` array**, text
          first, because that is the single `position_ids` the model accepts;
        * `guidance` must be None — this architecture has no guidance embedder to feed.

        Args:
            tokens: `(B, image_seq, in_channels)` — reference spans first, target span last,
                i.e. `[refs | target]`. This once said the opposite, and the stale line
                outlived the code: `tools/krea2/sample_krea2.py` kept slicing the head as the
                target and spent three evaluation rounds denoising reference latents.
                `ReferenceSequence.target_offset` is the authority — never assume 0.
            token_ids: `(B, image_seq, 3)` or `(image_seq, 3)` position ids aligned with `tokens`.
            text_embeds: `(B, text_seq, num_text_layers, text_hidden_dim)`.
            text_ids: `(B, text_seq, 3)` or `(text_seq, 3)`. Krea 2 puts text at the origin.
            timestep: `(B,)` in **[0, 1]**.
            guidance: must be None.
            text_mask: `(B, text_seq)` bool. None means every text position is valid, which is
                wrong for the real encoder — it pads in the middle of its template.
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be (B, seq, C), got {tuple(tokens.shape)}")
        if text_embeds.ndim != 4:
            raise ValueError(
                f"Krea 2 takes stacked text hidden states (B, L, num_text_layers, dim), got "
                f"{tuple(text_embeds.shape)}. FLUX.2's flattened (B, L, 3H) is a different layout."
            )
        if guidance is not None:
            raise ValueError(
                "Krea2Transformer2DModel takes no guidance argument; the pipeline uses real CFG "
                "with a negative prompt. Train the unconditional branch with caption dropout."
            )
        if timestep.ndim != 1 or timestep.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"timestep must be (B,) matching batch {tokens.shape[0]}, got "
                f"{tuple(timestep.shape)}"
            )
        # The model scales by 1000 internally. Same silent-schedule-shift trap as FLUX.2.
        if timestep.numel() and float(timestep.detach().abs().max()) > 1.0 + 1e-4:
            raise ValueError(
                "timestep must be normalised to [0, 1]; Krea 2's timestep embedding multiplies by "
                "1000 itself (Krea2TimestepEmbedding.forward)"
            )

        image_ids = _as_unbatched_ids(token_ids, "token_ids", expect=tokens.shape[1])
        text_position_ids = _as_unbatched_ids(text_ids, "text_ids", expect=text_embeds.shape[1])
        position_ids = torch.cat([text_position_ids, image_ids], dim=0)

        if text_mask is not None and text_mask.shape != text_embeds.shape[:2]:
            raise ValueError(
                f"text_mask {tuple(text_mask.shape)} must be (B, text_seq) matching text_embeds "
                f"{tuple(text_embeds.shape[:2])}"
            )

        return {
            "hidden_states": tokens,
            "encoder_hidden_states": text_embeds,
            "timestep": timestep,
            "position_ids": position_ids,
            "encoder_attention_mask": text_mask,
            "return_dict": False,
        }

    @staticmethod
    def take_target_span(
        output: torch.Tensor, target_len: int, *, target_offset: int = 0
    ) -> torch.Tensor:
        """Slice the target span out of the image sequence.

        The model removes the text span itself, so ``target_offset`` is relative to the first image
        token. Krea 2's layout puts the target **last** — `[refs | target]`, matching the edit LoRAs
        that work — so the offset is normally ``reference_len`` and this takes the tail. Passing the
        wrong offset returns reference tokens as the prediction, which trains against the wrong
        pixels without raising.
        """
        stop = target_offset + target_len
        if stop > output.shape[1]:
            raise ValueError(
                f"target span [{target_offset}, {stop}) exceeds output sequence {output.shape[1]}"
            )
        return output[:, target_offset:stop]

    @staticmethod
    def requires_guidance(config: dict[str, Any]) -> bool:
        """Krea 2 has no guidance embedder. Present so the family surface matches FLUX.2's."""
        return False


def _as_unbatched_ids(ids: torch.Tensor, name: str, *, expect: int) -> torch.Tensor:
    """Reduce `(B, seq, 3)` or `(seq, 3)` to the `(seq, 3)` the model wants.

    Krea 2's `position_ids` is not batched, so a batch can only be built from samples that agree on
    every id. Rather than silently taking element 0 and training the rest of the batch on the wrong
    geometry, this checks that they agree.
    """
    if ids.ndim == 3:
        if ids.shape[0] > 1 and not bool(torch.equal(ids, ids[:1].expand_as(ids))):
            raise ValueError(
                f"{name} differs across the batch, but Krea 2 takes one unbatched (seq, 3) "
                f"position_ids for every sample. Use batch_size=1, or bucket by shape so a batch "
                f"shares one geometry."
            )
        ids = ids[0]
    if ids.ndim != 2 or ids.shape[-1] != 3:
        raise ValueError(f"Krea 2 position ids are 3-D (T, H, W), got {name}={tuple(ids.shape)}")
    if ids.shape[0] != expect:
        raise ValueError(f"{name} has {ids.shape[0]} rows, expected {expect}")
    return ids


__all__ = ["DEFAULT_LORA_TARGETS", "Krea2Family"]
