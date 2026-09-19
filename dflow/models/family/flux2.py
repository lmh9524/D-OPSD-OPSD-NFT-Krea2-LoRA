"""FLUX.2 family adapter.

Covers every FLUX.2 variant — dev-32B and the klein 4B/9B models — because they share
one architecture class. What differs between them lives in ``config.json`` (layer counts,
``joint_attention_dim``, whether guidance is embedded) and in which text encoder the
pipeline pairs them with. Nothing here hardcodes those: the diffusers class defaults
describe dev-32B, so baking them in would be wrong for klein.

Facts this module encodes, read from diffusers 0.39.0 (the pinned release):

* ``forward`` takes ``timestep`` **normalised to [0, 1]** and multiplies by 1000 itself
  (``transformer_flux2.py:1236``). Passing 0..1000 silently shifts the whole schedule.
* Position ids are 4-D ``(T, H, W, L)`` with ``axes_dims_rope`` splitting the head
  dimension four ways. Each axis has a job: **T distinguishes reference images**, H/W are
  spatial, and **L carries text position** (text ids are ``(0, 0, 0, arange(L))``).
* The model strips text tokens from its own output (``transformer_flux2.py:1372``), so the
  returned sequence is ``[target; references]`` and the caller slices the target span.
* ``guidance=None`` is safe even when ``guidance_embeds=True``: the guidance embedder is
  simply skipped (``transformer_flux2.py:1004-1015``). But a model *trained* with guidance
  and fine-tuned without it has had its conditioning changed, so this module refuses to
  guess — see ``prepare_inputs``.
* Reference-specific modulation (``ref_fixed_timestep``) only applies under
  ``kv_cache_mode="extract"``. In the plain bidirectional path — what we train — reference
  tokens share the target's timestep, so nothing extra is passed.
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
from dflow.vendor.flux2 import Flux2Transformer2DModel

#: From diffusers' FLUX.2 DreamBooth guide. ``to_qkv_mlp_proj`` matters: single-stream
#: blocks fuse QKV *and* the feed-forward into one projection, so a LoRA that targets only
#: ``to_q``/``to_k``/``to_v`` silently skips 48 of the 56 blocks.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.to_qkv_mlp_proj",
)


class Flux2Family:
    """Adapter for ``Flux2Transformer2DModel``."""

    name = "flux2"
    latent_layout = "BCHW"
    # Latents are patchified and packed into tokens by the task, not by the model
    # (``patch_size=1``; ``x_embedder`` is a Linear over already-packed channels).
    patchify_outside = True

    def load_config(
        self, path: str, *, subfolder: str = "transformer", revision: str | None = None
    ) -> dict[str, Any]:
        config = Flux2Transformer2DModel.load_config(
            path, subfolder=subfolder, revision=revision
        )
        return dict(config)

    def build_meta(self, config: dict[str, Any]) -> nn.Module:
        """Instantiate on the meta device.

        Meta init is what lets parallelism be applied *before* weights exist, so a 9B or
        32B model is never materialised whole on one device.
        """
        with torch.device("meta"):
            return Flux2Transformer2DModel.from_config(config)

    def parallel_spec(self, model: nn.Module) -> ParallelSpec:
        return ParallelSpec(
            block_module_names=find_block_module_names(model),
            keep_fp32_patterns=find_keep_fp32_patterns(model),
            sequence_dim=1,
            has_native_cp_plan=getattr(model, "_cp_plan", None) is not None,
        )

    def default_lora_targets(self) -> tuple[str, ...]:
        return DEFAULT_LORA_TARGETS

    #: ``compute_empirical_mu`` converges to its resolution-only term by this step count, so
    #: passing it yields the schedule with the few-step correction switched off.
    MANY_STEP_LIMIT = 200

    def noise_shift_mu(self, *, image_tokens: int, inference_steps: int | None = None) -> float:
        """FLUX.2's shift, imported so training cannot drift from inference.

        Pass the **target** token count. ``pipeline_flux2_klein.py:815`` computes
        ``image_seq_len = latents.shape[1]`` while reference latents are a separate variable
        only concatenated inside the denoising loop — references are read-only context and do
        not move the noise schedule.

        ``inference_steps=None`` (the default, and what SFT wants) gives the **resolution-only**
        schedule. That is not an approximation: ``compute_empirical_mu``'s step-independent term
        is *exactly* FLUX.1's ``calculate_shift``. Its slope
        ``(1.15 - 0.5) / (4096 - 256) = 1.6927e-4`` is the constant ``a2``, and
        ``0.5 - a2 * 256 = 0.45667`` is ``b2``. So the formula is FLUX.1's resolution shift plus
        a few-step correction layered on top.

        That decomposition is why the default is not a deployment step count. The step term is
        **inference discretisation** — with ten steps you push mu up to place them better — not a
        statement about where the velocity field needs training signal. At 4096 tokens the
        resolution-only value is mu 1.15 (shift 3.16), matching what FLUX.1 and SD3 train at,
        while the 50-step value would be 2.02 (shift 7.6) and concentrate training at far higher
        noise than any comparable model uses.

        Pass an explicit step count only when deliberately training *for* a few-step schedule,
        as step distillation does.
        """
        from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu

        steps = self.MANY_STEP_LIMIT if inference_steps is None else inference_steps
        return compute_empirical_mu(image_seq_len=image_tokens, num_steps=steps)

    def prepare_inputs(
        self,
        *,
        tokens: torch.Tensor,
        token_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Build the kwargs for one forward pass.

        Args:
            tokens: ``(B, seq, in_channels)`` — the packed sequence, target span first,
                reference spans after it. Concatenating is the task's job, not ours.
            token_ids: ``(B, seq, 4)`` position ids aligned with ``tokens``.
            text_embeds: ``(B, txt, joint_attention_dim)``.
            text_ids: ``(B, txt, 4)``.
            timestep: ``(B,)`` in **[0, 1]**.
            guidance: ``(B,)`` or None. Required when the checkpoint embeds guidance.
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be (B, seq, C), got {tuple(tokens.shape)}")
        if token_ids.shape[:2] != tokens.shape[:2]:
            raise ValueError(
                f"token_ids {tuple(token_ids.shape)} must align with tokens "
                f"{tuple(tokens.shape)} on (batch, sequence)"
            )
        if token_ids.shape[-1] != 4:
            raise ValueError(
                f"FLUX.2 position ids are 4-D (T, H, W, L), got last dim "
                f"{token_ids.shape[-1]}"
            )
        if text_ids.shape[:2] != text_embeds.shape[:2]:
            raise ValueError(
                f"text_ids {tuple(text_ids.shape)} must align with text_embeds "
                f"{tuple(text_embeds.shape)} on (batch, sequence)"
            )
        if timestep.ndim != 1 or timestep.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"timestep must be (B,) matching batch {tokens.shape[0]}, got "
                f"{tuple(timestep.shape)}"
            )
        # The model scales by 1000 internally. Catching the 0..1000 convention here is
        # worth the check: it does not raise downstream, it just trains on the wrong
        # schedule.
        if timestep.numel() and float(timestep.detach().abs().max()) > 1.0 + 1e-4:
            raise ValueError(
                "timestep must be normalised to [0, 1]; FLUX.2's forward multiplies by "
                "1000 itself (transformer_flux2.py:1236)"
            )

        return {
            "hidden_states": tokens,
            "encoder_hidden_states": text_embeds,
            "timestep": timestep,
            "img_ids": token_ids,
            "txt_ids": text_ids,
            "guidance": guidance,
            "return_dict": False,
        }

    @staticmethod
    def take_target_span(output: torch.Tensor, target_len: int) -> torch.Tensor:
        """Slice the target span out of ``[target; references]``.

        The model has already removed text tokens, so index 0 is the first target token.
        References are read-only context and carry no loss.
        """
        if target_len > output.shape[1]:
            raise ValueError(
                f"target_len={target_len} exceeds output sequence {output.shape[1]}"
            )
        return output[:, :target_len]

    @staticmethod
    def requires_guidance(config: dict[str, Any]) -> bool:
        """Whether this checkpoint embeds a guidance scale.

        Checked explicitly rather than defaulted, because passing ``guidance=None`` to a
        guidance-embedded model is accepted by the forward pass and merely changes the
        conditioning — exactly the class of mistake that shows up as "quality is slightly
        worse" three thousand steps later.
        """
        return bool(config.get("guidance_embeds", False))


__all__ = ["DEFAULT_LORA_TARGETS", "Flux2Family"]
