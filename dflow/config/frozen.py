"""Frozen auxiliary components: VAE and text encoders.

These are never trained. What is configured here is *how to run them* — precision,
memory placement, and where their inputs come from. Anything that changes training
semantics (posterior sampling mode, caption dropout, guidance strategy) belongs at the
call site in ``experiments/``, where it is visible in the ~120 lines you actually read.

Pure declaration. This module must not import anything else from ``dflow``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .runtime import DType

Placement = Literal["device", "cpu"]


@dataclass(kw_only=True, slots=True)
class VAEConfig:
    # False when the dataset already provides latents: the VAE is then never loaded.
    enabled: bool = True
    path: str = ""  # empty: use the backbone's model path
    subfolder: str = "vae"

    # Encoding in bf16 is fine. Decoding for validation output is not: bf16 decode
    # produces banding and blocking, so it defaults to fp32.
    encode_dtype: DType = "bfloat16"
    decode_dtype: DType = "float32"

    placement: Placement = "device"
    tiling: bool = False
    tile_sample_min: tuple[int, ...] = (512, 512)
    slicing: bool = False


@dataclass(kw_only=True, slots=True)
class TextEncoderConfig:
    name: str = "qwen3"
    path: str = ""  # empty: use the backbone's model path
    subfolder: str = "text_encoder"
    tokenizer_subfolder: str = "tokenizer"
    dtype: DType = "bfloat16"
    max_length: int = 512

    # FLUX.2 does not use the final hidden state: it stacks several intermediate layers
    # and feeds the concatenation to the transformer, which is why the transformer's
    # `joint_attention_dim` equals 3 x the encoder's hidden size. klein uses
    # (9, 18, 27); dev uses (10, 20, 30). Getting this wrong silently changes the
    # conditioning.
    out_layers: tuple[int, ...] = (9, 18, 27)

    placement: Placement = "device"

    #: Feed the reference images through the text encoder as well as the VAE.
    #:
    #: Qwen3-VL is a vision-language model, so this conditions through a pathway that is **already
    #: pretrained** — unlike the RoPE T axis, which Krea 2 has only ever seen at zero. The working
    #: community edit LoRA calls the pair "in-context VAE tokens + image-grounded Qwen3-VL
    #: encoding"; the second half is what this framework was missing.
    #:
    #: Off by default: FLUX.2's encoder is text-only and has no vision tower to feed.
    ground_references: bool = False
    #: Longest side fed to the vision tower, sampled per call from ``[jitter_min, max_px]``.
    #:
    #: The published recipe is 768/384 for **two** references. Vision tokens grow quadratically
    #: with the side, so nine references at that setting would put ~3900 tokens into the text
    #: stream on their own; 384/192 keeps nine at roughly the budget two cost there.
    #:
    #: The jitter is not decoration — it teaches scale robustness, so inference may ground at a
    #: different resolution without a train/serve mismatch. It is also why these embeddings cannot
    #: be cached: a cache freezes one scale.
    grounding_max_px: int = 384
    grounding_jitter_min: int = 192
    #: How many leading references to ground. 0 grounds all of them.
    #:
    #: **This is a cost ceiling, not a preference.** Qwen3-VL's vision tower in transformers 5.15
    #: costs a flat ~2.5 s per image on an H100 — independent of the image's patch count, so it is
    #: fixed CPU overhead rather than compute, and neither flash-attention-2 nor a smaller grounding
    #: resolution moves it. Nine references would therefore add ~22 s to every step and put a 4000
    #: step run past a day.
    #:
    #: One is the useful one to spend it on. Garment appearance already reaches the model through
    #: the in-context VAE tokens, which demonstrably works — colour transfer is reliable. Identity
    #: is what has not worked: swapping the person moved the output less than reseeding did. That is
    #: the pathway worth routing through a pretrained encoder, and slot 0 is where identity lives.
    max_grounded_references: int = 1
    #: Replace Qwen3-VL's patch-embedding Conv3d with the equivalent linear layer.
    #:
    #: Its kernel equals its stride equals the input's spatial extent — `(2, 16, 16)` over a
    #: `(N, 3, 2, 16, 16)` input — so every output is 1x1x1 and the convolution *is* a matrix
    #: multiply over the flattened 1536-value patch, which is exactly the layout `pixel_values`
    #: already has. PyTorch has no cuDNN kernel for that shape and falls back to
    #: `slow_conv_dilated3d`: 490k `cudaMemcpyAsync` calls and 2.57 s of CPU per image on an H100,
    #: against 512 ms of actual GPU work. As a linear layer it is 13 ms.
    #:
    #: The consequence is larger than it sounds. A training step was 2.92 s, of which 2.57 s was
    #: this one convolution — the 13B transformer's forward and backward were 0.35 s. It also made
    #: grounding more than one reference unaffordable (23 s per step for nine) and so kept
    #: `max_grounded_references` at 1, which is the last unverified difference from the recipe this
    #: work is modelled on.
    #:
    #: Off by default because it is not bit-identical. The substitution's error against the
    #: convolution is relL2 0.0024, the same order as the convolution's own bf16 rounding (0.0022
    #: against fp32) — but Qwen3-VL amplifies perturbations at that scale unevenly, up to relL2 0.9
    #: on one test sample in twenty. The encoder is frozen, so what matters is that training and
    #: sampling use the *same* path, not that either matches a reference bit for bit; a checkpoint
    #: trained with this on must be sampled with it on.
    fast_patch_embed: bool = False
    #: Where to load the multimodal processor from. Krea 2's own ``tokenizer/`` ships only the text
    #: tokenizer, so the vision path needs the processor from the encoder's origin repo.
    processor_path: str = "Qwen/Qwen3-VL-4B-Instruct"


@dataclass(kw_only=True, slots=True)
class TextConfig:
    # False when prompt embeddings are precomputed: the encoder is then never loaded.
    # For klein that saves the largest single block of frozen weights on one card.
    enabled: bool = True
    # A list because some families use several encoders (FLUX.1: T5 + CLIP pooled).
    # klein uses exactly one and has no pooled branch.
    encoders: tuple[TextEncoderConfig, ...] = field(
        default_factory=lambda: (TextEncoderConfig(),)
    )
    cache_dir: str | None = None


@dataclass(kw_only=True, slots=True)
class FrozenConfig:
    vae: VAEConfig = field(default_factory=VAEConfig)
    text: TextConfig = field(default_factory=TextConfig)


__all__ = ["FrozenConfig", "Placement", "TextConfig", "TextEncoderConfig", "VAEConfig"]
