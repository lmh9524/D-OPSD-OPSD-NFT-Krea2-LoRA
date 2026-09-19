"""Krea 2 text encoder: lifecycle, not maths — same contract as ``text.py``, different upstream.

Krea 2 shares nothing with FLUX.2 on this path. It taps **twelve** Qwen3-VL-4B decoder layers instead
of three, stacks them on their own axis instead of flattening them, and wraps the prompt in a
template that pads *in the middle*:

    FLUX.2 klein                              Krea 2
    Qwen3 chat template                       Qwen-Image template, [prefix | prompt | PAD | suffix]
    3 taps -> stack -> (B, L, 3H)             12 taps -> stack -> (B, L, 12, 2560)
    no attention mask                         mask is required — padding sits between real tokens
    positions implicit                        mRoPE positions built by hand from cumulative valid

Four things a from-scratch implementation gets wrong without raising:

===========================  ==========================================================
the template                 the suffix goes **after** the padding, not before it
mRoPE positions              built from ``cumsum`` of the mask, so padding consumes no
                             position; the default raw-index positions put the suffix at
                             ~max_length and shift its phase
the prefix drop              the first 34 tokens (system prefix) are sliced off *both*
                             the hidden states and the mask
the stack axis               ``stack(..., dim=2)`` -> ``(B, L, layers, dim)``; any other
                             axis gives the same element count, scrambled
===========================  ==========================================================

So the maths is delegated, exactly as ``text.py`` delegates to a FLUX.2 static method. The
complication is that ``Krea2Pipeline.get_text_hidden_states`` is an **instance** method: it reads
nine attributes off ``self``. Rather than copy 40 lines that would then drift, this module builds a
shim carrying exactly those attributes and calls the unbound function through it. If an upgrade adds
an attribute, the call raises ``AttributeError`` at setup — loudly, and before any weights load —
rather than silently encoding differently. ``tests/test_upstream_contract.py`` pins the attribute
list.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dflow.common.hub import localize
from dflow.config import TextEncoderConfig

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

#: Vision block inserted per reference image, immediately after the system prefix and before the
#: instruction. Verbatim from the trainer that produced the working community edit LoRA
#: (`lbouaraba/krea2edit-trainer`, `_encode_image_prompt`).
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

#: Grounding resolution, sampled per call from ``[jitter_min, max_px]``. Jitter is not decoration:
#: it teaches scale robustness, so inference may use another grounding scale without a train/serve
#: mismatch. It is also why prompt embeddings cannot be cached — a cache freezes one scale.
GROUNDING_MAX_PX = 768
GROUNDING_JITTER_MIN = 384

#: Every attribute ``get_text_hidden_states`` reads off ``self``. Asserted against the real
#: signature in ``tests/test_upstream_contract.py``, so an upstream change surfaces as a failing
#: test rather than as different conditioning.
SHIM_ATTRIBUTES: tuple[str, ...] = (
    "tokenizer",
    "text_encoder",
    "text_encoder_select_layers",
    "prompt_template_encode_prefix",
    "prompt_template_encode_suffix",
    "prompt_template_encode_start_idx",
    "prompt_template_encode_num_suffix_tokens",
    "_execution_device",
)


def _hidden_states_fn():
    """The upstream encoder, unbound. Isolated so the dependency is visible in one place."""
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

    return Krea2Pipeline.get_text_hidden_states


@dataclass(frozen=True, slots=True)
class Krea2TextConditioning:
    """What the transformer needs for the text stream.

    Note the mask: it is not an optimisation. Krea 2's padding sits *between* real tokens, so
    dropping the mask lets the model attend to padding as if it were prompt.
    """

    embeds: torch.Tensor  # (B, L, num_text_layers, text_hidden_dim)
    mask: torch.Tensor  # (B, L) bool

    def to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
        return Krea2TextConditioning(
            embeds=self.embeds.to(device=device, dtype=dtype),
            mask=self.mask.to(device=device),
        )


class _PipelineShim:
    """Carries exactly the attributes ``get_text_hidden_states`` reads, and nothing else."""

    def __init__(self, *, tokenizer, text_encoder, select_layers, device) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.text_encoder_select_layers = tuple(select_layers)
        self._execution_device = device
        # Verbatim from Krea2Pipeline.__init__. Copied rather than imported because they are
        # instance attributes assigned in the constructor, not class-level constants there is
        # anything to import.
        self.prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_num_suffix_tokens = 5


def _use_linear_patch_embed(model: nn.Module) -> None:
    """Swap Qwen3-VL's patch-embedding Conv3d for the linear layer it already is.

    ``kernel_size == stride == (temporal, patch, patch)`` over an input viewed as
    ``(N, C, temporal, patch, patch)``, so every output position is 1x1x1 and the convolution is a
    matrix multiply over the flattened patch — which is the layout ``pixel_values`` arrives in. The
    weight reshape is the identity on memory; the arithmetic is the same dot product.

    Worth doing because PyTorch has no cuDNN kernel for that shape and falls back to
    ``slow_conv_dilated3d``, which costs 2.57 s of CPU per image on an H100 against 512 ms of GPU
    work, and dominated the training step it was called from.

    A no-op if the model has no vision tower, or if the swap has already happened.
    """
    visual = getattr(model, "visual", None) or getattr(getattr(model, "model", model), "visual", None)
    embed = getattr(visual, "patch_embed", None)
    proj = getattr(embed, "proj", None)
    if proj is None:
        return
    if tuple(proj.kernel_size) != tuple(proj.stride):
        raise ValueError(
            f"patch embedding has kernel {tuple(proj.kernel_size)} and stride "
            f"{tuple(proj.stride)}; the linear form is only equivalent when they match"
        )

    class _LinearPatchEmbed(nn.Module):
        def __init__(self, proj: nn.Conv3d) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                proj.weight.detach().reshape(proj.weight.shape[0], -1).contiguous(),
                requires_grad=False,
            )
            self.bias = nn.Parameter(proj.bias.detach().clone(), requires_grad=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(
                hidden_states.to(self.weight.dtype), self.weight, self.bias
            )

    visual.patch_embed = _LinearPatchEmbed(proj)


class Krea2TextEncoder:
    """Frozen Qwen3-VL text encoder, owned by the training run."""

    def __init__(
        self,
        model: nn.Module | None,
        tokenizer,
        config: TextEncoderConfig,
        *,
        device: torch.device,
        select_layers: tuple[int, ...],
        processor_path: str = "Qwen/Qwen3-VL-4B-Instruct",
        grounding_max_px: int = GROUNDING_MAX_PX,
        grounding_jitter_min: int = GROUNDING_JITTER_MIN,
        max_grounded: int = 0,
        fast_patch_embed: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = _DTYPES[config.dtype]
        self.tokenizer = tokenizer
        self.select_layers = tuple(select_layers)
        self.model = None if model is None else model.eval().requires_grad_(False)
        if self.model is not None and config.placement == "device":
            self.model.to(device=device)
        if self.model is None:
            raise ValueError(
                "Krea2TextEncoder needs a model. Precomputed embeddings are not wired up for "
                "Krea 2: its conditioning is (L, 12, 2560) per prompt, ~63 MB in bf16 at "
                "max_length 512, five times klein's 12.6 MB."
            )
        self._processor = None
        self.processor_path = processor_path
        self.grounding_max_px = grounding_max_px
        self.grounding_jitter_min = grounding_jitter_min
        self.max_grounded = max_grounded
        if fast_patch_embed and self.model is not None:
            _use_linear_patch_embed(self.model)
        self._shim = _PipelineShim(
            tokenizer=tokenizer,
            text_encoder=self.model,
            select_layers=self.select_layers,
            device=device,
        )

    @classmethod
    def load(
        cls,
        config: TextEncoderConfig,
        *,
        path: str,
        device: torch.device,
        select_layers: tuple[int, ...],
        revision: str | None = None,
    ) -> Krea2TextEncoder:
        from transformers import AutoModel, AutoTokenizer

        source = config.path or path
        # ``AutoModel``, not ``AutoModelForCausalLM``. The checkpoint declares
        # ``architectures: ["Qwen3VLModel"]`` and ``model_index.json`` names the same class, which
        # is the bare backbone — ``AutoModel`` resolves ``model_type: qwen3_vl`` to exactly it.
        # Asking for a CausalLM instead builds an ``lm_head`` this path never calls, and the
        # pipeline's ``get_text_hidden_states`` only ever reads ``outputs.hidden_states``.
        model = AutoModel.from_pretrained(
            localize(source, subfolder=config.subfolder),
            revision=revision,
            dtype=_DTYPES[config.dtype],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            localize(source, subfolder=config.tokenizer_subfolder), revision=revision
        )
        return cls(
            model, tokenizer, config, device=device, select_layers=select_layers,
            processor_path=config.processor_path,
            max_grounded=config.max_grounded_references,
            grounding_max_px=config.grounding_max_px,
            grounding_jitter_min=config.grounding_jitter_min,
            fast_patch_embed=config.fast_patch_embed,
        )

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def hidden_size(self) -> int:
        """Width of one tapped layer.

        ``Qwen3VLConfig`` is a **composite** config — a vision tower plus a text tower — so it has
        no top-level ``hidden_size``; the number the transformer must match lives under
        ``text_config``. Checked in that order rather than assuming, because klein's
        ``Qwen3Config`` does carry it at the top level and the two encoders share this wrapper's
        shape of API.
        """
        config = self.model.config
        for holder in (config, getattr(config, "text_config", None)):
            size = getattr(holder, "hidden_size", None)
            if size is not None:
                return int(size)
        raise AttributeError(
            f"{type(config).__name__} exposes no hidden_size, at the top level or under "
            f"text_config; cannot verify the encoder matches the transformer's text_hidden_dim"
        )

    @property
    def num_layers(self) -> int:
        return len(self.select_layers)

    @torch.no_grad()
    def encode(self, prompts: str | list[str]) -> Krea2TextConditioning:
        """Encode prompts to ``(embeds, mask)``.

        Empty strings are legitimate input: Krea 2 embeds no guidance scale and relies on real
        classifier-free guidance with a negative prompt, so the unconditional branch has to be
        exercised during training too.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("encode() needs at least one prompt")

        embeds, mask = _hidden_states_fn()(
            self._shim,
            prompt=list(prompts),
            max_sequence_length=self.config.max_length,
            device=self.device,
        )
        return Krea2TextConditioning(embeds=embeds.to(dtype=self.dtype), mask=mask)

    def _multimodal_processor(self):
        """A real ``Qwen3VLProcessor``, loaded lazily and cached.

        Krea 2's own ``tokenizer/`` holds only the text tokenizer — no image processor — so the
        vision path needs the processor from the encoder's origin repo. Its ``patch_size=16`` and
        ``merge_size=2`` are asserted against the checkpoint's ``vision_config``, because a
        mismatch would silently change how many image tokens each reference expands to.
        """
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.processor_path)
            vision = getattr(self.model.config, "vision_config", None)
            image_processor = self._processor.image_processor
            for name, expected in (
                ("patch_size", getattr(vision, "patch_size", None)),
                ("merge_size", getattr(vision, "spatial_merge_size", None)),
            ):
                actual = getattr(image_processor, name, None)
                if expected is not None and actual is not None and actual != expected:
                    raise ValueError(
                        f"{self.processor_path}'s image processor has {name}={actual} but the "
                        f"checkpoint's vision_config says {expected}; reference images would "
                        f"expand to a different number of image tokens than it was trained with"
                    )
        return self._processor

    def _grounding_image(self, image: torch.Tensor, rng):
        """``(3, H, W)`` in [-1, 1] -> PIL, downscaled to a jittered grounding resolution.

        The grounding path is **semantic**: fine detail reaches the transformer through the VAE
        reference tokens, not through Qwen3-VL. Capping the longest side therefore cuts vision
        tokens quadratically for little semantic loss, which is what keeps the text stream short
        enough to train.
        """
        from PIL import Image

        # Accept a collated ``(1, 3, H, W)`` slot as well as a bare ``(3, H, W)`` image: the dataset
        # yields one tensor per reference slot and collate keeps the batch axis even at batch_size=1.
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError(
                    f"grounding takes one image at a time, got a batch of {image.shape[0]}"
                )
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"expected (3, H, W) or (1, 3, H, W), got {tuple(image.shape)}")

        array = ((image.detach().float().clamp(-1, 1) + 1) * 127.5).round()
        pil = Image.fromarray(array.to(torch.uint8).permute(1, 2, 0).cpu().numpy())
        cap = self.grounding_max_px
        if cap > 0:
            low = self.grounding_jitter_min
            target = rng.randint(low, cap) if 0 < low < cap else cap
            if max(pil.size) > target:
                scale = target / max(pil.size)
                pil = pil.resize(
                    (max(1, round(pil.width * scale)), max(1, round(pil.height * scale))),
                    Image.LANCZOS,
                )
        return pil

    @torch.no_grad()
    def encode_grounded(self, prompt: str, images: list[torch.Tensor], *, rng=None):
        """Encode one prompt **with its reference images inside the text encoder**.

        This is the second conditioning path the working community edit LoRA uses, alongside the
        in-context VAE tokens. Qwen3-VL is a vision-language model, so the reference can enter
        through a pathway that is already pretrained — unlike the RoPE T axis, which Krea 2 has only
        ever seen at zero.

        Two alignment details decide whether this trains or silently corrupts:

        * **The vision blocks sit after the system prefix and before the instruction.** The 34-token
          prefix drop then still removes exactly the system prefix; putting the images first would
          make it cut into image tokens instead.
        * **The processor's own multimodal mRoPE is used.** The text-only path hand-builds
          ``position_ids`` from ``cumsum(attention_mask)``; that logic is wrong once image tokens
          are present, so the model's inputs are passed through untouched and it computes them.

        Returns the same ``(embeds, mask)`` contract as :meth:`encode`, at natural length.
        """
        import random

        rng = rng or random.Random()
        processor = self._multimodal_processor()
        # Only the leading references are grounded; see ``max_grounded_references`` for why the
        # ceiling exists and why slot 0 is what it is spent on.
        selected = images[: self.max_grounded] if self.max_grounded else images
        pils = [self._grounding_image(image, rng) for image in selected]
        vision = VISION_BLOCK * len(pils)
        text = (
            self._shim.prompt_template_encode_prefix
            + vision
            + prompt
            + self._shim.prompt_template_encode_suffix
        )
        inputs = processor(text=[text], images=pils or None, return_tensors="pt").to(self.device)
        states = self.model(**inputs, output_hidden_states=True)
        stacked = torch.stack(
            [states.hidden_states[i] for i in self.select_layers], dim=2
        )[:, self._shim.prompt_template_encode_start_idx :]

        mask = inputs.get("attention_mask")
        mask = (
            mask[:, self._shim.prompt_template_encode_start_idx :].bool()
            if mask is not None
            else torch.ones(stacked.shape[:2], dtype=torch.bool, device=stacked.device)
        )
        if mask.shape[1] != stacked.shape[1]:
            raise ValueError(
                f"grounded mask is {mask.shape[1]} long but the tapped states are "
                f"{stacked.shape[1]} — the prefix drop and the mask disagree"
            )
        return Krea2TextConditioning(embeds=stacked.to(dtype=self.dtype), mask=mask)

    def check_compatibility(self, text_hidden_dim: int, num_text_layers: int) -> None:
        """Fail loudly on an encoder/transformer mismatch, at setup rather than mid-run.

        Two independent numbers must agree here, where klein had one: the encoder width and the
        number of taps. A wrong tap count produces a shape error deep inside the text fusion
        stage's ``projector``, whose ``in_features`` is ``num_text_layers``.
        """
        if self.hidden_size != text_hidden_dim:
            raise ValueError(
                f"text encoder hidden size {self.hidden_size} != transformer's "
                f"text_hidden_dim={text_hidden_dim}"
            )
        if self.num_layers != num_text_layers:
            raise ValueError(
                f"{self.num_layers} tapped layers configured but the transformer expects "
                f"num_text_layers={num_text_layers}. Krea 2 taps twelve: "
                f"(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)."
            )


__all__ = [
    "GROUNDING_JITTER_MIN",
    "GROUNDING_MAX_PX",
    "SHIM_ATTRIBUTES",
    "VISION_BLOCK",
    "Krea2TextConditioning",
    "Krea2TextEncoder",
]
