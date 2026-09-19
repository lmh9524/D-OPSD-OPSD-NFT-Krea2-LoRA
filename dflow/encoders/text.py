"""Text encoder wrapper: lifecycle, not maths.

Prompt encoding is delegated to ``Flux2KleinPipeline._get_qwen3_prompt_embeds``, which is a
``@staticmethod`` taking the encoder and tokenizer explicitly — so it is callable with no
pipeline machinery at all. Delegating is not laziness; re-implementing it would go wrong in
two ways that produce no error:

* the prompt is wrapped in **Qwen3's chat template**
  (``apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)``), not
  tokenised directly the way FLUX.1 and Wan do;
* the three selected hidden layers are stacked **per token position** —
  ``stack(dim=1) -> permute(0, 2, 1, 3) -> (B, L, 3H)`` — and any other axis order yields
  the same shape with scrambled content.

The identity that ties it together, asserted in ``tests/test_real_klein_config.py``:
``transformer.joint_attention_dim == 3 * text_encoder.hidden_size`` (12288 == 3 x 4096 for
klein-9B).

What this module owns is what a training run needs and a pipeline does not: whether to load
a 16 GB encoder at all, its placement, and where cached embeddings come from. Caption
dropout is *not* here — it is a training decision that belongs in ``step_fn``, applied to
prompt strings before they reach this class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dflow.common.hub import localize
from dflow.config import TextEncoderConfig
from dflow.encoders.cache import TextEmbedCache, prompt_key

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _prompt_embed_fn():
    """The upstream static encoder. Isolated so the dependency is visible in one place."""
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    return Flux2KleinPipeline._get_qwen3_prompt_embeds


def _text_ids_fn():
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    return Flux2KleinPipeline._prepare_text_ids


@dataclass(frozen=True, slots=True)
class TextConditioning:
    """What the transformer needs for the text stream."""

    embeds: torch.Tensor  # (B, L, 3 * hidden_size)
    ids: torch.Tensor  # (B, L, 4) — text uses the fourth axis for position

    def to(self, *, device: torch.device | None = None, dtype: torch.dtype | None = None):
        return TextConditioning(
            embeds=self.embeds.to(device=device, dtype=dtype),
            ids=self.ids.to(device=device),
        )


class TextEncoder:
    """Frozen text encoder, owned by the training run."""

    def __init__(
        self,
        model: nn.Module | None,
        tokenizer,
        config: TextEncoderConfig,
        *,
        device: torch.device,
        cache: TextEmbedCache | None = None,
        embed_dim: int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = _DTYPES[config.dtype]
        self.tokenizer = tokenizer
        self.cache = cache
        self._cached_embed_dim = embed_dim
        # None when every prompt is cached: the 16 GB encoder is then never loaded at all, which is
        # the largest single memory saving available on one card.
        self.model = None if model is None else model.eval().requires_grad_(False)
        if self.model is not None and config.placement == "device":
            self.model.to(device=device)
        if self.model is None and cache is None:
            raise ValueError("TextEncoder needs either a model or a cache")

    @classmethod
    def load(
        cls,
        config: TextEncoderConfig,
        *,
        path: str,
        device: torch.device,
        revision: str | None = None,
        cache_dir: str | None = None,
        required_prompts: list[str] | None = None,
    ) -> TextEncoder:
        """Load the encoder, or skip it entirely when the cache already covers every prompt.

        ``required_prompts`` is what makes skipping safe: the cache is verified complete *before*
        the decision, so a miss surfaces at startup rather than after the encoder is gone.
        """
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        cache = TextEmbedCache(cache_dir) if cache_dir else None
        if cache is not None and cache.exists and required_prompts is not None:
            keys = [
                prompt_key(
                    prompt, out_layers=config.out_layers, max_length=config.max_length
                )
                for prompt in required_prompts
            ]
            cache.require(keys)
            width = cache.get(keys[0])
            return cls(
                None,
                None,
                config,
                device=device,
                cache=cache,
                embed_dim=int(width.shape[-1]) if width is not None else None,
            )

        # Resolved to local directories first: transformers' subfolder lookup fails offline
        # against a partial snapshot, which is the normal state for a 53 GB gated repo.
        source = config.path or path
        model = Qwen3ForCausalLM.from_pretrained(
            localize(source, subfolder=config.subfolder),
            revision=revision,
            dtype=_DTYPES[config.dtype],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            localize(source, subfolder=config.tokenizer_subfolder), revision=revision
        )
        return cls(model, tokenizer, config, device=device, cache=cache)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def hidden_size(self) -> int:
        if self.model is None:
            if self._cached_embed_dim is None:
                raise RuntimeError("no model and no cached width to infer hidden_size from")
            return self._cached_embed_dim // len(self.config.out_layers)
        return int(self.model.config.hidden_size)

    @property
    def embed_dim(self) -> int:
        """Width the transformer sees: one slice per stacked layer."""
        if self._cached_embed_dim is not None:
            return self._cached_embed_dim
        return self.hidden_size * len(self.config.out_layers)

    @torch.no_grad()
    def encode(self, prompts: str | list[str]) -> TextConditioning:
        """Encode prompts to ``(embeds, ids)``.

        Empty strings are legitimate input: klein embeds no guidance scale and relies on
        real classifier-free guidance with an empty negative prompt, so the unconditional
        branch has to be exercised during training too.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("encode() needs at least one prompt")

        if self.cache is not None:
            cached = self._from_cache(list(prompts))
            if cached is not None:
                return cached
        if self.model is None:
            raise RuntimeError(
                "the text encoder was not loaded and these prompts are not cached; re-run "
                "tools/data/precompute_text_embeds.py or enable the encoder"
            )

        embeds = _prompt_embed_fn()(
            text_encoder=self.model,
            tokenizer=self.tokenizer,
            prompt=list(prompts),
            dtype=self.dtype,
            device=self.device,
            max_sequence_length=self.config.max_length,
            hidden_states_layers=list(self.config.out_layers),
        )
        ids = _text_ids_fn()(embeds).to(self.device)
        return TextConditioning(embeds=embeds, ids=ids)

    def _from_cache(self, prompts: list[str]) -> TextConditioning | None:
        """Assemble a batch from the cache, or None if any prompt is missing."""
        assert self.cache is not None
        rows = []
        for prompt in prompts:
            key = prompt_key(
                prompt, out_layers=self.config.out_layers, max_length=self.config.max_length
            )
            row = self.cache.get(key, device=self.device)
            if row is None:
                return None
            rows.append(row.to(dtype=self.dtype))
        embeds = torch.stack(rows)
        return TextConditioning(embeds=embeds, ids=_text_ids_fn()(embeds).to(self.device))

    def check_compatibility(self, joint_attention_dim: int) -> None:
        """Fail loudly on an encoder/transformer width mismatch.

        Cheap to check once at setup, and the alternative is a shape error thousands of
        steps in — or worse, no error at all if a wrong ``out_layers`` length happens to
        match.
        """
        if self.embed_dim != joint_attention_dim:
            raise ValueError(
                f"text encoder produces {self.embed_dim} channels "
                f"({len(self.config.out_layers)} layers x {self.hidden_size}) but the "
                f"transformer expects joint_attention_dim={joint_attention_dim}. Check "
                f"out_layers: klein uses (9, 18, 27), dev uses (10, 20, 30)."
            )


__all__ = ["TextConditioning", "TextEncoder"]
