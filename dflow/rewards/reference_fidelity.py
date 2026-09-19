"""CLIP-I reference fidelity: does the generated image match the case's ground truth?

The reward the ref2img RL variant was held back for want of. Flow-GRPO for ref2img is deliberately
not shipped because no bundled reward judges *whether the references were used* — an aesthetic reward
trains a pretty image that ignores the garments, and optimises the wrong thing (see ``docs/rl-design``
and the ``krea2-ref2img`` SKILL). This is that missing reward: the cosine similarity between the CLIP
image embedding of the generated image and that of the case's ground-truth/reference image, the
standard "CLIP-I" score. Higher means the generation is semantically closer to the target look.

It reuses the **same CLIP image encoder pattern as ``aesthetic.py``** — one ``CLIPModel``,
``CLIPImageProcessor`` from the backbone's own ``preprocessor_config.json``, ``get_image_features``
for the projected embedding — so a run enabling both rewards loads one CLIP, and a score here is
comparable to any published CLIP-I number. Cosine is dimension-agnostic, so unlike the aesthetic head
there is no projection-width constraint; the only invariant that matters is that both embeddings come
from one model, which they do by construction.

## Three things that are wrong without raising (mirrors ``aesthetic.py``)

**The L2 normalisation.** Cosine similarity is the dot product of unit vectors. Skip the norm and the
"cosine" is an unbounded dot product that tracks embedding magnitude, not direction — a plausible
number uncorrelated with similarity. So :func:`_embed` normalises in one place and a test asserts it.

**The preprocessing.** CLIP's is a bicubic resize to 224, a centre crop, a rescale by 1/255 and
CLIP's *own* mean/std — not ImageNet's. Reimplementing it in torch would drift every score
systematically, exactly the risk ``encoders/vae.py`` warns about, so ``CLIPImageProcessor`` does it.
The generated image and the ground truth both go through it, so a folding difference between the two
sides cannot creep in.

**Which CLIP output.** ``get_image_features`` — the projected embedding — for both sides, so the two
live in the same space. Reading a different attribute for one side would compare across spaces and
score nonsense that still lies in [-1, 1].

## The metadata contract

The ground truth cannot be known from the generated pixels, so it arrives per sample through
``metadata`` under ``config.reference_key``. The value is a ground-truth image — or a list of them,
whose embeddings are averaged (then re-normalised) — as a uint8 tensor in [0, 255], either
``(C, H, W)`` or ``(H, W, C)``. A sample missing the field **raises**: a silently zero cosine is
indistinguishable from a genuinely dissimilar image, the same rule ``OCRReward`` follows for its
target string. The task's ``schema.py`` owns the field; ``ReferenceFidelityRewardConfig.reference_key``
names it so the contract is visible from both ends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from dflow.config.rl import ReferenceFidelityRewardConfig
from dflow.rewards.base import validate_pixels


class ReferenceFidelityReward:
    """CLIP-I cosine between a generated image and the case's ground truth."""

    name = "reference_fidelity"

    def __init__(
        self,
        clip: nn.Module,
        processor: Any,
        config: ReferenceFidelityRewardConfig,
        *,
        device: torch.device,
    ) -> None:
        self.clip = clip.to(device).eval()
        self.processor = processor
        self.config = config
        self.device = device

    @classmethod
    def load(
        cls,
        config: ReferenceFidelityRewardConfig,
        *,
        device: torch.device,
    ) -> ReferenceFidelityReward | None:
        """Load the CLIP image encoder, or return ``None`` when disabled.

        ``None`` rather than a stub, matching ``AestheticReward.load`` and ``VAEEncoder.load``: the
        not-loaded case stays visible at the call site instead of hiding a no-op behind a method call.
        """
        if not config.enabled:
            return None

        from transformers import CLIPImageProcessor, CLIPModel

        clip = CLIPModel.from_pretrained(config.clip_model)
        processor = CLIPImageProcessor.from_pretrained(config.clip_model)
        return cls(clip, processor, config, device=device)

    # ------------------------------------------------------------------------------- scoring

    @staticmethod
    def _to_chw_uint8(image: Any) -> torch.Tensor:
        """Coerce one ground-truth image to ``(C, H, W)`` uint8, or raise.

        Accepts ``(C, H, W)`` or ``(H, W, C)`` — the two layouts a task might attach — and refuses a
        float tensor on dtype, the same trap ``validate_pixels`` guards: a [0, 1] float would embed
        as a near-black image and score a plausible, wrong cosine.
        """
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if image.ndim != 3:
            raise ValueError(
                f"a reference image must be (C, H, W) or (H, W, C), got {tuple(image.shape)}"
            )
        if image.dtype != torch.uint8:
            raise ValueError(
                f"reference images must be uint8 in [0, 255], got {image.dtype}. Convert where the "
                f"model's output range is known — a float tensor embeds as a near-black image."
            )
        # (H, W, C) -> (C, H, W). Three channels is the discriminator; a 3-row image is pathological.
        if image.shape[0] != 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.shape[0] != 3:
            raise ValueError(
                f"a reference image must have 3 channels, got shape {tuple(image.shape)}"
            )
        return image

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, D)`` unit-norm CLIP image embeddings for ``(B, C, H, W)`` uint8 pixels.

        The normalisation is here, once, so both the generated and ground-truth sides are unit-norm
        and their dot product is a genuine cosine. Converted to PIL explicitly rather than trusting
        the processor to infer the channel axis and ``do_rescale`` from a uint8 tensor — both are
        version-dependent inferences that silently score a black image, and PIL is what the reference
        implementations feed it (identical reasoning to ``aesthetic.py``).
        """
        from PIL import Image

        frames = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in images]
        inputs = self.processor(images=frames, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device, dtype=self.clip.dtype)

        features = self.clip.get_image_features(pixel_values=pixels)
        embeds = features if isinstance(features, torch.Tensor) else features.pooler_output
        if embeds.ndim != 2:
            raise ValueError(
                f"expected (B, D) projected CLIP embeddings, got {tuple(embeds.shape)}; "
                f"`get_image_features` has changed shape"
            )
        embeds = embeds.float()
        return embeds / embeds.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        """``(B,)`` CLIP-I cosine in [-1, 1]. Prompt-independent; the reference is the target."""
        validate_pixels(images, len(prompts))
        if len(metadata) != len(prompts):
            raise ValueError(f"got {len(metadata)} metadata entries for {len(prompts)} prompts")

        key = self.config.reference_key
        generated = self._embed_in_batches(images)

        references: list[torch.Tensor] = []
        for index, meta in enumerate(metadata):
            if key not in meta:
                raise KeyError(
                    f"metadata[{index}] has no {key!r}, so there is nothing to compare against. "
                    f"A ref2img sample must carry its ground-truth image under {key!r}; a zero "
                    f"cosine would be indistinguishable from a dissimilar image."
                )
            value = meta[key]
            frames = value if isinstance(value, (list, tuple)) else [value]
            if not frames:
                raise ValueError(f"metadata[{index}][{key!r}] is empty; expected one or more images")
            stacked = torch.stack([self._to_chw_uint8(frame) for frame in frames])
            # Average the (already unit-norm) reference embeddings, then re-normalise, so several
            # ground-truth views become one direction rather than several competing ones.
            reference = self._embed_in_batches(stacked).mean(dim=0)
            references.append(reference / reference.norm().clamp_min(1e-12))

        reference_embeds = torch.stack(references).to(generated.device)
        return (generated * reference_embeds).sum(dim=-1).float()

    def _embed_in_batches(self, images: torch.Tensor) -> torch.Tensor:
        """Embed ``(N, C, H, W)`` uint8 pixels in ``config.batch_size`` chunks."""
        out: list[torch.Tensor] = []
        for start in range(0, images.shape[0], self.config.batch_size):
            out.append(self._embed(images[start : start + self.config.batch_size]))
        return torch.cat(out)


__all__ = ["ReferenceFidelityReward"]
