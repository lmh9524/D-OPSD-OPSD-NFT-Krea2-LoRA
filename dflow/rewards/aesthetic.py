"""The LAION aesthetic predictor: CLIP ViT-L/14 embeddings through a small MLP.

What DDPO and flow_grpo score with, so a reward curve here is comparable to published ones. The
head is ``improved-aesthetic-predictor``'s ``sac+logos+ava1-l14-linearMSE.pth``, served from its
Hub mirror; it outputs roughly 1-10, and nothing rescales it — a raw scale keeps the numbers
comparable, and group-relative advantage removes the offset anyway.

## Three things that are wrong without raising

**The L2 normalisation.** The head was fitted on unit-norm CLIP embeddings. Feed it raw ones and it
still produces numbers in about the right range, uncorrelated with aesthetics. This is the single
most likely way to get a plausible, useless reward, so :func:`_embed` normalises in one place and a
test asserts the norm.

**The preprocessing.** CLIP's is a bicubic resize to 224, a centre crop, a rescale by 1/255, and
CLIP's *own* mean and standard deviation — not ImageNet's. Reimplementing it in torch would be
faster than the PIL round-trip, and it is what the drift argument in ``encoders/vae.py`` warns
about: a different resize kernel shifts every score systematically. So ``CLIPImageProcessor`` does
it, from the backbone's own ``preprocessor_config.json``. The reward runs once per trajectory, not
once per denoising step, so the cost lands in the right place.

**Which CLIP output.** ``get_image_features`` — the *projected* embedding, 768-dim for ViT-L/14 —
not the vision tower's 1024-dim hidden state. Both are available, both are plausible, and only one
matches the head's first layer. The dimension check in :meth:`AestheticReward.load` is what turns
that into an error instead of a shape mismatch deep in a forward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from dflow.config.rl import AestheticRewardConfig
from dflow.rewards.base import validate_pixels

#: The published head's parameter shapes, read from the checkpoint itself.
#:
#: Kept as a constant so ``tests/rewards/test_aesthetic.py`` can assert the module we build matches
#: without downloading 1.7 GB of CLIP. ``nn.Sequential`` indices are part of the contract: the
#: dropouts at 1, 3 and 5 hold no parameters but do occupy positions, and 7 follows 6 with no
#: dropout between them. Renumber and ``load_state_dict`` fails — which is the good case; build the
#: right shapes in the wrong order and it succeeds.
HEAD_SHAPES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("layers.0.weight", (1024, 768)),
    ("layers.0.bias", (1024,)),
    ("layers.2.weight", (128, 1024)),
    ("layers.2.bias", (128,)),
    ("layers.4.weight", (64, 128)),
    ("layers.4.bias", (64,)),
    ("layers.6.weight", (16, 64)),
    ("layers.6.bias", (16,)),
    ("layers.7.weight", (1, 16)),
    ("layers.7.bias", (1,)),
)

#: CLIP ViT-L/14's projection dimension, and therefore the head's input width.
EMBED_DIM = 768


class AestheticHead(nn.Module):
    """The predictor MLP, in the layout the published checkpoint's keys imply.

    The ``layers`` attribute is not a stylistic choice: every key in the checkpoint is
    ``layers.N.*``, so a bare ``nn.Sequential`` — whose keys are ``N.*`` — fails to load. That is
    the good failure, but it is still a failure, which is why ``HEAD_SHAPES`` pins the names as
    well as the shapes.

    Dropout probabilities are the published ones but inert: the head is never trained here and is
    always in ``eval()``. They are present because the module *indices* are part of the contract,
    not because the values matter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMBED_DIM, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings)


def build_head() -> AestheticHead:
    """The predictor MLP, ready for the published state dict."""
    return AestheticHead()


class AestheticReward:
    """CLIP + the LAION head, scoring uint8 pixels."""

    name = "aesthetic"

    def __init__(
        self,
        clip: nn.Module,
        processor: Any,
        head: AestheticHead,
        config: AestheticRewardConfig,
        *,
        device: torch.device,
    ) -> None:
        self.clip = clip.to(device).eval()
        self.processor = processor
        self.head = head.to(device).eval()
        self.config = config
        self.device = device

    @classmethod
    def load(
        cls,
        config: AestheticRewardConfig,
        *,
        device: torch.device,
    ) -> AestheticReward | None:
        """Load CLIP and the head, or return ``None`` when disabled.

        ``None`` rather than a stub, matching ``VAEEncoder.load``: the not-loaded case stays visible
        at the call site instead of hiding a no-op behind a method call.
        """
        if not config.enabled:
            return None

        from huggingface_hub import hf_hub_download
        from transformers import CLIPImageProcessor, CLIPModel

        clip = CLIPModel.from_pretrained(config.clip_model)
        projection = int(clip.config.projection_dim)
        if projection != EMBED_DIM:
            raise ValueError(
                f"{config.clip_model} projects to {projection} dimensions, but the aesthetic head "
                f"expects {EMBED_DIM}. A different backbone loads cleanly and scores nonsense."
            )
        processor = CLIPImageProcessor.from_pretrained(config.clip_model)

        head = build_head()
        weights = hf_hub_download(config.head_repo, config.head_filename)
        # weights_only: the checkpoint is a pickled state dict from a third-party repo, and
        # torch.load's default would execute whatever it contains.
        state = torch.load(weights, map_location="cpu", weights_only=True)
        head.load_state_dict(state)

        return cls(clip, processor, head, config, device=device)

    # ------------------------------------------------------------------------------- scoring

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, 768)`` unit-norm CLIP image embeddings for uint8 pixels.

        The normalisation is here, once. The head was fitted on unit-norm embeddings and produces
        plausible, meaningless numbers without it.

        Converted to PIL explicitly rather than handing the processor a uint8 tensor and trusting
        it to infer the channel axis and apply ``do_rescale``. Both are version-dependent
        inferences on a path where getting them wrong scores a black image without complaint, and
        PIL is what the reference implementations feed it.
        """
        from PIL import Image

        frames = [
            Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in images
        ]
        inputs = self.processor(images=frames, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device, dtype=self.clip.dtype)

        # `transformers` is not pinned here the way diffusers is, and this return type has already
        # changed once: `get_image_features` used to hand back the projected tensor, and now
        # returns a `BaseModelOutputWithPooling` whose `pooler_output` the method has overwritten
        # with `visual_projection(...)`. Read the attribute explicitly rather than chaining
        # fallbacks -- and check the width, because the *unprojected* pooled output lives under
        # the same name upstream at 1024 dims, and feeding the head that would be a shape error
        # at best and a plausible wrong number at worst.
        features = self.clip.get_image_features(pixel_values=pixels)
        embeds = features if isinstance(features, torch.Tensor) else features.pooler_output
        if embeds.ndim != 2 or embeds.shape[-1] != EMBED_DIM:
            raise ValueError(
                f"expected ({pixels.shape[0]}, {EMBED_DIM}) projected CLIP embeddings, got "
                f"{tuple(embeds.shape)}. `get_image_features` has changed shape; the head is "
                f"fitted on the {EMBED_DIM}-dim projection, not the vision tower's hidden state."
            )
        embeds = embeds.float()
        return embeds / embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        metadata: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        """``(B,)`` aesthetic scores, roughly 1-10. Prompt-independent by construction."""
        validate_pixels(images, len(prompts))
        scores: list[torch.Tensor] = []
        for start in range(0, images.shape[0], self.config.batch_size):
            chunk = images[start : start + self.config.batch_size]
            scores.append(self.head(self._embed(chunk)).squeeze(-1))
        return torch.cat(scores).float()


__all__ = ["EMBED_DIM", "HEAD_SHAPES", "AestheticHead", "AestheticReward", "build_head"]
