"""Krea 2 VAE wrapper: lifecycle, not maths — but here the maths has no upstream to delegate to.

``Krea2Pipeline`` is **text-to-image only**. It decodes latents and never encodes an image, so unlike
``vae.py`` — which mirrors ``Flux2KleinPipeline._encode_vae_image`` — there is no encode path to copy
or call. What exists is the decode path, and encode is defined here as its exact inverse. That
inversion is the whole content of this module, so it is written out step by step:

``pipeline_krea2.py``, decoding::

    latents = self._unpack_latents(latents, height, width)              # (B, z_dim, 1, H, W)
    latents_mean = tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
    latents_std  = 1.0 / tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)
    latents = latents / latents_std + latents_mean
    image = vae.decode(latents)[0][:, :, 0]

Note ``latents_std`` is the *reciprocal* of the config field, and the code then *divides* by it — so
the two inversions cancel and the operation is a multiply by ``config.latents_std``::

    z_vae  = z_norm * config.latents_std + config.latents_mean          (decode)
    z_norm = (z_vae - config.latents_mean) / config.latents_std         (encode, this module)

Three things this gets right that a from-scratch version gets wrong without raising:

===============================  ==========================================================
normalisation order              **normalise first, patchify second** — the opposite of
                                 FLUX.2. Decode unpacks *before* normalising, and the
                                 statistics have ``z_dim`` entries, not ``z_dim * p * p``.
                                 Doing it FLUX.2's way broadcasts the wrong 16 numbers
                                 across 64 channels and still runs.
the frame axis                   ``AutoencoderKLQwenImage`` is a **video** VAE. Images go in
                                 as ``(B, C, 1, H, W)`` and come back with a frame axis that
                                 decode drops via ``[:, :, 0]``.
the reciprocal                   ``latents_std`` in the pipeline is ``1 / config.latents_std``
                                 and is then divided by. Reading the line without the
                                 preceding one inverts the scale.
===============================  ==========================================================

``tests/test_encoders_krea2.py`` pins the inverse by round-tripping through the pipeline's own
decode-side arithmetic, which is the only check that stays true if upstream changes the convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dflow.common.hub import localize
from dflow.config import VAEConfig

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

SampleMode = str  # "sample" | "mode"


def patchify(latents: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    """Fold ``patch_size``-square patches into channels: ``(B, C, H, W) -> (B, C*p*p, H/p, W/p)``.

    The channel order matches ``Krea2Pipeline._pack_latents``, which permutes ``(0, 2, 4, 1, 3, 5)``
    and flattens to ``(B, HW/p^2, C*p*p)``: within one token the index is
    ``c * p * p + row * p + column``. Emitting ``(B, C*p*p, H/p, W/p)`` here lets the task's generic
    ``pack()`` produce that token layout unchanged, which is the same division of labour ``vae.py``
    uses for FLUX.2.
    """
    if latents.ndim != 4:
        raise ValueError(f"latents must be (B, C, H, W), got {tuple(latents.shape)}")
    batch, channels, height, width = latents.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"latent extent {height}x{width} is not divisible by patch_size={patch_size}"
        )
    latents = latents.view(
        batch, channels, height // patch_size, patch_size, width // patch_size, patch_size
    )
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(
        batch, channels * patch_size * patch_size, height // patch_size, width // patch_size
    )


def unpatchify(latents: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    """Inverse of :func:`patchify`."""
    batch, channels, height, width = latents.shape
    folded = channels // (patch_size * patch_size)
    latents = latents.reshape(batch, folded, patch_size, patch_size, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, folded, height * patch_size, width * patch_size)


class Krea2VAEEncoder:
    """Frozen ``AutoencoderKLQwenImage``, owned by the training run rather than by a pipeline."""

    def __init__(
        self, model: nn.Module, config: VAEConfig, *, device: torch.device, patch_size: int = 2
    ) -> None:
        self.config = config
        self.device = device
        self.patch_size = patch_size
        self.encode_dtype = _DTYPES[config.encode_dtype]
        self.model = model.eval().requires_grad_(False)
        if config.placement == "device":
            self.model.to(device=device)
        if config.tiling:
            self.model.enable_tiling()
        if config.slicing:
            self.model.enable_slicing()

    @classmethod
    def load(
        cls,
        config: VAEConfig,
        *,
        path: str,
        device: torch.device,
        patch_size: int = 2,
        revision: str | None = None,
    ) -> Krea2VAEEncoder | None:
        """Load the VAE, or ``None`` when the dataset already provides latents."""
        if not config.enabled:
            return None
        from diffusers import AutoencoderKLQwenImage

        source = config.path or path
        # ``torch_dtype``, not ``dtype``: ``ModelMixin.from_pretrained`` forwards unrecognised
        # kwargs into the model's ``__init__``, so ``dtype=`` reaches ``AutoencoderKLQwenImage``
        # as an unexpected argument rather than being read as a precision request. Same spelling
        # as ``vae.py`` uses for FLUX.2.
        model = AutoencoderKLQwenImage.from_pretrained(
            localize(source, subfolder=config.subfolder),
            revision=revision,
            torch_dtype=_DTYPES[config.encode_dtype],
        )
        return cls(model, config, device=device, patch_size=patch_size)

    @property
    def z_dim(self) -> int:
        return int(self.model.config.z_dim)

    @property
    def spatial_compression(self) -> int:
        """Pixels per token side: the VAE's own downsampling times the packing patch size.

        ``vae_scale_factor = 2 ** len(temperal_downsample)`` in the pipeline — a *temporal* field
        used for the spatial factor, which reads like a bug and is not: the block count happens to
        be the same. Read the same way here so the two cannot disagree.
        """
        return 2 ** len(self.model.config.temperal_downsample) * self.patch_size

    @property
    def latent_channels(self) -> int:
        """Channels *after* packing — what the transformer's ``in_channels`` must equal."""
        return self.z_dim * self.patch_size * self.patch_size

    def _statistics(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, std)`` shaped to broadcast over ``(B, z_dim, F, H, W)``."""
        config = self.model.config
        view = (1, config.z_dim, 1, 1, 1)
        mean = torch.tensor(config.latents_mean, device=self.device, dtype=dtype).view(view)
        std = torch.tensor(config.latents_std, device=self.device, dtype=dtype).view(view)
        return mean, std

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        *,
        mode: SampleMode = "sample",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``(B, 3, H, W)`` in [-1, 1] -> ``(B, z_dim * p * p, H/f, W/f)``.

        ``mode`` selects the posterior: ``"sample"`` draws from it, ``"mode"`` takes its mean.
        Which to use is a *training* decision, so it is an argument here and chosen at the call site
        in ``experiments/`` rather than baked in.
        """
        if images.ndim != 4:
            raise ValueError(f"images must be (B, 3, H, W), got {tuple(images.shape)}")

        # The *model's* dtype, not the config's, matching ``vae.py``. ``load()`` builds it at
        # ``encode_dtype``, but a caller constructing this wrapper around an already-loaded VAE
        # would otherwise feed bf16 activations into fp32 convolutions — which raises deep inside
        # ``F.conv3d`` with a message about bias types, far from the cause.
        images = images.to(device=self.model.device, dtype=self.model.dtype)
        # AutoencoderKLQwenImage is a video VAE: it wants a frame axis.
        posterior = self.model.encode(images.unsqueeze(2)).latent_dist
        latents = posterior.sample(generator=generator) if mode == "sample" else posterior.mode()

        mean, std = self._statistics(latents.dtype)
        latents = (latents - mean) / std

        # Drop the frame axis only after normalising: the statistics are shaped for 5-D.
        return patchify(latents[:, :, 0], patch_size=self.patch_size)

    @torch.no_grad()
    def denormalise(self, latents: torch.Tensor) -> torch.Tensor:
        """``(B, z_dim * p * p, h, w)`` -> the 5-D latents ``vae.decode`` expects.

        The exact arithmetic the pipeline performs before decoding, so a sampled latent from
        training can be handed straight to ``vae.decode`` for a sanity render.
        """
        unpacked = unpatchify(latents, patch_size=self.patch_size).unsqueeze(2)
        mean, std = self._statistics(unpacked.dtype)
        return unpacked * std + mean


    @torch.no_grad()
    def decode_to_pil(self, tokens: torch.Tensor, *, height: int, width: int):
        """Packed target tokens -> a PIL image, for previews and sanity renders.

        ``height``/``width`` are the *latent grid* extents the tokens came from, so this is the
        inverse of ``encode`` followed by the task's ``pack``. Decoding is the pipeline's own call,
        including its frame axis and the ``[:, :, 0]`` that drops it.
        """
        from PIL import Image

        batch, sequence, channels = tokens.shape
        if sequence != height * width:
            raise ValueError(f"{sequence} tokens cannot fill a {height}x{width} latent grid")
        grid = tokens.permute(0, 2, 1).reshape(batch, channels, height, width)

        latents = self.denormalise(grid.to(self.model.dtype))
        image = self.model.decode(latents, return_dict=False)[0][:, :, 0]
        image = (image.float() / 2 + 0.5).clamp(0, 1)
        array = (image[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
        return Image.fromarray(array)


__all__ = ["Krea2VAEEncoder", "patchify", "unpatchify"]
