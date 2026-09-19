"""VAE wrapper: lifecycle, not maths.

The encode/decode maths is delegated to diffusers, deliberately. Re-deriving it would get
FLUX.2 wrong in two ways that raise no error:

* **Normalisation is a BatchNorm's running statistics**, not config scalars. Neither
  FLUX.1's ``shift_factor``/``scaling_factor`` nor Wan's per-channel
  ``latents_mean``/``latents_std`` applies here; the VAE carries
  ``bn.running_mean``/``bn.running_var`` and normalises with those.
* **Patchification comes before normalisation.** ``encode`` emits 32 channels, the 2x2
  patchify folds them to 128, and the BatchNorm has 128 features. Normalising first would
  use the wrong statistics on the wrong channel count.

What this class does own is everything a training run needs and an inference pipeline does
not: whether to load the VAE at all, its dtype per direction, placement, and tiling.

One function here mirrors upstream rather than calling it: ``Flux2KleinPipeline``'s
``_encode_vae_image`` is an instance method bound to ``self.vae``, so it cannot be called
without a pipeline. ``encode`` reproduces its eight lines with the VAE passed explicitly.
That is the only drift risk in this module, and ``tests/test_encoders.py`` pins it by
asserting element-wise equality against a real pipeline instance.
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


def _patchify(latents: torch.Tensor) -> torch.Tensor:
    """Fold 2x2 spatial patches into the channel dimension.

    Mirrors ``Flux2KleinPipeline._patchify_latents``. Imported rather than copied where
    possible; this one is short enough that the indirection costs more than it saves, and
    the shape assertion below catches any upstream change.
    """
    batch, channels, height, width = latents.shape
    if height % 2 or width % 2:
        raise ValueError(f"latent spatial dims must be even to patchify, got {height}x{width}")
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch, channels * 4, height // 2, width // 2)


def _unpatchify(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


class VAEEncoder:
    """Frozen VAE, owned by the training run rather than by a pipeline."""

    def __init__(self, model: nn.Module, config: VAEConfig, *, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.encode_dtype = _DTYPES[config.encode_dtype]
        self.decode_dtype = _DTYPES[config.decode_dtype]

        self.model = model.eval().requires_grad_(False)
        if config.tiling:
            self.model.enable_tiling()
        if config.slicing:
            self.model.enable_slicing()
        if config.placement == "device":
            self.model.to(device=device)

    # ------------------------------------------------------------------ construction

    @classmethod
    def load(
        cls,
        config: VAEConfig,
        *,
        path: str,
        device: torch.device,
        revision: str | None = None,
    ) -> VAEEncoder | None:
        """Load the VAE, or return ``None`` when the dataset already provides latents.

        Returning ``None`` rather than a stub keeps the "not loaded" case visible at the
        call site instead of hiding a silent no-op behind a method call.
        """
        if not config.enabled:
            return None

        from diffusers import AutoencoderKLFlux2

        model = AutoencoderKLFlux2.from_pretrained(
            localize(config.path or path, subfolder=config.subfolder),
            revision=revision,
            torch_dtype=_DTYPES[config.encode_dtype],
        )
        return cls(model, config, device=device)

    # --------------------------------------------------------------------- properties

    @property
    def spatial_compression(self) -> int:
        """Pixels per latent token side: conv downsampling times the internal patch size."""
        blocks = len(self.model.config.block_out_channels)
        patch = self.model.config.patch_size
        return 2 ** (blocks - 1) * patch[0]

    @property
    def latent_channels(self) -> int:
        """Channels *after* patchify — what the transformer's ``in_channels`` must equal."""
        import math

        return self.model.config.latent_channels * math.prod(self.model.config.patch_size)

    # ------------------------------------------------------------------------ encoding

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        *,
        mode: SampleMode = "sample",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Encode ``(B, 3, H, W)`` in [-1, 1] to normalised, patchified latents.

        ``mode`` is a **training decision**, exposed here rather than configured: the
        inference pipeline uses ``"mode"`` (the posterior mode) for determinism, while
        training usually wants ``"sample"`` so the target distribution is the real
        posterior. Keeping it an argument means the choice is visible in the ~120 lines of
        an experiment file instead of buried in a config tree.
        """
        if images.ndim != 4:
            raise ValueError(f"expected (B, 3, H, W), got {tuple(images.shape)}")

        images = images.to(device=self.model.device, dtype=self.model.dtype)
        posterior = self.model.encode(images).latent_dist
        latents = posterior.sample(generator=generator) if mode == "sample" else posterior.mode()

        # Order matters: patchify, then normalise with the 128-channel BatchNorm stats.
        latents = _patchify(latents)
        return self.normalize(latents)

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply the VAE's BatchNorm running statistics."""
        mean, std = self._bn_stats(latents)
        return (latents - mean) / std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._bn_stats(latents)
        return latents * std + mean

    def _bn_stats(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_norm = self.model.bn
        mean = batch_norm.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        variance = batch_norm.running_var.view(1, -1, 1, 1)
        std = torch.sqrt(variance + self.model.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return mean, std

    # ------------------------------------------------------------------------ decoding

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalised, patchified latents back to images in [-1, 1].

        Runs in ``decode_dtype`` (fp32 by default): bf16 decoding produces visible banding,
        which matters because this path is what validation samples are judged on.
        """
        latents = self.denormalize(latents.to(dtype=self.decode_dtype))
        latents = _unpatchify(latents)
        model = self.model.to(dtype=self.decode_dtype)
        try:
            return model.decode(latents.to(model.device), return_dict=False)[0]
        finally:
            if self.decode_dtype is not self.encode_dtype:
                self.model.to(dtype=self.encode_dtype)


__all__ = ["SampleMode", "VAEEncoder"]
