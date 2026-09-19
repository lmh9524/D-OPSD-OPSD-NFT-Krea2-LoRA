"""Image loading for multi-reference samples — matching inference exactly.

Geometry is **delegated to** ``Flux2ImageProcessor``, the same object the pipeline uses, rather than
reimplemented. The steps are the pipeline's, from ``pipeline_flux2_klein.py:770-780``:

1. scale down only if the area exceeds the cap, by ``sqrt(cap / area)`` — so the aspect ratio is
   exact, not approximated;
2. floor each side to a multiple of 16 (the VAE's effective compression);
3. centre-crop the ≤15px remainder;
4. normalise to [-1, 1].

An earlier version of this module centre-cropped to a fixed square. That is wrong in two ways at
once, and neither raises: it discards composition (44% of a 16:9 frame), and it trains on a shape
distribution inference never produces, so the H/W position ids span a different range than the model
will see. Aspect ratio is preserved here for the same reason encoders delegate their maths — the
dominant risk is training diverging from inference.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

#: The VAE's effective spatial compression: four conv down blocks (8x) times its internal 2x2 patch.
#: A side that is not a multiple of this cannot be patchified evenly.
MULTIPLE_OF = 16


@functools.cache
def _processor():
    """The pipeline's own image processor. Cached: constructing it re-reads a config."""
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

    return Flux2ImageProcessor(vae_scale_factor=MULTIPLE_OF)


def augment_identity(image, *, rng):
    """Break pixel-identity between an identity reference and the target it supervises.

    A person reference is cropped **out of the target**, so without this the two agree exactly —
    same lighting, same angle, same JPEG artefacts. The loss can then be satisfied by matching
    pixels rather than by learning who the person is, and the model collapses the moment inference
    hands it a separate photograph.

    Each transform removes one channel of that agreement:

    ============  ==========================================================
    crop + scale  the reference no longer aligns spatially with the target
    flip          a mirrored face is the same person from another view
    rotation      breaks exact orientation agreement
    colour        removes shared white balance and exposure
    JPEG          removes shared compression artefacts
    ============  ==========================================================

    Deliberately absent: anything that changes *identity*. Heavy hue rotation would recolour skin
    and hair; strong blur would erase the features the reference exists to carry.
    """
    import io

    from PIL import Image, ImageEnhance

    width, height = image.size
    scale = rng.uniform(0.82, 1.0)
    box_w, box_h = int(width * scale), int(height * scale)
    left = rng.randint(0, max(0, width - box_w))
    top = rng.randint(0, max(0, height - box_h))
    image = image.crop((left, top, left + box_w, top + box_h))

    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    angle = rng.uniform(-7.0, 7.0)
    if abs(angle) > 0.5:
        image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

    for enhancer, low, high in (
        (ImageEnhance.Brightness, 0.9, 1.1),
        (ImageEnhance.Contrast, 0.9, 1.1),
        (ImageEnhance.Color, 0.85, 1.15),
        (ImageEnhance.Sharpness, 0.7, 1.3),
    ):
        image = enhancer(image).enhance(rng.uniform(low, high))

    if rng.random() < 0.7:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(55, 92))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


def load_image(
    path: Path | str,
    *,
    max_area: int,
    augment=None,
    fit_inside: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Load one image as ``(3, H, W)`` in [-1, 1], aspect ratio preserved.

    ``H`` and ``W`` are multiples of 16 and ``H * W <= max_area`` (up to the flooring), so the shape
    varies per image — which is exactly what the model is trained and served with.

    ``fit_inside=(height, width)`` resizes to fit **inside** those pixel dimensions instead of to an
    area cap, preserving aspect ratio. That is how the working Krea 2 edit recipe sizes a reference:
    it lands at roughly the target's own grid, so a reference token at (h, w) is comparable in scale
    to the target token at (h, w) and the shared coordinate actually means something.

    An area cap sized for many references does the opposite. At 192x192 a reference is a 12x12 latent
    against a 42x24 target — a seventh of the tokens, a third of the linear resolution. Garment
    pattern does not survive that, and centring the small grid inside the large one leaves the
    spatial correspondence nominal.
    """
    from PIL import Image

    processor = _processor()
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if augment is not None:
            # Before the area cap, so a crop-jittered reference is still resized to the same budget.
            image = augment(image)
        width, height = image.size
        if fit_inside is not None:
            limit_h, limit_w = fit_inside
            scale = min(limit_h / height, limit_w / width)
            if scale < 1.0:
                width, height = max(1, round(width * scale)), max(1, round(height * scale))
                image = image.resize((width, height), Image.LANCZOS)
        elif width * height > max_area:
            image = processor._resize_to_target_area(image, max_area)
            width, height = image.size

        width = (width // MULTIPLE_OF) * MULTIPLE_OF
        height = (height // MULTIPLE_OF) * MULTIPLE_OF
        if width == 0 or height == 0:
            raise ValueError(
                f"{path} is {handle.size[0]}x{handle.size[1]}, smaller than {MULTIPLE_OF}px on a "
                f"side after flooring; inference rejects images under 64px too"
            )
        tensor = processor.preprocess(image, height=height, width=width, resize_mode="crop")

    return tensor.squeeze(0)  # preprocess returns (1, 3, H, W)


def token_count(latents: torch.Tensor) -> int:
    """Tokens a patchified latent contributes: its spatial extent.

    ``VAEEncoder.encode`` already patchifies, so ``(B, 128, h, w)`` packs to ``h * w`` tokens.
    """
    return latents.shape[-2] * latents.shape[-1]


__all__ = ["MULTIPLE_OF", "augment_identity", "load_image", "token_count"]
