"""Images with sidecar caption files — the simplest layout that trains something real.

    dataset/
      001.png
      001.txt        <- caption; empty or missing means an empty prompt
      002.jpg
      002.txt

Fixed resolution for now: centre-crop to a square after a short-side resize. Aspect-ratio
bucketing belongs in ``data/bucket.py`` and lands with the multi-reference task, where variable
sequence length is unavoidable anyway.

Retry is shared machinery (``data/retry.py``) rather than a loop here, because in a bucketed world
a naive "resample anything" retry can cross buckets and break collation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from dflow.config.dataset import ImageFolderConfig

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


class ImageFolderDataset(Dataset):
    """Emits ``{"image": (3, H, W) in [-1, 1], "prompt": str}``."""

    def __init__(self, config: ImageFolderConfig) -> None:
        self.config = config
        root = Path(config.root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root is not a directory: {root}")

        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise ValueError(
                f"no images under {root} (looked for {sorted(IMAGE_SUFFIXES)})"
            )
        if config.resolution % 16:
            # 16 is the VAE's effective compression; a non-multiple cannot be patchified evenly.
            raise ValueError(f"resolution must be a multiple of 16, got {config.resolution}")

    def __len__(self) -> int:
        return len(self.paths)

    def caption_for(self, path: Path) -> str:
        sidecar = path.with_suffix(self.config.caption_suffix)
        if not sidecar.is_file():
            return ""
        return sidecar.read_text(encoding="utf-8").strip()

    def __getitem__(self, index: int) -> dict[str, object]:
        from PIL import Image

        path = self.paths[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            tensor = _to_square_tensor(image, self.config.resolution)
        return {"image": tensor, "prompt": self.caption_for(path)}


def _to_square_tensor(image, resolution: int) -> torch.Tensor:
    """Short-side resize then centre crop, scaled to [-1, 1]."""
    from PIL import Image

    width, height = image.size
    scale = resolution / min(width, height)
    resized = image.resize(
        (max(resolution, round(width * scale)), max(resolution, round(height * scale))),
        Image.BICUBIC,
    )
    width, height = resized.size
    left = (width - resolution) // 2
    top = (height - resolution) // 2
    cropped = resized.crop((left, top, left + resolution, top + resolution))

    array = torch.frombuffer(cropped.tobytes(), dtype=torch.uint8).clone()
    tensor = array.view(resolution, resolution, 3).permute(2, 0, 1).float()
    return tensor.div_(127.5).sub_(1.0)


__all__ = ["IMAGE_SUFFIXES", "ImageFolderConfig", "ImageFolderDataset"]
