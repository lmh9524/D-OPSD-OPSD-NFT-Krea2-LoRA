"""Multi-reference dataset, read from a JSONL manifest.

The reference *slots* are fixed in number, so the slot -> RoPE-offset mapping is stable. Samples with
more references contribute a subset **in manifest order**; samples with fewer are cycled. Both keep
every sample usable, which matters because reference sets are expensive to curate and discarding the
short ones would throw away real data.

Order is load-bearing, and this is not obvious. References never reach the text encoder -- only the
VAE -- so the model has no semantic description of them. The only thing distinguishing reference *i*
is its offset on the T axis of the position ids. Any ability to follow "the character from image 1"
therefore comes from the DiT having learned to bind ordinal language to those offsets, which a shuffled
slot order destroys without raising anything.

Shapes vary per image, because aspect ratio is preserved (see ``transforms.py``). So ``references`` is
a **list** of per-slot tensors rather than one stacked tensor: two references in the same sample can
have different extents, and ``collate`` stacks per slot across the batch.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from dflow.config.dataset import Ref2ImgConfig
from dflow.tasks.ref2img.transforms import MULTIPLE_OF, augment_identity, load_image


@dataclass(frozen=True, slots=True)
class Sample:
    target: str
    references: tuple[str, ...]
    prompt: str


def read_manifest(path: Path) -> list[Sample]:
    """Parse the JSONL manifest, reporting the offending line rather than a bare KeyError."""
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                samples.append(
                    Sample(
                        target=record["target"],
                        references=tuple(record.get("refs", ())),
                        prompt=record.get("prompt", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not samples:
        raise ValueError(f"{path} contains no samples")
    return samples


def choose_references(
    available: tuple[str, ...],
    *,
    slots: int,
    dropout: float,
    rng: random.Random,
    pad_short: bool = True,
    pinned: int = 0,
) -> list[str]:
    """Fill ``slots`` reference slots from ``available``, **preserving manifest order**.

    More than needed: a random subset, sorted back into manifest order because slot position carries
    meaning (see the module docstring). Sampling indices and sorting them, rather than ``rng.sample``
    over the paths, is what keeps that true. Note the subset is **fixed per sample**, not per epoch —
    ``Ref2ImgDataset`` seeds this from the index alone so a resumed run replays the same data, which
    means a reference dropped here is never seen at all. Prefer ``pad_short=False`` with a slot count
    that covers the dataset over relying on resampling to reach the tail.

    Fewer than needed: cycled when ``pad_short``, so a one-reference sample still fills every slot.
    With ``pad_short=False`` the sample simply contributes what it has and ``slots`` becomes an
    upper bound — the sequence is shorter, which costs nothing because reference count is only a
    sequence-length change. That is the better default at ``batch_size=1``: on Garments2Look a
    fixed nine slots would make **37% of all reference slots duplicates**, paying attention over
    copies and teaching the model that repeats are normal when inference never sends any.

    ``pinned`` protects the leading slots from subsampling. The identity reference lives at slot 0
    and must never be the one dropped when a sample has more garments than slots.

    ``dropout`` replaces a slot with a repeat of the preceding one. It cannot blank a slot: the DiT
    takes no attention mask, so an "empty" reference would still be attended to as noise. Repeating
    perturbs which image sits in which slot, so leave it at 0 when captions use ordinal language.
    """
    if not available:
        raise ValueError("a ref2img sample needs at least one reference")

    pinned = min(pinned, slots, len(available))
    head, tail = list(available[:pinned]), available[pinned:]
    budget = slots - pinned

    if len(tail) >= budget:
        indices = sorted(rng.sample(range(len(tail)), budget))
        chosen = head + [tail[index] for index in indices]
    elif not pad_short:
        chosen = head + list(tail)
    else:
        # Cycle whatever is available — the tail if there is one, otherwise the pinned head itself,
        # which is the degenerate case of a sample whose only reference is its identity crop.
        source = tail or tuple(head)
        chosen = head + [source[index % len(source)] for index in range(budget)]

    if dropout > 0.0:
        for slot in range(max(1, pinned), len(chosen)):
            if rng.random() < dropout:
                chosen[slot] = chosen[slot - 1]
    return chosen


class Ref2ImgDataset(Dataset):
    """Emits :class:`dflow.tasks.ref2img.schema.Ref2ImgBatch` before collation."""

    def __init__(self, config: Ref2ImgConfig) -> None:
        self.config = config
        self.root = Path(config.root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root is not a directory: {self.root}")

        manifest = self.root / config.manifest
        if not manifest.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        self.samples = read_manifest(manifest)

        for name, area in (
            ("target_max_area", config.target_max_area),
            ("reference_max_area", config.reference_max_area),
        ):
            if area < MULTIPLE_OF**2:
                raise ValueError(
                    f"{name} must be at least {MULTIPLE_OF**2} (one latent token), got {area}"
                )
        if config.num_references < 1:
            raise ValueError("num_references must be >= 1; use the t2i task for no references")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        # Seeded per index so an epoch is reproducible while different epochs still vary: the
        # sampler's permutation changes which index maps to which step, not this choice.
        rng = random.Random(index)
        references = choose_references(
            sample.references,
            slots=self.config.num_references,
            dropout=self.config.reference_dropout,
            rng=rng,
            pad_short=self.config.pad_short_samples,
            pinned=self.config.pinned_references,
        )
        pinned = min(self.config.pinned_references, len(references))
        # Drawn from an *unseeded* generator, unlike the selection above, and this is the one place
        # the dataset is deliberately not replayable. Identity augmentation is noise injection
        # rather than data selection: re-drawing it every epoch is the whole point, since a jitter
        # fixed per index would agree with the target just as exactly as no jitter at all, merely
        # offset. Resume exactness covers *which samples in which order*, which is unaffected.
        noise = random.Random()

        def reference_augment(slot: int):
            if not self.config.augment_identity or slot >= pinned:
                return None
            return lambda image: augment_identity(image, rng=noise)

        target = load_image(self.root / sample.target, max_area=self.config.target_max_area)
        # References are sized against the *target's* extent when configured, not an area cap, so
        # they land at a comparable grid. See ``load_image``'s ``fit_inside``.
        fit_inside = tuple(target.shape[-2:]) if self.config.reference_fit_target else None
        return {
            "target": target,
            # A list, not a stacked tensor: references preserve their own aspect ratios, so two
            # slots in one sample can differ in extent.
            "references": [
                load_image(
                    self.root / path,
                    max_area=self.config.reference_max_area,
                    augment=reference_augment(slot),
                    fit_inside=fit_inside,
                )
                for slot, path in enumerate(references)
            ],
            "prompt": sample.prompt,
        }


__all__ = ["Ref2ImgDataset", "Sample", "choose_references", "read_manifest"]
