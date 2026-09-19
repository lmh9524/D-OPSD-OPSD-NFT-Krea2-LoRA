"""Multi-image-reference generation.

Conditioning is by **sequence concatenation** with 4-D position ids, so no weight surgery is needed
and plain LoRA works. See ``conditioning.py`` for why the id layout is the part that must be exact.
"""

from dflow.tasks.ref2img.conditioning import (
    T_SCALE,
    ReferenceSequence,
    build_reference_ids,
    build_sequence,
    pack,
    unpack,
)
from dflow.tasks.ref2img.dataset import Ref2ImgDataset, Sample, choose_references, read_manifest
from dflow.tasks.ref2img.schema import Ref2ImgBatch, validate
from dflow.tasks.ref2img.transforms import load_image, token_count

__all__ = [
    "T_SCALE",
    "Ref2ImgBatch",
    "Ref2ImgDataset",
    "ReferenceSequence",
    "Sample",
    "build_reference_ids",
    "build_sequence",
    "choose_references",
    "load_image",
    "pack",
    "read_manifest",
    "token_count",
    "unpack",
    "validate",
]
