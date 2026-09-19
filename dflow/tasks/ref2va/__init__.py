"""MiniMax-H3 `ref2va`: references (images, videos, audio) -> video + audio.

Only the packed-sequence layout lives here so far. The transformer is
`dflow.vendor.minimax_h3`; the family adapter is `dflow.models.family.minimax_h3`.
"""

from dflow.tasks.ref2va.layout import (
    AudioReference,
    ImageReference,
    PackedLayout,
    Reference,
    VideoReference,
    build_ref2va_layout,
    video_t_span,
)

__all__ = [
    "AudioReference",
    "ImageReference",
    "PackedLayout",
    "Reference",
    "VideoReference",
    "build_ref2va_layout",
    "video_t_span",
]
