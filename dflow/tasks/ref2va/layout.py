"""The packed-sequence layout a MiniMax-H3 `ref2va` forward requires.

`MiniMaxH3Transformer3DModel.forward` does not build a layout: it takes already-patchified rows plus
the index tensors that say where each modality sits in one packed 1-D sequence. This module is that
builder, and it is the reason `MiniMaxH3Family.prepare_inputs` raises — the protocol's image-shaped
arguments cannot describe it.

Layout, in row order::

    [ text | ref_0 | ref_1 | ... | target_audio | target_video ]

Every reference block is contiguous, and a video reference puts its **audio rows before its visual
rows**. The order is semantic: it fixes the `"<Picture i>"` / `"<Audio j>"` labels the prompt uses
*and* it advances the shared rotary clock, so a different order is a different request.

### The clock

`t` is one continuous axis shared by text, video and audio. It starts at 0 for text, and the media
clock starts at `text_len` — **so the length of the prompt shifts every media coordinate**. Each
block then advances a cursor by its own temporal extent:

===============  =================================================
image reference  ``1.0`` — a single frame
video reference  ``max(audio_t, video_t_span(latent_t))``
audio reference  ``audio_t``
===============  =================================================

`video_t_span` is **not** `latent_t`. A causal video VAE does not compress uniformly: the first
latent frame of each group of `T_GROUP` encodes one real frame and the rest encode `4`, so token
*k* spans ``FRAME_RESCALE * FRAME_PER_TOKEN[k % T_GROUP]``. Treating the axis as uniform puts every
reference after the first at the wrong time.

### The spatial grid

`h` / `w` are **not** row and column indices. Each visual block is mapped onto a normalised grid
centred on 0 and scaled by `INTERP`, derived from its *own* `sqrt(h * w)` — so a 480x832 reference
and a 832x480 one land on comparably sized grids, and aspect ratio is carried by the grid's extent
rather than by raw index counts. A reference uses its own area, never the target's.

Audio rows are `audio_t * audio_channels` and carry the channel on the **`w` axis**: the first
`audio_t` rows sit at the block's leftmost `w` and the rest at its rightmost, which is what keeps
two channels apart for a model that has no channel axis in its position ids.

### No padding

The reference implementation pads the sequence to a multiple of 64 for FlashAttention and splits
the tail off with `cu_seqlens = [0, used, S]`. The diffusers port this repo vendors has no use for
that — it runs unmasked over one document — so nothing here pads. Under context parallelism the
sequence length must instead be divisible by the region size; see the family module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: Modality tags `forward` expects on every row.
TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2

#: The causal video VAE's compression pattern: within each group of `T_GROUP` latent frames the
#: first encodes one real frame and the rest encode four.
T_GROUP = 5
FRAME_PER_TOKEN: tuple[int, ...] = (1, 4, 4, 4, 4)
#: Rescales a frame count onto the rotary clock.
FRAME_RESCALE = 5.0 / 3.0
#: Half-extent of the normalised spatial grid.
INTERP = 32
#: The transformer's spatial patch. `patch_size` is `(1, 2, 2)`, so a latent `h x w` becomes
#: `(h // 2) * (w // 2)` rows.
PATCH = 2


@dataclass(frozen=True, slots=True)
class ImageReference:
    """A still image reference. Occupies one frame on the clock."""

    latent_h: int
    latent_w: int


@dataclass(frozen=True, slots=True)
class VideoReference:
    """A video reference, optionally with its soundtrack (`audio_t == 0` means silent)."""

    latent_t: int
    latent_h: int
    latent_w: int
    audio_t: int = 0


@dataclass(frozen=True, slots=True)
class AudioReference:
    """A bare audio clip."""

    audio_t: int


Reference = ImageReference | VideoReference | AudioReference


@dataclass(frozen=True, slots=True)
class PackedLayout:
    """Everything `forward` needs that is not a latent.

    ``position_ids``/``token_tags`` describe the whole sequence; the three index tensors give, for
    each modality, the rows it occupies **in the order its latents must be supplied**. The target
    slices are exposed so a task can scatter its own noised latents and build a timestep map.
    """

    position_ids: torch.Tensor  # (seq_len, 3) float32
    token_tags: torch.Tensor  # (seq_len,) long
    video_indices: torch.Tensor  # (num_video_rows,) long
    audio_indices: torch.Tensor  # (num_audio_rows,) long
    text_indices: torch.Tensor  # (num_text_rows,) long
    seq_len: int
    target_video: slice
    target_audio: slice

    def timestep_indices(
        self, *, conditioning: int = 0, video: int = 1, audio: int = 2
    ) -> torch.Tensor:
        """Per-row index into the `timestep` vector `forward` takes.

        H3 serves several noise levels in one call: the target video and target audio are noised
        while the text and every reference row are clean. The defaults therefore describe a
        `timestep` of ``[0.0, sigma_video, sigma_audio]``. Pass equal ``video``/``audio`` for a
        single shared level.
        """
        indices = torch.full((self.seq_len,), conditioning, dtype=torch.long)
        indices[self.target_video] = video
        indices[self.target_audio] = audio
        return indices


def _axis(dim: int, sqrt_area: float) -> torch.Tensor:
    """One normalised spatial axis of `dim // PATCH` coordinates, centred on 0."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) * 0.5
    count = dim // PATCH
    step = ratio / count
    return (left + step * torch.arange(count, dtype=torch.float64)) * INTERP


def _frame_grid(latent_h: int, latent_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`((h//2)*(w//2), 2)` of `(h, w)` coordinates for one frame, plus the `w` axis itself."""
    sqrt_area = math.sqrt(latent_h * latent_w)
    h_axis, w_axis = _axis(latent_h, sqrt_area), _axis(latent_w, sqrt_area)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_axis


def video_t_span(latent_t: int) -> float:
    """How far `latent_t` latent frames advance the clock, honouring the causal VAE's pattern."""
    return sum(FRAME_RESCALE * FRAME_PER_TOKEN[k % T_GROUP] for k in range(latent_t))


def _video_t_grid(latent_t: int, origin: float) -> torch.Tensor:
    """`t` of each latent frame: the cumulative span of the frames before it, from ``origin``."""
    spans = torch.tensor(
        [FRAME_RESCALE * FRAME_PER_TOKEN[k % T_GROUP] for k in range(latent_t)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _video_grid(latent_t: int, frame: torch.Tensor, origin: float) -> torch.Tensor:
    """`(latent_t * frame_rows, 3)` — the frame grid repeated at each frame's `t`."""
    grid = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
    grid[:, :, 0] = _video_t_grid(latent_t, origin)[:, None]
    grid[:, :, 1:] = frame[None]
    return grid.reshape(-1, 3)


def _audio_w(w_axis: torch.Tensor, audio_t: int, channels: int) -> torch.Tensor:
    """Channel separation on the `w` axis: channel 0 leftmost, the rest rightmost."""
    return torch.cat(
        [
            torch.full((audio_t,), float(w_axis[0]), dtype=torch.float64),
            torch.full((audio_t * (channels - 1),), float(w_axis[-1]), dtype=torch.float64),
        ]
    )


def build_ref2va_layout(
    *,
    text_len: int,
    target_latent_t: int,
    target_latent_h: int,
    target_latent_w: int,
    target_audio_t: int,
    references: list[Reference] | None = None,
    audio_channels: int = 2,
) -> PackedLayout:
    """Build the packed layout for one `ref2va` request.

    Sizes are in **latent** units; the transformer's `(1, 2, 2)` patch is applied here, so a latent
    `h x w` contributes `(h // 2) * (w // 2)` rows per frame.
    """
    if text_len < 0 or target_latent_t < 0 or target_audio_t < 0:
        raise ValueError("text_len, target_latent_t and target_audio_t must be non-negative")
    for name, value in (("target_latent_h", target_latent_h), ("target_latent_w", target_latent_w)):
        if value % PATCH:
            raise ValueError(f"{name}={value} must be a multiple of the spatial patch {PATCH}")
    if audio_channels < 1:
        raise ValueError("audio_channels must be at least 1")
    references = list(references or ())

    target_frame_rows = (target_latent_h // PATCH) * (target_latent_w // PATCH)
    target_video_rows = target_latent_t * target_frame_rows
    target_audio_rows = target_audio_t * audio_channels

    reference_rows = 0
    for ref in references:
        if isinstance(ref, ImageReference):
            reference_rows += (ref.latent_h // PATCH) * (ref.latent_w // PATCH)
        elif isinstance(ref, VideoReference):
            reference_rows += ref.latent_t * (ref.latent_h // PATCH) * (ref.latent_w // PATCH)
            reference_rows += ref.audio_t * audio_channels
        elif isinstance(ref, AudioReference):
            reference_rows += ref.audio_t * audio_channels
        else:
            raise TypeError(f"unknown reference type: {type(ref).__name__}")

    seq_len = text_len + reference_rows + target_audio_rows + target_video_rows
    ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    tags = torch.full((seq_len,), TAG_TEXT, dtype=torch.long)

    # Text occupies the first rows and starts the clock; h and w stay at 0.
    ids[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)

    video_parts: list[torch.Tensor] = []
    audio_parts: list[torch.Tensor] = []
    cursor = text_len
    clock = float(text_len)

    for ref in references:
        if isinstance(ref, ImageReference):
            frame, _ = _frame_grid(ref.latent_h, ref.latent_w)
            span = slice(cursor, cursor + frame.shape[0])
            ids[span, 0] = clock
            ids[span, 1:] = frame
            tags[span] = TAG_VIDEO
            video_parts.append(torch.arange(span.start, span.stop))
            cursor, clock = span.stop, clock + 1.0

        elif isinstance(ref, VideoReference):
            frame, w_axis = _frame_grid(ref.latent_h, ref.latent_w)
            audio_rows = ref.audio_t * audio_channels
            audio_span = slice(cursor, cursor + audio_rows)
            visual_span = slice(audio_span.stop, audio_span.stop + ref.latent_t * frame.shape[0])

            if ref.audio_t:
                ids[audio_span, 0] = (
                    clock + torch.arange(ref.audio_t, dtype=torch.float64)
                ).repeat(audio_channels)
                # a video reference's audio is separated on **its own** w axis, not the target's
                ids[audio_span, 2] = _audio_w(w_axis, ref.audio_t, audio_channels)
                tags[audio_span] = TAG_AUDIO
                audio_parts.append(torch.arange(audio_span.start, audio_span.stop))

            ids[visual_span] = _video_grid(ref.latent_t, frame, clock)
            tags[visual_span] = TAG_VIDEO
            video_parts.append(torch.arange(visual_span.start, visual_span.stop))

            cursor = visual_span.stop
            clock += max(float(ref.audio_t), video_t_span(ref.latent_t))

        else:  # AudioReference
            audio_rows = ref.audio_t * audio_channels
            span = slice(cursor, cursor + audio_rows)
            if ref.audio_t:
                _, target_w = _frame_grid(target_latent_h, target_latent_w)
                ids[span, 0] = (clock + torch.arange(ref.audio_t, dtype=torch.float64)).repeat(
                    audio_channels
                )
                # a bare audio clip has no grid of its own, so it borrows the target's
                ids[span, 2] = _audio_w(target_w, ref.audio_t, audio_channels)
                tags[span] = TAG_AUDIO
                audio_parts.append(torch.arange(span.start, span.stop))
            cursor = span.stop
            clock += float(ref.audio_t)

    target_audio = slice(cursor, cursor + target_audio_rows)
    target_video = slice(target_audio.stop, target_audio.stop + target_video_rows)

    frame, w_axis = _frame_grid(target_latent_h, target_latent_w)
    ids[target_video] = _video_grid(target_latent_t, frame, clock)
    tags[target_video] = TAG_VIDEO
    if target_audio_t:
        ids[target_audio, 0] = (clock + torch.arange(target_audio_t, dtype=torch.float64)).repeat(
            audio_channels
        )
        ids[target_audio, 2] = _audio_w(w_axis, target_audio_t, audio_channels)
        tags[target_audio] = TAG_AUDIO

    video_parts.append(torch.arange(target_video.start, target_video.stop))
    audio_parts.append(torch.arange(target_audio.start, target_audio.stop))

    return PackedLayout(
        position_ids=ids.to(torch.float32),
        token_tags=tags,
        video_indices=torch.cat(video_parts),
        audio_indices=torch.cat(audio_parts),
        text_indices=torch.arange(text_len),
        seq_len=seq_len,
        target_video=target_video,
        target_audio=target_audio,
    )


__all__ = [
    "AudioReference",
    "ImageReference",
    "PackedLayout",
    "Reference",
    "VideoReference",
    "build_ref2va_layout",
    "video_t_span",
    "TAG_AUDIO",
    "TAG_TEXT",
    "TAG_VIDEO",
]
