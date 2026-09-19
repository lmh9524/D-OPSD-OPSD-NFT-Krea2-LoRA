"""Build the multi-reference sequence: the one place training must match inference exactly.

FLUX.2 conditions on reference images by **sequence concatenation**, not channel concatenation. That
is why multi-reference needs no weight surgery and works with plain LoRA — but it also means the
model tells target from reference purely by position ids, so getting those wrong produces a run that
trains happily and learns nothing useful.

The layout, from ``pipeline_flux2_klein.py``:

    hidden_states = cat([target_tokens, ref_1_tokens, ref_2_tokens, ...], dim=1)
    img_ids       = cat([target_ids,    reference_ids],                  dim=1)
    output        = model(...)[:, :target_len]      # references are read-only context

Position ids are 4-D ``(T, H, W, L)`` and each axis has a job: **T distinguishes reference images**
(target at 0, reference *i* at ``10 * (i + 1)``), H and W are spatial, and L carries text position.
``axes_dims_rope`` splits the head dimension four ways to match.

The id builders are **imported from diffusers, not reimplemented**. Copying them would leave a
second implementation to drift out of step with inference after an upgrade; importing makes drift
impossible, and ``tests/test_upstream_contract.py`` pins the signatures we rely on.

Both target and reference latents arrive already patchified by ``VAEEncoder.encode`` — 32 VAE
channels folded to 128 over half the spatial extent. The pipeline does the same for both paths
(``prepare_latents`` builds ``(B, 32 * 4, H/2, W/2)``; ``prepare_image_latents`` yields
``(1, 128, 32, 32)`` per 512px reference), so a 1024px target is 4096 tokens and a 512px reference
is 1024.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Spacing between reference images on the T axis. Matches ``_prepare_image_ids``' default; wider
#: than 1 so references stay far apart in RoPE space, and far from the target at T=0.
T_SCALE = 10


def _latent_ids_fn():
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

    return Flux2Pipeline._prepare_latent_ids


def _image_ids_fn():
    from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

    return Flux2Pipeline._prepare_image_ids


def pack(latents: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W) -> (B, H * W, C)``, matching ``_pack_latents``."""
    batch, channels, height, width = latents.shape
    return latents.reshape(batch, channels, height * width).permute(0, 2, 1)


def unpack(tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Inverse of :func:`pack`."""
    batch, sequence, channels = tokens.shape
    if sequence != height * width:
        raise ValueError(f"{sequence} tokens cannot fill a {height}x{width} grid")
    return tokens.permute(0, 2, 1).reshape(batch, channels, height, width)


@dataclass(frozen=True, slots=True)
class ReferenceSequence:
    """The concatenated sequence and its position ids."""

    tokens: torch.Tensor  # (B, target_len + reference_len, C)
    ids: torch.Tensor  # (B, target_len + reference_len, 4) or (seq, 3) for Krea 2
    target_len: int
    #: Where the target span starts. 0 puts the target first (FLUX.2's layout, matching
    #: ``_prepare_image_ids``); ``reference_len`` puts it last, which is what Krea 2 edit LoRAs
    #: trained on — `[text | refs | target]`, output sliced from the tail.
    target_offset: int = 0

    @property
    def reference_len(self) -> int:
        return self.tokens.shape[1] - self.target_len

    def replace_target(self, target_tokens: torch.Tensor) -> ReferenceSequence:
        """Swap in a different target span, keeping references and ids.

        Used once per step: the reference tokens are clean latents and the target span is the noisy
        one, so the noise is applied to the target *before* concatenation and only that part changes.
        """
        if target_tokens.shape[1] != self.target_len:
            raise ValueError(
                f"target span is {self.target_len} tokens, got {target_tokens.shape[1]}"
            )
        start = self.target_offset
        stop = start + self.target_len
        return ReferenceSequence(
            tokens=torch.cat(
                [self.tokens[:, :start], target_tokens, self.tokens[:, stop:]], dim=1
            ),
            ids=self.ids,
            target_len=self.target_len,
            target_offset=self.target_offset,
        )


def build_reference_ids(
    reference_latents: list[torch.Tensor], *, t_scale: int = T_SCALE
) -> torch.Tensor:
    """``(1, total_reference_tokens, 4)`` — reference *i* at ``T = t_scale * (i + 1)``.

    Ids depend only on shapes, so one batch element is enough to derive them.
    """
    if not reference_latents:
        raise ValueError("build_reference_ids needs at least one reference")
    return _image_ids_fn()([latents[:1] for latents in reference_latents], scale=t_scale)


def build_sequence(
    *,
    target_latents: torch.Tensor,
    reference_latents: list[torch.Tensor],
    t_scale: int = T_SCALE,
) -> ReferenceSequence:
    """Concatenate a target and its references into one sequence with matching ids.

    Args:
        target_latents: ``(B, C, H, W)``, already patchified and normalised.
        reference_latents: one ``(B, C, H, W)`` per reference slot. May be empty, which degrades
            cleanly to plain text-to-image.
    """
    if target_latents.ndim != 4:
        raise ValueError(f"target latents must be (B, C, H, W), got {tuple(target_latents.shape)}")
    batch = target_latents.shape[0]
    for index, latents in enumerate(reference_latents):
        if latents.ndim != 4:
            raise ValueError(
                f"reference {index} must be (B, C, H, W), got {tuple(latents.shape)}"
            )
        if latents.shape[0] != batch:
            raise ValueError(
                f"reference {index} has batch {latents.shape[0]}, target has {batch}"
            )
        if latents.shape[1] != target_latents.shape[1]:
            raise ValueError(
                f"reference {index} has {latents.shape[1]} channels, target has "
                f"{target_latents.shape[1]}"
            )

    target_tokens = pack(target_latents)
    target_ids = _latent_ids_fn()(target_latents).to(target_latents.device)
    target_len = target_tokens.shape[1]

    if not reference_latents:
        return ReferenceSequence(tokens=target_tokens, ids=target_ids, target_len=target_len)

    reference_tokens = torch.cat([pack(latents) for latents in reference_latents], dim=1)
    reference_ids = build_reference_ids(reference_latents, t_scale=t_scale)
    reference_ids = reference_ids.to(target_latents.device).expand(batch, -1, -1)

    return ReferenceSequence(
        tokens=torch.cat([target_tokens, reference_tokens], dim=1),
        ids=torch.cat([target_ids, reference_ids], dim=1),
        target_len=target_len,
    )


# ------------------------------------------------------------------------------ Krea 2
#
# Krea 2's layout is **not** FLUX.2's, and the differences were established by reading a working
# community edit LoRA (`conradlocke/krea2-identity-edit`) together with the node pack that serves it
# (`lbouaraba/comfyui-krea2edit`). Its forward assembles:
#
#     combined = cat([context] + src_imgs + [tgt_img], dim=1)      # [text | refs | target]
#     ref_ids  = [_imgids(bs, i + 1, gh, gw) for i, ...]           # frame 1, 2, 3 ...
#     pos      = cat([zeros(txtlen, 3)] + ref_ids + [_imgids(bs, 0, h_, w_)])
#     out      = final[:, txtlen + srclen : txtlen + srclen + tgtlen]
#
# Three things differ from what this file did first, and each is silent when wrong:
#
# **T frames are 1, 2, 3 — not 10, 20, 30.** FLUX.2's spacing is safe there because klein was
# *trained* with it. Krea 2's T axis has only ever seen 0, and its fastest rotary component turns
# 1.0 rad per unit: by T=7 that component has wrapped a full turn, so T=10 is an arbitrary phase far
# outside anything the weights have met. T=1 is a 0.16-turn extrapolation from the trained point.
#
# **The target comes last.** `[text | refs | target]`, and the output is sliced from the tail.
#
# **Reference H/W is registered to the target grid**, not restarted at (0, 0). A reference smaller
# than the target is centred inside it at stride 1, with a fractional offset because RoPE is
# continuous and half-token positions are exact. A reference token at (h, w) therefore shares its
# spatial coordinate with the target token at (h, w), which is what makes "keep this region" cheap
# to learn — and is precisely the thing Krea 2 could not learn when the two grids were independent.


REGISTRATIONS = ("center", "origin", "disjoint")


def build_krea2_latent_ids(
    latents: torch.Tensor,
    *,
    frame: int = 0,
    target_grid: tuple[int, int] | None = None,
    registration: str = "center",
) -> torch.Tensor:
    """``(H * W, 3)`` ids for one image span at ``frame`` on the T axis.

    ``registration`` decides what the H/W coordinates of a *reference* span claim about where its
    content belongs in the target. The choice is not cosmetic — it is the difference between a
    positional signal that agrees with the caption and one that contradicts it.

    ``center``
        Centre the span inside ``target_grid`` at stride 1, with a fractional offset (RoPE is
        continuous, so the exact half is free and an integer floor would put an odd-sized reference
        half a token off). This registers the reference to the target, which is right when the
        reference is roughly the target's own size and shows the same scene.

        It is wrong for product cutouts. Measured on Garments2Look against a 42-row target: the
        identity crop lands on rows 7.5-33.5, the top on 8.5-32.5, the *trousers* on 8.5-32.5 and
        the bag on 13-28. Torso is rows 6-21, legs 21-38, feet 38-41. So every garment's H
        coordinate says "torso height" — true for a top, wrong by twelve rows for trousers, and
        disjoint from where shoes go. Tops transfer because the position agrees with the content;
        everything below the waist has to be learned *against* the positional signal.

    ``origin``
        Start every span at (0, 0). No claim beyond "these are the first rows", which for a
        sub-target reference still overlaps head and torso.

    ``disjoint``
        Shift H past the target's last row, so no reference token shares an H coordinate with any
        target token and the false correspondence is gone entirely. Separation between references is
        then carried by the T axis alone, which is what it is for. W stays centred: left-right does
        carry meaning (a bag hangs on one side), it is only the vertical claim that misleads.

    Without ``target_grid`` the grid starts at (0, 0) whatever the mode, which is also what happens
    when the reference is larger than the target and centring would push coordinates negative.
    """
    if latents.ndim != 4:
        raise ValueError(f"latents must be (B, C, H, W), got {tuple(latents.shape)}")
    if registration not in REGISTRATIONS:
        raise ValueError(f"registration must be one of {REGISTRATIONS}, got {registration!r}")
    height, width = latents.shape[-2:]
    offset_h = offset_w = 0.0
    if target_grid is not None and registration != "origin":
        target_h, target_w = target_grid
        if width <= target_w:
            offset_w = (target_w - width) / 2.0
        if registration == "disjoint":
            offset_h = float(target_h)
        elif height <= target_h:
            offset_h = (target_h - height) / 2.0

    ids = torch.zeros(height, width, 3, device=latents.device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (
        torch.arange(height, device=latents.device, dtype=torch.float32) + offset_h
    )[:, None]
    ids[..., 2] = (
        torch.arange(width, device=latents.device, dtype=torch.float32) + offset_w
    )[None, :]
    return ids.reshape(height * width, 3)


def build_krea2_text_ids(text_len: int, *, device: torch.device) -> torch.Tensor:
    """``(text_len, 3)`` of zeros — Krea 2 puts every text token at the origin.

    Unlike FLUX.2 there is no text position axis: the text stream is ordered by its place in the
    sequence and by the encoder's own mRoPE, not by these ids.
    """
    return torch.zeros(text_len, 3, device=device, dtype=torch.float32)


def build_krea2_sequence(
    *,
    target_latents: torch.Tensor,
    reference_latents: list[torch.Tensor],
    registration: str = "center",
    t_scale: int = 1,
) -> ReferenceSequence:
    """Concatenate references and a target into one Krea 2 sequence: ``[refs | target]``.

    Ids come back **unbatched** as ``(seq, 3)``, which is what the model accepts, and the returned
    sequence reports ``target_offset`` so the caller slices the tail rather than the head.

    ``t_scale`` spaces the references along the T axis: reference *i* sits at ``t_scale * (i + 1)``,
    so 1 gives the frames 1, 2, 3 the community recipe uses and 10 gives 10, 20, 30. The axis is
    what distinguishes one reference from another, and Krea 2 has only ever seen it at zero, so how
    far apart the values sit is a free parameter nothing pretrained constrains. Adjacent integers
    are the smallest possible separation; wider spacing gives each reference a more distinct rotary
    phase at the cost of pushing the last one further outside the range the model has seen.

    Args:
        target_latents: ``(B, C, H, W)``, already patchified and normalised.
        reference_latents: one ``(B, C, H, W)`` per reference slot. May be empty, which degrades
            cleanly to plain text-to-image — the only configuration any stock checkpoint has seen.
    """
    if target_latents.ndim != 4:
        raise ValueError(f"target latents must be (B, C, H, W), got {tuple(target_latents.shape)}")
    batch = target_latents.shape[0]
    for index, latents in enumerate(reference_latents):
        if latents.ndim != 4:
            raise ValueError(f"reference {index} must be (B, C, H, W), got {tuple(latents.shape)}")
        if latents.shape[0] != batch:
            raise ValueError(
                f"reference {index} has batch {latents.shape[0]}, target has {batch}"
            )
        if latents.shape[1] != target_latents.shape[1]:
            raise ValueError(
                f"reference {index} has {latents.shape[1]} channels, target has "
                f"{target_latents.shape[1]}"
            )

    target_grid = tuple(target_latents.shape[-2:])
    target_tokens = pack(target_latents)
    target_ids = build_krea2_latent_ids(target_latents, frame=0)
    target_len = target_tokens.shape[1]

    if not reference_latents:
        return ReferenceSequence(tokens=target_tokens, ids=target_ids, target_len=target_len)

    reference_tokens = torch.cat([pack(latents) for latents in reference_latents], dim=1)
    reference_ids = torch.cat(
        [
            build_krea2_latent_ids(
                latents, frame=t_scale * (index + 1), target_grid=target_grid,
                registration=registration,
            )
            for index, latents in enumerate(reference_latents)
        ],
        dim=0,
    )

    return ReferenceSequence(
        tokens=torch.cat([reference_tokens, target_tokens], dim=1),
        ids=torch.cat([reference_ids, target_ids], dim=0),
        target_len=target_len,
        target_offset=reference_tokens.shape[1],
    )


__all__ = [
    "T_SCALE",
    "ReferenceSequence",
    "build_krea2_latent_ids",
    "build_krea2_sequence",
    "build_krea2_text_ids",
    "build_reference_ids",
    "build_sequence",
    "pack",
    "unpack",
]
