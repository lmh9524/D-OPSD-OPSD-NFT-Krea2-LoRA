"""Per-dataset configuration.

Lives in ``config/`` (L0) rather than beside the dataset implementation, so recipes — which are
also L0 — can reference it without depending on L3. The dataset class imports it from here.

Pure declaration. This module must not import anything else from ``dflow``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class ImageFolderConfig:
    """Images with sidecar caption files."""

    root: str = ""
    #: Must be a multiple of 16, the VAE's effective compression.
    resolution: int = 1024
    caption_suffix: str = ".txt"
    #: Probability of replacing the caption with "". klein embeds no guidance scale and relies on
    #: real CFG with an empty negative prompt, so the unconditional branch degrades if training
    #: never shows the model one.
    caption_dropout: float = 0.05


@dataclass(kw_only=True, slots=True)
class Ref2ImgConfig:
    """Multi-image-reference generation, read from a JSONL manifest.

    One object per line::

        {"target": "a.png", "refs": ["a1.png", "a2.png"], "prompt": "..."}

    Paths are relative to ``root``. An explicit manifest rather than a folder convention because a
    reference set is a relationship between files, which a directory layout cannot express.

    **Sizes are areas, not side lengths, and aspect ratio is preserved.** That is what inference
    does: ``pipeline_flux2_klein.py:770-780`` scales an image down only if it exceeds the area cap
    (``scale = sqrt(target_area / area)``, so the ratio is exact), floors each side to a multiple of
    16, and centre-crops the remainder. Forcing a square would discard composition — 44% of a 16:9
    frame — and train on a shape distribution inference never produces.

    The consequence is that samples have different token counts, so a batch only collates when its
    samples happen to agree. At 9B on one card ``batch_size=1`` is the operating point anyway;
    beyond that, bucketing by shape is the answer and ``data/bucket.py`` is ready for it.
    """

    root: str = ""
    manifest: str = "train.jsonl"

    #: Area cap for the target, matching inference's 1024x1024 default. 1M pixels is ~4096 tokens.
    target_max_area: int = 1024 * 1024
    #: Area cap per reference. **The dominant cost knob**: attention is quadratic in the concatenated
    #: length, and each reference at the full cap adds ~4096 tokens. Halving the area halves them.
    reference_max_area: int = 512 * 512

    #: Reference slots. With ``pad_short_samples`` this is a fixed count and short samples are
    #: cycled to fill it; without, it is an **upper bound** and a short sample simply contributes
    #: fewer references. Either way, samples with more contribute a subset in manifest order.
    num_references: int = 2
    #: Cycle short samples up to ``num_references``. Fixed slots keep a batch collatable, which is
    #: what this was for — but at ``batch_size=1`` it only buys duplicates. On Garments2Look at
    #: nine slots, 37% of every reference slot would be a copy: attention paid over repeats, and a
    #: model taught that duplicates are normal when inference never sends any.
    pad_short_samples: bool = True
    #: Leading slots never dropped when subsampling, and never touched by ``reference_dropout``.
    #: Set to 1 when slot 0 carries a person-identity reference.
    pinned_references: int = 0
    #: Augment the pinned identity references — crop jitter, flip, rotation, colour, JPEG. Their
    #: crops come from the target, so without this a reference agrees with the region it supervises
    #: pixel for pixel, and matching pixels satisfies the loss without learning identity.
    augment_identity: bool = False
    #: Size references to fit inside the *target's* pixel extent rather than to
    #: ``reference_max_area``. This is how the working Krea 2 edit recipe does it: a reference then
    #: lands at roughly the target's own grid, so the shared (h, w) coordinate carries real
    #: correspondence. An area cap chosen to keep nine references affordable produces a 12x12 latent
    #: against a 42x24 target — a seventh of the tokens, a third of the linear resolution — where
    #: garment pattern simply is not present to be copied.
    reference_fit_target: bool = False

    #: What a reference's H/W position ids claim about where its content belongs in the target.
    #:
    #: ``center`` registers each reference to the middle of the target grid. That is right when a
    #: reference is roughly the target's size and shows the same scene, and wrong for product
    #: cutouts: measured against a 42-row target, the trousers reference lands on rows 8.5-32.5
    #: while trousers belong on 21-38, and shoes would land nowhere near the feet. Tops transfer
    #: because their position happens to agree with their content; everything below the waist has
    #: to be learned against the positional signal.
    #:
    #: ``disjoint`` moves references past the target's last row so no false H correspondence exists
    #: and the T axis alone separates them; measured at matched step count it is 6.5x ``center`` and
    #: the only mode positive in both lower bands. ``origin`` starts every span at (0, 0).
    reference_registration: str = "center"

    #: Spacing between references on the RoPE T axis: reference *i* sits at ``t_scale * (i + 1)``.
    #:
    #: 1 gives frames 1, 2, 3 — the community recipe's choice, and the smallest separation possible.
    #: Nothing pretrained constrains this: Krea 2 has only ever seen T at zero, so the adapter is
    #: learning the axis from scratch either way and the spacing is free. Wider values give each
    #: reference a more distinct rotary phase; the cost is that the last reference sits further
    #: outside the range the model has seen.
    reference_t_scale: int = 1

    #: Replace a slot with a repeat of the preceding one, with this probability.
    #:
    #: Keep it at 0 when captions use ordinal language ("the character from image 1"). References
    #: reach the model through the VAE, and through the text encoder only when
    #: ``ground_references`` is on, so a slot is otherwise identified purely by its RoPE offset.
    #: Perturbing which image sits in which slot breaks that correspondence, silently.
    reference_dropout: float = 0.0

    caption_dropout: float = 0.05


@dataclass(kw_only=True, slots=True)
class T2IRLConfig:
    """Prompts for a text-to-image RL run, read from a JSONL manifest.

    One object per line; every field other than ``prompt`` becomes reward metadata::

        {"prompt": "a sign reading HELLO", "text": "HELLO"}

    No target images, which is what makes this an RL task rather than SFT with the images ignored.
    """

    root: str = ""
    manifest: str = "train.jsonl"

    #: The extent to generate at, in pixels. Both must be multiples of the VAE's compression.
    #:
    #: In RL this comes from the config rather than from the data, because there is no target image
    #: to take a shape from — so unlike ref2img, token count is fixed across a run and
    #: ``batch_size`` above 1 needs no bucketing.
    height: int = 512
    width: int = 512


__all__ = ["ImageFolderConfig", "Ref2ImgConfig", "T2IRLConfig"]
