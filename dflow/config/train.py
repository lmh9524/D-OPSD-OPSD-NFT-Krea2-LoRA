"""Training-loop, checkpoint and logging configuration.

Pure declaration. This module must not import anything else from ``dflow``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class TrainingConfig:
    seed: int = 42
    steps: int = 10_000
    grad_accum_steps: int = 1
    autocast_dtype: Literal["bfloat16", "float16"] | None = "bfloat16"
    max_grad_norm: float | None = 1.0


@dataclass(kw_only=True, slots=True)
class CheckpointConfig:
    save_enabled: bool = True
    directory: str = "checkpoints"
    interval: int = 1000
    keep_latest: int = 2
    # None, a path, or "latest".
    resume: str | None = None
    # Also write a diffusers-format copy (save_pretrained / LoRA) alongside the DCP
    # shards, so the result is loadable by a stock pipeline without a conversion step.
    export_diffusers: bool = True

    #: Render one sample from the training set every time a checkpoint is written.
    #:
    #: On by default because the alternative is waiting for a run to finish before learning it went
    #: nowhere — a twelve-hour lesson that a two-minute render gives at step 2000. It is nearly free
    #: during training: the transformer, VAE and text encoder are already resident, so a preview
    #: costs a few forward passes and no extra weights, where a separate process would need to load
    #: ~34 GB it has no room for.
    #:
    #: The loop itself stays ignorant of this — ``fit()`` takes an ``on_checkpoint`` callback and the
    #: experiment builds it, the same way ``step_fn`` works.
    preview: bool = True
    #: How often (in steps) to render a preview, decoupled from the checkpoint-save ``interval``
    #: above. ``None`` couples them — a preview rides every save (the original behaviour). Set it
    #: smaller than ``interval`` to watch training more often than you commit weights, e.g. preview
    #: every 100 while saving every 200. When a preview step coincides with a save step it renders
    #: once, not twice. Honoured by ``fit_distill`` (the D-OPSD loop) and ``fit``.
    preview_interval: int | None = None
    #: Denoising steps for the preview. ``None`` asks the backbone family for its own recommended
    #: setting, which is the only value that is right by default.
    #:
    #: 8 is the *distilled* operating point. A preview renders whatever backbone is in memory, and
    #: training a LoRA for Krea-2-Turbo still means training on Krea-2-Raw, which is not distilled.
    #: krea-ai/krea-2's README gives the two settings explicitly: Raw is ``--steps 52 --cfg 3.5``,
    #: Turbo is ``--steps 8 --cfg 0.0 --mu 1.15``. Rendering Raw at Turbo's settings produces a
    #: generic, condition-weak image no matter what the adapter learned, and reading that as "the
    #: LoRA ignores its references" costs a run.
    preview_steps: int | None = None
    #: CFG scale for the preview, in the pipeline's own ``pred + s * (pred - negative)`` form, so
    #: the effective strength is ``1 + s``. ``None`` asks the family; 0 disables CFG, which is right
    #: only for a distilled backbone.
    preview_guidance: float | None = None
    #: Cap on the preview's pixel area, independent of what the run trains at.
    #:
    #: A preview allocates on top of a training step's peak. At 384x672 that fits; at 576x1008 it
    #: asked for 5.85 GiB more than was left and every checkpoint in a nine-hour run went by with
    #: no visual signal at all, the failure swallowed into a warning because a preview must never
    #: end a run. Capping the render keeps it affordable at any training resolution, and a progress
    #: signal does not need the full extent — it needs to be the same sample every time.
    preview_max_area: int = 384 * 672
    #: Index into the dataset to render. Fixed, so previews across checkpoints are comparable.
    preview_index: int = 0
    #: Directory of a HELD-OUT ref2img dataset (train.jsonl + images/) to preview instead of the
    #: training set — real test cases the model never trained on. ``None`` falls back to
    #: ``preview_index`` into the training dataset.
    preview_dataset: str | None = None
    #: How many cases from ``preview_dataset`` to render at each checkpoint.
    preview_count: int = 5

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f"checkpoint interval must be >= 1, got {self.interval}")
        if self.preview_interval is not None and self.preview_interval < 1:
            raise ValueError(
                f"preview_interval must be >= 1 or None, got {self.preview_interval}"
            )


@dataclass(kw_only=True, slots=True)
class LoggingConfig:
    directory: str = "logs"
    filename: str = "train.log"
    interval: int = 1
    tensorboard: bool = True


__all__ = ["CheckpointConfig", "LoggingConfig", "TrainingConfig"]
