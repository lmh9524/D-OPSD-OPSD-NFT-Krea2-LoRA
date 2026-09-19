"""Top-level config for Krea 2 multi-image-reference LoRA.

Outfit-level virtual try-on: several garment images in, one model image wearing them out. Same task
shape as ``flux2_ref2img_lora``, different backbone — and one difference that is not a detail:

**Krea 2 has never been trained with reference images.** ``Krea2Pipeline`` is text-to-image only and
pins the RoPE T axis at zero, so a run started from this recipe is teaching a new conditioning
modality rather than adapting an existing one. The defaults below are set for that: a higher LoRA
rank than the FLUX.2 recipe, a longer schedule, and a warmup long enough that the zero-initialised
adapters do not get a large gradient before they mean anything. See ``dflow/models/family/krea2.py``.

Defaults target one 80 GB card. Sizes are **areas**, aspect ratio is preserved, so token counts vary
per sample and ``batch_size`` stays 1 — doubly so here, since Krea 2's ``position_ids`` is unbatched
and a batch would have to agree on geometry exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dflow.config.data import DataConfig
from dflow.config.dataset import Ref2ImgConfig
from dflow.config.diffusion import FlowMatchConfig
from dflow.config.frozen import FrozenConfig, TextConfig, TextEncoderConfig
from dflow.config.model import BackboneConfig, LoRAConfig, ModelConfig
from dflow.config.optim import LRSchedulerConfig, OptimizerConfig
from dflow.config.runtime import ActivationCheckpointConfig, CompileConfig, RuntimeConfig
from dflow.config.train import CheckpointConfig, LoggingConfig, TrainingConfig

#: Krea 2 taps twelve Qwen3-VL-4B decoder layers. Mirrors ``models/registry.py``; the recipe carries
#: it too because ``TextEncoderConfig`` is what the encoder actually reads.
KREA2_TEXT_LAYERS: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


@dataclass(kw_only=True, slots=True)
class Krea2Ref2ImgLoRARecipe:
    run_directory: str = "runs/krea2_ref2img_lora"

    backbone: BackboneConfig = field(
        default_factory=lambda: BackboneConfig(
            # Raw, not Turbo: fine-tuning a distilled checkpoint adapts an already-collapsed
            # trajectory. Override with --backbone.model.family krea2-turbo if the few-step
            # behaviour is itself what must be preserved.
            model=ModelConfig(family="krea2-raw"),
            # Rank 64, where the FLUX.2 recipe uses 32. Reference conditioning is a new capability
            # here rather than an adaptation of a pretrained one, and rank is the budget for it.
            lora=LoRAConfig(enabled=True, rank=64, alpha=64.0),
        )
    )
    frozen: FrozenConfig = field(
        default_factory=lambda: FrozenConfig(
            text=TextConfig(
                encoders=(
                    TextEncoderConfig(name="qwen3-vl", out_layers=KREA2_TEXT_LAYERS, max_length=512),
                )
            )
        )
    )
    dataset: Ref2ImgConfig = field(
        default_factory=lambda: Ref2ImgConfig(
            target_max_area=1024 * 1024,
            reference_max_area=384 * 384,
            # Garments2Look averages 4.48 garments per outfit. Attention is quadratic in the
            # concatenated length, so four references at 384^2 is the memory compromise; raise
            # --dataset.num-references and lower --dataset.reference-max-area together.
            num_references=4,
            # Must stay 0: reference slots bind to RoPE offsets positionally, and nothing in the
            # prompt names them. Perturbing which garment sits in which slot teaches noise.
            reference_dropout=0.0,
            caption_dropout=0.05,
        )
    )
    data: DataConfig = field(default_factory=DataConfig)

    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            activation_checkpoint=ActivationCheckpointConfig(enabled=True),
            compile=CompileConfig(enabled=False),
        )
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(steps=8000, grad_accum_steps=4)
    )
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-4))
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: LRSchedulerConfig(type="constant", warmup_steps=500)
    )
    #: Left at the default fixed shift, which for Krea 2 is not merely a default: Turbo's pipeline
    #: pins ``mu = 1.15`` at every resolution, and ``exp(1.15)`` is exactly that default. Pass
    #: ``--flow-match.shift None`` to follow Raw's resolution-dependent schedule instead.
    flow_match: FlowMatchConfig = field(default_factory=FlowMatchConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(interval=500, keep_latest=3)
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_paths(self) -> None:
        if self.logging.directory == LoggingConfig().directory:
            self.logging.directory = f"{self.run_directory}/logs"
        if self.checkpoint.directory == CheckpointConfig().directory:
            self.checkpoint.directory = f"{self.run_directory}/checkpoints"


__all__ = ["KREA2_TEXT_LAYERS", "Krea2Ref2ImgLoRARecipe"]
