"""Top-level config for FLUX.2 klein multi-image-reference LoRA.

Phase-1 target task. Sizes are **areas**, and aspect ratio is preserved, matching inference — so token
counts vary per sample and `batch_size=1` is the operating point until bucketing is wired in.

Defaults for a single 80 GB card: a 1M-pixel target (~4096 tokens) with two references capped at
512x512 (~1024 tokens each) gives ~6144. Attention is quadratic in the total, so reference area is the
first knob to turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dflow.config.data import DataConfig
from dflow.config.dataset import Ref2ImgConfig
from dflow.config.diffusion import FlowMatchConfig
from dflow.config.frozen import FrozenConfig
from dflow.config.model import BackboneConfig, LoRAConfig, ModelConfig
from dflow.config.optim import LRSchedulerConfig, OptimizerConfig
from dflow.config.runtime import ActivationCheckpointConfig, CompileConfig, RuntimeConfig
from dflow.config.train import CheckpointConfig, LoggingConfig, TrainingConfig


@dataclass(kw_only=True, slots=True)
class Flux2Ref2ImgLoRARecipe:
    run_directory: str = "runs/flux2_ref2img_lora"

    backbone: BackboneConfig = field(
        default_factory=lambda: BackboneConfig(
            model=ModelConfig(family="flux2-klein-base-9b"),
            lora=LoRAConfig(enabled=True, rank=32, alpha=32.0),
        )
    )
    frozen: FrozenConfig = field(default_factory=FrozenConfig)
    dataset: Ref2ImgConfig = field(
        default_factory=lambda: Ref2ImgConfig(
            target_max_area=1024 * 1024, reference_max_area=512 * 512, num_references=2
        )
    )
    data: DataConfig = field(default_factory=DataConfig)

    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            # Mandatory here, not optional: 3072 tokens across 32 blocks does not fit otherwise.
            activation_checkpoint=ActivationCheckpointConfig(enabled=True),
            compile=CompileConfig(enabled=False),
        )
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(steps=2000, grad_accum_steps=1)
    )
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-4))
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: LRSchedulerConfig(type="constant", warmup_steps=100)
    )
    flow_match: FlowMatchConfig = field(default_factory=FlowMatchConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(interval=500, keep_latest=2)
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_paths(self) -> None:
        if self.logging.directory == LoggingConfig().directory:
            self.logging.directory = f"{self.run_directory}/logs"
        if self.checkpoint.directory == CheckpointConfig().directory:
            self.checkpoint.directory = f"{self.run_directory}/checkpoints"


__all__ = ["Flux2Ref2ImgLoRARecipe"]
