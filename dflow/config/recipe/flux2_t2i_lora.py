"""Top-level config for FLUX.2 klein text-to-image LoRA.

A recipe is the whole configuration of one runnable experiment. Defaults here are the ones that
make a single-GPU run work, so the entry point needs only ``--data.root`` and ``--model.path``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dflow.config.data import DataConfig
from dflow.config.dataset import ImageFolderConfig
from dflow.config.diffusion import FlowMatchConfig
from dflow.config.frozen import FrozenConfig
from dflow.config.model import BackboneConfig, LoRAConfig, ModelConfig
from dflow.config.optim import LRSchedulerConfig, OptimizerConfig
from dflow.config.runtime import (
    ActivationCheckpointConfig,
    CompileConfig,
    RuntimeConfig,
)
from dflow.config.train import CheckpointConfig, LoggingConfig, TrainingConfig


@dataclass(kw_only=True, slots=True)
class Flux2T2ILoRARecipe:
    run_directory: str = "runs/flux2_t2i_lora"

    backbone: BackboneConfig = field(
        default_factory=lambda: BackboneConfig(
            model=ModelConfig(family="flux2-klein-base-9b"),
            # LoRA rather than full fine-tuning: 9B in bf16 plus fp32 AdamW moments does not fit
            # on one 80 GB card. Full fine-tuning needs the cluster.
            lora=LoRAConfig(enabled=True, rank=16, alpha=16.0),
        )
    )
    frozen: FrozenConfig = field(default_factory=FrozenConfig)
    dataset: ImageFolderConfig = field(default_factory=ImageFolderConfig)
    data: DataConfig = field(default_factory=DataConfig)

    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            activation_checkpoint=ActivationCheckpointConfig(enabled=True),
            # Compilation is off by default: it pays back over a long run but costs minutes up
            # front, which makes the first smoke test slower to diagnose.
            compile=CompileConfig(enabled=False),
        )
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(steps=1000, grad_accum_steps=1)
    )
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-4))
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: LRSchedulerConfig(type="constant", warmup_steps=50)
    )
    flow_match: FlowMatchConfig = field(default_factory=FlowMatchConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(interval=250, keep_latest=2)
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_paths(self) -> None:
        """Point logging and checkpoints inside the run directory unless overridden."""
        if self.logging.directory == LoggingConfig().directory:
            self.logging.directory = f"{self.run_directory}/logs"
        if self.checkpoint.directory == CheckpointConfig().directory:
            self.checkpoint.directory = f"{self.run_directory}/checkpoints"


__all__ = ["Flux2T2ILoRARecipe"]
