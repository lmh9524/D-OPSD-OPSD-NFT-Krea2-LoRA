"""Top-level config for FLUX.2 klein text-to-image Flow-GRPO.

A recipe is the whole configuration of one runnable experiment. The defaults here are chosen to
make a **single-GPU** run start, which for RL means smaller than the SFT defaults in several places
at once — see the comments, each of which is a memory or throughput argument rather than a taste.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dflow.config.data import DataConfig, DataLoaderConfig
from dflow.config.dataset import T2IRLConfig
from dflow.config.frozen import FrozenConfig
from dflow.config.model import BackboneConfig, LoRAConfig, ModelConfig
from dflow.config.optim import LRSchedulerConfig, OptimizerConfig
from dflow.config.rl import (
    AestheticRewardConfig,
    GroupConfig,
    PPOConfig,
    RewardConfig,
    SDEConfig,
)
from dflow.config.runtime import ActivationCheckpointConfig, CompileConfig, RuntimeConfig
from dflow.config.train import CheckpointConfig, LoggingConfig, TrainingConfig


@dataclass(kw_only=True, slots=True)
class Flux2T2IFlowGRPORecipe:
    run_directory: str = "runs/flux2_t2i_flowgrpo"

    backbone: BackboneConfig = field(
        default_factory=lambda: BackboneConfig(
            model=ModelConfig(family="flux2-klein-base-9b"),
            lora=LoRAConfig(enabled=True, rank=16, alpha=16.0),
        )
    )
    frozen: FrozenConfig = field(default_factory=FrozenConfig)

    #: 512px rather than SFT's 1024. Token count is quadratic in attention cost and a step is
    #: ``group_size * inference_steps`` forwards, so this is the difference between a minute and
    #: several per step.
    dataset: T2IRLConfig = field(default_factory=lambda: T2IRLConfig(height=512, width=512))

    #: **One prompt per rank per step.** A group is already ``group.size`` trajectories, so the
    #: effective rollout batch is ``batch_size * group.size``. Raise this only after the group fits
    #: comfortably; ``fit_rl`` requires at least ``dp_size`` prompts per step.
    data: DataConfig = field(
        default_factory=lambda: DataConfig(loader=DataLoaderConfig(batch_size=1))
    )

    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            activation_checkpoint=ActivationCheckpointConfig(enabled=True),
            compile=CompileConfig(enabled=False),
        )
    )

    #: Far fewer steps than SFT: each one is a rollout plus several updates, so 300 optimizer
    #: updates is already a few hours. flow_grpo's published runs move visibly inside 200.
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(steps=300, grad_accum_steps=1)
    )

    #: An order of magnitude below the SFT default. The policy gradient is noisier than a
    #: regression target and a group of eight is a small sample, so a large step overshoots before
    #: the reward signal has averaged out.
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-5))
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: LRSchedulerConfig(type="constant", warmup_steps=10)
    )

    sde: SDEConfig = field(default_factory=SDEConfig)
    group: GroupConfig = field(default_factory=GroupConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    #: Aesthetic only by default: it needs no per-sample metadata, so the recipe runs against a
    #: manifest of bare prompts. Add ``--reward.ocr.enabled True`` once the manifest carries a
    #: ``text`` field per prompt.
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            aesthetic=AestheticRewardConfig(enabled=True)
        )
    )

    #: Classifier-free guidance during rollout, as an *effective* scale — klein's combination is
    #: ``pred + scale * (pred - negative)``, so 0 means one forward per step and no negative branch.
    #:
    #: Off by default for two reasons. It doubles every rollout *and* every replay forward, and
    #: whatever it is set to, the replay must use the identical value or the log-probs describe a
    #: different policy — which is why the experiment routes both through one factory rather than
    #: letting the two drift.
    guidance: float = 0.0

    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(interval=50, keep_latest=2)
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_paths(self) -> None:
        """Point logging and checkpoints inside the run directory unless overridden."""
        if self.logging.directory == LoggingConfig().directory:
            self.logging.directory = f"{self.run_directory}/logs"
        if self.checkpoint.directory == CheckpointConfig().directory:
            self.checkpoint.directory = f"{self.run_directory}/checkpoints"


__all__ = ["Flux2T2IFlowGRPORecipe"]
