"""Configuration dataclasses.

L0: pure declaration. Nothing in this package imports any other ``dflow`` module, so
configs can be constructed, serialised and tested without touching torch.distributed,
diffusers or a GPU.
"""

from .data import BucketConfig, DataConfig, DataLoaderConfig, RetryConfig
from .dataset import ImageFolderConfig, Ref2ImgConfig, T2IRLConfig
from .diffusion import FlowMatchConfig
from .distill import DOPSDConfig
from .frozen import FrozenConfig, TextConfig, TextEncoderConfig, VAEConfig
from .model import BackboneConfig, EMAConfig, LoRAConfig, ModelConfig
from .optim import LRSchedulerConfig, OptimizerConfig
from .rl import (
    AestheticRewardConfig,
    DiffusionNFTConfig,
    GroupConfig,
    OCRRewardConfig,
    PPOConfig,
    ReferenceFidelityRewardConfig,
    RewardConfig,
    SDEConfig,
)
from .runtime import (
    ActivationCheckpointConfig,
    CompileConfig,
    CPConfig,
    DistributedConfig,
    DType,
    FSDPConfig,
    RuntimeConfig,
)
from .train import CheckpointConfig, LoggingConfig, TrainingConfig

__all__ = [
    "ActivationCheckpointConfig",
    "AestheticRewardConfig",
    "BackboneConfig",
    "BucketConfig",
    "CPConfig",
    "CheckpointConfig",
    "CompileConfig",
    "DOPSDConfig",
    "DiffusionNFTConfig",
    "DType",
    "DataConfig",
    "DataLoaderConfig",
    "DistributedConfig",
    "EMAConfig",
    "FSDPConfig",
    "FlowMatchConfig",
    "FrozenConfig",
    "GroupConfig",
    "ImageFolderConfig",
    "LRSchedulerConfig",
    "LoRAConfig",
    "LoggingConfig",
    "ModelConfig",
    "OCRRewardConfig",
    "OptimizerConfig",
    "PPOConfig",
    "Ref2ImgConfig",
    "ReferenceFidelityRewardConfig",
    "RetryConfig",
    "RewardConfig",
    "RuntimeConfig",
    "SDEConfig",
    "T2IRLConfig",
    "TextConfig",
    "TextEncoderConfig",
    "TrainingConfig",
    "VAEConfig",
]
