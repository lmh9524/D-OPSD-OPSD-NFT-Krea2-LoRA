"""Text-to-image RL: prompts and reward metadata, no supervision target.

A task of its own rather than an image dataset with the images ignored, because the absence of a
target is structural — it is what makes the loop an RL loop.
"""

from dflow.tasks.t2i_rl.dataset import PromptDataset, PromptSample, read_manifest
from dflow.tasks.t2i_rl.schema import T2IRLBatch, validate

__all__ = [
    "PromptDataset",
    "PromptSample",
    "T2IRLBatch",
    "read_manifest",
    "validate",
]
