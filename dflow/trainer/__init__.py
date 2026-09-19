"""L5: training loops and the progress state they carry.

``state.py`` is shared by every training method. ``loop.py`` (SFT) is not — RL and
distillation have different loop shapes and get their own modules rather than
subclassing this one.
"""

from dflow.trainer.distill import RolloutStepFn, fit_distill
from dflow.trainer.loop import StepFn, fit
from dflow.trainer.state import RESUMABLE, STATE_VERSION, ResumableComponent, StepMetrics, TrainState

__all__ = [
    "RESUMABLE",
    "STATE_VERSION",
    "ResumableComponent",
    "RolloutStepFn",
    "StepFn",
    "StepMetrics",
    "TrainState",
    "fit",
    "fit_distill",
]
