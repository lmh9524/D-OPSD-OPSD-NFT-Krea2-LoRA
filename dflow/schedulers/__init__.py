"""L3: noise schedules. Model-agnostic.

``mu`` is supplied by the model family, not derived here -- see
``dflow/schedulers/flow_matching.py``. Inference stepping stays with diffusers for SFT
previews and sampling; ``flow_sde.py`` is the rollout-side stepper RL needs in-loop.
"""

from dflow.schedulers.flow_matching import (
    FlowMatchScheduler,
    apply_time_shift,
    build_scheduler,
    shift_from_mu,
)
from dflow.schedulers.flow_sde import (
    SDESchedule,
    SDEStep,
    build_sde_schedule,
    ode_step,
    sde_step,
    step_logprob,
)

__all__ = [
    "FlowMatchScheduler",
    "SDESchedule",
    "SDEStep",
    "apply_time_shift",
    "build_scheduler",
    "build_sde_schedule",
    "ode_step",
    "sde_step",
    "shift_from_mu",
    "step_logprob",
]
