"""Noise-schedule configuration.

Pure declaration. This module must not import anything else from ``dflow``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class FlowMatchConfig:
    """Training-side flow matching.

    FLUX.2's *inference* does not use a constant shift. Its pipelines call
    ``compute_empirical_mu(image_seq_len, num_steps)`` and hand the result to
    ``FlowMatchEulerDiscreteScheduler.set_timesteps(mu=...)``; the scheduler's
    ``base_shift``/``max_shift`` config values are never read (they belong to FLUX.1's
    ``calculate_shift`` mechanism, which lives in pipelines rather than schedulers).

    Whether *training* follows that per-resolution schedule is a separate decision, and
    ``shift`` is where it is made. See its docstring for the two positions.
    """

    num_train_timesteps: int = 1000

    #: How raw timesteps are drawn before shifting. ``logit_normal`` is the SD3/FLUX
    #: convention and concentrates samples in the middle of the trajectory.
    distribution: Literal["logit_normal", "uniform", "mode"] = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    #: Only used by ``distribution="mode"``.
    mode_scale: float = 1.29

    #: Constant schedule shift, in the units other codebases report (``shift == exp(mu)``).
    #: **The default: one value for every sample, regardless of resolution.** ``None``
    #: switches to dynamic, deriving ``mu`` per sample from the target's own token count.
    #:
    #: Both positions are held by real implementations, and the disagreement is genuine:
    #:
    #: * **Fixed.** The training distribution need not equal the inference schedule. A
    #:   sampler's shift decides where its 50 steps land; training decides where the
    #:   velocity field needs signal. DiffSynth-Studio trains FLUX.2 at a flat ``mu = 0.8``
    #:   (shift 2.23) with the comment "if you ask me why I set mu=0.8, I can only say that
    #:   it yields better training results".
    #: * **Dynamic.** klein sets ``use_dynamic_shifting: true``, so inference really does
    #:   run a different schedule per resolution. Pin one value and only that resolution is
    #:   aligned; every other size trains where the sampler does not spend its steps.
    #:
    #: The default 3.1582 is the **dynamic value at a 1024x1024 target** (4096 tokens,
    #: ``mu`` 1.1500) — so a full-size sample trains exactly as it would under dynamic, and
    #: it is what FLUX.1 and SD3 train 1024px at. Smaller samples, which dynamic would give
    #: a gentler shift (256^2 -> 1.65, 512^2 -> 1.88), now share it. The identity is pinned
    #: by ``tests/test_family_flux2.py``, so the constant cannot drift into a bare preference.
    #:
    #: Set ``None`` for the dynamic schedule. Mutually exclusive with ``schedule_steps``.
    shift: float | None = 3.1582

    #: Step count to derive the schedule shift for. Only read when ``shift is None``.
    #: ``None`` means **resolution only**, which is what SFT wants.
    #:
    #: FLUX.2's shift formula decomposes into FLUX.1's resolution shift plus a few-step
    #: correction -- its step-independent term is *exactly* ``calculate_shift``. The step term
    #: is inference discretisation (with ten steps you push the schedule up to place them
    #: better), not a claim about where the velocity field needs training signal. At a
    #: 1024x1024 target, resolution-only gives shift 3.16, matching what FLUX.1 and SD3 train
    #: at; a 50-step value would give 7.6 and train at far higher noise than comparable models.
    #:
    #: Set an integer only when deliberately training *for* a few-step schedule, as step
    #: distillation does.
    #:
    #: How the shift is computed is *not* configured here -- it belongs to the model. Ask
    #: ``family.noise_shift_mu(image_tokens=..., inference_steps=...)`` and hand the result to
    #: the scheduler. Which token count to pass is a call-site decision; for FLUX.2 it is the
    #: **target** count (``pipeline_flux2_klein.py:815`` uses ``latents.shape[1]``, with
    #: reference latents concatenated only inside the loop).
    schedule_steps: int | None = None

    #: Optional loss weighting over timesteps. ``none`` matches plain flow matching.
    weighting: Literal["none", "sigma_sqrt", "cosmap"] = "none"

    #: Clamp sampled sigmas away from the exact endpoints, where the velocity target is
    #: degenerate.
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        # Both knobs move mu, and a fixed shift wins, so setting both means one of them is
        # silently doing nothing. Caught here rather than at the call site so it fires while
        # parsing flags, before 9B of weights load.
        if self.shift is not None:
            if self.shift <= 0.0:
                raise ValueError(
                    f"shift must be positive (mu = log(shift)), got {self.shift}. "
                    f"Pass --flow-match.shift None for the dynamic schedule."
                )
            if self.schedule_steps is not None:
                raise ValueError(
                    f"shift={self.shift} pins the schedule, so schedule_steps="
                    f"{self.schedule_steps} would be ignored. Set --flow-match.shift None to "
                    f"derive mu per sample, or drop --flow-match.schedule-steps."
                )


__all__ = ["FlowMatchConfig"]
