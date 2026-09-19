"""On-policy self-distillation (D-OPSD) configuration.

Pure declaration. This module must not import anything else from ``dflow``.

D-OPSD (arXiv 2605.05204) turns supervised fine-tuning of a *step-distilled* model into an
on-policy self-distillation problem, so a new concept/style/try-on capability can be learned
**without collapsing the few-step inference schedule** that distillation bought. The same model
plays two roles that differ only in conditioning:

* **student** — conditioned on the deployed context (for ref2img: prompt + garment references),
  rolls out its own few-step trajectory. This is exactly the inference pathway.
* **teacher** — an EMA copy of the student given a *stronger* context: additionally the
  ground-truth target image. It provides the supervision, stop-gradient.

At every visited rollout state both branches predict a velocity; the student is regressed onto
the teacher's implied clean latent (x0). Because the trajectory is the student's own and states
are detached between steps, the few-step dynamics are preserved.

For Krea 2 specifically, two facts (verified against the framework) shape the defaults:

* Krea 2's own text encoder **is Qwen3-VL**, so grounding the teacher on the target image
  (``teacher_ground_target``) goes through a *pretrained* multimodal pathway with no
  feature-space mismatch — the paper's "swap the VLM's LLM weights" hack is unnecessary here.
* Krea 2 has **no pretrained reference conditioning** (the RoPE T axis is pinned at 0), so the
  student can only use garment references once it has been taught to — i.e. after the phase-1
  ref2img LoRA. ``init_lora_from`` is therefore how a real run starts: both adapters are
  initialised from that trained LoRA so the student is already a ref2img model and the teacher
  can actually use the extra reference. Without it, ``teacher_ref_target`` has nothing to stand
  on and is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True, slots=True)
class DOPSDConfig:
    """Knobs for the D-OPSD training loop. The loop shape (rollout in-step) lives in the
    experiment's ``step_fn``; this only declares its parameters."""

    #: K — number of few-step rollout steps used **during training**. The paper trains at 4 and
    #: still infers at the model's native count (Krea 2 Turbo: 8). Fewer steps is cheaper and is
    #: what preserves the few-step distribution; raise only deliberately.
    num_steps: int = 4

    # ---- teacher construction --------------------------------------------------------------
    #: ``ema`` (best, per the paper: momentum 0.9999; the naive live-copy teacher collapses) keeps
    #: a second frozen LoRA adapter that is the EMA of the student adapter. ``frozen_base`` uses the
    #: base model with the adapter disabled — simpler, no EMA, "stable and effective" per the paper,
    #: but a weaker teacher and (for Krea 2) unable to use references at all.
    teacher: Literal["ema", "frozen_base"] = "ema"
    #: EMA momentum for the teacher adapter: ``teacher = decay*teacher + (1-decay)*student``.
    ema_decay: float = 0.9999
    ema_update_every: int = 1
    #: Name of the frozen EMA adapter (the trainable/deployable one keeps ``LoRAConfig.adapter_name``,
    #: normally ``"default"``, so the exported LoRA is the student).
    teacher_adapter_name: str = "teacher"

    # ---- teacher context: how the ground-truth target enters the teacher -------------------
    #: Ground the target image through the text encoder (Krea 2: its own Qwen3-VL) as in-context
    #: supervision. The pretrained, low-risk pathway for Krea 2 — the recommended default.
    teacher_ground_target: bool = True
    #: Additionally append the target's VAE latents as one extra reference span in the teacher's
    #: sequence (the "gt-ref" strategy). Highest identity fidelity, but only meaningful once the
    #: teacher can use references — requires ``init_lora_from``. Refused otherwise.
    teacher_ref_target: bool = False
    #: Text appended to the *teacher* prompt only (an edit instruction, e.g. naming the target).
    #: Empty by default; the grounded image is usually signal enough.
    edit_suffix: str = ""

    # ---- loss ------------------------------------------------------------------------------
    #: ``x0`` (student/teacher clean-latent MSE) matches the reference implementation and converges
    #: faster; it is a t-weighted velocity loss. ``velocity`` is the form written in the paper.
    loss_space: Literal["x0", "velocity"] = "x0"

    # ---- runtime ---------------------------------------------------------------------------
    #: Use the memory-flat trainer (``fit_distill``): back-propagate each rollout step separately, so
    #: peak activation memory is O(1) in ``num_steps`` instead of O(K). Numerically the same as the
    #: default single-backward ``fit`` path; turn on (``--distill.low-mem``) when the K retained
    #: forward graphs do not fit — which the long try-on sequence at K=4 tends to hit on one card.
    low_mem: bool = False

    # ---- initialisation --------------------------------------------------------------------
    #: Local dir or HF repo of the phase-1 ref2img LoRA. Both the student and teacher adapters are
    #: initialised from it, so the student is already reference-capable at step 0. Strongly
    #: recommended for Krea 2 (base cannot use references). ``None`` starts adapters from their
    #: zero-output init: text-only concept learning, or — with ``teacher_ground_target=True`` and
    #: ``teacher_ref_target=False`` — the **D-OPSD-from-base bootstrap** that teaches r2i from scratch,
    #: the teacher cheating via Qwen3-VL grounding of the target (a pretrained pathway the bare t2i
    #: base can already use, since the base cannot use a GT *reference span* yet).
    init_lora_from: str | None = None

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}")
        if self.ema_update_every < 1:
            raise ValueError("ema_update_every must be >= 1")
        if not self.teacher_ground_target and not self.teacher_ref_target:
            raise ValueError(
                "the teacher needs the target somehow: enable teacher_ground_target "
                "(recommended for Krea 2) and/or teacher_ref_target."
            )
        if self.teacher_ref_target and self.init_lora_from is None and not self.teacher_ground_target:
            raise ValueError(
                "teacher_ref_target from a base model (init_lora_from=None) carries no signal at "
                "step 0: Krea 2 pins the RoPE T axis at 0, so the base can *accept* a GT reference "
                "span but cannot yet *use* it — the gt-ref is inert until the student has learned "
                "reference conditioning. To bootstrap r2i from base, add teacher_ground_target=True "
                "(the pretrained Qwen3-VL grounding path provides the step-0 signal, and gt-ref then "
                "activates as the student learns the span). Or set init_lora_from to a "
                "reference-capable LoRA."
            )
        if self.teacher == "frozen_base" and self.teacher_ref_target:
            raise ValueError(
                "a frozen-base teacher (adapter disabled) cannot use references — Krea 2's base "
                "has no pretrained reference conditioning. Use teacher='ema' with init_lora_from."
            )


__all__ = ["DOPSDConfig"]
