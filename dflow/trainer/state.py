"""Training progress: the state that must survive a restart.

Split from ``loop.py`` for two reasons.

**It makes "what must be saved" an explicit, testable list.** When counters live as
scattered trainer attributes, what actually reaches the checkpoint is implicit — you
have to read the save function to find out, and adding a counter without saving it does
not raise. vflow shows the end state of that: its resume path reads

    training_state.get("num_consumed_batches",
        training_state.get("data_step",                 # a key that was renamed
            global_step * grad_accum_steps))           # an even older derivation

plus a ``"version": 3`` field. Here the answer is the field list of one dataclass, and
``RESUMABLE`` names every component a checkpoint must round-trip.

**It is shared where ``loop.py`` is not.** RL and distillation have different loop
shapes and will not reuse ``fit()``, but they need the same notion of progress and the
same resume guarantees. Keeping state here stops them from re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATE_VERSION = 2


@dataclass(frozen=True, slots=True)
class ResumableComponent:
    """One thing a checkpoint must contain for an exact restart."""

    key: str
    owner: str
    why: str


#: Every piece of state a checkpoint must round-trip. ``checkpoint/manager.py`` is
#: responsible for all of them, and ``tests/test_resume.py`` asserts the manifest and the
#: manager agree — so adding state here without handling it there fails loudly.
RESUMABLE: tuple[ResumableComponent, ...] = (
    ResumableComponent(
        key="train_state",
        owner="dflow.trainer.state.TrainState",
        why="step counters; consumed_batches for SFT data resume, consumed_rollout_batches for RL",
    ),
    ResumableComponent(
        key="model",
        owner="dflow.checkpoint.dcp",
        why="weights; DCP shards for full fine-tunes, adapter-only for LoRA",
    ),
    ResumableComponent(
        key="optimizer",
        owner="dflow.checkpoint.dcp",
        why="moment estimates; dropping them restarts Adam's warmup and dents quality",
    ),
    ResumableComponent(
        key="lr_scheduler",
        owner="torch.optim.lr_scheduler",
        why="schedule position; a reset schedule silently changes the effective LR",
    ),
    ResumableComponent(
        key="ema",
        owner="dflow.common.ema.EMA",
        why="shadow parameters, which are the published weights when EMA is on",
    ),
    ResumableComponent(
        key="rng_states",
        owner="dflow.runtime.seed",
        why="per-rank generator state; one shared seed makes every rank draw the same noise",
    ),
)


@dataclass(kw_only=True, slots=True)
class TrainState:
    """Where training got to.

    ``global_step`` and ``consumed_batches`` are tracked independently and cannot be
    derived from each other:

    * a crash mid-accumulation leaves micro-batches consumed without a completed step;
    * a sample that fails to load is retried, so consumption is not a fixed multiple of
      the step count.

    The invariant that does hold is
    ``consumed_batches >= global_step * grad_accum_steps``.
    """

    #: Completed optimizer updates.
    global_step: int = 0
    #: Micro-batches fed to the model. **The source of truth for data resume.**
    #:
    #: Not the sampler's internal position: with ``num_workers > 0`` the DataLoader
    #: prefetches, so the sampler runs ahead of what the model consumed. Restoring the
    #: sampler position would skip the prefetched-but-unconsumed batches.
    consumed_batches: int = 0
    #: Rollout batches drawn from the prompt dataset. **The source of truth for data resume
    #: under RL**, where ``consumed_batches`` is not.
    #:
    #: The two cannot be collapsed. ``fit()`` feeds one dataset batch per micro-batch, so
    #: ``consumed_batches`` positions its sampler. ``fit_rl()`` draws one batch of prompts, rolls
    #: out a group per prompt, and then takes ``inner_epochs * grad_accum_steps`` micro-batches
    #: from that one draw — so ``consumed_batches`` runs far ahead of the prompt sampler and
    #: restoring from it would skip most of the dataset.
    #:
    #: Zero for an SFT run, which is what makes the field additive rather than a fork.
    consumed_rollout_batches: int = 0

    #: Informational only; never used to reconstruct data order.
    epoch: int = 0

    def on_micro_batch(self) -> None:
        self.consumed_batches += 1

    def on_rollout_batch(self) -> None:
        self.consumed_rollout_batches += 1

    def on_optimizer_step(self) -> None:
        self.global_step += 1

    def validate(self, *, grad_accum_steps: int) -> None:
        if (
            self.global_step < 0
            or self.consumed_batches < 0
            or self.consumed_rollout_batches < 0
            or self.epoch < 0
        ):
            raise ValueError(f"TrainState counters must be non-negative: {self}")
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        minimum = self.global_step * grad_accum_steps
        if self.consumed_batches < minimum:
            raise ValueError(
                f"consumed_batches={self.consumed_batches} is below the minimum implied by "
                f"global_step={self.global_step} and grad_accum_steps={grad_accum_steps} "
                f"({minimum}). The checkpoint is inconsistent; resuming would replay data."
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "global_step": self.global_step,
            "consumed_batches": self.consumed_batches,
            "consumed_rollout_batches": self.consumed_rollout_batches,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = state.get("version")
        if version not in (1, STATE_VERSION):
            raise ValueError(
                f"unsupported train state version {version!r} (expected {STATE_VERSION}). "
                "Write an explicit migration rather than guessing at missing keys."
            )
        required = {"global_step", "consumed_batches", "epoch"}
        if version == STATE_VERSION:
            required = required | {"consumed_rollout_batches"}
        missing = required - state.keys()
        if missing:
            raise ValueError(f"train state is missing keys: {sorted(missing)}")
        self.global_step = int(state["global_step"])
        self.consumed_batches = int(state["consumed_batches"])
        self.epoch = int(state["epoch"])
        # Version 1 predates RL. Its runs were SFT by construction, so the RL counter is zero --
        # an explicit migration, per this module's own rule, rather than a defaulted lookup that
        # would also silently accept a truncated version-2 payload.
        self.consumed_rollout_batches = int(state.get("consumed_rollout_batches", 0))

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> TrainState:
        instance = cls()
        instance.load_state_dict(state)
        return instance


@dataclass(kw_only=True, slots=True)
class StepMetrics:
    """Per-step scalars produced by the loop, not by ``step_fn``.

    ``step_fn`` returns a loss and nothing else; everything here is measured by the
    loop, so every experiment logs the same quantities in the same way.
    """

    loss: float = 0.0
    lr: float = 0.0
    grad_norm: float | None = None
    step_time: float = 0.0
    peak_memory_gib: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {
            "loss": self.loss,
            "lr": self.lr,
            "step_time": self.step_time,
        }
        if self.grad_norm is not None:
            out["grad_norm"] = self.grad_norm
        if self.peak_memory_gib is not None:
            out["peak_memory"] = self.peak_memory_gib
        out.update(self.extra)
        return out


__all__ = ["RESUMABLE", "STATE_VERSION", "ResumableComponent", "StepMetrics", "TrainState"]
