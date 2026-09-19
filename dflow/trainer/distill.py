"""``fit_distill()`` — the on-policy self-distillation (D-OPSD) loop, memory-flat in the rollout.

A separate loop file, not a flag on ``fit()``. ``SKILL.md`` settles that: *loop shape changed → new
trainer*, and distillation rolls out a few-step trajectory inside the step. What it shares with
``fit()`` is ``TrainState`` and the discipline — it owns only the things that **fail silently when
written wrong**, and everything model- or task-specific arrives as a closure built in
``experiments/``.

## Why this exists alongside ``fit()``

D-OPSD can run on plain ``fit()``: the ``step_fn`` accumulates the K per-rollout-step losses into one
scalar and ``fit`` does a single ``backward()``. Correct, but that single backward keeps **all K
forward graphs alive at once** — K× the activation memory. At try-on sequence lengths (a ~768² target
plus several 384² references, run K times) that is what pushes a single card over.

Because the rollout states are **detached between steps** (D-OPSD is on-policy but not
back-prop-through-sampling), the K loss terms are independent subgraphs. So they can be backwarded
**one at a time**, each graph freed before the next forward — gradients accumulate into ``.grad``
exactly as a single backward of their sum would, but peak memory is O(1) in K, not O(K).

``fit_distill`` expresses that by taking the rollout as a **generator** (``rollout_step_fn``) that
*yields* one scalar loss per visited state; the loop backwards each yield as it arrives. The rollout
logic — encode, teacher forward under ``no_grad``, the Euler step, the x0/velocity loss — stays in the
experiment's closure, exactly as ``step_fn`` does for ``fit()``.

## What this file owns (the ``fit()`` list, adapted)

1. **Per-step backward with graphs freed** — the memory contract above; the loop, not the closure,
   calls ``backward`` so the freeing is guaranteed.
2. **``loss / grad_accum``**, and the closure yielding each term already divided by K — forgetting
   either multiplies the effective learning rate.
3. **CP gradient reduction before clipping** — the other order leaves the clip threshold meaningless.
4. **Non-finite checks** on every yielded loss before it reaches the optimizer.
5. **Metric all-reduce** so each rank does not report only its own shard.
6. **The EMA teacher update and its persistence** — ``ema.step`` after the optimizer step, and the
   teacher adapter round-tripped through the checkpoint's ``ema`` slot (it is frozen, so the model
   shards would not save it).
7. **Atomic checkpointing and exact resume** via ``consumed_batches``.

## Gradient sync under FSDP

Each yielded loss is backwarded separately, so with FSDP2 every backward reduce-scatters its own
contribution and accumulates into ``.grad`` — correct (reduce-scatter is linear), at K× the
communication of a single backward. On a single card, where ``fully_shard`` is a no-op, there is no
communication and the flag is inert. Sync is therefore left enabled on every backward rather than
deferred to a notional "last" one, which a lazy generator cannot identify without holding a graph.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from dflow.common.ema import LoRATeacherEMA
from dflow.common.logger import TrainLogger
from dflow.runtime.context import MeshBundle
from dflow.runtime.cp import reduce_cp_gradients
from dflow.runtime.precision import autocast
from dflow.trainer.loop import CheckpointHook, _run_checkpoint_hook
from dflow.trainer.state import StepMetrics, TrainState

if TYPE_CHECKING:  # pragma: no cover
    from dflow.checkpoint.manager import CheckpointManager

#: ``(mesh, model, batch) -> Iterator[scalar loss]``. A D-OPSD rollout as a generator: it yields one
#: scalar loss per visited state, **already divided by the rollout length K** so the yields sum to the
#: step's total loss. Yielding (rather than returning the sum) is what lets ``fit_distill`` free each
#: forward graph before the next — memory stays flat in K. Everything else it needs — the VAE, the
#: text encoder, the family, the generator, the schedule — it closes over, exactly like ``fit``'s
#: ``step_fn``. Between yields the closure must detach the trajectory state, so the graphs are
#: genuinely independent.
RolloutStepFn = Callable[[MeshBundle, nn.Module, Any], Iterator[torch.Tensor]]


def _as_scalar(loss: torch.Tensor, step: int) -> torch.Tensor:
    if loss.ndim != 0:
        if loss.numel() != 1:
            raise ValueError(
                f"rollout_step_fn must yield scalar losses, got shape {tuple(loss.shape)}. Reduce "
                f"each term inside the closure so the reduction is visible where the loss is defined."
            )
        loss = loss.reshape(())
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(
            f"non-finite loss at step {step}. Stopping rather than letting it propagate into every "
            f"parameter through the optimizer."
        )
    return loss


def fit_distill(
    *,
    mesh: MeshBundle,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    dataloader: Any,
    checkpoints: CheckpointManager,
    logger: TrainLogger,
    generator: torch.Generator,
    rollout_step_fn: RolloutStepFn,
    steps: int,
    grad_accum_steps: int = 1,
    autocast_dtype: str | None = "bfloat16",
    max_grad_norm: float | None = 1.0,
    ema: LoRATeacherEMA | None = None,
    lora: bool = True,
    on_checkpoint: CheckpointHook | None = None,
) -> TrainState:
    """Run D-OPSD to ``steps`` optimizer updates, resuming if configured.

    ``ema`` is the EMA teacher (``LoRATeacherEMA``): stepped after each optimizer update and persisted
    through the checkpoint. ``None`` means a frozen-base teacher, whose forward the closure produces
    with ``model.disable_adapter()`` — nothing for this loop to update.
    """
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    state = TrainState()
    resume_from = checkpoints.resolve_resume()
    if resume_from is not None:
        state = checkpoints.load(
            resume_from,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            generator=generator,
            lora=lora,
        )
        state.validate(grad_accum_steps=grad_accum_steps)
        logger.info(f"resumed from {resume_from} at step {state.global_step}")

    # Positioned from consumed_batches, never the sampler's own cursor: num_workers > 0 prefetches,
    # so the sampler runs ahead of what the model consumed. Same contract as fit().
    sampler = getattr(dataloader, "batch_sampler", None)
    if sampler is not None and hasattr(sampler, "set_step"):
        sampler.set_step(state.consumed_batches)
    batches: Iterator[Any] = iter(dataloader)

    last_saved = state.global_step
    last_previewed = state.global_step
    while state.global_step < steps:
        started = time.perf_counter()
        if mesh.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(mesh.device)

        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=mesh.device, dtype=torch.float32)
        rollout_steps = 0

        for _ in range(grad_accum_steps):
            # Every yielded backward reduce-scatters under FSDP (correct, K x comm; inert on one
            # card). Enabled rather than deferred: a lazy generator cannot mark its last yield
            # without holding that graph, which is the memory this loop exists to avoid.
            if hasattr(model, "set_requires_gradient_sync"):
                model.set_requires_gradient_sync(True)

            batch = next(batches)
            state.on_micro_batch()

            produced = 0
            with autocast(mesh.device, autocast_dtype):
                for loss_term in rollout_step_fn(mesh, model, batch):
                    loss_term = _as_scalar(loss_term, state.global_step)
                    # Backward here, in the loop, so the term's graph is freed before the closure
                    # computes the next state. The yields already sum to the step's loss, so this
                    # divides only by grad_accum — dividing by K is the closure's job.
                    (loss_term / grad_accum_steps).backward()
                    loss_total += loss_term.detach().float()
                    produced += 1
            if produced == 0:
                raise ValueError(
                    "rollout_step_fn yielded no losses — a rollout must visit at least one state. "
                    "Check that num_steps >= 1 and the generator is not empty."
                )
            rollout_steps += produced

        loss_total /= grad_accum_steps

        reduce_cp_gradients(mesh, model)

        grad_norm: float | None = None
        if max_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            )
            if isinstance(norm, DTensor):
                norm = norm.full_tensor()
            grad_norm = float(norm.detach())

        optimizer.step()
        lr_scheduler.step()
        state.on_optimizer_step()
        if ema is not None:
            ema.step(model, global_step=state.global_step)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
            loss_total /= mesh.world_size

        metrics = StepMetrics(
            loss=float(loss_total),
            lr=float(lr_scheduler.get_last_lr()[0]),
            grad_norm=grad_norm,
            step_time=time.perf_counter() - started,
            peak_memory_gib=(
                torch.cuda.max_memory_allocated(mesh.device) / 1024**3
                if mesh.device.type == "cuda"
                else None
            ),
            extra={"rollout/backwards_per_step": float(rollout_steps)},
        )
        logger.log_metrics(metrics.as_dict(), step=state.global_step)

        should_save = (
            checkpoints.config.save_enabled
            and state.global_step != last_saved
            and state.global_step % checkpoints.config.interval == 0
        )
        saved_path = None
        if should_save:
            saved_path = checkpoints.save(
                state=state,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                generator=generator,
                lora=lora,
            )
            last_saved = state.global_step
            logger.info(f"saved {saved_path}")

        # Preview on its own cadence, decoupled from the save interval: preview_interval lets you
        # watch training more often than you commit weights (e.g. preview every 100, save every
        # 200). Falls back to the save interval when unset, so the default rides each save exactly
        # as before. The hook ignores the checkpoint path, so a preview-only step passes the last
        # saved path (or None) harmlessly, and coincident save+preview steps render once.
        preview_interval = checkpoints.config.preview_interval or checkpoints.config.interval
        should_preview = (
            on_checkpoint is not None
            and state.global_step != last_previewed
            and state.global_step % preview_interval == 0
        )
        if should_preview:
            _run_checkpoint_hook(on_checkpoint, model, state.global_step, saved_path, logger)
            last_previewed = state.global_step

    if checkpoints.config.save_enabled and state.global_step != last_saved:
        path = checkpoints.save(
            state=state,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            generator=generator,
            lora=lora,
        )
        logger.info(f"saved final {path}")
        if on_checkpoint is not None and state.global_step != last_previewed:
            _run_checkpoint_hook(on_checkpoint, model, state.global_step, path, logger)
            last_previewed = state.global_step

    return state


__all__ = ["RolloutStepFn", "fit_distill"]
