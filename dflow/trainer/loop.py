"""``fit()`` — the SFT loop.

It owns only what **fails silently when written wrong**. Everything task- or model-specific is in
``step_fn``, which the caller builds; the loop never reaches back into ``models/``, ``tasks/``,
``losses`` or ``encoders/`` (enforced by ``tools/checks/check_layering.py``). ``step_fn`` closes over the
encoders and family it needs, so its signature stays ``(mesh, model, batch) -> loss`` and this file
stays ignorant of L3.

What is here and why each item earns its place:

* **gradient-sync control** — with FSDP2, syncing on every micro-batch is merely slow, but getting
  the flag inverted makes gradients wrong;
* **``loss / grad_accum``** — forgetting it multiplies the effective learning rate by the
  accumulation count;
* **CP gradient reduction before clipping** — the other order leaves the clip threshold meaningless
  (currently a no-op at ``cp == 1``, called anyway so the position is already fixed);
* **finite-loss checks** — a NaN that reaches the optimizer poisons every parameter, and the run
  keeps going;
* **metric all-reduce** — un-reduced loss makes each rank report only its own shard;
* **atomic checkpointing and exact resume** — see ``checkpoint/manager.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from dflow.common.ema import EMA
from dflow.common.logger import TrainLogger
from dflow.runtime.context import MeshBundle
from dflow.runtime.cp import reduce_cp_gradients
from dflow.runtime.precision import autocast
from dflow.trainer.state import StepMetrics, TrainState

if TYPE_CHECKING:  # pragma: no cover
    # Type-only: fit() merely calls save/load/resolve_resume, so importing the concrete manager at
    # runtime would create a cycle (checkpoint -> trainer.state -> trainer -> loop -> checkpoint)
    # and needlessly bind the loop to one implementation.
    from dflow.checkpoint.manager import CheckpointManager

#: ``(mesh, model, batch) -> scalar loss``. Everything else it needs, it closes over.
StepFn = Callable[[MeshBundle, nn.Module, Any], torch.Tensor]

#: ``(model, step, path) -> None``, called after a checkpoint is written. Same contract as
#: ``step_fn``: the loop knows nothing about what it does, and everything it needs — a VAE, a text
#: encoder, a family adapter — is captured in a closure built in ``experiments/``. That is what lets
#: an in-training preview exist without ``loop.py`` importing ``models/`` or ``encoders/``.
#:
#: Failures here are logged and swallowed. A preview is a convenience; losing a twelve-hour run
#: because a rendering helper raised is not a trade worth making.
CheckpointHook = Callable[[nn.Module, int, str], None]


def _as_scalar(loss: torch.Tensor, step: int) -> torch.Tensor:
    if loss.ndim != 0:
        if loss.numel() != 1:
            raise ValueError(
                f"step_fn must return a scalar loss, got shape {tuple(loss.shape)}. Reduce it "
                f"inside step_fn so the reduction is visible where the loss is defined."
            )
        loss = loss.reshape(())
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(
            f"non-finite loss at step {step}. Stopping rather than letting it propagate into "
            f"every parameter through the optimizer."
        )
    return loss


def fit(
    *,
    mesh: MeshBundle,
    step_fn: StepFn,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    dataloader: Any,
    checkpoints: CheckpointManager,
    logger: TrainLogger,
    generator: torch.Generator,
    steps: int,
    grad_accum_steps: int = 1,
    autocast_dtype: str | None = "bfloat16",
    max_grad_norm: float | None = 1.0,
    ema: EMA | None = None,
    lora: bool = False,
    on_checkpoint: CheckpointHook | None = None,
) -> TrainState:
    """Run training to ``steps`` optimizer updates, resuming if configured."""
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

    # The sampler is positioned from consumed_batches, never from its own saved cursor: with
    # num_workers > 0 the loader prefetches, so the sampler runs ahead of what the model consumed.
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

        for micro in range(grad_accum_steps):
            is_last = micro == grad_accum_steps - 1
            if hasattr(model, "set_requires_gradient_sync"):
                model.set_requires_gradient_sync(is_last)

            batch = next(batches)
            state.on_micro_batch()

            with autocast(mesh.device, autocast_dtype):
                loss = _as_scalar(step_fn(mesh, model, batch), state.global_step)
            (loss / grad_accum_steps).backward()
            loss_total += loss.detach().float()

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

        if dist.is_initialized():
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

        # Preview on its own cadence, decoupled from the save interval (see fit_distill): render
        # more often than you commit weights. Falls back to the save interval when unset, so the
        # default rides each save. The hook ignores the path; coincident save+preview render once.
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


def _run_checkpoint_hook(
    hook: CheckpointHook | None,
    model: nn.Module,
    step: int,
    path: str,
    logger: TrainLogger,
) -> None:
    """Run the post-checkpoint hook, restoring training mode whatever it does.

    ``model.eval()`` inside a preview would otherwise leak into the next step and silently change
    dropout and normalisation behaviour for the rest of the run — the class of bug this loop exists
    to prevent, so the restore is here rather than left to the caller.
    """
    if hook is None:
        return
    # A preview allocates on top of a training step's peak, and the margin can be under a GiB:
    # grounding nine references lengthens the text stream from ~173 to ~1150 tokens and took one
    # run to 78.7 GiB of 79.18. Releasing the allocator's cached-but-unused blocks first costs a
    # few milliseconds once per checkpoint and is the difference between a preview and an
    # `OutOfMemoryError` swallowed into a warning.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    was_training = model.training
    try:
        hook(model, step, path)
    except Exception as error:  # noqa: BLE001 - a preview must never end a run
        logger.warning(f"checkpoint hook failed at step {step}: {type(error).__name__}: {error}")
    finally:
        model.train(was_training)


__all__ = ["CheckpointHook", "StepFn", "fit"]
