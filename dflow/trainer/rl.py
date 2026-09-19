"""``fit_rl()`` — the Flow-GRPO loop.

A separate loop file, not a flag on ``fit()``. `SKILL.md` settles that: *loop shape changed → new
trainer*, and RL samples inside the step and needs rewards. A boolean that switched loop structure
would be the smell that says it is really two loops.

What it shares with ``fit()`` is ``TrainState`` and the discipline: it owns only the things that
**fail silently when written wrong**, and everything model- or task-specific arrives as a closure
built in ``experiments/``.

## The three closures, and why not one

An RL step has three phases with different needs, so ``step_fn``'s single closure splits three ways:

* ``rollout_fn`` runs under ``no_grad`` and needs the VAE, the text encoder and the SDE schedule;
* ``reward_fn`` needs a decode to pixels and a scorer;
* ``policy_step_fn`` needs the family adapter and the objective, and is the direct analogue of
  ``step_fn`` — it is where the PPO loss is computed.

``policy_step_fn`` returns metrics alongside its loss, because unlike SFT the loss here says almost
nothing: what tells you an RL run is healthy is ``clipfrac`` low but non-zero and ``ppo_kl`` small.

## What this file owns

1. **Phase order.** Rollout, then reward for the *whole* batch, then advantage, then the inner
   epochs. Normalising before every reward is in computes a baseline over a partial group.
2. **The advantage call.** Rank-local and exact under the whole-group layout; see
   ``rl/advantage.py`` for why ``global_std`` is the only path that communicates.
3. **Inner-epoch minibatching.** Where PPO's sample efficiency comes from, and the only reason the
   ratio ever leaves 1. A reshuffle that crossed group boundaries, or an accumulation that forgot
   to divide, changes the effective learning rate without raising.
4. **Metric reduction across ranks.** ``fit()`` all-reduces its loss; RL adds reward statistics,
   advantage statistics, clip fractions and PPO-KL, and every one of them reports only its own
   shard if the reduction is forgotten.
5. **`consumed_rollout_batches`.** RL's basis for data resume — ``consumed_batches`` runs ahead of
   the prompt sampler by ``inner_epochs * grad_accum_steps``. See ``trainer/state.py``.
6. The rest of ``fit()``'s list, for the same reasons: the gradient-sync flag, ``loss /
   grad_accum``, CP reduction before clipping, non-finite checks, atomic checkpointing.

## Prompts per step must be at least ``dp_size``

The data-parallel dimension distributes prompts, not group members, so a rank with no prompt has
nothing to roll out and contributes no gradient while still participating in every collective. That
is checked at setup rather than discovered as a hang.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from dflow.common.logger import TrainLogger
from dflow.config.rl import GroupConfig, PPOConfig
from dflow.rl.advantage import group_advantage
from dflow.rl.trajectory import RolloutGroup
from dflow.runtime.context import MeshBundle
from dflow.runtime.cp import reduce_cp_gradients
from dflow.runtime.precision import autocast
from dflow.trainer.state import StepMetrics, TrainState

if TYPE_CHECKING:  # pragma: no cover
    from dflow.checkpoint.manager import CheckpointManager

#: ``(mesh, model, batch) -> [RolloutGroup]``, one per prompt in the batch. Closes over the VAE,
#: the text encoder, the family and the SDE schedule.
RolloutFn = Callable[[MeshBundle, nn.Module, Any], Sequence[RolloutGroup]]

#: ``(groups, batch) -> (P, G)`` rewards, aligned with ``groups`` and their trajectory order.
#: Closes over the scorer, and over whatever decodes latents to pixels.
RewardFn = Callable[[Sequence[RolloutGroup], Any], torch.Tensor]

#: ``(mesh, model, group, advantages) -> (loss, metrics)``. The analogue of ``step_fn``: replays the
#: window under current weights and returns the clipped objective. Closes over the family, the
#: conditioning builder and the ``PPOConfig``.
PolicyStepFn = Callable[
    [MeshBundle, nn.Module, RolloutGroup, torch.Tensor], tuple[torch.Tensor, dict[str, float]]
]


def _as_scalar(loss: torch.Tensor, step: int) -> torch.Tensor:
    if loss.ndim != 0:
        if loss.numel() != 1:
            raise ValueError(
                f"policy_step_fn must return a scalar loss, got shape {tuple(loss.shape)}. Reduce "
                f"it inside the closure so the reduction is visible where the loss is defined."
            )
        loss = loss.reshape(())
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(
            f"non-finite loss at step {step}. Stopping rather than letting it propagate into "
            f"every parameter through the optimizer."
        )
    return loss


def _reduce_metrics(values: dict[str, float], mesh: MeshBundle) -> dict[str, float]:
    """Average scalar metrics across ranks. Un-reduced, each rank reports only its own shard."""
    if not (dist.is_available() and dist.is_initialized()) or not values:
        return values
    keys = sorted(values)
    packed = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=mesh.device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= mesh.world_size
    return dict(zip(keys, packed.tolist(), strict=True))


def fit_rl(
    *,
    mesh: MeshBundle,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    dataloader: Any,
    checkpoints: CheckpointManager,
    logger: TrainLogger,
    generator: torch.Generator,
    rollout_fn: RolloutFn,
    reward_fn: RewardFn,
    policy_step_fn: PolicyStepFn,
    steps: int,
    group: GroupConfig,
    ppo: PPOConfig,
    grad_accum_steps: int = 1,
    autocast_dtype: str | None = "bfloat16",
    max_grad_norm: float | None = 1.0,
    lora: bool = False,
) -> TrainState:
    """Run Flow-GRPO to ``steps`` optimizer updates, resuming if configured."""
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
            ema=None,
            generator=generator,
            lora=lora,
        )
        state.validate(grad_accum_steps=grad_accum_steps)
        logger.info(f"resumed from {resume_from} at step {state.global_step}")

    # consumed_rollout_batches, not consumed_batches: one rollout draw feeds
    # inner_epochs * grad_accum micro-batches, so the latter runs far ahead of the prompt sampler.
    sampler = getattr(dataloader, "batch_sampler", None)
    if sampler is not None and hasattr(sampler, "set_step"):
        sampler.set_step(state.consumed_rollout_batches)
    batches: Iterator[Any] = iter(dataloader)

    last_saved = state.global_step
    while state.global_step < steps:
        started = time.perf_counter()
        if mesh.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(mesh.device)

        batch = next(batches)
        state.on_rollout_batch()

        # ---- rollout: no gradient, one group per prompt -----------------------------------
        #
        # Under the **same** autocast as the policy step below, and that is load-bearing rather
        # than tidy. Autocast decides which ops run in which precision, so a rollout outside it and
        # a replay inside it are two numerically different paths through the same weights — which
        # reintroduces exactly the rollout/replay mismatch this design exists to avoid, and does it
        # invisibly. Measured on klein-4B before this line was added: with `inner_epochs=1`, where
        # the policy cannot have moved and the ratio must be 1, `ratio_std` reached 1.1e-3 and
        # `clipfrac` reached 0.5 — the clip firing on nothing but numerical noise.
        rollout_started = time.perf_counter()
        with autocast(mesh.device, autocast_dtype):
            groups = list(rollout_fn(mesh, model, batch))
        if not groups:
            raise ValueError(
                f"rollout_fn returned no groups. Every rank needs at least one prompt: this rank "
                f"would contribute no gradient while still joining every collective. Raise the "
                f"prompt batch size to at least dp_size={mesh.dp_size}."
            )
        for rolled in groups:
            if rolled.group_size != group.size:
                raise ValueError(
                    f"rollout produced {rolled.group_size} trajectories for prompt "
                    f"{rolled.prompt_index} but group.size is {group.size}"
                )
        rollout_seconds = time.perf_counter() - rollout_started

        # ---- reward: the whole batch before any normalisation -----------------------------
        reward_started = time.perf_counter()
        rewards = reward_fn(groups, batch)
        if rewards.shape != (len(groups), group.size):
            raise ValueError(
                f"reward_fn returned {tuple(rewards.shape)}, expected "
                f"({len(groups)}, {group.size}) — one scalar per trajectory, grouped by prompt"
            )
        reward_seconds = time.perf_counter() - reward_started

        # global_std reduces over the default (world) group. CP ranks hold *identical* rewards,
        # so including them would count each reward cp_size times and skew the Bessel correction.
        # Refused rather than approximated: CP is a no-op at degree 1 today, and the DP-only
        # subgroup this would need does not exist as a mesh dimension yet.
        if group.global_std and mesh.cp_size > 1:
            raise NotImplementedError(
                f"group.global_std with cp_size={mesh.cp_size} would count every reward "
                f"{mesh.cp_size} times, since CP ranks share a prompt and therefore a reward. "
                f"Use within-group normalisation, or add a DP-only process group first."
            )
        advantages, advantage_stats = group_advantage(rewards.to(mesh.device), group)

        # ---- inner epochs: several updates from one rollout ------------------------------
        metrics: dict[str, float] = {}
        loss_total = torch.zeros((), device=mesh.device, dtype=torch.float32)
        grad_norm: float | None = None
        updates = 0

        for _ in range(ppo.inner_epochs):
            order = torch.randperm(
                len(groups), generator=generator, device=generator.device
            ).tolist()
            for chunk_start in range(0, len(order), grad_accum_steps):
                chunk = order[chunk_start : chunk_start + grad_accum_steps]
                optimizer.zero_grad(set_to_none=True)
                accumulated = torch.zeros((), device=mesh.device, dtype=torch.float32)

                for position, index in enumerate(chunk):
                    is_last = position == len(chunk) - 1
                    if hasattr(model, "set_requires_gradient_sync"):
                        model.set_requires_gradient_sync(is_last)
                    state.on_micro_batch()

                    with autocast(mesh.device, autocast_dtype):
                        loss, step_metrics = policy_step_fn(
                            mesh, model, groups[index], advantages[index]
                        )
                        loss = _as_scalar(loss, state.global_step)
                    (loss / len(chunk)).backward()
                    accumulated += loss.detach().float()
                    for key, value in step_metrics.items():
                        metrics[key] = metrics.get(key, 0.0) + value

                reduce_cp_gradients(mesh, model)

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
                loss_total += accumulated / len(chunk)
                updates += 1

        loss_total /= max(updates, 1)
        metrics = {key: value / max(updates, 1) for key, value in metrics.items()}

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
            loss_total /= mesh.world_size
        metrics = _reduce_metrics({**metrics, **advantage_stats.as_dict()}, mesh)
        metrics["perf/rollout_time"] = rollout_seconds
        metrics["perf/reward_time"] = reward_seconds
        metrics["ppo/updates_per_rollout"] = float(updates)

        logger.log_metrics(
            StepMetrics(
                loss=float(loss_total),
                lr=float(lr_scheduler.get_last_lr()[0]),
                grad_norm=grad_norm,
                step_time=time.perf_counter() - started,
                peak_memory_gib=(
                    torch.cuda.max_memory_allocated(mesh.device) / 1024**3
                    if mesh.device.type == "cuda"
                    else None
                ),
                extra=metrics,
            ).as_dict(),
            step=state.global_step,
        )

        should_save = (
            checkpoints.config.save_enabled
            and state.global_step != last_saved
            and state.global_step % checkpoints.config.interval == 0
        )
        if should_save:
            path = checkpoints.save(
                state=state,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=None,
                generator=generator,
                lora=lora,
            )
            last_saved = state.global_step
            logger.info(f"saved {path}")

    if checkpoints.config.save_enabled and state.global_step != last_saved:
        path = checkpoints.save(
            state=state,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=None,
            generator=generator,
            lora=lora,
        )
        logger.info(f"saved final {path}")

    return state


__all__ = ["PolicyStepFn", "RewardFn", "RolloutFn", "fit_rl"]
