"""``fit_opsd_nft()`` — the fused OPSD-NFT (DiffusionNFT) loop.

A separate loop file, not a flag on ``fit_rl`` or ``fit_distill``. ``SKILL.md`` settles that: *loop
shape changed → new trainer*. DiffusionNFT shares Flow-GRPO's outer shape (roll out, reward the
whole batch, compute a group-relative signal, then update) but its inner update is a different
algorithm — no PPO ratio, no SDE window, no log-probability — and it maintains a frozen ``old``
policy that this loop refreshes, which ``fit_rl`` does not. A boolean that switched between the two
would be the smell that says it is really two loops, exactly the trap the design rejects.

What it shares with the other trainers is ``TrainState`` and the discipline: it owns only the things
that **fail silently when written wrong**, and everything model-, task- and reward-specific arrives
as a closure built in ``experiments/``. So this file imports from ``dflow.rl``, ``dflow.config``,
``dflow.runtime``, ``dflow.common`` and ``dflow.trainer.{state,loop}`` — never ``models/``,
``tasks/``, ``encoders/``, ``losses`` or ``rewards``. The model, VAE, text encoder, family and
reward all arrive through the three closures.

## The three closures, and why not one

Like Flow-GRPO, an OPSD-NFT step has three phases with different needs:

* ``rollout_fn`` runs under ``no_grad`` with the **frozen ``old`` adapter** and produces one
  :class:`NFTRolloutGroup` per prompt — a group of *clean* target latents from a deterministic
  few-step rollout. It closes over the VAE, the text encoder, the family and the schedule.
* ``reward_fn`` decodes those clean latents to pixels and scores them, returning ``(P, G)``. It
  closes over the reward and the decode. This is where reference fidelity enters.
* ``nft_step_fn`` does the update for one trajectory: re-noise its clean latent at a sampled ``t``,
  run **three** forwards — trainable (``default``, grad), frozen ``old`` (no grad), and the frozen
  reference (base model via ``disable_adapter``, no grad) — slice the target span, and call
  ``diffusion_nft_loss``. It closes over the family, the conditioning and the ``DiffusionNFTConfig``.
  It returns ``(loss, metrics)`` because, as in RL, the loss says little on its own.

## What this file owns (the ``fit_rl`` list, adapted)

1. **Phase order.** Rollout, then reward for the *whole* batch, then advantage, then the update.
   Normalising before every reward is in computes a baseline over a partial group.
2. **The advantage call and the reward-prob map.** ``group_advantage`` centres each group on its own
   mean (that is what makes it GRPO); ``nft_reward_prob`` turns the advantage into the optimality
   probability the objective weights branches by. Both under the whole-group-on-one-rank layout.
3. **``loss / grad_accum`` and gradient-sync control.** Forgetting the divide multiplies the
   effective learning rate; an inverted sync flag makes gradients wrong under FSDP2.
4. **CP gradient reduction before clipping.** The other order leaves the clip threshold meaningless.
5. **Non-finite checks** on every micro loss before it reaches the optimizer.
6. **Metric all-reduce** so each rank does not report only its own shard.
7. **The ``old`` policy refresh.** ``ema.step`` after the optimizer step refreshes ``old`` from the
   trainable ``default`` — a hard copy at ``old_policy_decay == 0`` (fully on-policy), a slow EMA
   otherwise. The ``old`` adapter is frozen, so a LoRA checkpoint does not save it in the model
   shards; the manager round-trips it through the ``ema`` slot, exactly as ``fit_distill`` does for
   its teacher.
8. **``consumed_rollout_batches``** as the basis for data resume, since ``consumed_batches`` runs
   ahead of the prompt sampler by ``group_size * ceil(timestep_fraction * num_train_timesteps)``.
9. **Atomic checkpointing** and the post-checkpoint hook.

## Prompts per step must be at least ``dp_size``

The data-parallel dimension distributes prompts, not group members, so a rank with no prompt has
nothing to roll out and contributes no gradient while still joining every collective. Checked at
setup in the experiment rather than discovered as a hang.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from dflow.common.ema import LoRATeacherEMA
from dflow.common.logger import TrainLogger
from dflow.config.rl import DiffusionNFTConfig, GroupConfig
from dflow.rl.advantage import group_advantage
from dflow.rl.objective import nft_reward_prob
from dflow.rl.trajectory import NFTRolloutGroup
from dflow.runtime.context import MeshBundle
from dflow.runtime.cp import reduce_cp_gradients
from dflow.runtime.precision import autocast
from dflow.trainer.loop import CheckpointHook, _run_checkpoint_hook
from dflow.trainer.state import StepMetrics, TrainState

if TYPE_CHECKING:  # pragma: no cover
    from dflow.checkpoint.manager import CheckpointManager

#: ``(mesh, model, batch) -> [NFTRolloutGroup]``, one per prompt. Runs under ``no_grad`` with the
#: frozen ``old`` adapter and returns clean target latents. Closes over the VAE, the text encoder,
#: the family and the few-step schedule.
NFTRolloutFn = Callable[[MeshBundle, nn.Module, Any], Sequence[NFTRolloutGroup]]

#: ``(groups, batch) -> (P, G)`` rewards, aligned with ``groups`` and their trajectory order. Closes
#: over the reward and whatever decodes latents to pixels.
NFTRewardFn = Callable[[Sequence[NFTRolloutGroup], Any], torch.Tensor]

#: ``(mesh, model, sample, reward_prob) -> (loss, metrics)``. The DiffusionNFT update for one
#: trajectory: re-noise its clean latent at a sampled ``t``, run the trainable / frozen-old /
#: reference forwards, and call ``diffusion_nft_loss``. ``sample`` is one :class:`NFTSample`;
#: ``reward_prob`` is that trajectory's scalar optimality probability. Closes over the family, the
#: conditioning and the ``DiffusionNFTConfig``.
NFTStepFn = Callable[
    [MeshBundle, nn.Module, "NFTSample", torch.Tensor], tuple[torch.Tensor, dict[str, float]]
]


class NFTSample:
    """One trajectory handed to ``nft_step_fn``: its clean latent and how to find its conditioning.

    A plain carrier, not a dataclass with a fixed schema, because *what* conditioning a trajectory
    needs is the closure's business (which prompt, which references, which text embeddings) and the
    loop must not know it — the same reason ``policy_step_fn`` in ``fit_rl`` is handed a
    ``RolloutGroup`` and reads back its own ``context[prompt_index]``. The loop fills exactly two
    fields; the closure reads them plus whatever it kept keyed by ``prompt_index``.
    """

    __slots__ = ("prompt_index", "member_index", "clean_latent")

    def __init__(self, *, prompt_index: int, member_index: int, clean_latent: torch.Tensor) -> None:
        #: The prompt this trajectory came from — the key into the closure's per-prompt conditioning.
        self.prompt_index = prompt_index
        #: Which member of the group (0..G-1). Informational; lets the closure pick per-member noise.
        self.member_index = member_index
        #: ``(target_len, C)`` — this trajectory's clean target latent, detached. The closure
        #: re-noises it at a sampled ``t`` and regresses the trainable policy toward it.
        self.clean_latent = clean_latent


def _as_scalar(loss: torch.Tensor, step: int) -> torch.Tensor:
    if loss.ndim != 0:
        if loss.numel() != 1:
            raise ValueError(
                f"nft_step_fn must return a scalar loss, got shape {tuple(loss.shape)}. Reduce it "
                f"inside the closure so the reduction is visible where the loss is defined."
            )
        loss = loss.reshape(())
    if not torch.isfinite(loss.detach()):
        raise FloatingPointError(
            f"non-finite loss at step {step}. Stopping rather than letting it propagate into every "
            f"parameter through the optimizer."
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


def fit_opsd_nft(
    *,
    mesh: MeshBundle,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    dataloader: Any,
    checkpoints: CheckpointManager,
    logger: TrainLogger,
    generator: torch.Generator,
    rollout_fn: NFTRolloutFn,
    reward_fn: NFTRewardFn,
    nft_step_fn: NFTStepFn,
    ema: LoRATeacherEMA,
    steps: int,
    group: GroupConfig,
    nft: DiffusionNFTConfig,
    grad_accum_steps: int = 1,
    autocast_dtype: str | None = "bfloat16",
    max_grad_norm: float | None = 1.0,
    lora: bool = True,
    on_checkpoint: CheckpointHook | None = None,
) -> TrainState:
    """Run OPSD-NFT to ``steps`` optimizer updates, resuming if configured.

    ``ema`` is the ``old``-policy refresher (:class:`LoRATeacherEMA`, ``src="default"``,
    ``dst="old"``): stepped after each optimizer update and persisted through the checkpoint's ``ema``
    slot, which is how the frozen ``old`` adapter survives a resume. At ``nft.old_policy_decay == 0``
    it is a hard copy — the rollout policy becomes the last trained policy, fully on-policy.
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

    # consumed_rollout_batches, not consumed_batches: one rollout draw feeds
    # group_size * timesteps_per_sample micro-batches, so the latter runs far ahead of the prompt
    # sampler. Same contract as fit_rl.
    sampler = getattr(dataloader, "batch_sampler", None)
    if sampler is not None and hasattr(sampler, "set_step"):
        sampler.set_step(state.consumed_rollout_batches)
    batches: Iterator[Any] = iter(dataloader)

    timesteps_per_sample = max(1, math.ceil(nft.timestep_fraction * nft.num_train_timesteps))

    last_saved = state.global_step
    while state.global_step < steps:
        started = time.perf_counter()
        if mesh.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(mesh.device)

        batch = next(batches)
        state.on_rollout_batch()

        # ---- rollout: no gradient, frozen `old` adapter, one group of clean latents per prompt ----
        #
        # Under the same autocast as the update below, and load-bearing rather than tidy for the same
        # reason fit_rl states it: autocast decides op precision, so a rollout outside it and a
        # re-noised forward inside it are two numerically different paths through the same weights.
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

        # ---- reward: the whole batch before any normalisation --------------------------------
        reward_started = time.perf_counter()
        rewards = reward_fn(groups, batch)
        if rewards.shape != (len(groups), group.size):
            raise ValueError(
                f"reward_fn returned {tuple(rewards.shape)}, expected "
                f"({len(groups)}, {group.size}) — one scalar per trajectory, grouped by prompt"
            )
        reward_seconds = time.perf_counter() - reward_started

        if group.global_std and mesh.cp_size > 1:
            raise NotImplementedError(
                f"group.global_std with cp_size={mesh.cp_size} would count every reward "
                f"{mesh.cp_size} times, since CP ranks share a prompt and therefore a reward. "
                f"Use within-group normalisation, or add a DP-only process group first."
            )
        advantages, advantage_stats = group_advantage(rewards.to(mesh.device), group)
        # The optimality probability the objective weights branches by: r for the positive branch,
        # 1 - r for the negative. Per trajectory, same (P, G) layout as the advantages.
        reward_prob = nft_reward_prob(advantages, adv_clip_max=nft.adv_clip_max)

        # ---- update: for each (trajectory, sampled timestep), one NFT micro-step ---------------
        #
        # Flatten the groups into per-trajectory samples, then take `timesteps_per_sample` micro
        # steps for each — the closure samples a fresh t per call. This is the DiffusionNFT analogue
        # of Flow-GRPO's inner epochs, and the only place gradient enters. A reshuffle that crossed
        # group boundaries would not change the loss here (the advantage is already baked into
        # reward_prob), but the accumulation must still divide by the chunk size.
        samples: list[tuple[NFTSample, torch.Tensor]] = []
        for group_position, rolled in enumerate(groups):
            for member in range(rolled.group_size):
                sample = NFTSample(
                    prompt_index=rolled.prompt_index,
                    member_index=member,
                    clean_latent=rolled.final_latents[member].detach(),
                )
                prob = reward_prob[group_position, member]
                for _ in range(timesteps_per_sample):
                    samples.append((sample, prob))

        order = torch.randperm(len(samples), generator=generator, device=generator.device).tolist()

        metrics: dict[str, float] = {}
        loss_total = torch.zeros((), device=mesh.device, dtype=torch.float32)
        grad_norm: float | None = None
        updates = 0
        micro_total = 0

        for chunk_start in range(0, len(order), grad_accum_steps):
            chunk = order[chunk_start : chunk_start + grad_accum_steps]
            optimizer.zero_grad(set_to_none=True)
            accumulated = torch.zeros((), device=mesh.device, dtype=torch.float32)

            for position, index in enumerate(chunk):
                is_last = position == len(chunk) - 1
                if hasattr(model, "set_requires_gradient_sync"):
                    model.set_requires_gradient_sync(is_last)
                state.on_micro_batch()

                sample, prob = samples[index]
                with autocast(mesh.device, autocast_dtype):
                    loss, step_metrics = nft_step_fn(mesh, model, sample, prob)
                    loss = _as_scalar(loss, state.global_step)
                (loss / len(chunk)).backward()
                accumulated += loss.detach().float()
                micro_total += 1
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

            if state.global_step >= steps:
                break

        # Refresh the frozen `old` policy from the trainable one ONCE per rollout, *not* per optimizer
        # step. `old` must stay the frozen rollout policy for the whole update phase, because that is
        # the anchor the NFT branches contrast against (v_pos = β·v_θ + (1-β)·v_old,
        # v_neg = (1+β)·v_old - β·v_θ). Refreshing it mid-phase — as a per-chunk call would, and with
        # the default hard copy (decay 0) especially — leaves `old` a single optimizer step behind
        # `default`, collapsing the contrast toward a plain regression to the sampled latents and
        # throwing away the reward signal. verl-omni refreshes once per training step for the same
        # reason. Hard copy at decay 0 (next rollout fully on-policy), slow EMA otherwise;
        # LoRATeacherEMA's update_every still applies the interval.
        ema.step(model, global_step=state.global_step)

        loss_total /= max(updates, 1)
        # Per-micro-step metrics (reward_prob, losses...) were summed once per micro-step, so average
        # over the micro-step count, not the optimizer-step count — dividing by `updates` over-reports
        # every metric by grad_accum (e.g. reward_prob 0.5 shows as 1.0 at grad_accum=2). loss_total
        # above is a separate, correctly-averaged accumulator and is unaffected.
        metrics = {key: value / max(micro_total, 1) for key, value in metrics.items()}

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
            loss_total /= mesh.world_size
        metrics = _reduce_metrics({**metrics, **advantage_stats.as_dict()}, mesh)
        metrics["perf/rollout_time"] = rollout_seconds
        metrics["perf/reward_time"] = reward_seconds
        metrics["nft/updates_per_rollout"] = float(updates)

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
                ema=ema,
                generator=generator,
                lora=lora,
            )
            last_saved = state.global_step
            logger.info(f"saved {path}")
            _run_checkpoint_hook(on_checkpoint, model, state.global_step, path, logger)

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
        _run_checkpoint_hook(on_checkpoint, model, state.global_step, path, logger)

    return state


__all__ = ["NFTRewardFn", "NFTRolloutFn", "NFTSample", "NFTStepFn", "fit_opsd_nft"]
