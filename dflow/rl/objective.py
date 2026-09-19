"""The clipped policy objective, and the metrics that tell you an RL run has gone wrong.

One function, because the pieces that look trivial are the ones that fail silently:

* **the detach.** ``old_logprobs`` are recorded constants. Gradient through them would flip the
  sign of half the objective.
* **the advantage broadcast.** An advantage is per *trajectory*; a log-prob is per *(step,
  trajectory)*. Both are 2-D by the time they arrive, and a transposed advantage still broadcasts
  — applying trajectory 3's advantage to step 3 of every trajectory. So the shape is checked
  rather than broadcast.
* **the clip direction.** ``max`` of the unclipped and clipped losses, not ``min``: the objective
  is a *pessimistic* bound, and ``min`` trains happily in the wrong direction.
* **the sign.** The loss is ``-advantage * ratio``. Getting it backwards produces a curve that
  falls just as convincingly.

## The scale of ``clip_ratio``

``step_logprob`` reduces over token and channel dimensions with a **mean**, so a log-prob is a
per-element average and the ratio sits very close to 1. Published diffusion recipes are calibrated
to that — verl-omni's Qwen-Image Flow-GRPO script passes ``clip_ratio=1e-5``. A threshold copied
from LLM RL, where log-probs are summed over tokens, will never fire. ``PPOConfig.clip_ratio``
carries the same warning, because this is the number most likely to be transplanted without it.

## Read ``clipfrac`` and ``ppo_kl``, not the loss

The loss is a weighted average of advantages and tells you almost nothing on its own. What tells
you the run is healthy is that ``ppo_kl`` stays small and ``clipfrac`` is low but non-zero. A
``clipfrac`` at zero for the whole run means the policy is barely moving, or ``clip_ratio`` is on
the wrong scale; one near 1 means ``inner_epochs`` is too high for the step size and every update
is being truncated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dflow.config.rl import DiffusionNFTConfig, PPOConfig


@dataclass(frozen=True)
class ObjectiveStats:
    """Per-update diagnostics. The loss alone does not say whether RL is working."""

    ppo_kl: float
    clipfrac: float
    clipfrac_high: float
    clipfrac_low: float
    ratio_mean: float
    ratio_std: float

    def as_dict(self) -> dict[str, float]:
        return {
            "ppo/kl": self.ppo_kl,
            "ppo/clipfrac": self.clipfrac,
            "ppo/clipfrac_high": self.clipfrac_high,
            "ppo/clipfrac_low": self.clipfrac_low,
            "ppo/ratio_mean": self.ratio_mean,
            "ppo/ratio_std": self.ratio_std,
        }


def ppo_clip_loss(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    config: PPOConfig,
) -> tuple[torch.Tensor, ObjectiveStats]:
    """The clipped surrogate, as a scalar to call ``backward()`` on.

    Args:
        logprobs: ``(W, G)`` under current weights — from ``replay_logprobs``, with gradient.
        old_logprobs: ``(W, G)`` as recorded during the rollout. Detached here.
        advantages: ``(G,)`` — one per trajectory, broadcast across the window. A ``(W, G)``
            tensor is accepted too, for an objective that ever weights steps differently.
        config: clip epsilon and the advantage clamp.

    Returns the loss and the diagnostics. Reductions happen here so every experiment reports the
    same quantities the same way, which is the same reason ``StepMetrics`` lives in
    ``trainer/state.py``.
    """
    if logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"logprobs {tuple(logprobs.shape)} and old_logprobs {tuple(old_logprobs.shape)} "
            f"must match — they describe the same actions under two policies"
        )
    if logprobs.ndim != 2:
        raise ValueError(
            f"logprobs must be (window, group), got {tuple(logprobs.shape)}"
        )
    window, group = logprobs.shape

    if advantages.ndim == 1:
        if advantages.shape[0] != group:
            raise ValueError(
                f"advantages has {advantages.shape[0]} entries for {group} trajectories. One "
                f"advantage per trajectory; a transposed tensor would still broadcast and apply "
                f"the wrong one to every step."
            )
        advantages = advantages.reshape(1, group).expand(window, group)
    elif advantages.shape != logprobs.shape:
        raise ValueError(
            f"advantages {tuple(advantages.shape)} must be ({group},) or "
            f"{tuple(logprobs.shape)}, not something that merely broadcasts"
        )

    # Recorded under weights that no longer exist. Detached so gradient reaches the policy only
    # through `logprobs`.
    old = old_logprobs.detach()
    clamped = advantages.detach().clamp(-config.adv_clip_max, config.adv_clip_max)

    log_ratio = logprobs - old
    ratio = torch.exp(log_ratio)
    unclipped = -clamped * ratio
    clipped = -clamped * ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
    # max, not min: the surrogate is a pessimistic bound on the improvement.
    loss = torch.maximum(unclipped, clipped).mean()

    with torch.no_grad():
        excess = ratio - 1.0
        stats = ObjectiveStats(
            # The k1 estimator, matching verl and verl-omni: mean(-log_ratio).
            ppo_kl=float(-log_ratio.mean()),
            clipfrac=float((excess.abs() > config.clip_ratio).float().mean()),
            clipfrac_high=float((excess > config.clip_ratio).float().mean()),
            clipfrac_low=float((-excess > config.clip_ratio).float().mean()),
            ratio_mean=float(ratio.mean()),
            ratio_std=float(ratio.std()) if ratio.numel() > 1 else 0.0,
        )
    return loss, stats


# ============================================================================ DiffusionNFT
#
# A second objective in this file, and it earns its place here for the same reason ``ppo_clip_loss``
# does: every line is invisible to a shape check and to "the loss went down". But it is a *different*
# objective, not a mode of PPO — no clip, no ratio, no log-probability at all. DiffusionNFT
# ("Diffusion Negative-aware FineTuning") is forward-process and likelihood-free: it rolls out clean
# latents with a frozen ``old`` policy, maps each group-relative advantage to an optimality
# probability ``r`` in [0, 1], and regresses the trainable policy's velocity at a freshly sampled
# ``t`` toward a positive branch (weighted ``r``) and away from an implicit negative branch (weighted
# ``1 - r``), both compared in x0-space against the rollout's own clean latent.
#
# The maths is ported verbatim from verl-omni's ``DiffusionNFTLoss.compute_loss`` and the OPSD
# reference (``DiffusionOPSD/scripts/train_opsd_ri_sd3.py``), which agree byte-for-byte on the
# canonical x0-space form. Kept **pure** — no model, no encoder, no family import — so the whole
# objective unit-tests against analytic tensors, exactly like ``ppo_clip_loss``. The three velocity
# predictions (trainable / frozen-old / frozen-reference) are produced by the experiment's closure
# via adapter switching and handed in here already sliced to the target span.
#
# What is silent when wrong, and therefore why this is one tested function rather than inline maths:
#
# * **the two detaches.** ``old`` and ``ref`` are frozen policies; gradient through either turns the
#   negative branch into a second copy of the positive one and the KL term into a no-op.
# * **the positive/negative mix.** ``v_pos = beta*forward + (1-beta)*old`` and ``v_neg =
#   (1+beta)*old - beta*forward`` are reflections of ``forward`` about ``old``. Swap a sign and the
#   policy is pushed *toward* the low-reward branch — a curve that falls just as convincingly.
# * **the x0 conversion.** ``x0 = xt - t*v`` uses the rectified-flow identity; ``t`` must be the same
#   broadcastable timestep the rollout noised at, or the target is a different clean latent.
# * **the adaptive weight.** Normalising each squared error by its own detached magnitude is what
#   keeps positive and negative on one scale; computed under ``no_grad`` so it does not move the
#   gradient. Forgetting the detach makes it a second, wrong, loss term.
# * **fp32.** The squared terms are computed in fp32 regardless of autocast, mirroring ``flow_mse``:
#   bf16 residuals lose precision exactly where the loss is smallest.


@dataclass(frozen=True)
class NFTStats:
    """Per-update DiffusionNFT diagnostics. Like PPO's, the loss alone says little.

    Watch ``reward_prob`` (its spread across a batch is the advantage signal — a value stuck at 0.5
    means every advantage is ~zero and the run has stopped learning), ``ref_kl`` (small; a rising
    value means the policy is drifting off the pretrained manifold), and that ``pos_loss`` and
    ``neg_loss`` stay finite and comparable in magnitude — a diverging ``neg_loss`` is ``mix_beta``
    set too high, the failure this objective's trust region exists to prevent.
    """

    policy_loss: float
    positive_loss: float
    negative_loss: float
    ref_kl: float
    reward_prob_mean: float
    old_deviation: float

    def as_dict(self) -> dict[str, float]:
        return {
            "nft/policy_loss": self.policy_loss,
            "nft/positive_loss": self.positive_loss,
            "nft/negative_loss": self.negative_loss,
            "nft/ref_kl": self.ref_kl,
            "nft/reward_prob_mean": self.reward_prob_mean,
            "nft/old_deviation": self.old_deviation,
        }


def nft_reward_prob(advantages: torch.Tensor, *, adv_clip_max: float) -> torch.Tensor:
    """Map group-relative advantages to DiffusionNFT optimality probabilities in [0, 1].

    ``clamp(clamp(adv, ±adv_clip_max) / adv_clip_max / 2 + 0.5, 0, 1)`` — verl-omni's
    ``_advantage_to_reward_prob`` and the OPSD reference's inline ``r`` map, identically. A zero
    advantage maps to 0.5 (the positive and negative branches weighted equally, so no net push); the
    most positive advantage the clamp allows maps to 1 (pure positive branch), the most negative to 0
    (pure negative). Monotone increasing by construction, which is the property the objective relies
    on and ``tests/test_nft_objective.py`` pins.

    Pure and shape-preserving: whatever ``advantages`` shape comes in — ``(P, G)`` from
    ``group_advantage`` or a flat ``(N,)`` — comes back unchanged, so the caller controls the layout.
    """
    if adv_clip_max <= 0.0:
        raise ValueError(f"adv_clip_max must be positive, got {adv_clip_max}")
    clamped = advantages.clamp(-adv_clip_max, adv_clip_max)
    prob = clamped / adv_clip_max / 2.0 + 0.5
    return prob.clamp(0.0, 1.0)


def diffusion_nft_loss(
    *,
    forward_prediction: torch.Tensor,
    old_prediction: torch.Tensor,
    ref_forward_prediction: torch.Tensor,
    x0: torch.Tensor,
    xt: torch.Tensor,
    t_expanded: torch.Tensor,
    reward_prob: torch.Tensor,
    config: DiffusionNFTConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The DiffusionNFT forward-process policy loss, as a scalar to call ``backward()`` on.

    All prediction tensors are **velocities over the target span**, shape ``(B, ...)`` and identical
    to each other. Only ``forward_prediction`` carries gradient; ``old_prediction`` and
    ``ref_forward_prediction`` are detached here (belt and braces — the closure produces them under
    ``no_grad``, but a caller that forgot would otherwise train through a frozen policy silently).

    Args:
        forward_prediction: ``(B, ...)`` velocity from the **trainable** policy (adapter ``default``).
            The only tensor with gradient.
        old_prediction: ``(B, ...)`` velocity from the frozen **old** policy (adapter ``old``), the
            policy that rolled out the clean latents. Detached.
        ref_forward_prediction: ``(B, ...)`` velocity from the frozen **reference** (base model,
            adapter disabled). Detached. Anchors the KL regulariser.
        x0: ``(B, ...)`` the rollout's clean target latent — the regression target for both branches.
        xt: ``(B, ...)`` the re-noised state the three predictions were evaluated at.
        t_expanded: broadcastable to ``x0`` — the timestep ``xt`` was noised at, for the
            ``x0 = xt - t*v`` conversion. ``(B, 1, 1, ...)`` or ``(B,)`` both broadcast.
        reward_prob: ``(B,)`` (or any shape reducing to it) optimality probabilities from
            :func:`nft_reward_prob`. ``r`` weights the positive branch, ``1 - r`` the negative.
        config: the DiffusionNFT knobs — ``mix_beta``, ``ref_kl_coef``, ``adv_clip_max``,
            ``adaptive_weight_min``.

    Returns the scalar loss and a metrics dict (``NFTStats.as_dict()``). Reductions happen here so
    every experiment logs the same quantities the same way — the same reason ``ppo_clip_loss`` owns
    its stats.
    """
    if forward_prediction.shape != old_prediction.shape:
        raise ValueError(
            f"forward_prediction {tuple(forward_prediction.shape)} and old_prediction "
            f"{tuple(old_prediction.shape)} must match — the same velocity under two policies"
        )
    if forward_prediction.shape != ref_forward_prediction.shape:
        raise ValueError(
            f"forward_prediction {tuple(forward_prediction.shape)} and ref_forward_prediction "
            f"{tuple(ref_forward_prediction.shape)} must match"
        )
    if forward_prediction.shape != x0.shape or forward_prediction.shape != xt.shape:
        raise ValueError(
            f"forward_prediction {tuple(forward_prediction.shape)}, x0 {tuple(x0.shape)} and xt "
            f"{tuple(xt.shape)} must all match — predictions, target and state are the same span"
        )

    beta = config.mix_beta
    old = old_prediction.detach()
    ref = ref_forward_prediction.detach()

    # r is per-sample: reduce anything wider to (B,), matching verl-omni's flatten(1).mean(1).
    reward_weight = reward_prob
    if reward_weight.ndim > 1:
        reward_weight = reward_weight.flatten(1).mean(dim=1)
    reward_weight = reward_weight.to(device=x0.device, dtype=x0.dtype)
    if reward_weight.shape[0] != x0.shape[0]:
        raise ValueError(
            f"reward_prob reduces to {tuple(reward_weight.shape)} but the batch is {x0.shape[0]}; "
            f"one optimality probability per sample"
        )

    reduce_dims = tuple(range(1, x0.ndim))
    # v_pos and v_neg are reflections of the trainable prediction about the frozen old one.
    positive_prediction = beta * forward_prediction + (1.0 - beta) * old
    implicit_negative_prediction = (1.0 + beta) * old - beta * forward_prediction

    x0_positive = xt - t_expanded * positive_prediction
    x0_negative = xt - t_expanded * implicit_negative_prediction

    # Adaptive per-sample normalisation, under no_grad so it does not enter the gradient — fp32 to
    # mirror flow_mse (bf16 residuals lose precision where the loss is smallest).
    with torch.no_grad():
        positive_weight = (
            (x0_positive.float() - x0.float())
            .abs()
            .mean(dim=reduce_dims, keepdim=True)
            .clamp_min(config.adaptive_weight_min)
        )
        negative_weight = (
            (x0_negative.float() - x0.float())
            .abs()
            .mean(dim=reduce_dims, keepdim=True)
            .clamp_min(config.adaptive_weight_min)
        )

    positive_loss = ((x0_positive.float() - x0.float()) ** 2 / positive_weight).mean(dim=reduce_dims)
    negative_loss = ((x0_negative.float() - x0.float()) ** 2 / negative_weight).mean(dim=reduce_dims)

    policy_per_sample = (
        reward_weight * positive_loss / beta + (1.0 - reward_weight) * negative_loss / beta
    )
    policy_loss = (policy_per_sample * config.adv_clip_max).mean()

    # The KL anchor is in velocity space (not x0), matching both reference implementations.
    ref_kl = ((forward_prediction.float() - ref.float()) ** 2).mean(dim=reduce_dims).mean()
    loss = policy_loss + config.ref_kl_coef * ref_kl

    with torch.no_grad():
        stats = NFTStats(
            policy_loss=float(policy_loss.detach()),
            positive_loss=float(positive_loss.mean().detach()),
            negative_loss=float(negative_loss.mean().detach()),
            ref_kl=float(ref_kl.detach()),
            reward_prob_mean=float(reward_weight.mean().detach()),
            old_deviation=float(((forward_prediction - old) ** 2).mean().detach()),
        )
    return loss, stats.as_dict()


__all__ = [
    "NFTStats",
    "ObjectiveStats",
    "diffusion_nft_loss",
    "nft_reward_prob",
    "ppo_clip_loss",
]
