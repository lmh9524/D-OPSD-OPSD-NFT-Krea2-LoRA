"""RL post-training configuration.

L0: pure declaration. See ``docs/rl-design.md`` for the design these describe.

The validation here earns its place the same way ``FlowMatchConfig.__post_init__`` does: every
constraint below, violated, produces a run that trains on the wrong thing rather than one that
raises. A window that reaches past the schedule silently trains fewer steps than asked; a
``group_size`` of one makes every advantage exactly zero and the run is a no-op that still burns
GPU hours. Both fire while ``tyro`` parses flags, long before 9B of weights load.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SDEConfig:
    """The rollout: how the deterministic sampler becomes a stochastic policy.

    Rectified flow sampling is an ODE, so it has no actions to take a gradient through.
    Flow-GRPO converts it to an SDE with the same marginals, which makes each denoising step a
    Gaussian draw with a tractable log-probability. ``dflow/schedulers/flow_sde.py`` is the maths.
    """

    #: Scale on the injected noise. 0 recovers the ODE exactly, which is why the degeneracy is a
    #: test rather than a comment -- but it also makes the step's variance zero, so a log-prob is
    #: undefined and RL cannot run there. flow_grpo ships 0.7; verl-omni's Qwen-Image recipe uses
    #: 1.2. Higher means more exploration and a rollout further from what inference produces.
    noise_level: float = 0.7

    #: Denoising steps in the rollout. This is the *sampler's* step count, not a training quantity:
    #: it decides the sigma schedule the trajectory walks.
    inference_steps: int = 10

    #: **Denoising reduction.** How many consecutive steps are stochastic -- and therefore the
    #: number that carry a log-prob, get stored, and enter the loss. Steps outside the window are
    #: plain deterministic ODE steps: no noise, no log-prob, no gradient, nothing recorded.
    #:
    #: One mechanism, not two. It is tempting to read "sample with K steps, train on W" as a
    #: subsample of a fully stochastic trajectory, but making the whole trajectory stochastic and
    #: then training a subset is a different (and more expensive) algorithm: it would store K+1
    #: latents and put exploration noise on steps that never receive a gradient.
    #:
    #: Cost is linear in this: a step is ``group_size * inference_steps`` no-grad forwards plus
    #: ``group_size * window_size`` forward-backwards.
    #:
    #: ``None`` makes the whole trajectory stochastic, stopping one step short of the end, where
    #: ``sigma_next == 0`` puts the log-prob at its most delicate.
    window_size: int | None = 2

    #: The half-open span of step indices the window may cover. The start is drawn uniformly so
    #: that ``[start, start + window_size)`` stays inside it, which is why the range is expressed
    #: as the covered span rather than as bounds on the start: ``(0, 5)`` with ``window_size=2``
    #: means the window is somewhere in the first five steps, whatever its size.
    #:
    #: Early steps decide global composition, which is usually where a reward has leverage.
    window_range: tuple[int, int] = (0, 5)

    def __post_init__(self) -> None:
        if self.noise_level <= 0.0:
            raise ValueError(
                f"noise_level must be positive, got {self.noise_level}. Zero recovers the "
                f"deterministic ODE, whose steps have no variance and therefore no log-prob, so "
                f"there is no policy to take a gradient of."
            )
        if self.inference_steps < 2:
            raise ValueError(f"inference_steps must be >= 2, got {self.inference_steps}")

        start, stop = self.window_range
        if not 0 <= start < stop <= self.inference_steps:
            raise ValueError(
                f"window_range must satisfy 0 <= start < stop <= inference_steps="
                f"{self.inference_steps}, got {self.window_range}"
            )
        if self.window_size is not None:
            if self.window_size < 1:
                raise ValueError(f"window_size must be >= 1 or None, got {self.window_size}")
            if self.window_size > stop - start:
                raise ValueError(
                    f"window_size={self.window_size} does not fit in window_range="
                    f"{self.window_range}, which spans {stop - start} steps. A window that cannot "
                    f"fit would silently train fewer steps than asked."
                )

    @property
    def trained_steps(self) -> int:
        """Steps per trajectory that carry a gradient."""
        return self.inference_steps - 1 if self.window_size is None else self.window_size


@dataclass
class GroupConfig:
    """Group-relative advantage: the baseline that replaces a critic.

    Each prompt is sampled ``size`` times and the rewards within that group are centred, so the
    baseline is the group mean and no value network is trained.

    **A prompt's whole group lives on one rank.** The data-parallel dimension distributes prompts,
    not group members, which makes the normalisation rank-local and therefore exact with no
    collective at all. Splitting a group across ranks would parallelise the rollout better and put
    an all-gather in the advantage path, where an ordering that differs by rank gives each rank a
    different baseline while the run proceeds normally. See ``docs/rl-design.md``.
    """

    #: Trajectories per prompt. Two is the minimum at which a group mean carries any information.
    size: int = 8

    #: Normalise by the standard deviation over the whole batch instead of within each group.
    #:
    #: Within-group (the default) is GRPO as published and needs no communication under the
    #: whole-group layout. Global is steadier when groups are small enough that their own standard
    #: deviation is mostly noise, but it **does** need an all-reduce across the DP dimension, so
    #: ``group_advantage`` takes a process group and reduces only in this mode.
    global_std: bool = False

    #: Divide the centred reward by a standard deviation at all. ``False`` is Dr.GRPO, which argues
    #: the division introduces a length/difficulty bias; the centring is what removes the baseline.
    normalise_by_std: bool = True

    #: Guard against dividing by a vanishing standard deviation, which happens whenever every
    #: sample in a group scores identically -- common early, and with a saturating reward.
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError(
                f"group size must be >= 2, got {self.size}. With one sample per prompt the group "
                f"mean is that sample, so every advantage is exactly zero and the run is a no-op."
            )
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")


@dataclass
class PPOConfig:
    """The clipped policy objective.

    ``clip_ratio`` is **not** on the scale LLM RL uses, and the reason is worth reading before
    copying a number from anywhere. ``step_logprob`` reduces over the token and channel dimensions
    with a **mean**, following flow_grpo and verl-omni, so a log-prob is a per-element average and
    the ratio ``exp(new - old)`` sits very close to 1. Published diffusion recipes are calibrated
    to that: verl-omni's Qwen-Image Flow-GRPO script passes ``clip_ratio=1e-5``.

    Reducing with a sum instead would scale every log-ratio by the element count -- of order
    ``4096 * 128`` for a 1024px target -- and any inherited clip threshold becomes meaningless
    without anything raising. That is why the reduction lives in one tested function rather than at
    the call site.
    """

    #: PPO clip epsilon, on the per-element-averaged log-prob scale described above.
    clip_ratio: float = 1e-4

    #: Clamp the advantage before it multiplies the ratio. A single outlier reward otherwise
    #: dominates a whole group.
    adv_clip_max: float = 5.0

    #: Optimizer updates per rollout. This is where PPO's sample efficiency comes from, and the
    #: only reason the ratio ever leaves 1.0: on the first pass the current policy *is* the policy
    #: that produced the trajectories, so the ratio is 1 and the clip never fires.
    inner_epochs: int = 1

    def __post_init__(self) -> None:
        if self.clip_ratio <= 0.0:
            raise ValueError(f"clip_ratio must be positive, got {self.clip_ratio}")
        if self.adv_clip_max <= 0.0:
            raise ValueError(f"adv_clip_max must be positive, got {self.adv_clip_max}")
        if self.inner_epochs < 1:
            raise ValueError(f"inner_epochs must be >= 1, got {self.inner_epochs}")


@dataclass
class AestheticRewardConfig:
    """The LAION aesthetic predictor: CLIP ViT-L/14 image embeddings through a small MLP.

    What DDPO and flow_grpo score with, so numbers here are comparable to published runs. About
    1.7 GB resident, which fits alongside a training step — unlike a VLM judge, which does not.
    """

    enabled: bool = False

    #: Weight in the composite reward.
    weight: float = 1.0

    #: CLIP backbone. Its projection dimension must be 768, which is what the published head's
    #: first layer expects; a different backbone loads and produces plausible nonsense.
    clip_model: str = "openai/clip-vit-large-patch14"

    #: Where the MLP head comes from. The Hub mirror of
    #: ``improved-aesthetic-predictor``'s ``sac+logos+ava1-l14-linearMSE.pth``.
    head_repo: str = "trl-lib/ddpo-aesthetic-predictor"
    head_filename: str = "aesthetic-model.pth"

    #: Images per CLIP forward. The scorer runs once per trajectory, not once per step, so this
    #: trades a little latency against peak memory during the reward phase.
    batch_size: int = 8

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class OCRRewardConfig:
    """Text-rendering reward: ``1 - normalised edit distance(detected, target)``.

    The engine is injected rather than chosen here. A VLM judge of the kind verl-omni uses is
    ~15 GB and cannot sit alongside a 52 GiB training step, so anything that large belongs behind
    ``rewards/http.py``; a dedicated detector-plus-recogniser is a few hundred megabytes.

    **The normalisation is what needs pinning, not the engine.** Case folding, whitespace handling
    and how multiple detected boxes are joined each move the reward substantially and none of them
    raise, so they are explicit fields with a test asserting specific pairs.
    """

    enabled: bool = False
    weight: float = 1.0

    #: Which metadata field carries the string the image should contain. Owned by the task's
    #: ``schema.py``, named here so the contract is visible from both ends.
    target_key: str = "text"

    #: Fold case before comparing. Renderers are not reliably case-faithful and the reward is
    #: usually about whether the glyphs are there at all.
    case_sensitive: bool = False

    #: Collapse runs of whitespace to one space and strip the ends. An OCR engine's spacing
    #: depends on box geometry, not on the text.
    collapse_whitespace: bool = True

    #: How detected boxes are joined into one string before comparing. Reading order is the
    #: engine's business; this is only the separator.
    join: str = " "

    #: Languages the engine should recognise. Only read by the bundled engine; an injected one
    #: carries its own configuration.
    languages: tuple[str, ...] = ("en",)

    def __post_init__(self) -> None:
        if not self.target_key:
            raise ValueError("target_key must name a metadata field")


@dataclass
class ReferenceFidelityRewardConfig:
    """Reference fidelity: CLIP-I cosine between the generated image and the case's ground truth.

    The reward Flow-GRPO for ref2img was held back for want of — a scorer that judges *whether the
    references were used*, not just whether the image is pretty. It is the CLIP image-image cosine
    between the generated image and the case's ground-truth/reference image(s), the standard "CLIP-I"
    metric, over the *same* CLIP image encoder the aesthetic reward already loads.

    ## The metadata contract

    A generated image is scored against a target the reward cannot know from the pixels alone, so the
    ground truth arrives per sample through ``metadata`` under :attr:`reference_key`. The value is a
    ground-truth image (or a list of them, averaged) as a uint8 ``(C, H, W)`` or ``(H, W, C)`` tensor
    in [0, 255], matching the pixel contract the scorer's inputs already obey. A sample missing the
    field **raises** — a silently zero reward is indistinguishable from a bad image, the same rule
    ``OCRReward`` follows. The task's ``schema.py`` owns the field; it is named here so the contract
    is visible from both ends, exactly like ``OCRRewardConfig.target_key``.

    ## What it cannot measure

    CLIP-I is a whole-image semantic cosine, so it rewards "looks like the same outfit/scene", not
    pixel copy-paste and not identity under a *different* photo of the same person — the same blind
    spot the ``gain`` metric has (see ``krea2-ref2img`` SKILL). Answering identity needs a face
    embedding, which is out of scope here.
    """

    enabled: bool = False

    #: Weight in the composite reward.
    weight: float = 1.0

    #: Which metadata field carries the ground-truth/reference image(s) to compare against. Owned by
    #: the task's ``schema.py``; named here so the contract is visible from both ends.
    reference_key: str = "reference_image"

    #: CLIP backbone whose *image* tower produces both embeddings. The same default as the aesthetic
    #: reward, so a run enabling both loads one CLIP. Unlike the aesthetic head there is no
    #: projection-dimension constraint — cosine is dimension-agnostic — but the two embeddings must
    #: come from one model, which they do by construction.
    clip_model: str = "openai/clip-vit-large-patch14"

    #: Images per CLIP forward during scoring. The reward runs once per trajectory, not once per
    #: denoising step, so this trades a little latency against peak memory in the reward phase.
    batch_size: int = 8

    def __post_init__(self) -> None:
        if not self.reference_key:
            raise ValueError("reference_key must name a metadata field")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class RewardConfig:
    """The composite reward. Components are summed by weight, and each is logged on its own.

    Two rewards pulling against each other is the normal case, not the exception: an aggregate that
    moved tells you nothing about which component moved, which is why ``CompositeReward`` reports
    the breakdown rather than only the total.
    """

    aesthetic: AestheticRewardConfig = field(default_factory=AestheticRewardConfig)
    ocr: OCRRewardConfig = field(default_factory=OCRRewardConfig)
    reference_fidelity: ReferenceFidelityRewardConfig = field(
        default_factory=ReferenceFidelityRewardConfig
    )

    def __post_init__(self) -> None:
        if not (self.aesthetic.enabled or self.ocr.enabled or self.reference_fidelity.enabled):
            raise ValueError(
                "no reward is enabled. RL without a reward optimises nothing — enable at least "
                "one of reward.aesthetic, reward.ocr or reward.reference_fidelity."
            )


@dataclass(kw_only=True, slots=True)
class DiffusionNFTConfig:
    """The DiffusionNFT (OPSD-NFT) forward-process, likelihood-free objective.

    This is not PPO. DiffusionNFT never computes a log-probability and never needs an SDE rollout:
    it rolls out **clean latents** with the frozen ``old`` policy, turns each group-relative
    advantage into an *optimality probability* ``r`` in [0, 1], and regresses the trainable policy's
    velocity at a freshly sampled ``t`` toward the positive (``r``) or away from the negative
    (``1 - r``) branch in x0-space. See ``dflow/rl/objective.py`` for the maths and
    ``docs/rl-design.md`` for the design.

    Every field below, set wrong, changes the objective without raising — the same test as every
    other validated config here — so ``__post_init__`` fires while ``tyro`` parses flags.

    ## Why ``mix_beta`` is the delicate one

    ``mix_beta`` (``beta``) mixes the trainable and frozen predictions into the positive target
    ``beta*forward + (1-beta)*old`` and the *implicit* negative ``(1+beta)*old - beta*forward``, and
    the per-sample loss is divided by ``beta`` again. Small ``beta`` keeps the update close to the
    ``old`` policy (the trust region), which is what a forward-process method uses instead of a clip;
    the reference implementations run 0.1. Too large and the negative branch's ``x0`` extrapolates
    far past the data manifold and the run diverges — quietly, because the loss is still finite.
    """

    #: ``beta`` — the mix between the trainable ``forward`` prediction and the frozen ``old`` one.
    #: The DiffusionNFT trust-region knob; verl-omni and the OPSD reference both use 0.1. Must lie in
    #: (0, 1]: it is a divisor of the per-sample loss, so 0 is undefined, and above 1 the implicit
    #: negative prediction has a negative coefficient on ``old`` that is off the algorithm's design.
    mix_beta: float = 0.1

    #: Weight on the KL-to-reference regulariser ``||forward - ref||^2``, where ``ref`` is the
    #: frozen base model (adapter disabled). Keeps the trainable policy from drifting off the
    #: pretrained manifold; the reference implementations use 1e-4. Zero disables the term.
    ref_kl_coef: float = 1e-4

    #: Clamp on the advantage before it becomes ``reward_prob``, and the scalar the policy loss is
    #: multiplied by so its magnitude is comparable across ``adv_clip_max`` choices. A single outlier
    #: reward otherwise dominates a whole group. Matches ``PPOConfig.adv_clip_max``.
    adv_clip_max: float = 5.0

    #: Floor on the per-sample adaptive weight ``|x0_pred - x0|.mean``, applied before dividing the
    #: squared error by it. Guards against a vanishing denominator when a prediction happens to land
    #: on ``x0`` exactly (common early, and for a degenerate group). Computed under ``no_grad``.
    adaptive_weight_min: float = 1e-5

    #: Fraction of the rollout's few-step schedule whose timesteps a single update trains on. The
    #: rollout visits ``num_train_timesteps`` steps; each optimiser update re-noises the clean latent
    #: at ``ceil(fraction * num_train_timesteps)`` of them (sampled per sample). 1.0 trains on every
    #: visited step. Lower trades signal per update for throughput, matching verl-omni's
    #: ``timestep_fraction``.
    timestep_fraction: float = 1.0

    #: Denoising steps in the ``old``-policy rollout that produces the clean target latents. This is
    #: the *sampler's* step count — the few-step inference schedule DiffusionNFT trains against — not
    #: a training-density quantity, and it sets the pool of timesteps ``timestep_fraction`` samples.
    num_train_timesteps: int = 8

    #: EMA decay for refreshing the frozen ``old`` policy from the trainable one after each optimiser
    #: step: ``old = decay*old + (1-decay)*forward``. **0.0 is a hard copy** (``old`` becomes the
    #: current policy every ``old_policy_update_interval`` steps), which is the on-policy default and
    #: what ``DiffusionNFT`` Algorithm 1 does — the rollout policy is the last trained policy. A
    #: positive decay keeps ``old`` as a slow EMA, steadier but further from on-policy.
    old_policy_decay: float = 0.0

    #: How often (in optimiser steps) the ``old`` policy is refreshed toward the trainable one. 1
    #: refreshes every step; a hard copy (``old_policy_decay == 0``) at interval 1 is fully
    #: on-policy. Raise it to amortise the refresh, at the cost of a staler rollout policy.
    old_policy_update_interval: int = 1

    #: The composite reward. DiffusionNFT's ``reward_prob`` is a group-relative *ranking* signal, so
    #: any reward whose ordering is meaningful works; for ref2img this is the reference-fidelity
    #: reward, enabled by default here so a bare config is runnable (``RewardConfig`` refuses one with
    #: nothing enabled). Held here so the experiment builds one reward the same way an RL run does.
    reward: RewardConfig = field(
        default_factory=lambda: RewardConfig(
            reference_fidelity=ReferenceFidelityRewardConfig(enabled=True)
        )
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.mix_beta <= 1.0:
            raise ValueError(
                f"mix_beta must be in (0, 1], got {self.mix_beta}. It divides the per-sample loss, "
                f"so 0 is undefined; above 1 the implicit negative target leaves the algorithm's "
                f"design. The reference implementations use 0.1."
            )
        if self.ref_kl_coef < 0.0:
            raise ValueError(f"ref_kl_coef must be non-negative, got {self.ref_kl_coef}")
        if self.adv_clip_max <= 0.0:
            raise ValueError(f"adv_clip_max must be positive, got {self.adv_clip_max}")
        if self.adaptive_weight_min <= 0.0:
            raise ValueError(
                f"adaptive_weight_min must be positive, got {self.adaptive_weight_min}; it is a "
                f"floor on a divisor"
            )
        if not 0.0 < self.timestep_fraction <= 1.0:
            raise ValueError(
                f"timestep_fraction must be in (0, 1], got {self.timestep_fraction}"
            )
        if self.num_train_timesteps < 2:
            raise ValueError(
                f"num_train_timesteps must be >= 2, got {self.num_train_timesteps}; a few-step "
                f"schedule needs at least two steps to integrate"
            )
        if not 0.0 <= self.old_policy_decay < 1.0:
            raise ValueError(
                f"old_policy_decay must be in [0, 1), got {self.old_policy_decay}. 0 is a hard copy "
                f"(fully on-policy); a positive value makes old a slow EMA of the trainable policy."
            )
        if self.old_policy_update_interval < 1:
            raise ValueError(
                f"old_policy_update_interval must be >= 1, got {self.old_policy_update_interval}"
            )


__all__ = [
    "AestheticRewardConfig",
    "DiffusionNFTConfig",
    "GroupConfig",
    "OCRRewardConfig",
    "PPOConfig",
    "ReferenceFidelityRewardConfig",
    "RewardConfig",
    "SDEConfig",
]
