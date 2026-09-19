"""Top-level config for Krea 2 ref2img **OPSD-NFT** (DiffusionNFT on the step-distilled Turbo).

This is the RL sibling of ``krea2_dopsd_lora.py``. Where D-OPSD self-distills a phase-1 try-on LoRA
against a stronger-conditioned EMA teacher, OPSD-NFT post-trains it with a **reward**: the frozen
``old`` policy rolls out clean try-on images, a reference-fidelity reward scores each against the
case's ground truth, and the group-relative advantage becomes DiffusionNFT's optimality probability,
which drives a forward-process, likelihood-free update. See ``dflow/rl/objective.py`` for the maths,
``dflow/trainer/opsd_nft.py`` for the loop, and ``docs/rl-design.md`` for the design.

The conditioning is **identical to the D-OPSD recipe**, and deliberately so — the student rolls out
at exactly the extents, registration and T-spacing it was trained on, so its deployed pathway is the
one being optimised:

* **Target on Turbo.** The whole point is preserving few-step behaviour, so the backbone is
  ``krea2-turbo`` (the distilled checkpoint), not Raw.
* **Start from the phase-1 LoRA.** Base Krea 2 cannot use garment references (its RoPE T axis is
  pinned at 0), so the *student* (adapter ``default``) and the frozen ``old`` policy must both start
  reference-capable. ``init_lora_from`` points at the trained phase-1 LoRA; both adapters are
  initialised from it. Its rank/alpha here **must match** the phase-1 LoRA, or the tensors will not
  load.

The reward is the piece ref2img RL was previously held back for: reference fidelity, enabled here by
default (``reward.reference_fidelity``). It scores CLIP-I cosine between the generated image and the
case's ground truth, which the experiment passes through reward metadata.

Sizes are areas; aspect ratio is preserved; ``batch_size`` stays 1 (Krea 2's ``position_ids`` is
unbatched). An OPSD-NFT step is ``group.size`` few-step rollouts (no grad) plus
``group.size * timesteps_per_sample`` forward-backwards, each of which runs **three** transformer
forwards (trainable / frozen-old / reference), so ``group.size``, ``nft.num_train_timesteps`` and
``nft.timestep_fraction`` are the first throughput/memory knobs to turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dflow.config.data import DataConfig
from dflow.config.dataset import Ref2ImgConfig
from dflow.config.diffusion import FlowMatchConfig
from dflow.config.frozen import FrozenConfig, TextConfig, TextEncoderConfig
from dflow.config.model import BackboneConfig, LoRAConfig, ModelConfig
from dflow.config.optim import LRSchedulerConfig, OptimizerConfig
from dflow.config.recipe.krea2_dopsd_lora import DEFAULT_PHASE1_LORA, NUM_REFERENCES
from dflow.config.recipe.krea2_ref2img_lora import KREA2_TEXT_LAYERS
from dflow.config.rl import (
    DiffusionNFTConfig,
    GroupConfig,
    ReferenceFidelityRewardConfig,
    RewardConfig,
)
from dflow.config.runtime import ActivationCheckpointConfig, CompileConfig, RuntimeConfig
from dflow.config.train import CheckpointConfig, LoggingConfig, TrainingConfig

#: Name of the frozen ``old`` (rollout) LoRA adapter. The trainable/deployable one keeps
#: ``LoRAConfig.adapter_name`` (normally ``"default"``), so the exported LoRA is the student.
OLD_ADAPTER_NAME = "old"


@dataclass(kw_only=True, slots=True)
class Krea2OPSDNFTRecipe:
    run_directory: str = "runs/krea2_opsd_nft_lora"

    #: Local dir or HF repo of the phase-1 ref2img LoRA. Both the trainable ``default`` adapter and
    #: the frozen ``old`` (rollout) adapter are initialised from it, so both are reference-capable at
    #: step 0 — mandatory for Krea 2, whose base cannot use references. Its rank/alpha must match
    #: ``backbone.lora``. ``None`` starts from zero-output init, which for try-on cannot roll out
    #: anything worth scoring; a warning fires and the run trains on a degenerate reward.
    init_lora_from: str | None = DEFAULT_PHASE1_LORA

    #: Name of the frozen ``old``/rollout adapter (see :data:`OLD_ADAPTER_NAME`).
    old_adapter_name: str = OLD_ADAPTER_NAME

    backbone: BackboneConfig = field(
        default_factory=lambda: BackboneConfig(
            # Turbo: the step-distilled checkpoint whose few-step schedule OPSD-NFT preserves.
            model=ModelConfig(family="krea2-turbo"),
            # Must match the phase-1 LoRA these adapters are initialised from (see init_lora_from).
            lora=LoRAConfig(enabled=True, rank=64, alpha=64.0),
        )
    )
    frozen: FrozenConfig = field(
        default_factory=lambda: FrozenConfig(
            text=TextConfig(
                encoders=(
                    TextEncoderConfig(
                        name="qwen3-vl",
                        out_layers=KREA2_TEXT_LAYERS,
                        # Matched to the phase-1 LoRA's conditioning.json, so the student's deployed
                        # pathway is identical to what it learned. Grounding is a pretrained pathway
                        # here because Krea 2's own encoder is Qwen3-VL. A mismatch trains a
                        # different model with no error.
                        max_length=1024,
                        ground_references=True,
                        max_grounded_references=0,  # 0 == ground every reference, as phase-1 did
                        grounding_max_px=384,
                        fast_patch_embed=True,
                    ),
                )
            )
        )
    )
    dataset: Ref2ImgConfig = field(
        default_factory=lambda: Ref2ImgConfig(
            # Matched to the phase-1 LoRA's conditioning.json, so the rollout walks exactly the
            # extents — and thus the H/W position-id ranges — the student was trained on.
            target_max_area=768 * 768,  # 589824
            reference_max_area=384 * 384,  # 147456
            reference_fit_target=True,
            reference_registration="disjoint",
            reference_t_scale=1,
            num_references=NUM_REFERENCES,  # 9: covers every case, so ordinal prompts stay honest
            pinned_references=1,  # slot 0 is the person; never subsample it away
            pad_short_samples=False,
            reference_dropout=0.0,
            # Turbo infers with CFG disabled, so caption dropout is unnecessary here.
            caption_dropout=0.0,
        )
    )
    data: DataConfig = field(default_factory=DataConfig)

    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            # Mandatory at these sequence lengths; the update runs three forwards per micro-step.
            activation_checkpoint=ActivationCheckpointConfig(enabled=True),
            compile=CompileConfig(enabled=False),
        )
    )

    #: Group-relative advantage. Eight rollouts per prompt is the published GRPO group size and the
    #: minimum at which the reference-fidelity ranking within a group carries real signal.
    group: GroupConfig = field(default_factory=lambda: GroupConfig(size=8))

    #: The DiffusionNFT objective. ``mix_beta`` 0.1 and ``ref_kl_coef`` 1e-4 are the reference
    #: implementations' values; ``old_policy_decay`` 0 makes the rollout policy the last trained one
    #: (fully on-policy). ``num_train_timesteps`` matches Turbo's 8-step inference schedule — the
    #: schedule OPSD-NFT trains against. The reward is reference fidelity, enabled below.
    nft: DiffusionNFTConfig = field(
        default_factory=lambda: DiffusionNFTConfig(
            mix_beta=0.1,
            ref_kl_coef=1e-4,
            adv_clip_max=5.0,
            adaptive_weight_min=1e-5,
            timestep_fraction=1.0,
            num_train_timesteps=8,
            old_policy_decay=0.0,
            old_policy_update_interval=1,
            reward=RewardConfig(
                reference_fidelity=ReferenceFidelityRewardConfig(enabled=True, weight=1.0)
            ),
        )
    )

    training: TrainingConfig = field(
        # Effective batch 4 (grad-accum 4 x local 1). RL needs fewer optimizer steps than SFT; 500
        # is a reasonable first budget given each step is many forwards.
        default_factory=lambda: TrainingConfig(steps=500, grad_accum_steps=4)
    )
    # 1e-5: the paper's LoRA learning rate for FLUX.2-klein; a good starting point for Krea 2 Turbo.
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-5))
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: LRSchedulerConfig(type="constant", warmup_steps=50)
    )
    # Turbo pins mu = 1.15 (shift 3.1582) at every resolution, which is FlowMatchConfig's default —
    # but the rollout uses the few-step schedule derived from nft.num_train_timesteps, so this field
    # only governs any fallback; the experiment asks the family for the K-step mu.
    flow_match: FlowMatchConfig = field(default_factory=FlowMatchConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(interval=100, keep_latest=3, preview=False)
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_paths(self) -> None:
        if self.logging.directory == LoggingConfig().directory:
            self.logging.directory = f"{self.run_directory}/logs"
        if self.checkpoint.directory == CheckpointConfig().directory:
            self.checkpoint.directory = f"{self.run_directory}/checkpoints"


__all__ = ["OLD_ADAPTER_NAME", "Krea2OPSDNFTRecipe"]
