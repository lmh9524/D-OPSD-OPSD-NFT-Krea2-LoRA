"""Top-level config for Krea 2 ref2img **D-OPSD** (on-policy self-distillation).

This continues a phase-1 Krea 2 ref2img try-on LoRA with D-OPSD, so the model keeps learning
try-on quality **without losing Turbo's few-step inference speed** — the failure mode plain SFT has
on a step-distilled model.

Read alongside ``krea2_ref2img_lora.py`` (the phase-1 recipe) and ``dflow/config/distill.py``.

Two Krea-2-specific facts drive the defaults:

* **Target on Turbo.** The whole point is preserving few-step behaviour, so the backbone is
  ``krea2-turbo`` (the distilled checkpoint), not Raw.
* **Start from the phase-1 LoRA.** Base Krea 2 cannot use garment references (its RoPE T axis is
  pinned at 0), so the *student* must already be a ref2img model. ``distill.init_lora_from`` points
  at the trained phase-1 LoRA; both the student and the EMA-teacher adapters are initialised from
  it. Its rank/alpha here **must match** the phase-1 LoRA, or the tensors will not load.

Sizes are areas; aspect ratio is preserved; ``batch_size`` stays 1 (Krea 2's ``position_ids`` is
unbatched). D-OPSD costs ~4x the FLOPs and ~2x the wall-clock of SFT per step (a K-step rollout plus
a teacher pass), so ``num_steps`` (K) and ``reference_max_area`` are the first memory/throughput
knobs to turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dflow.config.data import DataConfig
from dflow.config.dataset import Ref2ImgConfig
from dflow.config.diffusion import FlowMatchConfig
from dflow.config.distill import DOPSDConfig
from dflow.config.frozen import FrozenConfig, TextConfig, TextEncoderConfig
from dflow.config.model import BackboneConfig, LoRAConfig, ModelConfig
from dflow.config.optim import LRSchedulerConfig, OptimizerConfig
from dflow.config.recipe.krea2_ref2img_lora import KREA2_TEXT_LAYERS
from dflow.config.runtime import ActivationCheckpointConfig, CompileConfig, RuntimeConfig
from dflow.config.train import CheckpointConfig, LoggingConfig, TrainingConfig

#: Original run location for the phase-1 ref2img LoRA. Override with
#: ``--distill.init-lora-from`` (a local dir or an HF repo id).
DEFAULT_PHASE1_LORA = "/mnt/shared/lihaoran/ssd/ckps/krea2-tryon-lora"

#: The curated try-on data has 2-9 references per case and the prompts name them by ordinal
#: ("Image 2: the jacket"). Subsampling below a case's real count would leave the prompt referring
#: to images the model never sees, so this covers the maximum: every case keeps all its references
#: and the ordinal prompt stays honest. Lower it only if you also filter the data to <= N
#: references; it is a primary memory knob (attention is quadratic in the concatenated length).
NUM_REFERENCES = 9


@dataclass(kw_only=True, slots=True)
class Krea2DOPSDRecipe:
    run_directory: str = "runs/krea2_dopsd_lora"

    backbone: BackboneConfig = field(
        default_factory=lambda: BackboneConfig(
            # Turbo: the step-distilled checkpoint whose few-step schedule D-OPSD exists to preserve.
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
                        # Every field below is matched to the phase-1 LoRA's conditioning.json, so
                        # the student's deployed pathway is identical to what it learned. Grounding
                        # is a pretrained pathway here because Krea 2's own encoder is Qwen3-VL.
                        # A mismatch trains a different model with no error.
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
            # All four geometry knobs are matched to the phase-1 LoRA's conditioning.json, so the
            # student rolls out at exactly the extents — and thus the H/W position-id ranges — it
            # was trained on. This is also the memory driver: 768^2 target + 9x384^2 refs is a long
            # sequence, and fit()'s single backward retains it K times. If you OOM, drop these (and
            # accept the train/deploy geometry gap) or move to a per-step-backward trainer.
            target_max_area=768 * 768,  # 589824
            reference_max_area=384 * 384,  # 147456
            reference_fit_target=True,  # refs sized to the target grid, not an area cap
            reference_registration="disjoint",  # phase-1 registered reference H disjoint from target
            num_references=NUM_REFERENCES,  # 9: covers every case, so ordinal prompts stay honest
            # Slot 0 is the person in the R2I try-on data; pin it so subsampling never drops it.
            pinned_references=1,
            # Prompts name references by ordinal ("Image 2: the jacket"), so slot order is
            # load-bearing and slots must not be blanked or duplicated.
            pad_short_samples=False,
            reference_dropout=0.0,
            # Turbo infers with CFG disabled, so the unconditional branch is not exercised and
            # caption dropout is unnecessary here.
            caption_dropout=0.0,
        )
    )
    data: DataConfig = field(default_factory=DataConfig)

    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            # Mandatory: a K-step rollout retains K student graphs for one backward.
            activation_checkpoint=ActivationCheckpointConfig(enabled=True),
            compile=CompileConfig(enabled=False),
        )
    )
    distill: DOPSDConfig = field(
        default_factory=lambda: DOPSDConfig(
            num_steps=4,
            teacher="ema",
            ema_decay=0.9999,
            # Default teacher = the "gt-ref" strategy: the target enters as one extra reference span
            # (needs a reference-capable teacher, hence init_lora_from). Cheaper than grounding and
            # what the D-OPSD authors used for FLUX.2 edit. Add teacher_ground_target=True for an
            # extra Qwen3-VL semantic signal on top.
            teacher_ground_target=False,
            teacher_ref_target=True,
            edit_suffix="The final reference image is the exact target look to reproduce; match it.",
            loss_space="x0",
            init_lora_from=DEFAULT_PHASE1_LORA,
        )
    )
    training: TrainingConfig = field(
        # 1K steps at effective batch 4 (grad-accum 4 x local 1), matching the paper's LoRA setting.
        default_factory=lambda: TrainingConfig(steps=1000, grad_accum_steps=4)
    )
    # 1e-5: the paper's LoRA learning rate for FLUX.2-klein; a good starting point for Krea 2 Turbo.
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1e-5))
    lr_scheduler: LRSchedulerConfig = field(
        default_factory=lambda: LRSchedulerConfig(type="constant", warmup_steps=50)
    )
    # Turbo pins mu = 1.15 (shift 3.1582) at every resolution, which is FlowMatchConfig's default —
    # but D-OPSD rolls out on the *few-step* schedule, derived per step from distill.num_steps, so
    # this field only governs any fallback; the step function asks the family for the K-step mu.
    flow_match: FlowMatchConfig = field(default_factory=FlowMatchConfig)
    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(interval=200, keep_latest=3, preview=False)
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def resolve_paths(self) -> None:
        if self.logging.directory == LoggingConfig().directory:
            self.logging.directory = f"{self.run_directory}/logs"
        if self.checkpoint.directory == CheckpointConfig().directory:
            self.checkpoint.directory = f"{self.run_directory}/checkpoints"


__all__ = ["DEFAULT_PHASE1_LORA", "NUM_REFERENCES", "Krea2DOPSDRecipe"]
