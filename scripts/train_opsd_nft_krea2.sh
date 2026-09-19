#!/usr/bin/env bash
# Single-GPU OPSD-NFT (DiffusionNFT) reward post-training for Krea 2 ref2img try-on, on Turbo.
#
#   ./scripts/train_opsd_nft_krea2.sh <dataset-root> [extra tyro flags...]
#
# This continues a phase-1 Krea 2 ref2img try-on LoRA with the fused DiffusionNFT objective: the
# frozen `old` adapter rolls out clean try-on images on Turbo's few-step schedule, a reference-
# fidelity reward (CLIP-I vs the case's ground truth) scores them, and the group-relative advantage
# drives a forward-process, likelihood-free update of the trainable `default` adapter. See
# `experiments/krea2_ref2img_opsd_nft.py` and `dflow/rl/objective.py`.
#
# `<dataset-root>` holds `train.jsonl` (rows: {target, refs[], prompt}) plus the images it points at,
# the same layout the phase-1 ref2img and D-OPSD runs use.
#
# PREREQUISITES (see the repository README for setup):
#   * Krea 2 Turbo weights                 -> $CKPS/Krea-2-Turbo   (--backbone.model.path)
#   * the phase-1 ref2img LoRA             -> $CKPS/krea2-tryon-lora (--init-lora-from; BOTH adapters
#                                             start from it, so its rank/alpha must match below)
#   * the curated try-on dataset          -> <dataset-root>
#
# COST: an OPSD-NFT step is GROUP_SIZE few-step rollouts (no grad) plus GROUP_SIZE * K forward-
# backwards, and EACH backward runs THREE transformer forwards (trainable / frozen-old / reference).
# So expect it to be several times an SFT step. Watch `perf/rollout_time`, `reward/reference_fidelity`
# (should rise) and `advantage/degenerate_groups` (toward 1 == the reward has saturated and the run
# has quietly stopped learning) before raising GROUP_SIZE or K.
#
# GPU-only: the rollout, the three-forward update and the CLIP reward all need real weights.
set -euo pipefail

DATA_ROOT="${1:?usage: $0 <dataset-root> [flags...]}"
shift || true

# Override these paths for the local checkpoint layout.
CKPS="${CKPS:-/mnt/shared/lihaoran/ssd/ckps}"
BACKBONE="${BACKBONE:-$CKPS/Krea-2-Turbo}"
INIT_LORA="${INIT_LORA:-$CKPS/krea2-tryon-lora}"

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Knobs: override any via env, or append tyro flags after the dataset root.
python3 experiments/krea2_ref2img_opsd_nft.py \
    --dataset.root "$DATA_ROOT" \
    --backbone.model.family "${FAMILY:-krea2-turbo}" \
    --backbone.model.path "$BACKBONE" \
    --backbone.lora.rank "${LORA_RANK:-64}" \
    --backbone.lora.alpha "${LORA_ALPHA:-64}" \
    --init-lora-from "$INIT_LORA" \
    --group.size "${GROUP_SIZE:-8}" \
    --nft.mix-beta "${MIX_BETA:-0.1}" \
    --nft.ref-kl-coef "${REF_KL_COEF:-1e-4}" \
    --nft.num-train-timesteps "${NUM_TRAIN_TIMESTEPS:-8}" \
    --nft.old-policy-decay "${OLD_POLICY_DECAY:-0.0}" \
    --training.steps "${STEPS:-500}" \
    --training.grad-accum-steps "${GRAD_ACCUM:-4}" \
    --checkpoint.interval "${INTERVAL:-100}" \
    --run-directory "${RUN_DIR:-runs/krea2_opsd_nft_lora}" \
    "$@"
