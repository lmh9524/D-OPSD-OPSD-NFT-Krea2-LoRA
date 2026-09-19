#!/usr/bin/env bash
# Outfit-level virtual try-on LoRA on Krea 2, single GPU.
#
# Krea 2 has NO pretrained reference conditioning: Krea2Pipeline is text-to-image only and pins the
# RoPE T axis at 0. This trains that capability from scratch, so it needs more rank, more steps and
# more data than the FLUX.2 equivalent. See docs/krea2-reference.md.
#
# Sizes are AREAS and aspect ratio is preserved, matching inference. Token counts vary per sample,
# so batch_size stays 1 -- doubly so here, since Krea 2's position_ids is unbatched.
#
# <root> holds train.jsonl, written by tools/prepare_garments2look.py:
#   {"target": "looks-resized/women/x.jpg", "refs": ["images/women/dress/x_1.jpg"], "prompt": "..."}
set -euo pipefail

DATA_ROOT="${1:?usage: $0 <dataset-root> [flags...]}"
shift || true

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

python3 experiments/krea2_ref2img_lora.py \
    --dataset.root "$DATA_ROOT" \
    --dataset.target-max-area "${TARGET_AREA:-1048576}" \
    --dataset.reference-max-area "${REF_AREA:-147456}" \
    --dataset.num-references "${NUM_REFS:-4}" \
    --backbone.model.family "${KREA2_VARIANT:-krea2-raw}" \
    --run-directory "${RUN_DIR:-runs/krea2_ref2img_lora}" \
    "$@"
