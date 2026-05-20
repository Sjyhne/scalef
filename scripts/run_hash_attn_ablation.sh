#!/usr/bin/env bash
# HashGrid MLP baseline vs level-attention across all discovered scenes.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/run_experiment_suite.py \
  --suite hash_attn \
  --dataset "${DATASET:-satburst_synth}" \
  --device "${DEVICE:-cuda:0}" \
  --iters "${ITERS:-2000}" \
  --eval_crop_lr_size "${EVAL_CROP_LR_SIZE:-off}" \
  "$@"
