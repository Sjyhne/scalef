#!/usr/bin/env bash
# Exam ablation from updated_attention_experiment_agent_brief.md (matched hyperparameters).
#
# Suites:
#   attention      — methods A/B/C (Fourier/HashGrid mlp_tcnn baselines + HashGrid level-attention)
#   attention_full — A/B/C/D with area + s2_psf HR→LR (8 runs/scene; run names suffixed _area / _s2_psf)
#
# Examples:
#   DATASET=satburst_real ./scripts/run_attention_experiment_suite.sh
#   DATASET=worldstrat_sweet SUITE=attention_full DEVICE=cuda:0 ./scripts/run_attention_experiment_suite.sh
#   DATASET=worldstrat_bitter WORLDSTRAT_DATA_ROOT=worldstrat_datasets/worldstrat_bitter ./scripts/run_attention_experiment_suite.sh
#   SUITE=attention_full DEVICE=cuda:7 ./scripts/run_attention_experiment_suite.sh
#   SUITE=attention_full DEVICE=cuda:0 LIMIT_SAMPLES=1 ITERS=100 ./scripts/run_attention_experiment_suite.sh
#   ./scripts/run_attention_experiment_suite.sh --skip-existing
#
# Full-patch training on 1024 LR (4096 HR) needs far more GPU memory than 256 LR scenes;
# default MAX_LR_SIDE=256 skips s2_scene_center_v2_{512,1024}. Override with MAX_LR_SIDE=0
# to include all sizes, or MAX_LR_SIDE=512 for a middle ground.
set -euo pipefail
cd "$(dirname "$0")/.."

_worldstrat_default_root() {
  case "${1}" in
    worldstrat_sweet) echo "worldstrat_datasets/worldstrat_sweet" ;;
    worldstrat_bitter) echo "worldstrat_datasets/worldstrat_bitter" ;;
    worldstrat_test) echo "worldstrat_test_data" ;;
    *) return 1 ;;
  esac
}

if [[ "${DATASET:-}" == worldstrat_* ]]; then
  _ws_root="${WORLDSTRAT_DATA_ROOT:-$(_worldstrat_default_root "${DATASET}")}"
  if [[ ! -d "${_ws_root}" ]]; then
    echo "ERROR: WorldStrat data directory not found: ${_ws_root}" >&2
    echo "Set WORLDSTRAT_DATA_ROOT to the parent folder that contains <area>/hr/ and <area>/lr/." >&2
    echo "See docs/WORLDSTRAT_EXPERIMENTS.md (areas listed in worldstrat_datasets.txt)." >&2
    if [[ -L "${_ws_root}" ]]; then
      echo "Broken symlink: ${_ws_root} -> $(readlink "${_ws_root}" 2>/dev/null || echo '?')" >&2
    fi
    exit 1
  fi
  unset _ws_root
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

"$PYTHON" scripts/run_experiment_suite.py \
  --suite "${SUITE:-attention}" \
  --dataset "${DATASET:-satburst_synth}" \
  ${WORLDSTRAT_DATA_ROOT:+--worldstrat_data_root "$WORLDSTRAT_DATA_ROOT"} \
  --device "${DEVICE:-cuda:0}" \
  --df "${DF:-4}" \
  --lr_shift "${LR_SHIFT:-1.0}" \
  --aug "${AUG:-none}" \
  --num_samples "${NUM_SAMPLES:-16}" \
  --iters "${ITERS:-2000}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --learning_rate "${LEARNING_RATE:-2e-3}" \
  --weight_decay "${WEIGHT_DECAY:-0}" \
  --seed "${SEED:-6}" \
  --eval_crop_lr_size "${EVAL_CROP_LR_SIZE:-auto}" \
  --max_lr_side "${MAX_LR_SIDE:-256}" \
  ${LIMIT_SAMPLES:+--limit_samples "$LIMIT_SAMPLES"} \
  ${CUDA_EXPANDABLE:+--cuda_expandable_segments} \
  "$@"
