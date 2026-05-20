#!/usr/bin/env bash
# Export multi-size center patches from the same S2 time-series stack with a **shared mosaic origin**.
#
# All sizes use the same (y0, x0) on the common mosaic (from the 64px reference grid), so the
# top-left 64×64 LR window is the same geographic footprint in s2_scene_center_v2_{64,128,256,...}.
#
# Creates satburst-compatible folders:
#   ${DATA_ROOT}/s2_scene_center_v2_<SIZE>/scale_4_shift_1.0px_aug_none/
# for SIZE in {64,128,256,512,1024} (default DATA_ROOT=data_real).
#
# Requires:
# - s2_norway/geotiff (TCI GeoTIFFs) and s2_norway/geotiff_raw (B4/B3/B2 reflectance GeoTIFFs)
#   produced by download_s2_timeseries.py --format geotiff --write-raw-b432
# - rasterio + pillow installed in the current python environment.
#
# Override interpreter:
#   PYTHON=/path/to/python ./scripts/run_s2_scene_center_v2_export_sizes.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-}"
if [[ -z "${PY}" && -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
PY="${PY:-python3}"

TCI_FOLDER="${ROOT}/s2_norway/geotiff"
RAW_B432_FOLDER="${ROOT}/s2_norway/geotiff_raw"
DATA_ROOT="${DATA_ROOT:-data_real}"

DF=4
LR_SHIFT="1.0"
AUG="none"
REFERENCE_PATCH=64

ORIGIN_FILE="${ROOT}/${DATA_ROOT}/.s2_scene_center_v2_mosaic_origin.json"
SIZES=(64 128 256 512 1024)

for SZ in "${SIZES[@]}"; do
  SAMPLE_ID="s2_scene_center_v2_${SZ}"
  OUT_DIR="${ROOT}/${DATA_ROOT}/${SAMPLE_ID}/scale_${DF}_shift_${LR_SHIFT}px_aug_${AUG}"
  echo "==> Exporting ${SAMPLE_ID} (patch_size=${SZ}) -> ${OUT_DIR}"

  EXTRA=()
  if [[ -f "${ORIGIN_FILE}" ]]; then
    EXTRA=(--mosaic_origin_file "${ORIGIN_FILE}")
  else
    EXTRA=(
      --align_mosaic_origin_to_patch_size "${REFERENCE_PATCH}"
      --write_mosaic_origin "${ORIGIN_FILE}"
    )
  fi

  "${PY}" "${ROOT}/scripts/export_s2_patch_to_satburst_format.py" \
    --tci_folder "${TCI_FOLDER}" \
    --raw_b432_folder "${RAW_B432_FOLDER}" \
    --output_dir "${OUT_DIR}" \
    --patch_size "${SZ}" \
    --df "${DF}" \
    --convert_raw_tif_to_rgb \
    "${EXTRA[@]}" \
    "$@"
done

echo "Done. Shared mosaic origin: ${ORIGIN_FILE}"
echo "Re-run experiments with --eval_crop_lr_size 64 (suite default auto) after replacing data_real scenes."
