#!/usr/bin/env bash
# Export multi-size center patches from the same S2 time-series stack.
#
# Creates satburst_synth-compatible folders:
#   data/s2_scene_center_v2_<SIZE>/scale_4_shift_1.0px_aug_none/
# for SIZE in {64,128,256,512,1024}.
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

DF=4
LR_SHIFT="1.0"
AUG="none"

SIZES=(64 128 256 512 1024)

for SZ in "${SIZES[@]}"; do
  SAMPLE_ID="s2_scene_center_v2_${SZ}"
  OUT_DIR="${ROOT}/data/${SAMPLE_ID}/scale_${DF}_shift_${LR_SHIFT}px_aug_${AUG}"
  echo "==> Exporting ${SAMPLE_ID} (patch_size=${SZ}) -> ${OUT_DIR}"
  "${PY}" "${ROOT}/scripts/export_s2_patch_to_satburst_format.py" \
    --tci_folder "${TCI_FOLDER}" \
    --raw_b432_folder "${RAW_B432_FOLDER}" \
    --output_dir "${OUT_DIR}" \
    --patch_size "${SZ}" \
    --df "${DF}" \
    --convert_raw_tif_to_rgb \
    "$@"
done

echo "Done."

