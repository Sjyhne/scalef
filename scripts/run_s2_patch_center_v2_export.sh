#!/usr/bin/env bash
# Rebuild data/s2_patch_center_v2 from s2_norway GeoTIFFs (same layout as data/s2_patch_center/...).
# Optional: after downloading with download_s2_timeseries.py --write-raw-b432, add e.g.
#   --raw_b432_folder "${ROOT}/s2_norway/geotiff_raw"
# to also emit sample_XX_raw.npz (B4/B3/B2 reflectance) for post-SR RGB via s2_reflectance_utils.
# Requires: numpy, pillow, rasterio (e.g. pip install -e . in the project venv).
# Override interpreter: PYTHON=/path/to/python ./scripts/run_s2_patch_center_v2_export.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-}"
if [[ -z "${PY}" && -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi
PY="${PY:-python3}"
exec "${PY}" "${ROOT}/scripts/export_s2_patch_to_satburst_format.py" \
  --tci_folder "${ROOT}/s2_norway/geotiff" \
  --output_dir "${ROOT}/data/s2_patch_center_v2/scale_4_shift_1.0px_aug_none" \
  --patch_size 64 \
  --df 4 \
  --convert_raw_tif_to_rgb \
  "$@"
