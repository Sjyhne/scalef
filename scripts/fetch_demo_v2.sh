#!/usr/bin/env bash
# Re-fetch the 6-tile demo with pinned/lowest-cloud STAC + on-disk SCL rescore.
# Writes data/s2_revisits/national_2025_v2/ and does not touch national_2025/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOGDIR="$ROOT/production/national_2025/logs"
mkdir -p "$LOGDIR" "$ROOT/data/s2_revisits/national_2025_v2"
# 32VNM first — that is the visual test granule.
for mgrs in 32VNM 32VNL 32VPL 32VPM 32VPN 32VNN; do
  echo "===== fetch $mgrs $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  python -u "$ROOT/scripts/fetch_national_mgrs.py" \
    --mgrs "$mgrs" \
    --fetch-only \
    --tile \
    --s2-out "$ROOT/data/s2_revisits/national_2025_v2/$mgrs"
  echo "===== done $mgrs $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
done
echo "===== all demo v2 fetches done $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
