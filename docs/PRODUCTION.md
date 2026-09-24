# Norway / national production SR

How to super-resolve entire MGRS granules (or any place) **without NIB HR ground truth**.

Project goals, publish checklist, and evidence log: [`HASHGRID_SUPERF.md`](HASHGRID_SUPERF.md).

Development benches still use NIB for LPIPS. Production does not: each tile is an independent test-time INR fit on Sentinel-2 revisits only. Stopping uses `holdout_mse` (patience 8). Deliverable is georeferenced `sr_pred.tif` at 2.5 m.

Recipe (from `docs/METHOD.md`): **LR 512 × 512**, **k4** fused tiles (128×4), `s2_psf_m`, Charbonnier ε=0.01, `holdout_mse` stop. Do not raise hash table / `n_levels`.

---

## Pipeline

```text
1. Fetch revisits     scripts/fetch_s2_revisits.py   → data/s2_revisits/<mgrs_or_aoi>/
2. Tile granule       scripts/make_granule_tiles.py  → *_t512_y##_x##/ + manifest
                      (+ --mainland-only / --min-clear to drop ocean & cloudy cells)
3. Train + export     scripts/run_production.py      → single_samples/<parent>/sample/prod_k4_*/
4. Mosaic             scripts/mosaic_granule_sr.py   → production/mosaics/<parent>_sr_2p5m.tif
```

Multi-granule driver (tile + train + mosaic):

```bash
python scripts/run_granule_batch.py \
  --parents asker bergen rana tromso amli stavanger \
  --gpus 8 --skip-existing
```

Phase 4 planning and resumable national demo:

```bash
# 1. Probe candidate windows. SCL/B04/B08 mode is network-heavy, so it
# checkpoints each window/tile under --out-dir and resumes by default.
python scripts/compare_norway_season_windows.py \
  --mode both --scl-max-days 20 \
  --out-dir production/cloud_availability/season_compare_2025

# 2. Deterministically select a compact or merely connected 6-9 tile set.
python scripts/select_norway_mgrs_block.py \
  --season-artifact production/cloud_availability/season_compare_2025/season_compare.json \
  --heatmap-dir production/cloud_availability/lr512_norway_july2025_pm45 \
  --window jul_pm45 --count 6 --mode block --min-clear 6 \
  --out production/national_demo/mgrs_selection.json

# 3. Inspect an exact no-side-effect command plan, then execute/resume later.
python scripts/run_norway_national_demo.py \
  --selection-manifest production/national_demo/mgrs_selection.json \
  --heatmap-dir production/cloud_availability/lr512_norway_july2025_pm45 \
  --dry-run
```

The selection manifest records the chosen shared date range, normalized score,
per-tile reasons, mainland-qualified LR512 counts, candidate ranking, and
adjacency graph. `--mode block` adds a compactness preference; `connected`
requires connectivity but prioritizes availability. If mainland-qualified
counts have already been exported as JSON, pass `--qualified-counts`; otherwise
the selector computes them from the clear-count GeoTIFFs and mainland mask.
The selected season and heatmap `date_range` must match; regenerate the LR512
heatmaps for the recommended window before selecting it.

The national driver calls, in order, `fetch_s2_revisits.py`,
`make_granule_tiles.py --mainland-only --min-clear`, `run_production.py`,
`mosaic_granule_sr.py`, then `mosaic_national_sr.py` for available granule
mosaics. Its JSON log contains exact argv/shell commands, wall times, failures,
expected outputs, and resume decisions. Dry-run does not issue STAC requests or
start CPU/GPU processing.

National Norway (MGRS fetch unit, LR512 train unit — **mainland only**):

```bash
python scripts/run_granule_batch.py \
  --parents 32VNM 32VKL ... \
  --mainland-only \
  --min-clear 6 \
  --clear-counts-dir production/cloud_availability/lr512_norway_july2025_pm45 \
  --gpus 8 --force-tiles
```

The **74** mainland MGRS tiles in `production/cloud_availability/norway_mgrs_list.json` are the fetch / bookkeeping set. Training should **not** run all ~441 LR512 cells per granule: use `--mainland-only` so only cells that overlap the NE50 mainland+islands outline are written and trained. Optionally gate on the clear-count heatmap with `--min-clear`.

### 1. Fetch

Same stack as development: cloud-filtered L2A RGB(+NIR) revisits on one MGRS tile, `meta.json` with frames.

**Cloud policy for national / full-granule runs.** Filtering today is tied to the **fetch AOI** in `fetch_s2_revisits.py` (`--size-km` / bbox), not to each LR512 production tile. For Norway:

- Prefer fetching with an AOI that covers the **whole MGRS footprint** used for tiling (or accept scene-level STAC cloud only as a weak prefilter).
- Prefer the per-LR512 clear heatmap (`scripts/map_lr512_clear_counts.py`) + `--min-clear` at tile/train time.
- Training still does **not** mask residual cloud pixels in the loss — same as the NIB dataset path.

Until per-date re-scoring inside each 512 window exists, do not treat a small study-AOI fetch (Asker-style) as cloud-safe for the rest of that granule.

MGRS catalog: `production/cloud_availability/norway_mgrs_list.json` (mainland + nearshore islands; excludes Svalbard / Jan Mayen / open-ocean pads).

### 2. Tile into LR512 AOIs

```bash
python scripts/make_granule_tiles.py \
  --src data/s2_revisits/asker \
  --side 512
```

National / coastal granules — keep only mainland-overlapping cells:

```bash
python scripts/make_granule_tiles.py \
  --src data/s2_revisits/32VKL \
  --side 512 \
  --mainland-only \
  --min-clear 6 \
  --clear-counts-dir production/cloud_availability/lr512_norway_july2025_pm45
```

By default this tiles the **full MGRS raster** (not the small development `aoi_window`). Pass `--use-aoi-window` only to restrict to the meta crop.

Writes non-overlapping 512² windows under `data/s2_revisits/{parent}_t512_y{iy}_x{ix}/` and:

`data/s2_revisits/<parent>/granule_tiles_lr512_manifest.json`

Incomplete edge strips (height/width not divisible by 512) are skipped. A full ~10980² MGRS tile yields ~21×21 ≈ **441** AOIs before land/clear filters (cost model often quotes 484 for a padded grid). With `--mainland-only`, coastal granules keep far fewer.

Standalone filter for an existing full-grid manifest (no retile):

```bash
python scripts/land_mask_lr512.py \
  --manifest data/s2_revisits/32VKL/granule_tiles_lr512_manifest.json \
  --land-mask production/cloud_availability/norway_outline_ne50m.geojson \
  --min-clear 6 \
  --clear-counts-dir production/cloud_availability/lr512_norway_july2025_pm45
```

`run_production.py` also accepts `--mainland-only` / `--min-clear` to filter at train time if the manifest was built without the land mask.

### 3. Production train (no HR)

```bash
python scripts/run_production.py \
  --manifest data/s2_revisits/asker/granule_tiles_lr512_manifest.json \
  --gpus 8 \
  --skip-existing
```

Each worker calls `optimize.py` with `--allow_no_hr` and QGIS export on. Outputs per tile:

| Path | Role |
|--|--|
| `single_samples/<parent>/sample/prod_k4_<tile_id>/qgis/sr_pred.tif` | **SR product** (2.5 m, CRS of the S2 stack) |
| `.../qgis/s2_bilinear.tif` | Bilinear upsample baseline |
| `.../qgis/s2_lr.tif` | Native 10 m LR window |
| `.../metrics.json` | Timing, holdout stop; GT metrics are null/NaN |

Summary JSON: `single_samples/sweep_results/production_run.json`.

Smoke test one tile:

```bash
python scripts/run_production.py \
  --manifest data/s2_revisits/asker/granule_tiles_lr512_manifest.json \
  --limit 1 --gpus 1
```

Or a single AOI directly:

```bash
python optimize.py \
  --dataset asker --s2-dir data/s2_revisits/asker_t512_y00_x00 \
  --run_name prod_k4_smoke --allow_no_hr \
  --lr_degradation s2_psf_m --recon_loss charbonnier --charbonnier_eps 0.01 \
  --lr_tile 128 --lr_tiles_per_step 4 --lr_tile_mix within \
  --early_stop_metric holdout_mse --early_stop_patience 8 \
  --early_stop_min_iters 1000 --iters 5000 --eval_every 200 \
  --hr_render_tile 2048 --spatial_holdout 0.1
```

---

## Remaining limitations

- Weight checkpoints (re-render still means retrain)
- Partial edge tiles (only full 512²)
- Setup-time accounting in the cost model
- Per-date cloud re-score inside each LR512 window at train time (heatmap + `--min-clear` is the planning proxy)
- Cross-granule mosaic currently materializes the merged array in memory; use
  the selected 6-9 tile demo, not all Norway, until a streaming VRT/COG path is added.

Development NIB path remains: `make_complete_patch_tiles.py` + `bench_complete_patches.py` (requires HR).

---

## Flags that matter

| Flag | Production value |
|--|--|
| `--allow_no_hr` | **Required** when no NIB package exists for the tile |
| `--no_qgis_export` | **Off** for write-out (runner leaves export on) |
| `--early_stop_metric holdout_mse` | Only viable stop without GT (EMA 0.4 + reg 0.01 auto) |
| `--max_base_cloud_frac 0.02` | Frozen identity frame = closest to centre/NIB among frames with *cell* SCL cloud ≤2% (else clearest). Do not freeze chronological frame 0. |
| ICM identity plan | After the 2% pick: `scripts/plan_identity_icm.py` then `run_production.py --identity-plan …` (5% slack to the seasonal primary). Retrain only date-changed cells. |
| Date-cut ramp | Delivery mosaic: 0-overlap assembly + cosine ramp **only** on leftover date-boundary edges (`seam_stack_block.py --stage delivery` or `mosaic_seam_ramp.py` on those edges). Not full-tile colour matching. |
| `--hash_log2_hashmap_size` / `--hash_n_levels` | Leave defaults |
| `--mainland-only` | National: keep LR512 cells overlapping mainland outline |
| `--min-clear` + `--clear-counts-dir` | National: skip cells below clear-day floor |

### Ops notes

- **Setup time** ≈ 0.5 s/AOI (`scripts/measure_setup_time.py`) — negligible vs train.
- **GPU memory:** `metrics.json["gpu_memory"]["process_peak_used_gb"]` is sampled
  through NVML and includes the tiny-cuda-nn arena. PyTorch allocator peaks do not.
  `run_production.py` propagates per-tile values plus aggregate mean/max into its
  run summary.
- **Stopping replay / live traces:** `scripts/bench_stopping_rules.py` (offline) or `--run` for k4+`--force_hr_eval`.
- **Mosaic:** `scripts/mosaic_granule_sr.py` requires `BIGTIFF=YES` (full-granule SR exceeds classic TIFF 4 GB). Seam appendix (colour / overlap / halo, not delivery): `paper/APPENDIX_SEAMS.md`.
- **Verified full-granule delivery:** `production/mosaics/32VNM_icm_ramp_sr_2p5m.tif`
  (438 cells, 43,008² px, 16.19 GB). Plan, timings, and checks:
  `production/seams/32VNM_full/REPORT.md`.
- **Season window:** July ±45 d is a pilot. Compare candidates (cloud + snow + shadow + SCL vegetation + B04/B08 NDVI stability) with `scripts/compare_norway_season_windows.py`. Probe checkpoints resume by default; use `--no-resume` only to recompute. For green-season heatmaps use `map_lr512_clear_counts.py --max-snow-frac 0.05` (legacy maps did not gate SCL snow).
- **Next national-style trial:** lock the shared window from the season compare, then ~6 **adjacent** MGRS with `--min-clear 6–8`, not NIB-dated scattered sites.
