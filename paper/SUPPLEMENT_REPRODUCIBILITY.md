# Supplement notes recovered from the reported runs

Verified against the implementation and the records of the reported experiments. The evaluated fits were launched from git `3e2765f2a30279db65e70904d163b4d4f62366c0` (2026-09-25, dirty work tree at launch). Each `metrics.json` stores that provenance. Do not archive the later checkout as if it were that release.

## 1. Reference registration

Estimator: `scripts/audit_hr_lr_spatial_alignment.py`, `audit_s2_dir`. Method string `hr_phase_corr + local_hr_mse_refine`. Application at load: `s2_dataset.py`, `resolve_hr_eval_alignment` and `_apply_hr_spatial_shift`.

Inputs. The moving image is the base Sentinel-2 frame (the revisit moved to index 0), bands B04, B03, B02. Digital numbers are divided by 10 000 when the peak exceeds 1.5, then clipped to `[0, 1.5]` (`_l2a_to_reflectance`). The reference is the NIB orthophoto after per-band histogram matching to that base frame (`_harmonize_hr_histogram_match`). The audit sets `no_hr_spatial_align=True`, so a previously stored shift is not applied while the shift is estimated. Both images are converted to grayscale with weights 0.2989, 0.5870, 0.1140. Valid pixels are those with all three channels strictly positive. An 8 HR-pixel border is excluded from the search mask.

Search. The base frame is bilinearly upsampled to the HR grid (`cv2.INTER_LINEAR`). Phase correlation is `skimage.registration.phase_cross_correlation` on the full masked grayscale pair, `upsample_factor=100` (0.01 HR pixel). There is no separate integer search window: the Fourier estimate covers the image, and invalid pixels are zeroed rather than passed as a mask. Local refinement then searches a square of radius 2.0 HR pixels about that estimate, step 0.25 HR pixel (`_exhaustive_refine`). The score is masked grayscale MSE. Stored shifts are in HR pixels of the `df=4` (2.5 m) grid. The value applied to the orthophoto is the negation of the shift that aligns the upsampled Sentinel-2 image to the orthophoto.

Interpolation and failure. The orthophoto is shifted with `scipy.ndimage.shift`, order 1, `prefilter=False`, constant 0, then clipped to `[0, 1]`. The evaluation mask is shifted with order 0 and kept where the result exceeds 0.5 and the shifted orthophoto is still positive. An empty search mask raises and that field is not written. If no manifest entry resolves, no shift is applied. Histogram matching failure keeps the unharmonized orthophoto and still applies a resolved shift.

Independence. `eval/spatial_alignment.json` stores one entry per LR512 directory name under `tiles` (112 entries) and keeps city keys as fallback. Lookup order in `resolve_hr_eval_alignment` is: the revisit directory name, then `patch_grid.parent_tile_id`, then `{city}_lr512`, then the city name. A nested child therefore inherits only the frozen correction of its own parent when that parent id is recorded. The city-level keys remain a fallback if the parent entry is missing; they are not a second estimate. Shifts are scaled by `current_df / recorded_df` when the evaluation grid changes. Training revisits are not shifted.

## 2. Loss normalization

One optimizer step, shipped mix `within`, is one revisit. The dataloader uses batch size 1 and `shuffle=False`, so a pass visits each admitted frame once. Inside that step, `LrTileSampler` draws `k=4` windows of 128 LR pixels. The training loss is the masked mean of the elementwise Charbonnier residual on the pixels that are both inside the training mask and clear (`eval/lr_holdout.py`, `masked_mean` and `elementwise_recon`). The denominator is the count of selected spatial locations, and the residual is then divided by the channel count. Frames are weighted equally because each frame is one step. Windows of one frame are not weighted equally: their selected pixels are pooled. Cloudier frames still receive one step, taken over fewer pixels.

The validation score used for stopping is not that training loss and not an MSE. `compute_holdout_val_loss` evaluates the same Charbonnier residual (`ε=0.01`) on held-out blocks, then takes an unweighted mean of the per-chunk masked means. At most four frames are scored (`val_frame_ids`: every frame when there are four or fewer; otherwise indices `0, s, 2s, 3s` with `s = n // 4`). Chunks are weighted equally, so a frame with more held-out blocks contributes more terms.

Manuscript equation. The displayed objective averages frames by `1/T` and then sums over windows. The implementation matches the surrounding sentence (a masked mean over selected pixels), not the inner sum. The quantity called hold-out MSE in the text is this Charbonnier mean. Reported `early_stop.best_val_loss` values are the smoothed Charbonnier score. The fits are internally consistent with the training loss; the equation and the word MSE should be changed. No refit is required for that correction.

## 3. Checkpoint selection

Hold-out construction (`build_holdout_mask`). Fraction 0.1. Block side `--holdout_block 0`, which resolves to 8 LR pixels on a 512 field (`resolve_holdout_block`: `max(8, round(8 * side / 512))`). Blocks are drawn on the block grid with a CPU `torch.Generator` whose seed is the frame index, not the training seed. Seeds 6, 7 and 8 therefore share the hold-out mask. Pixels in a selected block are training pixels; the complement is validation. The mask is independent per frame, including the anchor frame.

Stopping (`EarlyStopState`). The metric name is `holdout_mse`. The value passed in is the Charbonnier validation mean above. EMA: `s_0 = v_0`, then `s_t = 0.4 v_t + 0.6 s_{t-1}` (`default_early_stop_ema`). Patience, the regression test, and the best-score comparison all use `s_t`. Absolute improvement tolerance is `--early_stop_min_delta` default `0.0005` (the help text says this default is for LPIPS; the same absolute value is applied to the Charbonnier score). Absolute regression tolerance is `0.01` (`default_early_stop_regression` for `holdout_mse`). Validation is every 200 iterations (`eval_every`). Checks before iteration 1000 are logged and do not update the checkpoint or the patience counter. After that, eight non-improving checks stop training. A regression of the EMA by more than 0.01 forces the patience counter to its limit.

Restoration loads the raw parameter dictionary saved on the iteration whose EMA was best (`restore_best`). It does not load an exponential average of the weights, and it does not select the iteration with the best raw score when that iteration differs from the EMA. On the recorded Ålgård seed-6 run the restored iteration is 4600, the stored best score is the EMA value 0.29536, and training ran to the 5000-iteration cap (`stopped: false`, `restored_best: true`).

## 4. Configuration

Decoder (`models/mlp_tcnn.py`). tiny-cuda-nn `FullyFusedMLP`, hidden activation ReLU, output activation None, width 256, depth 4 (`n_hidden_layers = 3`), fp16. If the fused network fails to construct, the same widths are built as `CutlassMLP`.

Geometric affine (`models/utils.py`, `get_learnable_affines`). One 2×3 matrix per frame, initialized to identity. Frame 0 has `requires_grad=False`. Radiometric map (`models/inr.py`, `ChannelAffine1x1`): per channel, scale initialized to 1 and bias to 0. Frame 0 is applied as the constants 1 and 0, so its parameters receive no gradient.

Seeds. `--seed` default 6 sets `torch`, CUDA, NumPy and Python RNGs, and the LR-tile sampler. `cudnn.deterministic=True`, `cudnn.benchmark=False`. The hold-out mask seed is the frame index, as above. Fourier matrix seed is unused for the hash-grid runs.

Optimizer. AdamW, learning rate `1e-3`, weight decay `0.05`, PyTorch defaults `β=(0.9, 0.999)`, `ε=1e-8`. `CosineAnnealingLR` with `T_max` equal to the iteration cap and `eta_min=1e-6`. Gradient scaling is on for CUDA: `GradScaler` initial scale `2^15`. `grad_accum` default 1. A forward that would exceed `2^23` query rows is split (`INRBase.MAX_FORWARD_ROWS`). Shipped batching is `k=4` tiles of 128 LR pixels from one frame.

Hash grid, argparse defaults actually used: 16 levels, 2 features per level, `log2` hashmap size 21, smoothstep interpolation, finest resolution equal to the LR grid (512), coarsest resolution `max(8, 512 // 4)`.

Gaussian widths (`models/s2_psf_forward.py`, `DEFAULT_S2_PSF_SIGMA_M_BY_BAND`). RGB values used by `s2_psf_m` are B02 2.8 m, B03 3.25 m, B04 4.2 m, stated in code as a legacy calibration at 10 m GSD. No measurement note, citation, or derivation script is in the repository; the source of those three numbers is unrecoverable from this archive. At the 2.5 m prediction grid they become `σ / 2.5` pixels (1.12, 1.30, 1.68). The kernel is a separable Gaussian, truncated at 4σ, reflect-padded, then area-downsampled by 4.

## 5. Timing

`training_time_seconds` is wall time from `time.time()` immediately after the optimizer, dataloader and tile sampler exist, through the end of the loop (`optimize.py`, `main`). Dataset loading and model construction are outside it. Hold-out mask construction is inside it. The loop includes validation checks. There is no discarded warm-up iteration.

`training_instrumentation.total_step_time_seconds` is separate. Each training step is bracketed by CUDA events around `train_one_iteration` only. Events are summed after `torch.cuda.synchronize`. Validation, checkpoint copies and host-side logging are not inside those events. The instrumentation field `timing_source` is `cuda_events` on the reported GPU runs. The Ålgård seed-6 wall time is 80.98 s for 5000 iterations, which is the figure class quoted as about 82 s. That number includes the validation work inside the wall-clock interval.

## 6. Dataset membership

The seventeen named sites are the keys of `FOCUS_PROJECT_BY_CITY` in `s2_dataset.py`, in the order used by `scripts/build_frozen17_table.py`:

| Site | NIB project folder |
| --- | --- |
| asker | `01_asker_akershus` |
| vennesla | `02_vennesla_agder` |
| trondheim | `03_trondheim_trondelag` |
| bergen | `04_bergen_vestland` |
| rana | `05_rana_nordland` |
| tromso | `06_tromso_troms` |
| amli | `07_amli_agder` |
| stavanger | `08_stavanger_rogaland` |
| algard | `01_eksport_7531643_1_tile01` |
| naerbo | `01_eksport_7531643_1_tile01` |
| flekkefjord | `02_eksport_7532043_1` |
| rafsbotn | `03_eksport_7532042_1_tile01` |
| nittedal | `04_eksport_7532041_1` |
| melhus | `05_eksport_7531640_1` |
| alta | `06_eksport_7528850_1` |
| karasjok | `08_eksport_7532042_1_tile03` |
| kautokeino | `09_eksport_7532042_1_tile04` |

The first eight are the original focus exports. The last nine are later NIB exports. Ålgård and Nærbø share one export folder. A written sampling design, beyond “these are the NIB projects for which a Sentinel-2 LR512 stack and an orthophoto export both exist,” is not in the repository.

The eleven nested-study parents are the lowest `parent_tile_id` in each project folder that has a complete-HR patch grid, fixed before scoring (`scripts/run_nested_float_validation.py`, `select_parents`; recorded in `paper/results/run_manifests/nested_floatval_v3_manifest.json`):

`algard_p01_02`, `alta_p04_04`, `asker_p02_02`, `bergen_p02_04`, `flekkefjord_p02_03`, `melhus_p01_01`, `nittedal_p02_01`, `rafsbotn_p01_02`, `tromso_p02_03`, `trondheim_p02_03`, `vennesla_p04_07`.

Rana, Åmli, Stavanger, Karasjok and Kautokeino have no nested grid in that manifest. Nærbø shares Ålgård’s project, so it does not add an eleventh-plus parent. The nested manifest contains 85 LR512 fields.

Orthophoto resolution. The file the loader opens is the delivered 1 m product `Eksport-nib.tif` inside the NIB export zip (`resolve_focus_hr_path`). Evaluation warps that raster onto the 2.5 m Sentinel-2 AOI grid with bilinear resampling for RGB and nearest-neighbour for the coverage mask (`_warp_rgb_to_grid`). If the warped peak exceeds 1.5 the values are divided by 255 and clipped to `[0, 1]`. The native capture GSD, and any resampling Kartverket applied before writing the 1 m export, are not recorded in this repository.

## 7. Pilot mosaic recipe (32VNM)

Verified against `production/seams/32VNM_stack/identity_plan.json` and `production/mosaics/32VNM_icm_ramp_sr_2p5m.tif.json`. This is the SCL identity-and-ramp pilot, not a later cloud-mask or overlap-feather run.

SCL. Cloud classes are 8 (medium), 9 (high) and 10 (cirrus). Shadow (3) is not part of the cloud fraction. Nodata is class 0. `cloud_frac` is the fraction of valid pixels. `valid_frac` is the fraction of all pixels that are not nodata (`eval/s2_cloud_mask.py`, `score_scl_cell`). The planner also rejects snow (class 11) above 0.05 of valid pixels. SCL is read at native resolution and nearest-neighbour resampled onto the 10 m B04 grid (`scripts/map_lr512_clear_counts.py`, `_read_scl_10m`).

Scene per day. `_one_per_day` keeps, for each UTC calendar day, the STAC item with the lowest `eo:cloud_cover`. An equal cloud-cover tie keeps the item encountered first. That encounter order is the STAC listing order and is not otherwise defined.

Identity (`scripts/plan_identity_icm.py`, `assign_icm`). Center date 2025-07-15. Hard cloud 0.02. Slack 0.05. The primary is the date closest to the center that is within the hard cap on at least half of the cells; ties break by coverage, then by the date string (`date_ladder`). Recorded primary: 2025-07-12. 143 cells. Each cell joins the primary when that date is in its stack and its cloud fraction is at most 0.05. Leftover cells then take the modal 4-neighbour date that they themselves pass at the slack cap. Neighbour order is east, west, south, north. Vote ties follow `Counter.most_common`, which keeps the first of those neighbours. Sweeps stop at 32 or when a sweep changes nothing (the plan recorded 3). A later polish pass may still change a cell, including a primary cell, when the change reduces the local cut count, then prefers the cell’s independent date, then prefers the date closer to the center, then the date string. A date absent from the cell is never assigned.

Land mask used when a mainland filter is requested: Natural Earth 1:50 m Norway outline `production/cloud_availability/norway_outline_ne50m.geojson` (`scripts/land_mask_lr512.py`). Default test is a 3×3 sample grid; `min_land_frac=0` keeps a cell if any sample hits land. Latitudes at or above 72° and negative longitudes are excluded by default (Svalbard and Jan Mayen). The 32VNM identity plan itself is a cell-date assignment, not a second land-mask pass.

Ramp (`scripts/mosaic_seam_ramp.py`, `_smooth_ramp`). On a date cut only, each side of the shared edge is adjusted over 128 HR pixels. At 2.5 m that is 320 m. The weight is `0.5 * (1 - cos(π u))` with `u` running from 1 at the edge to 0 at 128 pixels inward. Same-date edges are not ramped. The recorded mosaic sidecar has `ramp_px: 128`, `date_cuts_only: true`, 438 tiles and 90 ramped edges. Interior pixels are unchanged. The ramp operates on the already written 2.5 m reflectance tiles, in geographic mosaic order, not inside the fit.

The six-tile national-demo selector (`scripts/select_norway_mgrs_block.py`) is a different planner. Its score is `0.35 * season_clear + 0.45 * qualified_clear_fraction + 0.20 * mainland_area`, adjacency uses factor 1.65, and ties break by sorted MGRS id. It was not the selector that produced the 32VNM identity plan.

## 8. Benchmark release identifiers

Evaluated code revision recorded in the confirmatory metrics: `3e2765f2a30279db65e70904d163b4d4f62366c0`, dirty at launch. The commit message is “Rescore diagnostics under the surface-reflectance protocol and diagnose the Åmli SSIM loss.” Later source edits are not part of that launch.

What the runs loaded for alignment is `eval/confirmatory_eval_manifest.v2.json`, sha256 `4dd3306fed54cc0aa53c930f5e47008279a5be1e2a4136e9b8e38ee0dd402e6f` (matches the file on disk, and matches `eval_alignment.sha256` inside the Ålgård seed-6 metrics). That manifest’s recorded source hash for `eval/spatial_alignment.json` is `0a8a30e3b229f4a16e9fb337bc7bc88a026c6256b88c7226910356accf64895b`. The live `eval/spatial_alignment.json` now hashes to `c61f6f508a10543ab71caaa733fb4d234ead3bfb50da6ea85d1fce85dbc3418b` and contains 112 per-field tile entries. The reported named-site numbers are pinned to the v2 manifest, not to the live alignment file.

Command archive: `paper/results/run_manifests/`. The named-site mask refit is `confirmatory_v6_masks__run_manifest.json` (created 2026-09-25T18:11:56Z). Nested parents are `nested_floatval_v3_manifest.json`. Configurations in those commands are the shipped recipe: hash-grid, `s2_psf_m`, Charbonnier `ε=0.01`, `k=4` of 128, mix `within`, hold-out 0.1, block 0 (resolved to 8), patience 8, minimum 1000 iterations, cap 5000, `eval_every` 200, seed 6/7/8 where the confirmatory jobs say so.

Historical whole-window freeze. `eval/spatial_alignment.pre_lr512.json` has the same estimator and the 17 city entries, and it has no `tiles` object. That is the one-shift-per-site package. `paper/results/run_manifests/confirmatory_v2_lr512align__run_manifest.json` is the corresponding run archive. The current release is the per-LR512 correction plus the later mask and reflectance protocol (`confirmatory_v6_masks` and the v2 evaluation manifest). A freeze tarball was not written from the present work tree: that tree is not the evaluated checkout, and `scripts/build_freeze_manifest.py` would stamp it as `checkout_at_manifest`.
