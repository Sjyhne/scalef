# ScaleF current method

Decision record for the test-time INR super-resolution pipeline, as of 4 September 2026. Numbers are from the asker AOI unless noted. Selection metric is LPIPS (VGG, lower is better); PSNR and SSIM are reported but not used to rank.

The production recommendation:

> Fit one INR per **5.12 km (LR 512 × 512)** window, **k4 fused tiles** (128 px, 4 per step), physical Sentinel-2 MTF, Charbonnier loss, `holdout_mse` early stop with **patience 8**. Do not raise hash-table size or hash-grid depth. Do not move to 10–20 km INRs for quality.

k4 is the speed/quality knee, not the ceiling. Full-field at the same AOI is a bit better (mean LPIPS 0.363 vs k4 mean 0.376). A long-budget LR2048 `k8` can match full-field quality (0.360) if you wait ~15× longer. Neither is the trade this project asked for. Seed variance showed the old published k4 of 0.3700 was the luckiest of five draws under patience 3; ship patience 8 so that gap shrinks.

---

## 1. What we train

Each AOI is an independent test-time optimisation. There is no pretrained network and no weight sharing across AOIs.

**Inputs.** 16 cloud-filtered Sentinel-2 revisits of the same ground, 10 m GSD, RGB. A NIB aerial orthophoto at 2.5 m is used only as HR ground truth for development metrics; the Norway production run will not have it.

**Forward model.** The INR decodes an HR reflectance field. That field is warped by a per-frame learnable affine, blurred by a calibrated Sentinel-2 MTF (`s2_psf_m`), and area-downsampled by `df = 4` to LR. A per-frame colour transform absorbs residual radiometry. The reconstruction loss is computed against the observed LR frames.

**Output.** An HR RGB image at 2.5 m, plus georeferenced GeoTIFFs (`hr_gt.tif`, `sr_pred.tif`, `s2_bilinear.tif`) when export is on.

Scene: asker, UTM 32N, granule origin (499980, 6590220). The LR512 and LR2048 datasets are centre crops of the same MGRS tile, 5.12 km and 20.48 km on a side.

---

## 2. Architecture

| Piece | Choice | Why this, not the alternative |
|--|--|--|
| Encoder | Instant-NGP hashgrid via tinycudann | Fourier features need a much wider MLP to match detail; the hashgrid is the whole reason the step is milliseconds. |
| Decoder | Fully-fused TCNN MLP, depth 4, width 256, FP16 | Matches the encoder backend. FP16 is required for the fused kernels; it is also the source of the gradient-underflow bug below. |
| Hash ladder | Auto-sized from LR shape: `max = lr_size`, `base = max/4`, 16 levels × 2 features | Physically identical on every AOI size: 40 m → 10 m cells. See §5. |
| Hash table | `log2 = 21` (2.1 M entries) | Raising it does nothing once the table covers the finest level. See §5. |
| Finest-level multiplier | **1.0 (keep)** | `mult=2` (finest at 5 m) can win a few LPIPS points on Asker k4 (0.362 vs 0.370) but looks worse in QGIS against NIB — oversharp / unstable high frequency. Rejected after side-by-side GeoTIFF inspection (`hashmult{1,2}_k4_qgis`). Do not promote. |
| Alignment | One global affine per frame | Enough at 5 km, where it is one tile's warp. At 20 km it is a spatial average of a field that varies by metres; rotation is ~0°. See §6. |
| Colour | One 3×4 affine per frame | Removes residual radiometric mismatch the PSF does not explain. |

### Correctness fixes now in the training stack

These are not research choices. They are bugs that silently poisoned earlier fused-k results, and they are why several previous tables should be discarded.

1. **FP16 gradient underflow.** The reconstruction loss is a mean over every supervised LR element. At LR2048 `k4` that is ~3.15 M elements, so `dL/dout` underflows to exact zero in the TCNN FP16 decoder and the run produces a flat grey image with a frozen holdout curve. Fixed with `torch.amp.GradScaler` (`--grad_scaler`, on by default when the decoder is FP16). Diagnose with `scripts/diag_chunked_grad.py`. Every fused-k LR2048 number from before this fix is invalid.

2. **TCNN 32-bit indexing.** A single forward whose `rows × hidden_width` exceeds 2³² hits an illegal memory access. `_encode_and_decode` now chunks at `MAX_FORWARD_ROWS`. This is what made the 8192 × 8192 HR eval crash, and what made `k4` tile-512 at LR2048 crash.

3. **Tiled HR eval.** Final decode is stitched from 2048 × 2048 tiles (`eval/hr_render.py`). LPIPS/SSIM use a centre crop when the full HR tensor does not fit. Production still needs this even without HR ground truth, because we still write the SR image.

4. **Pathological `apply_affine` backward.** The batched `B ≥ 2` path used a GEMM that was ~4–5× slower than `k1` for no algorithmic reason. Both `apply_affine` and `apply_color_transform` are now vectorised elementwise ops.

5. **Gradient accumulation.** tinycudann allocates its own arena via `cuMemCreate`, invisible to `torch.cuda.max_memory_allocated`. A "9.6 GB" `k8` step was actually ~55 GB. `--grad_accum` splits the tile batch so `k8`/`k16` at LR2048 fit; micro-batch size 2 tiles is the largest that is safe on an 80 GB H100.

---

## 3. Training procedure

| Knob | Current | Why |
|--|--|--|
| Loss | Charbonnier, ε = 0.01 | Robust to hashgrid ringing. Default CLI is still MAE; every recent bench overrides to Charbonnier. Promote the default when convenient. |
| LR | 1e-3, Adam | Unchanged. |
| Iterations | Cap 5000 (benches) / 8000 (AOI ladder) | Holdout early-stop usually fires first. |
| Holdout | 10 % of LR pixels, in blocks (~8 px at LR512, ~32 px at LR2048), independent per frame | The only production-viable validation signal. Norway has no HR GT, so LPIPS/MAE/PSNR cannot be the stopping metric. |
| Stop metric | `holdout_mse` | LR-only. See the caveat in §4. |
| Stop score | **EMA α=0.4** of holdout (auto when metric is `holdout_mse`) | Cuts fused-k patience trips on tile-sampling jitter. Offline replay on 7-city long runs: p8+EMA mean LPIPS regret ≪ p3 raw. |
| Patience | 3 in the published tables; **8 going forward** | Patience 3 trips on holdout jitter from random tile sampling. See §4. |
| Regression trip-wire | **0.01** absolute on holdout EMA (auto) | Was 0 for holdout_mse; now forces stop on clear val blow-ups without waiting out patience. |
| Eval cadence | Every 200 iterations | HR eval is skipped entirely when the stop metric is `holdout_mse` (`_skip_periodic_hr_eval`). We only ever see the final HR number. |
| Tile mix | `within` | Tiles are drawn inside one AOI. Cross-frame mixing was measured separately and is not the default. |
| Seed | 6 | Default. Full-field is deterministic across seeds; fused-k is not, because of the stopping rule. |

Fused-k sampling (`models/lr_tile_sampler.py`): `--lr_tile 128 --lr_tiles_per_step K` on LR512. `K = 0` means all tiles (full coverage). At LR2048 the tile is 512 so that a micro-batch of 2 tiles stays inside the TCNN arena.

---

## 4. How we evaluate, and why the proxy is a problem

Three objectives sit in one run and they do not agree.

| Stage | Objective | Space |
|--|--|--|
| Train | Charbonnier ≈ L1 | LR |
| Stop / restore | held-out MSE | LR |
| Rank | LPIPS | HR |

`holdout_mse` is kept because it is the only metric that will exist in production. It is not kept because it tracks LPIPS well. Two disagreements are now measured:

- On the LR512 level ladder, `holdout_mse` ranks `8 × 2` features best; LPIPS ranks any width-32 config best.
- On fused-k, `holdout_mse` jitter from random tile sampling trips patience 3 at very different iterations across seeds, and **final LPIPS tracks the stopping iteration**, not the coverage setting.

Seed variance on LR512 (same config, seeds 6–10):

| config | n | mean LPIPS | sd | spread | published (seed 6) |
|--|--:|--:|--:|--:|--:|
| full | 3 | 0.36259 | 0.00002 | 0.00004 | 0.3626 |
| k8 | 3 | 0.36880 | 0.00292 | 0.00572 | 0.3650 |
| k4 | 5 | 0.37602 | 0.00787 | 0.01859 | 0.3700 |
| k2 | 3 | 0.38412 | 0.00233 | 0.00422 | 0.3813 |

Full-field is deterministic: every seed stopped at iteration 4400. `k4` seeds stopped anywhere from 2000 (LPIPS 0.3884) to 4800 (LPIPS 0.3698). The published `k4` number was the best of five. The ordering full < k8 < k4 < k2 survives in the means, so the shape of the quality/speed frontier is real, but the gaps are wider than the canvas reported (k4→full is 0.0134 in the mean, not 0.0074).

Going forward: patience 8, **EMA-smoothed holdout** (`--early_stop_ema` auto 0.4), holdout regression 0.01, and treat single-seed fused-k differences below ~0.005 as noise unless the run was allowed to finish. Replay tool: `scripts/bench_stopping_rules.py`.

---

## 5. Capacity: we have more than we need

The hashgrid auto-sizes with the AOI. LR512 and LR2048 therefore already have **physically identical ladders** (16 levels, 40 m → 10 m) and **matched per-area capacity** (2.9 M vs 47.1 M parameters, ratio 16.0 for 16× the ground). The one thing extra cells cost — hash collisions — is not binding at the `log2 = 22` the LR2048 coverage series actually used. Raising the table to `log23`/`log24` moved k1 LPIPS from 0.5140 to 0.5134.

The level ladder on LR512 k4, all 4800 iterations:

| levels × features | decoder in | encoder params | holdout MSE | LPIPS |
|--|--:|--:|--:|--:|
| 4 × 8 | 32 | 3.39 M | 0.3557 | **0.3695** |
| 8 × 4 | 32 | 3.07 M | 0.3554 | **0.3699** |
| 16 × 2 | 32 | 2.95 M | 0.3553 | **0.3696** |
| 8 × 2 | 16 | 1.54 M | 0.3551 | 0.3724 |
| 4 × 2 | 8 | 0.85 M | 0.3622 | 0.4024 |

Quality tracks **decoder input width**, not ladder depth. Four levels at 8 features is indistinguishable from sixteen at 2. Below width 16 it falls off. The realistic speedup of dropping to 8 × 2 is 13 % of step time for 0.003 LPIPS; matching quality (8 × 4) saves about 3 %.

The 12-level arm is a real anomaly (mean 0.3987 across 3 seeds, sd 0.0016), worse than configs with a quarter of its parameters. Unexplained; do not use 12 levels.

The LR2048 up-ladder at collision-free `log24` is flat: 16 levels 0.4113, 24 levels 0.4096, 32 × 1 0.4128. Adding levels there does not close the route gap.

**Rejected.** Bigger hash table, more levels, more features-as-capacity. None of them is the constraint.

---

## 6. Coverage, AOI size, and alignment

### Fused-k at LR512 (the production route)

After the gradient fix, coverage is monotone and cheap. Quality below uses seed-variance *means*; times are seed-6 training only:

| config | mean LPIPS | vs bilinear (0.4442) | train s / AOI | hours / granule, 8 GPU |
|--|--:|--:|--:|--:|
| k2 | 0.3841 | +0.060 | 45 | 0.76 |
| k4 | 0.3760 | +0.068 | 67 | 1.13 |
| k8 | 0.3688 | +0.075 | 114 | 1.92 |
| full | 0.3626 | +0.082 | 186 | 3.13 |

Hours assume 484 AOIs per 10980 px granule. Per-AOI setup is ~0.5 s (measured) and is paid once per AOI — ~0.06 h/granule serial, not a cost driver.

`k4` is the operating point: most of full-field's gain, roughly a third of the train time. The mean gap to full (0.013) is wider than the luckiest single seed showed (0.007); a slice of that is premature stopping under patience 3. It is not large enough to prefer full-field.

### The large-AOI "quality cliff" was mostly the stopping rule

Patience-3 LR2048 numbers (gradient fix on, collision-free encoder) looked like a failure:

| config | LPIPS | vs bilinear (0.4401) | train s | stopped at |
|--|--:|--:|--:|--:|
| k1 | 0.5140 | −0.074 | 217 | 5000 (cap) |
| k2 | 0.4670 | −0.027 | 435 | 4800 |
| k4 | 0.4282 | +0.012 | 731 | 4600 |
| k8 | 0.4121 | +0.028 | 1127 | 3600 |
| k16 | 0.3993 | +0.041 | 2749 | 4400 |

Those k1/k2 runs are worse than bilinear. They are also starved. The same `k8` with patience 8 ran to **14,600 iterations and landed at 0.3603** (+0.080, 4612 s) — matching LR512 full-field (0.3590 under patience 8). The method *can* scale in the weak sense: bigger AOI, enough steps, same LPIPS. It does not scale in the strong sense: same `k`, iters, and stop, same gain.

### AOI-size ladder (full coverage, patience 8)

| AOI | km | LPIPS | vs bilinear | train s | notes |
|--|--:|--:|--:|--:|--|
| LR256 | 2.56 | 0.3745 | +0.095 | 97 | harder crop (bil 0.470); cap 8000, no stop |
| LR512 | 5.12 | **0.3590** | +0.085 | 331 | best raw LPIPS; stop 7800 / best 6200 |
| LR1024 | 10.24 | 0.3728 | +0.071 | 1301 | |
| LR2048 | 20.48 | 0.3823 | +0.058 | 5010 | same window as the affine grid |

Raw LPIPS still prefers 5 km over 20 km (0.359 vs 0.382). The gap is 0.023, not the 0.037 we quoted under patience 3. Bilinear baselines still differ, and the GeoTIFF export on this ladder did not write, so these are **not** yet scored on identical ground. Treat the ordering as real and the gains as not strictly comparable.

NIB aerial GT does not fill every pixel of a 20 km window. The 2048 eval mask is 86.9 % valid; the SE 512 of that window is 8.3 %. S2 is complete; the aerial is not. Interior 512s with ~100 % valid GT still beat bilinear (typically +0.04 to +0.09). Thin-GT edge tiles and water (`y3_x2`) are not a fair quality read. Two SE 512s (`y2_x3`, `y3_x3`) first crashed in centre-spot eval because the mask had no `model_psnr`; that path now skips the spot.

### One global affine is a 20 km problem, not a 5 km problem

We cut the existing LR2048 window into a real partition (1×2048, 2×2 of 1024, 4×4 of 512), retrained full coverage with affine dumps, and compared warps in metres. Figures: `single_samples/sweep_results/affine_grid_hr_warp.png` (HR mosaics + mean warp) and `affine_grid_frames.png` (per-frame 512 fields).

- **Translation varies by metres.** On well-behaved frames the 16 small tiles disagree by 2–15 m (0.2–1.5 S2 pixels). The single 2048 affine sits near their average, not on any corner.
- **Rotation is ~0°** (≤0.1°). The extra affine degrees of freedom are not doing the work.
- **Some frames diverge** at 20 km (kilometre-scale outliers; frame 10 took the 2048 with it). Large AOIs are not only averaging — they can fail to find a stable warp.
- Mean warp plots exclude cells with `|shift| > 50 m` or `|α| > 5°`, then `nanmean` over frames. Frame 0 (frozen identity) is still in that mean and pulls it toward zero. Per-frame maps are the ones without averaging.

That is why we stay at 5 km: each INR's affine is then one tile's warp. A spatially-varying alignment is a rescue for 20 km INRs, not a better 512 method.

Excluded as explanations of the old cliff: collisions, hash headroom, per-area capacity, ladder resolution. What remains at matched budget is a milder extent effect (alignment + crop + leftover stop interaction), not an inability to represent the scene.

---

## 7. Production config

For a Sentinel-2 granule on 8× H100, today:

```
--dataset <name>
--s2-dir data/s2_revisits/<name>_lr512
--lr_degradation s2_psf_m
--recon_loss charbonnier --charbonnier_eps 0.01
--lr_tile 128 --lr_tiles_per_step 4 --lr_tile_mix within
--early_stop_metric holdout_mse
--early_stop_patience 8 --early_stop_min_iters 1000
# EMA α=0.4 and max_regression=0.01 apply automatically for holdout_mse
--iters 5000 --eval_every 200 --hr_render_tile 2048
--spatial_holdout 0.1
```

Do **not** pass `--no_qgis_export` on a production write-out. Do **not** raise `--hash_log2_hashmap_size` or `--hash_n_levels`. Leave GradScaler at its default. For stopping-rule studies only, add `--force_hr_eval` so LPIPS is logged while still stopping on holdout.

If quality is non-negotiable and time is not, drop `--lr_tile` / `--lr_tiles_per_step` and run full-field. That is the ceiling on this AOI, not a different method.

If time is non-negotiable, `k2` is the next step down. Do not go to LR2048 to save AOI count: setup is negligible (~0.5 s/AOI), and at matched quality the large window is ~15× slower to train, not cheaper.

---

## 8. What we are not doing, and why

| Idea | Status | Reason |
|--|--|--|
| Larger hash table | Rejected | Flat at log21–24 once the finest level fits. |
| More hash levels at LR2048 | Rejected | Flat at 16 / 24 / 32×1. |
| Fewer hash levels at LR512 | Optional, low value | Width-32 is what matters; 8×2 saves 13 % for 0.003 LPIPS. Not worth a production change on its own. |
| Deeper / wider MLP | Not swept | Decoder width 256 × depth 4 has not been the suspected constraint. Cheap to do later at LR512. |
| Large AOIs (LR2048) | Rejected for the speed/quality trade | Can match LR512 if you give it ~15 k steps (k8 → 0.360). Does not beat it, costs much more train time, and the global affine is a measured compromise. Revisit only if setup overhead dominates. |
| Spatially-varying affine | Not for production | Confirmed leftover at 20 km (metres of dx/dy, ~0° rotation). Unnecessary at 5 km, where each INR already has its own warp. |
| Stop on HR LPIPS | Impossible in production | No HR GT on the Norway run. Keep `holdout_mse` + EMA + patience 8. |
| Rank on PSNR | Rejected | The 7-city PSF sweep: the physically correct MTF won on LPIPS and lost on PSNR. Ranking on PSNR would have rejected the right forward operator. |
| DSen2 / box degradation | Rejected | Physical MTF (`s2_psf_m`) wins 5 of 7 cities. |
| Finest hash level at HR (mult = 4) | Rejected | Underdetermined: 8 frames, 16 HR unknowns per LR pixel. Outputs are high-frequency checkerboards. |
| Finest hash level at 2× LR (mult = 2) | Rejected | LPIPS can tick up (Asker k4: 0.362 vs 0.370) but QGIS vs NIB prefers mult=1 — looks oversharp / unstable. Keep default 1.0. |

---

## 9. Open work

In rough priority.

1. **Stopping rule (done for ship).** EMA α=0.4 + holdout regression 0.01 auto for `holdout_mse`. Offline full-field: p8+EMA lowest regret. Live k4 traces (asker/bergen, 5k): holdout still improving at budget — p8 does not premature-stop; p3 can. Fused-k seed-variance re-bench optional.
2. **Common-window LPIPS.** The AOI-size ladder did not write GeoTIFFs. Recompute 256 / 512 / 1024 / 2048 on the shared centre 2.56 km so gains are on the same ground.
3. **Development-only HR trajectory.** Periodic centre-crop LPIPS via `--force_hr_eval`, logged, not used to stop.
4. **12-level anomaly.** Reproducible, unexplained. Not blocking.
5. **Checkpoints.** Run directories keep PNGs, metrics, and now `affines.json`, not weights. Re-renders still require a retrain.

**Setup time (measured).** Production AOI setup (data + model + first step) is ~0.3–0.8 s on H100 for LR512 / `--allow_no_hr` (`scripts/measure_setup_time.py`). Serial over 441 AOIs ≈ **0.06 h/granule** — not the cost-model risk we feared. Large AOIs stay rejected on train time, not setup.

---

## 10. Scene and reproducibility

- City: asker. 16 S2 revisits, May 2024 window, NIB HR dated 2024-05-15.
- Code entry point: `optimize.py`. Dataset loader: `s2_dataset.py` via `--s2-dir`.
- Sweep drivers: `scripts/bench_capacity_coverage.py`, `bench_capacity_ladder.py`, `bench_seed_variance.py`, `bench_aoi_size.py`, `bench_affine_grid.py`.
- Affine grid: `scripts/make_affine_grid.py` (partition of the LR2048 window), dumps via `eval/learned_affines.py`, compare/plot with `scripts/compare_affine_grid.py` and `scripts/viz_affine_grid.py`.
- **Production (no HR GT):** `docs/PRODUCTION.md` — `scripts/make_granule_tiles.py` + `scripts/run_production.py` (`--allow_no_hr`).
- Results JSON: `single_samples/sweep_results/bench_*.json`.
- Figures: `spot_256_comparison.png`, `affine_grid_hr_warp.png`, `affine_grid_frames.png`.
- All benches in this note used FP32 eval, gradient loss scaling on, and `--hr_render_tile 2048`.
