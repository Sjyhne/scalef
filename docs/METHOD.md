# ScaleF current method

Decision record for the test-time INR super-resolution pipeline, as of 3 September 2026. Numbers are from the asker AOI unless noted. Selection metric is LPIPS (VGG, lower is better); PSNR and SSIM are reported but not used to rank.

The production recommendation, pending the AOI-size ladder still running:

> Fit one INR per **5.12 km (LR 512 × 512)** window, **k4 fused tiles** (128 px, 4 per step), physical Sentinel-2 MTF, Charbonnier loss, `holdout_mse` early stop. Do not use larger AOIs for quality. Do not raise hash-table size or hash-grid depth.

That recommendation is slightly weaker than it looked yesterday. Seed variance showed that `k4`'s published 0.3700 was the luckiest of five draws, and that a large part of the fused-k quality gap is premature stopping rather than missing coverage. The ordering still holds.

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
| Finest-level multiplier | 1.0 in the current benches | The earlier 7-city PSF sweep preferred `hash_max_resolution_mult = 2` (finest level at 5 m). That has not been re-validated on the coverage/capacity arms and is **not** in the numbers below. Treat it as a pending promotion, not current practice. |
| Alignment | One global affine per frame | Cheap and sufficient at 5 km. Leading suspect for the remaining 20 km quality gap. See §6. |
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
| Patience | 3 in the published tables; **8 going forward** | Patience 3 trips on holdout jitter from random tile sampling. See §4. |
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

Going forward: patience 8, and treat single-seed fused-k differences below ~0.005 as noise unless the run was allowed to finish.

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

## 6. Coverage and AOI size

### Fused-k at LR512 (the production route)

After the gradient fix, coverage is monotone and cheap. Using seed-6 published times and the seed-variance *means* for quality:

| config | mean LPIPS | vs bilinear (seed 6 bil = 0.4442) | train s / AOI | hours / granule, 8 GPU |
|--|--:|--:|--:|--:|
| k2 | 0.3841 | +0.060 | 45 | 0.76 |
| k4 | 0.3760 | +0.068 | 67 | 1.13 |
| k8 | 0.3688 | +0.075 | 114 | 1.92 |
| full | 0.3626 | +0.082 | 186 | 3.13 |

Hours assume 484 AOIs per 10980 px granule and **training time only**. Per-AOI setup is unmeasured and is paid 484 times, so this systematically favours whatever reduces AOI count.

`k4` remains the recommended operating point: most of full-field's gain, roughly a third of the compute. The gap to full is larger in the mean than the single-seed canvas showed, and a slice of it is recoverable by not stopping early. It is not large enough to prefer full-field.

### Large AOIs are worse, not faster-enough to compensate

LR2048 coverage, full collision-free encoder, gradient fix on:

| config | LPIPS | vs bilinear (0.4401) | train s | hours / granule |
|--|--:|--:|--:|--:|
| k1 | 0.5140 | −0.074 | 217 | 0.27 |
| k2 | 0.4670 | −0.027 | 435 | 0.54 |
| k4 | 0.4282 | +0.012 | 731 | 0.91 |
| k8 | 0.4121 | +0.028 | 1127 | 1.41 |
| k16 (full) | 0.3993 | +0.041 | 2749 | 3.44 |

k1 and k2 are worse than a free bilinear upsample. k16 is the fairest comparison to LR512 full: same coverage, same iterations, matched per-area capacity — and still 0.037 behind, at higher cost (3.44 h vs 3.13 h). Every cheap large-AOI option is dominated.

Bilinear baselines differ (0.4442 vs 0.4401), so the two routes are not scored on identical ground. That is a real confound and is why the AOI-size ladder exports GeoTIFFs: we will recompute metrics on the common centre 2.56 km window.

### What is left as an explanation

Excluded, with equal iterations and full coverage: collisions, hash headroom, per-area capacity, ladder resolution, per-pixel update scarcity.

Still open:

1. **One global affine over 20 km.** A single 6-parameter warp per frame is asked to absorb residual misregistration, terrain, and any spatially-varying error across 16× the ground. Predicted signature: quality degrades smoothly with AOI extent. The AOI-size ladder (LR256 / 512 / 1024 / 2048, full coverage, patience 8, GeoTIFF export on) is running to test this.
2. **Eval-crop mismatch.** Part of the 0.037 may simply be different ground. The GeoTIFFs settle it.
3. **Early-stop interaction.** Less likely at full coverage (the full-field seeds were deterministic) but the large-AOI arms still tile internally.

LR256 full, already in: LPIPS 0.3745, bilinear 0.4698, gain **+0.0953**, PSNR +1.75 dB, 97 s, 8000 iterations (patience 8 did not fire). Raw LPIPS is worse than LR512 full (0.3626) because the crop is different and harder (bilinear 0.470 vs 0.444). Gain-over-bilinear is the number to watch once all four sizes are scored on common ground.

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
--iters 5000 --eval_every 200 --hr_render_tile 2048
--spatial_holdout 0.1
```

Do **not** pass `--no_qgis_export` on a production write-out. Do **not** raise `--hash_log2_hashmap_size` or `--hash_n_levels`. Leave GradScaler at its default.

If quality is non-negotiable and time is not, drop `--lr_tile` / `--lr_tiles_per_step` and run full-field. That is the ceiling on this AOI, not a different method.

If time is non-negotiable, `k2` is the next step down. Do not go to LR2048 to save AOI count: on current numbers it is both worse and, at the coverage required to beat bilinear, not cheaper.

---

## 8. What we are not doing, and why

| Idea | Status | Reason |
|--|--|--|
| Larger hash table | Rejected | Flat at log21–24 once the finest level fits. |
| More hash levels at LR2048 | Rejected | Flat at 16 / 24 / 32×1. |
| Fewer hash levels at LR512 | Optional, low value | Width-32 is what matters; 8×2 saves 13 % for 0.003 LPIPS. Not worth a production change on its own. |
| Deeper / wider MLP | Not swept | Decoder width 256 × depth 4 has not been the suspected constraint. Cheap to do later at LR512. |
| Large AOIs (LR2048) | Rejected for quality | Best large-AOI result is worse than the worst small-AOI result, and full coverage costs more. Revisit only if per-AOI setup dominates the 484-AOI layout. |
| Stop on HR LPIPS | Impossible in production | No HR GT on the Norway run. Keep `holdout_mse`, fix its patience, later add a centre-crop HR log for development only. |
| Rank on PSNR | Rejected | The 7-city PSF sweep: the physically correct MTF won on LPIPS and lost on PSNR. Ranking on PSNR would have rejected the right forward operator. |
| DSen2 / box degradation | Rejected | Physical MTF (`s2_psf_m`) wins 5 of 7 cities. |
| Finest hash level at HR (mult = 4) | Rejected | Underdetermined: 8 frames, 16 HR unknowns per LR pixel. Outputs are high-frequency checkerboards. |
| Finest hash level at 2× LR (mult = 2) | Promising, not current | Won the 7-city PSF sweep. Not in the coverage/capacity numbers. Re-validate before promoting. |

---

## 9. Open work

In rough priority.

1. **AOI-size ladder** (running). Full coverage at 256 / 512 / 1024 / 2048, patience 8, GeoTIFFs on. Decides whether quality-vs-extent is smooth (alignment) and which AOI size actually belongs in §7. Also gives the first apples-to-apples score on common ground.
2. **Per-AOI setup time.** Unmeasured, multiplied by 484 vs 36. At even 30 s it adds ~0.5 h per granule on the LR512 route, comparable to the whole k2→k4 gap. This is the biggest unknown in the cost model and the only argument that could bring large AOIs back.
3. **Stopping rule.** Patience 8 is a patch. A better LR-only criterion (smoothed holdout, or a regression trip-wire that is not zero) would recover the fused-k quality we are currently throwing away, which is worth more than any capacity knob.
4. **Development-only HR trajectory.** Periodic centre-crop LPIPS, logged, not used to stop. Tells us whether `holdout_mse` and HR quality peak at the same iteration. Needed to design (3).
5. **`hash_max_resolution_mult = 2`.** Re-validate on the current stack (gradient scaling, patience 8, LR512 k4) before calling it the default. The 7-city evidence is real but predates the correctness fixes.
6. **12-level anomaly.** Reproducible, unexplained. Not blocking. Worth a look if anyone starts moving `n_levels` in anger.
7. **Checkpoints.** Run directories currently keep PNGs and metrics, not weights. Re-renders require a retrain. Cheap to fix; blocks a lot of "just decode this window again" work.

---

## 10. Scene and reproducibility

- City: asker. 16 S2 revisits, May 2024 window, NIB HR dated 2024-05-15.
- Code entry point: `optimize.py`. Dataset loader: `s2_dataset.py` via `--s2-dir`.
- Sweep drivers: `scripts/bench_capacity_coverage.py`, `bench_capacity_ladder.py`, `bench_seed_variance.py`, `bench_aoi_size.py`.
- Results JSON: `single_samples/sweep_results/bench_*.json`.
- Visual spot check: `scripts/viz_spot_comparison.py` → `single_samples/sweep_results/spot_256_comparison.png`.
- All benches in this note used FP32 eval, gradient loss scaling on, and `--hr_render_tile 2048`.
