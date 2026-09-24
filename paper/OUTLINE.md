# HG-SuperF — paper outline

Working title (provisional):

> **HG-SuperF: Scalable Test-Time Multi-Image Super-Resolution for National Sentinel-2 Maps**

Venue: TBD (remote sensing / geospatial ML).  
Evidence freeze: [`results/`](results/) · Method: [`../docs/METHOD.md`](../docs/METHOD.md) · Ops: [`../docs/PRODUCTION.md`](../docs/PRODUCTION.md).

---

## One-sentence pitch

We replace Fourier-feature SuperF with an Instant-NGP hashgrid and fused-k tiling so **test-time MISR SR becomes cheap enough for national 2.5 m Sentinel-2 maps**, and we evaluate on **real NIB orthophotos**—not SEN2NAIP.

## Contributions (map to sections)

| # | Claim | Where in paper | Evidence status |
|--|--|--|--|
| C1 | **HG-SuperF**: hashgrid INR + physical S2 MTF + LR-only early stop | §3 Method | Locked recipe in METHOD |
| C2 | **Scalability**: hash ≫ Fourier at national AOI size; k4 is speed/quality knee; LR512 is the ops sweet spot | §4 Experiments | `paper_ablations.json`, fair Fourier track |
| C3 | **Real MISR eval** on Norwegian NIB AOIs (+ bilinear) | §4.1–4.2 | 7-city B0 done; **17-AOI table still TODO** |
| C4 | **National-scale pipeline** (tile → train → mosaic) + clear-day scheduling | §4.4 / §5 | Production path + Norway heatmap + complete 438-cell `32VNM` GeoTIFF. **Finding:** freeze closest cell-SCL≤2% day as identity or mosaics show a 5.12 km brightness grid. **Delivery:** ICM (5% slack to seasonal primary) + ramp leftover date cuts. Colour/overlap/halo → appendix. Wall-to-wall Norway remains. |

**Non-claims (do not write):**
- “Hash always beats Fourier” without the fair small-AOI caveat.
- SEN2* leaderboard numbers.
- Raising hash capacity as a quality story.

---

## Section plan

1. **Abstract** — problem (national 2.5 m), gap (test-time SR too slow / Fourier unhappy at large AOIs), method (hash + k4 + holdout stop), result headline (≈80 s/AOI, ~10 h/MGRS-class extrapolate; beats bil on 7 NIB cities).
2. **Introduction** — national mapping need; MISR test-time SR lineage (SuperF); scalability bottleneck; contributions list.
3. **Related work** — single-image S2 SR; MISR / multi-date fusion; INR / Instant-NGP; operational mosaic pipelines.
4. **Method (HG-SuperF)** — forward model; hashgrid; fused-k; `s2_psf_m`; Charbonnier; spatial LR holdout early stop; production without HR.
5. **Experiments**
   - Setup: 7 frozen cities (+17 later); metrics LPIPS/PSNR/SSIM + **time + VRAM + h/MGRS**.
   - Main table: B0 vs bilinear vs B1 (harsh Fourier@512).
   - Fair Fourier: F vs H by AOI size (64→512) and scale.
   - k tiling Pareto (B2).
   - AOI size ladder (S_aoi*) + nested complete-HR ladder (throughput vs quality).
   - Ablations B3/B4 (short).
   - National demo / clear-day heatmap (figures).
6. **Discussion** — when Fourier still works; affine validity vs AOI size; cloud floor; limitations.
7. **Conclusion**

---

## Figure / table wishlist

| ID | Content | Data ready? |
|--|--|--|
| T1 | 7-city main results (B0, bil, B1) + s/AOI + h/MGRS | Yes (VRAM TBD backfill) |
| T2 | Fair Fourier vs hash by AOI (Asker) | Yes (merge s2 into cite set) |
| T3 | k1/k2/k4/full Pareto | Yes (sanity-check B2_k1 time) |
| T4 | S_aoi 256→2048 Asker | Yes |
| T5 | Nested size ladder project means | Partial (nest64 still finishing) |
| F1 | Method diagram (INR → affine → PSF → LR loss) | Draw |
| F2 | LPIPS vs seconds Pareto | From JSON |
| F3 | Same-spot visual 64/128/256/512 | Scriptable, no GPU |
| F4 | Norway clear-count heatmap | Almost (1 tile gap) |
| F5 | Mosaic delivery: independent 2% vs ICM+ramp (32VNM 143-cell block) | `production/seams/32VNM_stack/delivery_icm_ramp.png` |
| F6 | Full 32VNM granule: independent 2% vs ICM+ramp (438 cells) | `production/seams/32VNM_full/full_granule_delivery.png` |
| A1 | Seam appendix: colour affine, overlap feather/Laplacian, halo, 3×3 | `paper/APPENDIX_SEAMS.md` |

---

## Writing order (while GPUs busy)

1. ~~Abstract + Intro framing~~ (drafted).
2. ~~Method § in SuperF style~~ (`DRAFT.md` §3 — shared INR, hashgrid, MTF, holdout, fused-k).
3. Results narrative + T1–T4 from `paper/results/`.
4. Related work (literature pass; cite SuperF arXiv:2512.09115).
5. Fill gaps when GPUs free: 17-AOI table, VRAM, nest64 final, national block.
