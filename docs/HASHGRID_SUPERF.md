# HashGrid SuperF — planning & evidence

Living plan for the HashGrid / ScaleF project (aka **HG-SuperF / SuperF 2.0**).  
Method details: [`METHOD.md`](METHOD.md). Production ops: [`PRODUCTION.md`](PRODUCTION.md).  
Paper framing + draft: [`../paper/OUTLINE.md`](../paper/OUTLINE.md), [`../paper/DRAFT.md`](../paper/DRAFT.md).

Last updated: 7 September 2026 (scalability-first experiment track).

---

## 1. Goals

**North-star contribution:** make SuperF-style test-time MISR SR **scalable and fast** enough for national 2.5 m S2 maps (Norway / Denmark), without giving up quality on real NIB orthophotos.

| Priority | Goal |
|--|--|
| P0 | **Faster / cheaper than OG-SuperF** (hashgrid + fused-k tiling) at matched or better LPIPS on our NIB eval |
| P0 | **Throughput story**: s/AOI, peak VRAM, extrapolated h/MGRS (441 AOIs) and h/Norway-block |
| P0 | Evaluate on **our own** real HR (NIB / focus) — *not* SEN2NAIP / SEN2NEON |
| P0 | At least one **contiguous full-scale** national-style demo |
| P1 | Two HR national S2 maps (Norway, Denmark) |
| P2 | Cloud-robust SR; downstream task hook (Lucia) |

**Non-goals:** SEN2* benchmarks; raising hash capacity “for quality theatre.”

Every experiment below must report **quality + time + VRAM**. Quality alone is not enough for this paper’s claim.

---

## 2. Intended contributions

1. **HG-SuperF (SuperF 2.0)** — Instant-NGP SuperF that is **faster and cheaper to scale** than Fourier/OG-SuperF  
2. **Speed–quality Pareto** on real NIB/MISR (hash vs Fourier; k tiling; AOI size ladder)  
3. **National-scale 2.5 m S2 maps** — Norway and/or Denmark as the throughput demo  

---

## 2b. Scalability claims we must back with numbers

| Claim | Experiment | Report |
|--|--|--|
| Hash beats Fourier on wall-clock at matched quality | **B0 vs B1** on frozen NIB cities | LPIPS + s/AOI + peak VRAM |
| Fused-k tiling is the speed knob | **B2** k1 / k2 / k4 / full | Pareto: LPIPS vs s/AOI |
| LR512 is the national sweet spot | **S_aoi*** ladder on Asker (256→2048) | quality vs time; affine validity note |
| National cost is predictable | Extrapolate s/AOI → **h/MGRS** (~441 LR512 tiles) and h/block | table in paper |
| Mosaic ops don’t dominate | Overlap/feather timing vs train time | secondary |

---

## 3. Locked technical recipe (ship this)

From [`METHOD.md`](METHOD.md) / production runs:

| Knob | Value | Notes |
|--|--|--|
| AOI | **LR 512²** (5.12 km) | One INR per window |
| Sampling | **k4** (128 px × 4 / step) | Speed/quality knee; ~1 GB VRAM/AOI |
| Degradation | `s2_psf_m` | Physical S2 MTF |
| Loss | Charbonnier ε=0.01 | |
| Stop | `holdout_mse`, patience **8**, EMA 0.4 | No HR needed for stop |
| Hash table / levels | defaults | Do **not** raise for quality theatre |
| Coarse→fine hash | **Rejected** | Did not work |
| Production HR | `--allow_no_hr` | NIB only for eval AOIs |
| Mosaic (overlap trials) | **15% overlap + feather** (candidate) | Compare vs 0% / 10% mean / 10% feather on Asker |

**Frame floor (ops):** prefer **≥6–8** clear revisits per AOI after cloud filter; if short, widen date window or ease cloud before training.

---

## 4. Publish minimum (evidence checklist)

Use this as the gate for “ready to write.”

### Method + scalability (paper spine)

- [ ] **Hash vs Fourier** (B0 vs B1): LPIPS + **s/AOI + peak VRAM** on frozen NIB set  
- [ ] **k tiling Pareto** (B2): LPIPS vs s/AOI for k1 / k2 / k4 / full  
- [ ] **AOI size ladder** (S_aoi*): Asker LR256→2048 — quality, time, VRAM; argue LR512 for national  
- [ ] Extrapolated **h/MGRS** (~441 AOIs) and h/Norway-block from measured s/AOI  
- [ ] Ablation freeze for loss / PSF (B3 / B4) — secondary to speed story  
- [ ] Seed / stop stability note (patience 8 vs luck under patience 3)

### Real MISR eval (our data)

- [x] Evaluation packages downloaded + WorldCover stats  
- [x] Spatial alignment complete  
- [x] Harmonization path in place  
- [ ] Frozen eval split list (cities / AOIs / LR dirs) written down here or in `eval/`  
- [ ] Final metrics table on that split (HG-SuperF vs bilinear vs SuperF)

### Scale demo

- [x] Full-MGRS tile→train→mosaic pipeline (`PRODUCTION.md`)  
- [x] Multi-granule batch + BigTIFF mosaics  
- [x] Contiguous **3×3 MGRS** July trial (8/9; `32VNK` starved under whole-tile OCM)  
- [x] Per-LR512 **clear-day heatmap** (SCL proxy) for July ±45 d pilot + nationwide run in progress  
- [ ] One **shared-season** contiguous Norway block with frame floor enforced (re-fetch using heatmaps)  
- [ ] Optional: Denmark strip  

### Optional / stretch

- [ ] Cloud stress test (# cloudy frames until quality breaks)  
- [ ] Affine validity vs AOI size (local warp field viz)  
- [ ] Downstream task hook (Lucia / MMEarth-Bench)  

---

## 5. Open research questions

1. **Largest AOI where one affine per frame is enough?**  
   Visualize local transform parameters / variance (affine grid on Asker already sketched).
2. **Overlap & feather vs hard partition** — measured (appendix): feather/Laplacian do not kill date islands. Shipped mosaic is ICM identity + date-cut ramp (`paper/APPENDIX_SEAMS.md`).  
3. **Per-512 cloud scheduling vs whole-MGRS OCM** — heatmaps show `32VNK` is rich per-cell but whole-tile fetch starved it.  
4. **When do cloudy LR frames break SR?** Dose–response on #cloudy inputs.

---

## 6. Experiment tracks → evidence log

### A. Ablations & scalability (vs SuperF)

| Item | Status | Where |
|--|--|--|
| Recipe ablations (k, stop, PSF, Charbonnier) | Mostly done | `docs/METHOD.md`, sweeps |
| Paper B0–B4 + **S_aoi*** (quality **and** speed) | **Done** (52) | `paper/results/paper_ablations.json` |
| Fair Fourier **F_*** + hash **H_aoi*** (Asker) | **Done** (24) | `paper/results/paper_fourier_fair.json` |
| Hash vs Fourier wall-clock @512 (B0 vs B1) | Done (harsh) | Fourier loses; see fair track |
| AOI ladder Asker 256/512/1024/2048 | Done | `asker_lr*` |
| Extrapolated h/MGRS table | Partial | in JSON `hours_per_mgrs_extrapolated` |
| VRAM / time reference | Partial | ~70 s/AOI class on 8×H100; setup ~0.5 s |

### B. Own MISR + NIB eval

| Item | Status | Where |
|--|--|--|
| Focus + new NIB packages | Done | `data/nib_*`, `data/s2_revisits/aois.json` |
| Alignment JSON | Done (17 AOIs) | `eval/spatial_alignment.json` (8 focus + 9 new NIB) |
| Production `--allow_no_hr` | Done | `optimize.py` / `PRODUCTION.md` |
| Frozen metric table | **TODO** | |

### C. National / contiguous scale

| Item | Status | Where |
|--|--|--|
| 6 scattered MGRS (mixed dates) | Done | `production/mosaics/{asker,bergen,rana,tromso,amli,stavanger}_sr_2p5m.tif` |
| July 3×3 Åmli-centered | 8/9 mosaics | `production/mosaics/j25_*_sr_2p5m.tif` |
| Asker overlap 0 / 10 mean / 10 feather / 15 feather | Done | `production/mosaics/asker*_sr_2p5m.tif` |
| LR512 clear counts (3×3) | Done | `production/cloud_availability/lr512_july2025_pm45/` |
| Norway-wide LR512 clear map | **In progress** | `production/cloud_availability/lr512_norway_july2025_pm45/` → `norway_clear_counts.png` |
| **Season window pick** (cloud + snow + phenology) | **Started** | `scripts/compare_norway_season_windows.py` → `season_compare_2025/` |
| Re-fetch with ≥6–8 frames using map | **TODO** | |

---

## 7. Metrics (what we report)

**Every ablation row must include quality and cost.** Prefer LPIPS-vs-seconds Pareto plots over quality-only tables.

| Class | Metric | Role |
|--|--|--|
| Quality | LPIPS (rank), PSNR, SSIM | vs NIB on eval AOIs |
| Baseline | Bilinear; **Fourier / OG-SuperF-style** (B1) | Required for publish |
| Speed | `training_time_s` / AOI; iters to stop | Primary scalability evidence |
| Scale-up | **h/MGRS** = `s/AOI × 441 / 3600`; h/block | National cost story |
| Compute | Peak VRAM / AOI | ~1 GB with k4 tiles |
| Ops | Clear days / LR512 cell | Heatmaps; frame floor |

---

## 8. Near-term plan (ordered)

### 8.1 Cleanup (regenerable SR only)

Delete bulky **outputs**, not inputs:

| Keep | Remove |
|--|--|
| `data/s2_revisits/`, NIB packages, `eval/` | `production/mosaics/*.tif` (all SR mosaics) |
| `production/cloud_availability/` | `single_samples/*/sample/**` run dirs |
| `docs/`, sweep **JSON/logs** under `single_samples/sweep_results/` | Per-AOI `prod_k4_*` GeoTIFFs |

METHOD numbers stay in `docs/METHOD.md` — do **not** re-derive the whole recipe from scratch.

### 8.2 Paper ablation campaign (quality **and** speed)

Frozen eval cities (NIB available): **asker, bergen, rana, tromso, amli, vennesla, trondheim**.  
Each cell: `--force_hr_eval`, production stop; always log **LPIPS/PSNR/SSIM + training_time_s + peak VRAM**, then extrapolate h/MGRS.

#### Quality / method (secondary)

| ID | Variant | Purpose |
|--|--|--|
| **B0** | Ship: `hashgrid_tcnn` + k4 + `s2_psf_m` + Charbonnier + holdout_mse p8 | HG-SuperF baseline |
| **B3** | B0 with `mae` | Loss ablation |
| **B4** | B0 with `--lr_degradation area` | No-MTF / area downsample ablation |

#### Scalability (primary — paper claim)

| ID | Variant | Purpose |
|--|--|--|
| **B1** | B0 + `--input_projection fourier` (default `fourier_scale=10`) | **Hash vs Fourier at national AOI (512)** — expect Fourier weak; see caveat below |
| **B2_k1 / k2 / full** | Change fused-k | **Pareto**: LPIPS vs s/AOI (k4 = B0) |
| **S_aoi256 / 512 / 1024 / 2048** | Same recipe, Asker’s `*_lr{N}` stacks | **AOI size vs national cost**; argue LR512 |
| **F_aoi{N}_s{S}** | Fourier, full-field, AOI N, `fourier_scale=S` | Fair SuperF-style baseline where Fourier can work |
| **H_aoi{N}** | Hash full-field at same N | Matched-window control for F_* |

Driver: `scripts/run_paper_ablations.py`  
- Working dump → `single_samples/sweep_results/paper_*.json`  
- **Paper freeze** → [`paper/results/`](../paper/README.md) (canonical copies + `MANIFEST.json`)  
(JSON includes `hours_per_mgrs_extrapolated` from measured s/AOI × AOIs-per-MGRS.)

### Caveats / fair baselines

- **Fourier @ LR512 is not OG-SuperF’s happy place.** Random Fourier features + default `fourier_scale=10` look poor on 512² (B1 lost to bilinear on 7/7). That is expected: Fourier tends to need **much smaller windows** (e.g. ~64²) and a **tuned scale**. Do **not** paper-claim “hash destroys Fourier” from B1 alone.
- Fair track: Asker **LR64 / 128 / 256 / 512**, Fourier scales **2 / 5 / 10** (skip ≤1 and >10), plus hash controls. National claim: hash keeps quality when AOIs must be large for throughput; Fourier may only win where tiny patches are acceptable.
- First fair sweep also ran s∈{1,20,40}; treat those as exploratory. Prefer s=2/5/10 going forward (`F_aoi*_s2` may still need a short top-up run).

### 8.2b Ablation status board (keep current)

| Batch | Status | Out / log | Notes |
|--|--|--|--|
| **B0–B4 + S_aoi*** (7 cities) | **Done** 52/52 | `paper/results/paper_ablations.json` | Hash≫Fourier@512; k4 sweet spot; PSF helps; MAE≈Charb |
| **F_* + H_aoi*** fair Fourier (Asker) | **Done** 24/24 | `paper/results/paper_fourier_fair.json` | Scale + AOI ladder; hash controls |
| Nested complete-HR size ladder | **Done** (summary) | `paper/results/bench_complete_patches_size_ladder_nested.json` | Same footprints; nest per-size JSONs alongside |
| Norway LR512 clear heatmap | **Full-cover mapping** | 74 mainland tiles | island tips filled; map leftovers then stitch |
| National SR schedule | **mainland LR512 only** | `--mainland-only` (+ optional `--min-clear`) | skip ocean cells inside coastal MGRS; see `PRODUCTION.md` |

**Fair Fourier job list (Asker):**  
Canonical: `F_aoi{64,128,256,512}_s{2,5,10}` + `H_aoi{64,128,256,512}`  
(Exploratory s=1/20/40 already in JSON — don’t extend past 10.)  
Data: `asker_lr64` / `lr128` via `make_lr_size_variants.py`.

### 8.3 After ablations

1. Finish Norway clear-count mosaic (leave running).  
2. Build paper tables: B0 vs B1 (harsh); **fair F vs H by AOI**; B2 Pareto; S_aoi ladder.  
3. Re-fetch a shared-season block using heatmaps + frame floor.  
4. National mosaic demo with ICM identity + date-cut ramp (not overlap colour).  

---

## 9. Artifact index (quick paths)

```text
docs/METHOD.md                          # method decision record
docs/PRODUCTION.md                      # tile → train → mosaic
paper/APPENDIX_SEAMS.md                 # mosaic ablations (not delivery)
paper/results/                          # frozen paper tables only (see paper/README.md)
production/mosaics/                     # SR GeoTIFFs (2.5 m)
production/cloud_availability/          # clear-day heatmaps
data/s2_revisits/                       # LR stacks + meta + masks
eval/spatial_alignment*.json            # NIB↔S2 shifts (frozen copy also in paper/results/)
single_samples/sweep_results/           # working batch JSON / logs (not paper-canonical)
```

When adding evidence, prefer: one row in §6, path to numbers, and a one-line conclusion (keep / reject / revisit).
