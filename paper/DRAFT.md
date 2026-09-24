# HG-SuperF: Scalable Test-Time Multi-Image Super-Resolution for National Sentinel-2 Maps

*Working draft — numbers from `paper/results/` unless noted. Do not cite exploratory sweeps.*

---

## Abstract *(draft)*

Producing seamless **2.5 m** reflectance from **10 m Sentinel-2** at national scale requires a super-resolution method that is not only accurate on real aerial orthophotos, but **fast enough to run per local window across an entire country**. Multi-image, test-time implicit neural representations (INR) in the SuperF family can exploit multi-date revisits, yet Fourier-feature encodings scale poorly when each optimisation window must be large enough for operational throughput.

We present **HG-SuperF**: Instant-NGP **hashgrid** encoding, fused-k LR tiling, a calibrated Sentinel-2 MTF degradation, and **LR-only** early stopping via spatial pixel-block holdout—so production needs no HR ground truth. On seven Norwegian AOIs with NIB orthophoto evaluation, the shipped recipe (LR 512², k4, physical PSF, Charbonnier, holdout patience 8) reaches mean LPIPS **0.395** vs bilinear **0.458** in ≈**82 s** per AOI on an H100-class GPU (~**10 h** extrapolated per MGRS-sized workload at 441 LR512 cells). At the same national AOI size, a Fourier-feature baseline collapses below bilinear. A fair small-window study shows Fourier can compete on LR 64 patches; hashgrids keep quality as windows grow to the sizes national mapping actually needs.

---

## 1. Introduction *(framing)*

National agencies and research programmes increasingly want **wall-to-wall** high-resolution reflectance, not a handful of showcase chips. Sentinel-2 provides dense revisits at 10 m; aerial orthophotos (e.g. Norway’s NIB) provide sparse but trustworthy 2.5 m reference. Bridging that gap with **multi-image super-resolution (MISR)** at test time is attractive: each locality gets its own fit to the observed revisits, with no SEN2*-style domain gap baked into a pretrained net.

The obstacle is **cost**. Classic Fourier-feature INRs (OG-SuperF-style) need small windows and careful scale tuning to look good; the windows that make *national* schedules practical (kilometre-scale AOIs, modest VRAM, minutes not hours) are exactly where those encodings struggle. A method that wins a small-patch benchmark but cannot be scheduled over Norway is the wrong objective for this paper.

**HG-SuperF** reframes SuperF around scalability:

1. Replace Fourier features with an **Instant-NGP hashgrid** so each step stays milliseconds-scale as AOIs grow.
2. Use **fused-k tiling** as an explicit speed knob (k4 at LR512 ≈1 GB-class footprint).
3. Stop on **held-out LR pixels**, not HR metrics—required for Norway-wide production.
4. Evaluate on **our own NIB/MISR packages**, and report **quality + wall-clock + extrapolated national cost** together.

Contributions are listed in [`OUTLINE.md`](OUTLINE.md). The rest of this draft expands method and results from the frozen JSON.

---

## 2. Related work *(stubs)*

- **S2 single-image SR / SEN2NAIP-style benchmarks** — useful for architecture search; we deliberately do **not** compete there (domain and claim mismatch).
- **Multi-temporal / MISR fusion** — classical and deep; test-time INR SuperF is the closest parent.
- **INRs & Instant-NGP** — NeRF lineage; hash encodings for speed.
- **Operational mosaicking** — tiling, overlap, cloud masks; our production path is complementary to the representation change.

*(Literature pass still needed.)*

---

## 3. Methodology

We build directly on **SuperF** [Jyhne et al., 2025]: a test-time optimisation (TTO) MISR method that shares one implicit neural representation (INR) across multiple shifted low-resolution (LR) frames, jointly optimises sub-pixel alignment, and treats the LR frames as reconstruction targets rather than as network inputs. HG-SuperF keeps that problem formulation and changes the pieces that dominate **wall-clock cost** and **production viability** at national AOI sizes: the positional encoding, the image-formation (degradation) model, the sampling of the LR loss, and the stopping rule.

### 3.1 Problem setup

As in SuperF, we describe images as functions mapping continuous coordinates to intensities. Let $\mathbf{y}_{\mathrm{LR}}^{(1)},\dots,\mathbf{y}_{\mathrm{LR}}^{(T)}$ be $T$ cloud-filtered Sentinel-2 RGB revisits of the same ground footprint, observed on the discrete LR grid $\mathcal{W}$. Our goal is an approximation $\hat{\mathbf{y}}_{\mathrm{HR}}$ of the underlying high-resolution (HR) reflectance field on a finer grid $\mathcal{V}$ with $|\mathcal{V}| > |\mathcal{W}|$. Throughout we use an upsampling factor of $r{=}4$ (10 m → 2.5 m).

Each AOI—an independent TTO problem—is a local window of the national map. There is no pretrained network and no weight sharing across AOIs. The shipped operating point uses **LR $512{\times}512$** windows (5.12 km on a side); we study other sizes in §4.

We assume that each LR observation is a degraded, misaligned view of a shared HR field,

$$
\mathbf{y}_{\mathrm{LR}}^{(t)}(\mathbf{w})
\;\approx\;
\bigl(\varphi * \hat{\rho}^{(t)}(f_{\theta}(\hat{\mathbf{A}}^{(t)}\mathbf{v}))\bigr)(\mathbf{w}),
\qquad \mathbf{w}\in\mathcal{W},
$$

where $f_{\theta}$ is a coordinate-based network (the INR), $\hat{\mathbf{A}}^{(t)}$ is a learnable affine warp in homogeneous pixel coordinates, $\hat{\rho}^{(t)}$ is a per-frame radiometric (colour) transform, and $\varphi$ is a fixed sensor degradation (blur + downsample) from HR to LR. Following SuperF and handheld MISR practice, we freeze the base frame: $\hat{\mathbf{A}}^{(1)}=I$ and $\hat{\rho}^{(1)}$ is the identity, so all other frames align relatively to frame 1.

### 3.2 Shared INR with hashgrid encoding

SuperF parameterises $f_{\theta}$ as a ReLU MLP on **random Fourier features** $\gamma(\mathbf{v})$ [Tancik et al., 2020], with a domain-sensitive scale $\sigma$. That encoding recovers high frequencies from multi-frame constraints, but the cost and quality both depend strongly on $\sigma$ and on AOI size: the same Fourier setup that works on small chips degrades when each optimisation window must grow for national throughput (§4.2).

HG-SuperF replaces Fourier features with an **Instant-NGP multi-resolution hashgrid** [Müller et al., 2022], implemented via tinycudann, followed by a fully fused MLP decoder (depth 4, width 256, FP16). Coordinates $\mathbf{v}\in[0,1)^{2}$ are encoded by a hierarchy of hash tables; the decoder maps the concatenated features to RGB. The hash ladder is sized from the LR shape so that physical cell sizes stay matched across AOI scales (finest level at the LR pitch; 16 levels × 2 features by default; table size $\log_2{=}21$). Raising table capacity or finest-level density beyond this point does not improve NIB LPIPS in our ablations and is not part of the method.

The continuous INR is still queried on the **HR** coordinate grid (SuperF’s supersampling idea): alignment and radiometry act in continuous coordinates, and only the degraded prediction is compared to LR observations.

### 3.3 Geometric and radiometric alignment

Per frame $t$, we directly parameterise a 2×3 affine $\hat{\mathbf{A}}^{(t)}$ (translation and linear part; rotation is free but typically near zero at 5 km scales) and a per-band scale–shift colour map $\hat{\rho}^{(t)}$. Both are optimised jointly with $\theta$. Direct affine parameters—not a second MLP over the frame index—are inherited from SuperF’s improvement over NIR-style alignment.

At LR512, one global affine per frame is an adequate model of residual geolocation and orbit-induced shift. At much larger windows (e.g. LR2048), a single affine averages a spatially varying warp field; we treat that as an operational limit of the current formulation rather than a reason to enlarge AOIs for quality (§4.4).

The frozen identity frame is **not** “whichever revisit happens to be index 0.” Independent cells that lock to different residual-haze days produce a 5.12 km brightness grid in the mosaic: on 32VNM the SR cell offset versus neighbours tracks the bilinear (base-LR) offset at $r{=}0.99$. We therefore take as $t{=}1$ the revisit closest to the scene centre date whose *cell* SCL cloud fraction is ${\le}2\%$ (else the clearest day), and move it to index 0 so the identity affine/radiometry actually attach to that frame.

Independent 2% picks still leave date islands where a neighbour’s day is only slightly over the cap (or missing). Delivery therefore **grows same-date components** before training: the granule primary is the closest-to-centre day that is 2%-clear on ${\ge}50\%$ of cells; a cell joins that day (or a neighbour’s day) if the date is in-stack and cell SCL ${\le}5\%$. Leftover weather cuts are not re-coloured; they get a 1-D cosine ramp at mosaic time (§4.7). Overlap blending and per-tile colour affines are appendix checks, not the shipped mosaic.

### 3.4 Sentinel-2 image formation

SuperF’s synthetic satellite protocol and the public demo often use a simplified blur-then-average degradation. For real Sentinel-2 MISR we use a **calibrated band-wise MTF** in metres (`s2_psf_m`): the HR prediction is blurred with sensor-appropriate Gaussian kernels, then area-downsampled by $r{=}4$ onto the LR grid. An `area`-only ablation (no MTF) is worse on our NIB set (§4.6), so the physical PSF is part of the shipped forward model—not an optional extra.

### 3.5 Reconstruction loss and LR holdout early stopping

Let $\hat{\mathbf{y}}_{\mathrm{LR},\theta}^{(t)}$ denote the degraded, colour-corrected prediction on frame $t$. We minimise a robust Charbonnier penalty ($\varepsilon{=}0.01$) on **training** LR pixels,

$$
\mathcal{L}
=
\frac{1}{T}
\sum_{t=1}^{T}
\sum_{\mathbf{w}\in\mathcal{W}_{\mathrm{train}}^{(t)}}
\sqrt{
\bigl(\hat{\mathbf{y}}_{\mathrm{LR},\theta}^{(t)}(\mathbf{w})
-
\mathbf{y}_{\mathrm{LR}}^{(t)}(\mathbf{w})\bigr)^{2}
+\varepsilon^{2}
},
$$

averaged with a masked mean over selected pixels (not a mean that dilutes the signal by counting zeros).

National production has **no** HR orthophoto, so HR metrics (LPIPS/PSNR) cannot drive stopping. We therefore hold out a spatial fraction (default 10%) of LR pixels in **blocks** (side length auto-scaled with AOI; ≈8 LR px at LR512), independently per frame, and monitor held-out MSE. Blocks matter: the hashgrid’s finest level is comparable to the LR pitch, so isolated held-out pixels would be interpolated from trained neighbours. We do **not** hold out whole frames—removing the anchor frame frees the radiometric/geometric gauge.

Early stopping uses an EMA of holdout MSE (α=0.4), patience 8, a small absolute regression trip-wire, and a minimum iteration floor. On stop (or at the iteration cap), we restore the checkpoint with the best holdout score, then decode the full HR field. This is the only stop rule that transfers to Norway-wide runs without NIB.

### 3.6 Fused-$k$ tiling for scalable steps

Evaluating the full LR×HR correspondence every step is unnecessary once the AOI is large. **Fused-$k$ sampling** draws $k$ LR tiles of fixed side (128 px at LR512) per iteration and accumulates the loss over that mini-batch of tiles (`within`-AOI mixing). The shipped knee is **$k{=}4$**: near full-field LPIPS at a fraction of the time and roughly gigabyte-class peak activation memory per AOI. $k{=}0$ denotes full coverage (all tiles). Gradient scaling and forward chunking keep the FP16 fused MLP numerically stable when many LR elements enter the mean—these are engineering prerequisites for fused-$k$, not separate research claims.

### 3.7 Inference and mosaicking

After optimisation, $f_{\theta}$ is evaluated on the HR grid (tiled decode for large outputs) to write a 2.5 m RGB product. Optional GeoTIFF export attaches the AOI georeferencing for GIS use. Contiguous national coverage is obtained by scheduling many independent AOIs (mainland LR512 cells). The **shipped mosaic** is 0-overlap painter’s assembly after the ICM identity plan, then a cosine ramp (${\sim}320$ m) **only** on edges whose neighbours still disagree in identity date. Same-date abutments are left alone so genuine land-cover steps are not histogram-matched away. HR orthophotos are used **only** for offline evaluation metrics, never inside the production loss, stop, or colour fit.

---


## 4. Experiments

**Metrics.** LPIPS (VGG, primary), PSNR, SSIM; **training_time_s**; process-level NVML peak GPU memory (includes the tiny-cuda-nn arena); **h/MGRS** = `training_time_s × 441 / 3600` (reference cell count for an MGRS-sized LR512 workload).

**Frozen ablation cities (7):** asker, bergen, rana, tromso, amli, vennesla, trondheim.  
**Aligned NIB set (17):** see `results/spatial_alignment.json` — full table TODO.

### 4.1 Main result — national AOI recipe (B0) vs bilinear vs Fourier@512 (B1)

Mean over 7 cities (`paper_ablations.json`):

| Method | LPIPS ↓ | vs bil | Time / AOI | ≈ h/MGRS |
|--|--:|--:|--:|--:|
| Bilinear | 0.458 | — | — | — |
| **HG-SuperF (B0)** | **0.395** | **−0.064** | **82 s** | **~10.1** |
| Fourier@512 (B1) | 0.575 | +0.117 (worse than bil) | 96 s | ~11.8 |

B0 beats bilinear on all 7 cities. B1 loses to bilinear on all 7. **Do not** read B1 as “Fourier is useless”—see §4.2.

### 4.2 Fair Fourier — when small windows still work

Asker full-field hash (**H_aoi***) vs Fourier (**F_aoi*_s{2,5,10}**), from fair track + s2 top-up:

| AOI | Best F (among s∈{2,5,10}) | Hash H | Bilinear |
|--|--:|--:|--:|
| 64 | 0.392 (s10) | **0.375** | 0.484 |
| 128 | 0.438 (s10) | **0.386** | 0.491 |
| 256 | 0.489 (s10) | **0.377** | 0.470 |
| 512 | 0.557 (s2) | **0.362** | 0.444 |

**Story for the paper:** Fourier can beat bilinear on small chips; as AOI size grows toward the national schedule, hash remains strong and Fourier does not. Scalability is the claim, not a blanket encoding superiority at every scale.

### 4.3 Fused-k Pareto (B2)

7-city means:

| Variant | LPIPS | Time / AOI | ≈ h/MGRS |
|--|--:|--:|--:|
| k4 (**B0**) | 0.395 | **82 s** | **10.1** |
| k2 | 0.393 | 127 s | 15.5 |
| k1 | 0.391 | 224 s* | 27.5* |
| full | 0.391 | 225 s | 27.6 |

\*Verify `B2_k1` timing against logs before locking the table—k1≈full is suspicious.

**Takeaway:** k4 is the operating point we ship; full-field buys little LPIPS for large extra time at LR512.

### 4.4 AOI size ladder (Asker `S_aoi*`)

| LR side | LPIPS | Time | Notes |
|--|--:|--:|--|
| 256 | 0.377 | 67 s | Better LPIPS; more AOIs / km² |
| 512 (B0) | 0.370 | 81 s | Ship |
| 1024 | 0.402 | 226 s | |
| 2048 | 0.470 | 435 s | Worse than bil (0.440); affine stress |

Argue **LR512** as the national default: quality holds, time stays ~1 min, affine remains plausible.

### 4.5 Nested complete-HR size ladder *(same footprints)*

Project-mean LPIPS on nested children of the same LR512 complete tiles (partial nest64 freeze; full 5504 still training):

| LR train size | Project mean LPIPS | Bilinear |
|--|--:|--:|
| 64 | 0.356 | 0.413 |
| 128 | 0.372 | 0.432 |
| 256 | 0.379 | 0.438 |
| 512 | 0.394 | 0.445 |

Smaller windows look better **per chip**, but covering the same ground needs many more INRs (roughly ×64 at LR64 vs one LR512). Throughput, not chip LPIPS alone, decides the national recipe—consistent with §4.4.

### 4.6 Secondary ablations

| Variant | LPIPS | Time |
|--|--:|--:|
| B0 Charbonnier | 0.395 | 82 s |
| B3 MAE | 0.395 | 82 s |
| B4 area downsample (no MTF) | 0.410 | 80 s |

PSF helps; MAE ≈ Charbonnier on this set.

### 4.7 National scale *(figures, not a leaderboard)*

- Tile → train → mosaic pipeline (`PRODUCTION.md`).
- Norway-wide LR512 clear-day heatmap (July ±45 d SCL proxy)—finalize missing tile + freeze figure.
- Shared-season full-granule `32VNM` delivery complete: 438 populated LR512 cells on a 107.52 km square grid.
- **Finding (base-frame gauge):** a 15% cell-SCL admit still leaves 10–14% cloud on the chronological-first frame. Those cells are systematically brighter than neighbours (32VNM v2: 53 cells with $\Delta\mu{\ge}0.015$; frozen-frame cloud 9.7% vs 1.6% on quiet cells). Closest-to-centre days for the same cells are typically clear. This is a mosaic radiometric-gauge bug, not an INR hallucination—do not score it as a 17-city LPIPS ablation. Evidence pair: `32VNM_v2` (earliest-frame freeze) vs `32VNM_base2` (closest day with cell cloud ${\le}2\%$).
- **Delivery (ICM + date-cut ramp):** on a 143-cell 32VNM block, independent 2% still had 61 date-boundary edges. ICM (primary 2025-07-12, 5% slack) joined 8 cells and cut edges to 40; only 16 tiles needed a new train. A 128 HR-px ramp on remaining date cuts halved leftover $|\Delta\mu|$ (0.011→0.0048) and left same-date edges at MAE 0.0084. Main figure: `production/seams/32VNM_stack/delivery_icm_ramp.png`. Colour-affine leftovers, overlap feather/Laplacian, and halo consistency are appendix (`APPENDIX_SEAMS.md`)—they do not remove date islands and are not in the shipped mosaic.
- **Full granule:** the same plan changes 29/438 cells, grows the 12 Jul primary from 358→376 cells, and reduces date cuts 126→90. The ramp reduces remaining cut $|\Delta\mu|$ 0.0093→0.0047 while same-date MAE stays 0.00824. Output: `production/mosaics/32VNM_icm_ramp_sr_2p5m.tif`; timing and verification: `production/seams/32VNM_full/REPORT.md`.

---

## 5. Discussion *(bullets for later prose)*

- **Encoding × scale interaction** is the scientific point; B1 alone is incomplete.
- **Holdout vs HR metrics** disagree; production must trust LR holdout.
- **Affine per frame** is a 5 km assumption; 20 km shows the limit (S_aoi2048, affine-grid viz).
- **Cloud floor** and per-cell scheduling beat whole-MGRS OCM (32VNK lesson).
- **Base-frame gauge:** freeze the closest nearly cloud-free day, not stack order; otherwise the mosaic inherits a 5.12 km DC grid from residual SCL cloud on frame 0. After that, grow same-date components with 5% SCL slack (ICM) and ramp leftover weather cuts—do not independently match tile histograms.
- Limitations: RGB-only here; no Denmark yet; wall-to-wall Norway and acquisition-inclusive timing remain.

---

## 6. Conclusion *(one paragraph, draft)*

HG-SuperF makes test-time MISR SR **schedulable**: hashgrid + fused-k + LR holdout stop deliver NIB-beating LPIPS at ~80 s per 5 km AOI, while Fourier-style encodings that look plausible on tiny patches fail at the window sizes national maps require. A complete 32VNM granule validates the ICM+ramp delivery path; the remaining geographic work is multi-granule and ultimately wall-to-wall Norway, not another capacity chase.

---

## Appendix pointers

- Frozen JSON: `paper/results/` + `MANIFEST.json`
- Recipe decision record: `docs/METHOD.md`
- Open checklist: `docs/HASHGRID_SUPERF.md` §4
- Mosaic ablations (not delivery): [`APPENDIX_SEAMS.md`](APPENDIX_SEAMS.md)
