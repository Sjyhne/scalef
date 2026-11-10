# HG-SuperF — paper outline

Working title (provisional):

> **HG-SuperF for National Sentinel-2 Maps and Thin-Structure Segmentation**

Venue: IEEE JSTARS-style (remote sensing).  
Evidence: SR scale → [`results/`](results/) · Method → [`../docs/METHOD.md`](../docs/METHOD.md) · Ops → [`../docs/PRODUCTION.md`](../docs/PRODUCTION.md) · Downstream → `production/segbench/stratified/`.

---

## Dual pitch

1. **Engineering / scale** — Instant-NGP hashgrid test-time MISR SR fast enough for national **2.5 m** Sentinel-2 delivery (vs Fourier at operational AOI sizes; fused-k tiling; production mosaic path), evaluated on real NIB orthophotos.
2. **Downstream utility** — SR products tested on **thin-structure segmentation** (roads + buildings, FKB width bins thin / medium / wide) against bilinear, native 10 m, and temporal early-fusion on frozen Sør-Varanger.

## Contributions (map to sections)

| # | Claim | Where | Evidence |
|--|--|--|--|
| C1 | **HG-SuperF** recipe: hashgrid INR + physical S2 MTF + fused-k + production without HR fit target | §4 | Locked recipe in METHOD |
| C2 | **Scalability**: hash vs Fourier vs bilinear across AOI / nested subsample sizes; k4 knee; LR512 ops default | §4, §6 | Nested + fair Fourier + ablations |
| C3 | **Segbench** protocol: 1 m FKB → 2.5 m labels; width bins; SV-matched stratified train/val; frozen SV test | §5 | `segbench_*` bundles + split JSON |
| C4 | **Downstream**: shared U-Net on HG-SuperF / bilinear / base10 / temporal (+ Fourier SuperF if products exist); IoU + structure metrics | §6.4 | Stratified SV scores |

**Non-claims:** no blanket “hash always beats Fourier”; no SEN2* leaderboard; no verified 2.5 m optical resolution.

---

## Section plan

1. **Abstract**
2. **Introduction** — national 2.5 m; thin structures; SuperF lineage; cost + utility gaps; contributions
3. **Related work** — S2 SR / MISR / SuperF; mosaicking; road–building extraction; SR-for-downstream
4. **Scalable HG-SuperF** — forward model; hashgrid; fused-k; production path (seams → appendix)
5. **Thin-structure segmentation benchmark** — labels/width taxonomy; splits; inputs; shared U-Net; metrics
6. **Experiments**
   - **6.1 Setup**
   - **6.2 Named-site / main SR quality–cost** (HG vs bilinear; Fourier@large-AOI control as needed)
   - **6.2 Input size** — Fourier vs hash-grid across LR64…1024 (Overleaf `tab:area`)
   - **6.3 Full-field subsampling** — pixel fraction × LR patch size × **Fourier and hash-grid** (Overleaf `tab:sampling`)
   - **6.4 Hash-grid ablation**
   - **6.5 Downstream thin-structure segmentation** — size-class table (Overleaf `tab:segmentation`)
7. **Discussion**
8. **Conclusion**
9. **Appendix** — seams; protocol/reproducibility; extra qualitative

---

## Full-field subsampling table (Overleaf Results)

Matches `ScaleF_Overleaf/main.tex` `tab:sampling`: for each pixels-seen / patch-size cell, report **Fourier** and **Hash grid** rows (PSNR, SSIM, LPIPS, ZNCC, speed).

---

## Downstream table (~Experiment 4)

Shared U-Net; frozen SV test. Columns: **base10 · bilinear · HG-SuperF · temporal** (add Fourier SuperF only once nested/main SR products are locked and if bandwidth allows). Metrics: IoU + boundary / excess / thin-fill / width recall.

---

## Writing order

1. Lock this outline.
2. Nested three-way table (bilinear / Fourier / HG) under current protocol.
3. Scaling narrative §6.2–6.3 from `paper/results/` + nested refresh.
4. Downstream §6.4 once stratified scores are frozen.
5. Abstract / conclusion last.
