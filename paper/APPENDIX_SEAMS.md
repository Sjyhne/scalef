# Appendix — mosaic seam ablations (not the shipped recipe)

**Delivery (main text):** per-cell identity at cell SCL ≤2%, then ICM toward the seasonal primary with 5% slack, then a cosine ramp **only** on leftover date-cut edges. Figure: `production/seams/32VNM_stack/delivery_icm_ramp.png`.

Everything below was measured so we can say what we *did not* ship. Do not promote these to the main mosaic path.

## A. Stacked additions on the 143-cell block

Same extent as the delivery figure (32VNM y00–y12 × x10–x20). Four columns, one technique added each time:

`production/seams/32VNM_stack/stack_0_to_3.png`  
`production/seams/32VNM_stack/stack_0_to_3.json`  
`production/seams/32VNM_stack/STACK.md`

| Stage | Role | Outcome |
|-------|------|---------|
| 0 independent 2% | Control (`prod_k4_base2`) | 61 date-cut edges |
| 1 + ICM | **In delivery** | 40 edges; 8 cells join 12 Jul |
| 2 + coarse S2 colour on leftovers | **Appendix only** | Cut \|Δμ\| 0.0113→0.0102; interiors stay dated; skip |
| 3 + date-cut ramp | **In delivery** (applied on ICM, *without* stage 2) | Leftover \|Δμ\| → 0.0048; same-date edges unchanged |

Stage 2 is kept so reviewers can see that a joint LR gain/offset on leftovers does not replace ICM or the ramp.

## B. 3×3 overlap / blend (y03–y05 × x16–x18)

Independent 2% identity, then overlapping reconstructions (not mixed with ICM).

| Item | Path |
|------|------|
| Shared-stretch diagnosis | `production/seams/32VNM_3x3/diagnose_fixed_stretch.png` |
| Common-base 12 Aug (separate trains) | `production/seams/32VNM_3x3/diagnose_base12aug_fixed_stretch.png` |
| ovl32 hard / feather / colour+feather / Laplacian | `production/seams/32VNM_3x3/prod_k4_ovl32_0*.png` |
| ovl64 same | `production/seams/32VNM_3x3/prod_k4_ovl64_0*.png` |
| Report | `production/seams/32VNM_3x3/REPORT.md` |

**Takeaway:** overlap + feather softens a cut; colour+Laplacian does not kill a date island and Laplacian drops detail ~15×. Common-base 12 Aug works on this 3×3 because that day is 0% SCL on all nine—**not** true of the 143-cell block if the primary stays 12 Jul.

## C. Halo (training-time neighbour)

`production/halo/base2_halo_modes_y04_x17_x18.png`

West 32 LR px, modes full / lowfreq / mean, on the 12 Jul vs 12 Aug pair. Edge MAE did not move (~0.013). Halo cannot overrule an August-locked identity.

## D. Chronological frame 0 (v2) vs 2% identity (base2)

Full-granule finding already in §4.7: `production/mosaics/32VNM_v2_overview.png`, `32VNM_base2_vs_v2_overview.png`. Cells with neighbour Δμ ≥ 0.015: 53 → 8. This is why independent 2% is the *per-cell* default before ICM.

## E. What not to do

- Independently match entire tile SR histograms (washes out land cover).
- Force a cloudy identity just to share a date.
- Use NIB orthophotos for colour or stop (evaluation only).
- Ship overlap multiband or leftover colour affine as the national mosaic.
