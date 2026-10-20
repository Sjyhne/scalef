# Figure and clipping audit

This is a read-only audit of the reported runs. Do not refit the benchmark or
replace reported scores merely to run it. The manuscript's recorded protocols
take precedence over a later working tree: the current exporter and final-score
route use a 1.5 upper bound, whereas the manuscript records a 1.0 scoring clip.
Recover the producing code and saved run provenance before interpreting either.

## Re-render the figures from their producing runs

```bash
python scripts/render_paper_figures.py original-superf
python scripts/render_paper_figures.py encoding-scale
```

Check the CLI help for the available subcommand names. These commands need the
paths in the renderer, `metrics.json`, S2 `meta.json`, and float `qgis/*.tif`
exports. The original-SuperF figure also needs the original `pred_hwc.npy` files.
The renderer now reads the extension run's recorded `s2_boa_offset` mode:
`remove` means that the float export already has the offset removed; `keep`
requires removing it once for display. Missing or conflicting provenance raises
an error. Mixed processing baselines also require the actual base-frame offset;
there is no fallback to an assumed 0.1.

The encoding renderer now requires float exports and applies the same 0--0.25
mapping throughout. It does not fall back to percentile-stretched PNGs. If float
exports are absent, recover them from the saved final checkpoint using the exact
producing configuration, frozen masks and alignment. Do not invert PNG stretches.

For Asker, compare the raw warped orthophoto, histogram-matched orthophoto,
`qgis/hr_gt.tif`, and the rendered panel in that order. Check band order, no-data,
value ranges, offset removal, and the histogram match. Keep all six sites visible;
do not drop Asker just because the reference looks unusual. A display-only fix
does not require rescoring; a wrong reference or scoring transform does. Inspect
the actual arrays before choosing between those cases.

After regenerating, copy PDF, PNG and JSON sidecars together into the paper repo.
Only then remove the manuscript's temporary percentile-display exception. Check
that the six original-SuperF score badges still refer to the arrays actually
scored, rather than recomputing them from the displayed image.

## Count clipping and masking on fixed support

Prepare a JSON manifest with one entry per frame/array. Include all selected
revisits at all 17 named sites, the affected 32UNG cell, its two controls, and
representative bright production cells. Do not report these as independent sites.

```json
{
  "entries": [
    {
      "id": "site_tile_frame",
      "stage": "lr_input_before_loader_clip",
      "path": "/data/exact_source_frame.tif",
      "window": [9234, 6318, 512, 512],
      "bands": [1, 2, 3],
      "domain": "dn",
      "offset": 0.1,
      "source_commit": "REPLACE_WITH_PRODUCING_COMMIT",
      "clipping_history": "none; original delivered DN",
      "valid_mask": "/data/exact_crop_valid.npy",
      "clear_mask": "/data/exact_crop_clear.npy",
      "mask_semantics": "HW binary arrays, 1=eligible; clear mask before bright-pixel masking"
    }
  ]
}
```

The example window is the affected 32UNG cell; supply the actual crop for every
entry. Raster bands must be ordered R/G/B. NPY RGB arrays must be HWC. Masks must
be binary HW arrays on that exact cropped grid, using the producing run's valid
and clear definitions. The command does not resample masks or guess their class
meaning. Distinguish valid dark pixels from no-data before offset subtraction.
Use `domain: dn_over_10000` or `surface_reflectance` for those actual domains;
never infer the domain from a maximum value. `offset` is the amount to subtract
from DN/10000 (0.1 or 0 for the products used here).

```bash
python scripts/audit_radiometric_clipping.py \
  --manifest /data/radiometry_audit_manifest.json \
  --out paper/results/radiometry_clipping_audit.json
```

The output separates valid from clear-and-valid support and reports per-band and
any-band exceedances at DN/10000 = 1 and 1.5, physical reflectance = 1, lower
clipping, nonfinite values, display saturation, and means/standard deviations
before and after clipping on the same support. Threshold comparisons are strict
(`>`), matching the bright-pixel mask. These are counts and statistical
diagnostics, not an isolated normalization ablation.

For the scoring audit, supply prediction, bilinear and harmonized reference
arrays **before the historical scoring clip**, with the original evaluation mask.
Record any clipping already applied to an export. A zero exceedance count in a
clipped export cannot establish that the pre-clip values were unaffected.
If pre-clip arrays are missing, re-render the saved final parameters; this does
not require training again. Recompute PSNR/SSIM/LPIPS under the historical clip
and without that additional clip, holding masks, registration, histogram target,
offset conversion, metric data range and aggregation fixed. Do not rematch the
reference separately for each arm or alter LPIPS preprocessing between arms.

Summarize affected fractions per frame and per site, including the maximum.
Only consider a targeted training sensitivity experiment if clear observations
are materially clipped or the metric conclusions change. The existing Denmark
comparison changes both the loss mask and normalization statistics; separating
those effects would require statistics-only, loss-only and joint masking arms.
That experiment is optional if the paper keeps the narrower robustness claim.
