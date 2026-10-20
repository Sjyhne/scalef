"""Read-only radiometry checks; no fitting, scoring, or display policy changes."""
from __future__ import annotations

import numpy as np


def export_to_surface(rgb, *, offset_mode: str, offset: float):
    """Convert a QGIS export once, using the exporting run's recorded mode.

    ``remove`` means optimize.py already subtracted the base offset before export.
    ``keep`` means the export retains DN/10000 units. Zero remains the no-data
    sentinel in that route; negative surface values are preserved until display.
    """
    x = np.asarray(rgb, dtype=np.float32)
    if offset_mode == "remove":
        return x
    if offset_mode == "keep":
        return np.where(x != 0, x - float(offset), x)
    raise ValueError("Unknown export offset mode; inspect the run provenance instead of guessing")


def audit_array(rgb, mask, *, domain: str, offset: float):
    """Summarize values on an explicit, fixed support before any audit clipping.

    Input must be HWC RGB and the mask must already encode the desired valid or
    clear-and-valid support. This deliberately does not infer units from maxima.
    """
    x = np.asarray(rgb, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 3 or x.shape[-1] != 3 or mask.shape != x.shape[:2]:
        raise ValueError("Expected HWC RGB and a matching HW mask")
    if not np.isfinite(offset) or offset < 0:
        raise ValueError("offset must be finite and nonnegative")
    if domain == "dn":
        x = x / 10000.0
    elif domain == "surface_reflectance":
        x = x + offset
    elif domain != "dn_over_10000":
        raise ValueError("domain must be dn, dn_over_10000, or surface_reflectance")
    finite = np.isfinite(x).all(axis=-1)
    values = x[mask & finite]
    result = {"selected_pixels": int(mask.sum()),
              "nonfinite_selected_pixels": int((mask & ~finite).sum()),
              "finite_selected_pixels": int(len(values)), "threshold_domain": "dn_over_10000"}
    if not len(values):
        result["status"] = "empty_support"
        return result

    def counts(flags):
        return {"count_per_band": flags.sum(axis=0).tolist(),
                "fraction_per_band": flags.mean(axis=0).tolist(),
                "any_band_count": int(flags.any(axis=1).sum()),
                "any_band_fraction": float(flags.any(axis=1).mean())}

    def stats(a):
        return {"mean": a.mean(axis=0).tolist(),
                "sample_std": a.std(axis=0, ddof=1).tolist() if len(a) > 1 else None}

    result.update({"status": "ok", "min": values.min(axis=0).tolist(),
                   "max": values.max(axis=0).tolist(),
                   "percentiles_50_99_99_9": np.percentile(values, [50, 99, 99.9], axis=0).tolist(),
                   "before_clipping": stats(values),
                   "below_zero": counts(values < 0),
                   "above_dn_over_10000_1": counts(values > 1.0),
                   "above_dn_over_10000_1_5": counts(values > 1.5),
                   "above_surface_reflectance_1": counts(values > 1.0 + offset),
                   "after_clip_0_1_same_support": stats(np.clip(values, 0, 1)),
                   "after_clip_0_1_5_same_support": stats(np.clip(values, 0, 1.5)),
                   "display_outside_0_0_25": counts((values - offset < 0) | (values - offset > 0.25))})
    return result
