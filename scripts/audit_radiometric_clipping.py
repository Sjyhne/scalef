#!/usr/bin/env python3
"""Audit supplied pre-clip RGB arrays on explicit masks, without changing runs.

See docs/RADIOMETRY_AUDIT.md for the manifest and required provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.radiometry_audit import audit_array


def read_array(path, *, mask=False, window=None, bands=None):
    if path.suffix == ".npy":
        x = np.load(path, allow_pickle=False)
        if window is not None:
            col, row, width, height = window
            x = x[row:row + height, col:col + width]
        return x
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as src:
        w = Window(*window) if window is not None else None
        if w is not None and (w.col_off < 0 or w.row_off < 0 or w.col_off + w.width > src.width or w.row_off + w.height > src.height):
            raise ValueError(f"Window exceeds raster: {path}")
        return src.read(1, window=w) if mask else src.read(bands or [1, 2, 3], window=w).transpose(1, 2, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    if not manifest.get("entries"):
        raise ValueError("Manifest has no entries")
    rows = []
    for entry in manifest["entries"]:
        def resolve(name):
            p = Path(entry[name])
            return p if p.is_absolute() else args.manifest.parent / p
        for key in ("id", "stage", "domain", "offset", "source_commit", "clipping_history", "mask_semantics"):
            if key not in entry:
                raise ValueError(f"Missing manifest key: {key}")
        rgb = read_array(resolve("path"), window=entry.get("window"), bands=entry.get("bands"))
        # Masks must be prepared on this exact crop/grid; no implicit reprojection.
        valid = read_array(resolve("valid_mask"), mask=True)
        clear = read_array(resolve("clear_mask"), mask=True)
        for name, m in (("valid", valid), ("clear", clear)):
            if m.shape != rgb.shape[:2] or not np.isin(m, [0, 1]).all():
                raise ValueError(f"{entry['id']}: {name} mask must be binary and match the RGB crop")
        kwargs = {"domain": entry["domain"], "offset": float(entry["offset"])}
        rows.append({"source": entry,
                     "valid": audit_array(rgb, valid, **kwargs),
                     "clear_valid": audit_array(rgb, valid.astype(bool) & clear.astype(bool), **kwargs)})
    report = {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "entries": rows,
              "scope": "Counts and fixed-support statistics only; not a training ablation or metric sensitivity result.",
              "limitation": "Prior clipping is irreversible. Zero exceedances in clipped exports do not establish that pre-clip values were unaffected."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(rows)} entries to {args.out}")


if __name__ == "__main__":
    main()
