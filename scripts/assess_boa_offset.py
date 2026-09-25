#!/usr/bin/env python3
"""Assess the effect of the unremoved Sentinel-2 BOA_ADD_OFFSET on reported quality metrics.

For processing baseline >= 04.00 the stored L2A digital numbers include +1000, so the loader's
DN/10000 values are reflectance + 0.1. Per-frame standardization cancels a constant offset during
fitting, the prediction is destandardized with the base frame, and the reference is histogram-matched
to the base frame, so prediction, bilinear, and reference all carry the same +0.1. Removing it is
therefore an exact shift of all three (up to clipping at 1), which lets us rescore existing float
outputs without refitting.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import types
from pathlib import Path

import lpips
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("optimize", types.SimpleNamespace(get_lpips_model=None))
from eval.masked_metrics import mask_bbox_slices, masked_lpips, masked_psnr, masked_ssim  # noqa: E402
from scripts.rescore_nested_parent_reference import (  # noqa: E402
    NEST_ROOT, RUN_MANIFEST, RUN_TAG, SIDES, _covering_tile, _index_manifests, _run_dir, crop_for,
    project_mean, read, t,
)
from scripts.run_complete_patch_size_ladder import _project_of_city  # noqa: E402

OUT = ROOT / "paper" / "results" / "boa_offset_assessment.json"
OUT_PARENT = ROOT / "paper" / "results" / "boa_offset_assessment_parent_ref.json"
OFFSET = 0.1
OFFSET_DETECT = 0.05


def has_offset(bilinear: np.ndarray) -> bool:
    valid = np.all(bilinear > 0, axis=-1)
    return bool(np.percentile(bilinear[valid], 0.1) > OFFSET_DETECT)


def scores(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, fn) -> dict:
    y0, y1, x0, x1 = mask_bbox_slices(mask)
    p, g = t(pred)[:, :, y0:y1, x0:x1], t(gt)[:, :, y0:y1, x0:x1]
    m = torch.from_numpy(mask[y0:y1, x0:x1])
    with torch.no_grad():
        return {"lpips": masked_lpips(fn, p, g, m), "ssim": masked_ssim(p, g, m), "psnr": masked_psnr(p, g, m)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", choices=["window", "parent"], default="window")
    args = ap.parse_args()
    torch.set_num_threads(16)
    fn = lpips.LPIPS(net="vgg", verbose=False).eval()
    parents = set(json.loads(RUN_MANIFEST.read_text())["parents"])
    index = _index_manifests()
    children = [c for c in json.loads((NEST_ROOT / "nested_lr64_manifest.json").read_text())["tiles"]
                if c["parent_tile_id"] in parents]
    rows, offsets = [], {}
    for i, child in enumerate(children, 1):
        cov = {s: _covering_tile(child, s, index) for s in SIDES}
        dirs = {s: _run_dir(tt["parent_city"], tt["tile_id"], s, RUN_TAG) for s, tt in cov.items()}
        bil = read(dirs[64] / "qgis" / "s2_bilinear.tif")
        gt = read(dirs[64] / "qgis" / "hr_gt.tif")
        if args.reference == "parent":
            gt = crop_for(512, child, read(dirs[512] / "qgis" / "hr_gt.tif"))[: gt.shape[0], : gt.shape[1]]
        mask = np.all(gt > 1e-6, axis=-1)
        shifted = has_offset(bil)
        offsets[child["parent_tile_id"]] = shifted
        preds = {s: crop_for(s, child, read(dirs[s] / "qgis" / "sr_pred.tif"))[: gt.shape[0], : gt.shape[1]]
                 for s in SIDES}
        preds["bilinear"] = bil
        gt_fixed = np.where(mask[..., None], gt - OFFSET, 0.0) if shifted else gt
        for s, pred in preds.items():
            fixed = pred - OFFSET if shifted else pred
            if s == "bilinear":
                fixed = np.clip(fixed, 0.0, 1.0)
            cur, cor = scores(pred, gt, mask, fn), scores(fixed.astype(np.float32), gt_fixed.astype(np.float32), mask, fn)
            rows.append({"child": child["tile_id"], "parent": child["parent_tile_id"],
                         "project": _project_of_city(child["parent_city"]), "side": s, "offset_removed": shifted,
                         **{f"{k}_current": v for k, v in cur.items()}, **{f"{k}_corrected": v for k, v in cor.items()}})
        if i % 50 == 0:
            print(f"{i}/{len(children)}", flush=True)

    table = [{"side": s, **{f"{k}_{v}": project_mean(rows, s, f"{k}_{v}")
                            for k in ("lpips", "ssim", "psnr") for v in ("current", "corrected")}}
             for s in [*SIDES, "bilinear"]]
    (OUT_PARENT if args.reference == "parent" else OUT).write_text(json.dumps({"offset": OFFSET, "reference": args.reference, "parents_with_offset": offsets, "table": table, "rows": rows},
                              indent=1) + "\n")
    for r in table:
        print(r["side"], {k: round(v, 4) for k, v in r.items() if k != "side"})


if __name__ == "__main__":
    main()
