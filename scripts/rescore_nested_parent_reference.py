#!/usr/bin/env python3
"""Rescore the float-validated nested windows against the LR512 parent's reference.

The nested table scores each shared LR64 window against the LR64 child's reference,
which is histogram-matched to the base frame over the 640 m child. The encoding
diagnostic instead scores every size against the LR512 parent's reference (matched over
5.12 km). This script scores the same float predictions both ways to measure how much of
the size ladder depends on the scale of reference harmonization.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import lpips
import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import types  # noqa: E402

# optimize imports tinycudann, which cannot initialise without a GPU; only get_lpips_model is referenced.
sys.modules.setdefault("optimize", types.SimpleNamespace(get_lpips_model=None))
from eval.common_footprint import crop_hwc_frac, nest_window_frac  # noqa: E402
from eval.masked_metrics import mask_bbox_slices, masked_psnr  # noqa: E402
from scripts.rescore_nested_common_footprint import (  # noqa: E402
    NEST_ROOT, SIDES, _covering_tile, _index_manifests, _run_dir,
)
from scripts.run_nested_float_validation import MANIFEST as RUN_MANIFEST, RUN_TAG  # noqa: E402

OUT = ROOT / "paper" / "results" / "nested_parent_reference.json"


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.transpose(src.read().astype(np.float32), (1, 2, 0))


def crop_for(side: int, child: dict, image: np.ndarray) -> np.ndarray:
    if side == 64:
        return image
    k = side // 64
    return crop_hwc_frac(image, *nest_window_frac(int(child["nest_iy"]) % k, int(child["nest_ix"]) % k, 64, side))


def t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)


def score(pred: np.ndarray, gt: np.ndarray, fn) -> tuple[float, float]:
    p, g = t(pred), t(gt)
    mask = torch.all(g[0] > 1e-6, dim=0)
    y0, y1, x0, x1 = mask_bbox_slices(mask)
    with torch.no_grad():
        lp = float(fn(p[:, :, y0:y1, x0:x1] * 2 - 1, g[:, :, y0:y1, x0:x1] * 2 - 1).item())
    return lp, float(masked_psnr(p, g, mask))


def project_mean(rows: list[dict], side, key: str) -> float:
    by_parent: dict = {}
    for r in rows:
        if r["side"] == side:
            by_parent.setdefault((r["project"], r["parent"]), []).append(r[key])
    by_project: dict = {}
    for (proj, _), v in by_parent.items():
        by_project.setdefault(proj, []).append(statistics.fmean(v))
    return statistics.fmean(statistics.fmean(v) for v in by_project.values())


def main() -> None:
    torch.set_num_threads(16)
    fn = lpips.LPIPS(net="vgg", verbose=False).eval()
    parents = set(json.loads(RUN_MANIFEST.read_text())["parents"])
    index = _index_manifests()
    children = [c for c in json.loads((NEST_ROOT / "nested_lr64_manifest.json").read_text())["tiles"]
                if c["parent_tile_id"] in parents]
    val = {(r["child_tile_id"], r["side"]): r for r in json.loads(
        (ROOT / "paper" / "results" / "nested_float_validation.json").read_text())["rows"]["float_v3"]}
    from scripts.run_complete_patch_size_ladder import _project_of_city

    rows, cache = [], {}

    def load(p: Path) -> np.ndarray:
        if p not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[p] = read(p)
        return cache[p]

    for i, child in enumerate(children, 1):
        cov = {s: _covering_tile(child, s, index) for s in SIDES}
        dirs = {s: _run_dir(tt["parent_city"], tt["tile_id"], s, RUN_TAG) for s, tt in cov.items()}
        gt_child = load(dirs[64] / "qgis" / "hr_gt.tif")
        gt_parent = crop_for(512, child, load(dirs[512] / "qgis" / "hr_gt.tif"))
        bil = load(dirs[64] / "qgis" / "s2_bilinear.tif")
        if gt_parent.shape != gt_child.shape:
            gt_parent = gt_parent[: gt_child.shape[0], : gt_child.shape[1]]
        preds = {s: crop_for(s, child, load(dirs[s] / "qgis" / "sr_pred.tif")) for s in SIDES}
        preds["bilinear"] = bil
        for s, pred in preds.items():
            pred = pred[: gt_child.shape[0], : gt_child.shape[1]]
            lc, pc = score(pred, gt_child, fn)
            lpar, ppar = score(pred, gt_parent, fn)
            rows.append({"child": child["tile_id"], "parent": child["parent_tile_id"],
                         "project": _project_of_city(child["parent_city"]), "side": s,
                         "lpips_child_ref": lc, "psnr_child_ref": pc,
                         "lpips_parent_ref": lpar, "psnr_parent_ref": ppar,
                         "lpips_published": val.get((child["tile_id"], s), {}).get("lpips")})
        if i % 50 == 0:
            print(f"{i}/{len(children)}", flush=True)

    sides = [*SIDES, "bilinear"]
    table = [{"side": s, **{k: project_mean(rows, s, k) for k in
                            ("lpips_child_ref", "lpips_parent_ref", "psnr_child_ref", "psnr_parent_ref")}}
             for s in sides]
    check = [abs(r["lpips_child_ref"] - r["lpips_published"]) for r in rows
             if r["lpips_published"] is not None]
    payload = {"n_windows": len(children), "table": table,
               "sanity_child_ref_vs_published": {"mean_abs": statistics.fmean(check), "max_abs": max(check)},
               "rows": rows}
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    for r in table:
        print(r["side"], {k: round(v, 4) for k, v in r.items() if k != "side"})
    print("sanity", payload["sanity_child_ref_vs_published"])


if __name__ == "__main__":
    main()
