#!/usr/bin/env python3
"""Validate nested common-footprint scores against float outputs.

On the validation parents (scripts/run_nested_float_validation.py), each shared
LR64 window is scored three ways with the same LPIPS/PSNR code, the child's
reference, the child's bilinear baseline and the same base-date exclusion:

  float_v3  float32 qgis/{hr_gt,s2_bilinear,sr_pred}.tif of the floatval_v3 runs
  png_v3    display-PNG route of rescore_nested_common_footprint.py on the same runs
  png_v2    display-PNG route on the original lr512align_v2 runs (published table)

float_v3 vs png_v3 isolates PNG quantisation/stretch recovery; png_v3 vs png_v2
adds the halo fix and refitting variation.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.common_footprint import crop_hwc_frac, nest_window_frac  # noqa: E402
from eval.masked_metrics import mask_bbox_slices, masked_psnr  # noqa: E402
from optimize import get_lpips_model  # noqa: E402
from scripts.rescore_nested_common_footprint import (  # noqa: E402
    NEST_ROOT, SIDES, _aggregate, _base_date, _covering_tile, _hwc_to_bchw, _index_manifests,
    _infer_display_vmax, _load_rgb, _raw_base_lr, _run_dir,
)
from scripts.run_complete_patch_size_ladder import _project_of_city  # noqa: E402
from scripts.run_nested_float_validation import MANIFEST as RUN_MANIFEST, RUN_TAG  # noqa: E402

OUT = ROOT / "paper" / "results" / "nested_float_validation.json"
ROUTES = ("float_v3", "png_v3", "png_v2")
# Predeclared before scoring: keep the published (png_v2) nested table if, at every field
# size, its project-weighted mean LPIPS is within this of float_v3 and the grid-vs-bilinear
# gain keeps its sign.
LPIPS_TOLERANCE = 0.005


def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.transpose(src.read().astype(np.float32), (1, 2, 0))


def to_bchw(image: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).to(device)


def crop_for(side: int, child: dict, image: np.ndarray) -> np.ndarray:
    if side == 64:
        return image
    n = side // 64
    return crop_hwc_frac(image, *nest_window_frac(int(child["nest_iy"]) % n, int(child["nest_ix"]) % n, 64, side))


def score(pred, gt, bil, lpips_fn):
    mask = torch.all(gt[0] > 1e-6, dim=0)
    y0, y1, x0, x1 = mask_bbox_slices(mask)
    lp = lambda a: float(lpips_fn(a[:, :, y0:y1, x0:x1] * 2 - 1, gt[:, :, y0:y1, x0:x1] * 2 - 1).item())  # noqa: E731
    return lp(pred), masked_psnr(pred, gt, mask), lp(bil), masked_psnr(bil, gt, mask)


def float_route(child, covering, dirs, lpips_fn, device, cache):
    def load(p):
        if p not in cache:
            cache[p] = read_tif(p)
        return cache[p]
    cdir = dirs[64] / "qgis"
    gt = to_bchw(load(cdir / "hr_gt.tif"), device)
    bil = to_bchw(load(cdir / "s2_bilinear.tif"), device)
    return {side: score(to_bchw(crop_for(side, child, load(dirs[side] / "qgis" / "sr_pred.tif")), device),
                        gt, bil, lpips_fn) for side in SIDES}


def png_route(child, covering, dirs, lpips_fn, device, cache):
    def load(p):
        if p not in cache:
            cache[p] = _load_rgb(p)
        return cache[p]
    cdir = dirs[64]
    vmax64, _ = _infer_display_vmax(load(cdir / "lr_original.png"), _raw_base_lr(child, cdir))
    gt = (_hwc_to_bchw(load(cdir / "ground_truth.png")) * vmax64).to(device)
    display_bil = load(cdir / "bilinear_baseline.png")
    bil = (_hwc_to_bchw(display_bil) * vmax64).to(device)
    out = {}
    for side in SIDES:
        pred_display = crop_for(side, child, load(dirs[side] / "model_output_aligned.png"))
        rel = 1.0
        if side != 64:
            src = _hwc_to_bchw(crop_for(side, child, load(dirs[side] / "bilinear_baseline.png")))
            tgt = _hwc_to_bchw(display_bil)
            valid = (src > 0.05) & (src < 0.95) & (tgt > 0.05) & (tgt < 0.95)
            rel = float(torch.median(tgt[valid] / src[valid]))
        out[side] = score((_hwc_to_bchw(pred_display) * vmax64 * rel).to(device), gt, bil, lpips_fn)
    return out


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpips_fn = get_lpips_model(device)
    parents = set(json.loads(RUN_MANIFEST.read_text())["parents"])
    index = _index_manifests()
    children = [c for c in json.loads((NEST_ROOT / "nested_lr64_manifest.json").read_text())["tiles"]
                if c["parent_tile_id"] in parents]
    rows = {r: [] for r in ROUTES}
    excluded, missing = [], []
    cache: dict = {}
    for i, child in enumerate(children, 1):
        covering = {side: _covering_tile(child, side, index) for side in SIDES}
        dirs = {
            "v3": {s: _run_dir(t["parent_city"], t["tile_id"], s, RUN_TAG) for s, t in covering.items()},
            "v2": {s: _run_dir(t["parent_city"], t["tile_id"], s, "lr512align_v2") for s, t in covering.items()},
        }
        if not all((d / "qgis" / "sr_pred.tif").is_file() for d in dirs["v3"].values()):
            missing.append(child["tile_id"])
            continue
        per_version_dates = {v: {_base_date(d) for d in dd.values()} for v, dd in dirs.items()}
        if any(len(s) != 1 for s in per_version_dates.values()):
            excluded.append({"child": child["tile_id"],
                             "base_dates": {v: sorted(s) for v, s in per_version_dates.items()}})
            continue
        results = {
            "float_v3": float_route(child, covering, dirs["v3"], lpips_fn, device, cache),
            "png_v3": png_route(child, covering, dirs["v3"], lpips_fn, device, cache),
            "png_v2": png_route(child, covering, dirs["v2"], lpips_fn, device, cache),
        }
        for route, by_side in results.items():
            for side, (lp, ps, lpb, psb) in by_side.items():
                rows[route].append({
                    "child_tile_id": child["tile_id"], "parent_tile_id": child["parent_tile_id"],
                    "parent_city": child["parent_city"], "project_folder": _project_of_city(child["parent_city"]),
                    "side": side, "lpips": lp, "psnr": ps, "lpips_bilinear": lpb, "psnr_bilinear": psb,
                })
        if len(cache) > 256:
            cache.clear()
        if i % 100 == 0:
            print(f"{i}/{len(children)}", flush=True)

    def paired(a, b, key):
        ib = {(r["child_tile_id"], r["side"]): r[key] for r in rows[b]}
        out = {}
        for side in SIDES:
            d = [r[key] - ib[(r["child_tile_id"], side)] for r in rows[a] if r["side"] == side]
            out[str(side)] = {"n": len(d), "mean": statistics.fmean(d), "mean_abs": statistics.fmean(map(abs, d)),
                              "max_abs": max(map(abs, d))} if d else None
        return out

    summary = {r: _aggregate(rows[r])["table"] for r in ROUTES if rows[r]}

    def agreement(a: str, b: str) -> dict:
        tb = {t["lr_side"]: t for t in summary[b]}
        per_side = {}
        for t in summary[a]:
            u = tb[t["lr_side"]]
            d = t["mean_lpips_project"] - u["mean_lpips_project"]
            gain_a = t["mean_lpips_bilinear_project"] - t["mean_lpips_project"]
            gain_b = u["mean_lpips_bilinear_project"] - u["mean_lpips_project"]
            per_side[str(t["lr_side"])] = {"lpips_diff": d, "gain_a": gain_a, "gain_b": gain_b,
                                           "ok": abs(d) <= LPIPS_TOLERANCE and (gain_a > 0) == (gain_b > 0)}
        return {"per_side": per_side, "pass": all(v["ok"] for v in per_side.values())}

    decision = {
        "lpips_tolerance": LPIPS_TOLERANCE,
        "rule": "keep published png_v2 table iff |png_v2 - float_v3| <= tolerance and gain sign agrees at every side",
        "png_v2_vs_float_v3": agreement("png_v2", "float_v3"),
        "png_v3_vs_float_v3": agreement("png_v3", "float_v3"),
    }
    payload = {
        "decision": decision,
        "schema": "scalef.nested_float_validation.v1",
        "run_tag": RUN_TAG, "parents": sorted(parents), "n_children": len(children),
        "n_scored_windows": len(rows["float_v3"]) // len(SIDES),
        "excluded_base_date_mismatch": excluded, "missing": missing,
        "aggregation": "windows within parents, parents within projects, projects weighted equally",
        "table": summary,
        "paired_window_differences": {
            f"{a}-{b}": {k: paired(a, b, k) for k in ("lpips", "lpips_bilinear", "psnr")}
            for a, b in (("png_v3", "float_v3"), ("png_v2", "float_v3"), ("png_v2", "png_v3"))
        },
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"-> {OUT.relative_to(ROOT)}; windows={payload['n_scored_windows']} excluded={len(excluded)} missing={len(missing)}")
    for route, table in summary.items():
        print(route, [(t["lr_side"], round(t["mean_lpips_project"], 4), round(t["mean_lpips_bilinear_project"], 4)) for t in table])
    for k in ("png_v2_vs_float_v3", "png_v3_vs_float_v3"):
        print(k, "PASS" if decision[k]["pass"] else "FAIL",
              {s: round(v["lpips_diff"], 4) for s, v in decision[k]["per_side"].items()})
    for k, v in payload["paired_window_differences"].items():
        print(k, {s: (round(x["mean"], 4), round(x["max_abs"], 4)) for s, x in v["lpips"].items() if x})


if __name__ == "__main__":
    main()
