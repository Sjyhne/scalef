#!/usr/bin/env python3
"""Why do LR512 fields lose quality on some nested windows?

For each shared LR64 window of the float-validated nested parents, compare the LR512
penalty (LR512 minus LR64 LPIPS, float_v3 route) with how much of the window's
date-to-date change a per-frame, per-band affine colour transform cannot explain when it
is fitted over the whole LR512 field rather than over the window alone. ScaleF's
colour model is exactly such a transform (one gain and offset per frame and band), so
this ``transform-locality residual`` measures what a larger field cannot absorb.

Frames are the parent's LR stack; pixels flagged by the stack's per-frame LR cloud masks
are not available here, so the residual uses the median over frames to limit the effect
of cloudy dates. Also reports the penalty by window position (centre / interior / edge).
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "paper" / "results" / "nested_float_validation.json"
OUT = ROOT / "paper" / "results" / "nested_lr512_penalty.json"
S2 = ROOT / "data" / "s2_revisits"
W = 64


def load_stack(parent: str) -> tuple[np.ndarray, int]:
    meta = json.loads((S2 / parent / "meta.json").read_text())
    win = meta["aoi_window"]
    frames = []
    for f in meta["frames"]:
        with rasterio.open(S2 / parent / f["path"]) as src:
            a = src.read((1, 2, 3), window=Window(win["col_off"], win["row_off"], win["width"], win["height"]))
        frames.append(a.astype(np.float32) / 10000.0)
    stack = np.stack(frames)  # (F, 3, H, W)
    dates = [f["path"][4:12] for f in meta["frames"]]
    nib = meta.get("nib_acquisition_date", "").replace("-", "")
    base = min(range(len(dates)), key=lambda i: abs(int(dates[i]) - int(nib))) if nib else 0
    return stack, base


def affine_residual(x: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, eval_mask: np.ndarray) -> float:
    """RMS of y - (a x + b) over eval_mask, with a, b fitted by least squares on fit_mask."""
    xf, yf = x[fit_mask], y[fit_mask]
    A = np.stack([xf, np.ones_like(xf)], 1)
    coef, *_ = np.linalg.lstsq(A, yf, rcond=None)
    r = y[eval_mask] - (coef[0] * x[eval_mask] + coef[1])
    return float(np.sqrt(np.mean(r * r)))


def window_residuals(stack: np.ndarray, base: int) -> dict[tuple[int, int], dict]:
    F, _, H, Wd = stack.shape
    full = np.ones((H, Wd), bool)
    out = {}
    for iy in range(H // W):
        for ix in range(Wd // W):
            m = np.zeros((H, Wd), bool)
            m[iy * W:(iy + 1) * W, ix * W:(ix + 1) * W] = True
            glob, loc = [], []
            for f in range(F):
                if f == base:
                    continue
                g = np.mean([affine_residual(stack[f, c], stack[base, c], full, m) for c in range(3)])
                l = np.mean([affine_residual(stack[f, c], stack[base, c], m, m) for c in range(3)])
                glob.append(g)
                loc.append(l)
            out[(iy, ix)] = {"global": float(np.median(glob)), "local": float(np.median(loc)),
                             "excess": float(np.median(np.array(glob) - np.array(loc)))}
    return out


def position(iy: int, ix: int, n: int = 8) -> str:
    if iy in (0, n - 1) or ix in (0, n - 1):
        return "edge"
    if iy in (n // 2 - 1, n // 2) and ix in (n // 2 - 1, n // 2):
        return "centre"
    return "interior"


def main() -> None:
    val = json.loads(VALIDATION.read_text())
    rows = val["rows"]["float_v3"]
    by = {}
    for r in rows:
        by.setdefault(r["child_tile_id"], {})[r["side"]] = r
    records = []
    for parent in val["parents"]:
        stack, base = load_stack(parent)
        res = window_residuals(stack, base)
        for child, sides in by.items():
            if sides[64]["parent_tile_id"] != parent:
                continue
            iy = int(re.search(r"_y(\d+)", child).group(1))
            ix = int(re.search(r"_x(\d+)", child).group(1))
            rr = res[(iy, ix)]
            records.append({
                "child": child, "parent": parent, "project": sides[64]["project_folder"], "iy": iy, "ix": ix,
                "position": position(iy, ix),
                "penalty_512": sides[512]["lpips"] - sides[64]["lpips"],
                "penalty_256": sides[256]["lpips"] - sides[64]["lpips"],
                "psnr_penalty_512": sides[64]["psnr"] - sides[512]["psnr"],
                "gain_64": sides[64]["lpips_bilinear"] - sides[64]["lpips"],
                "resid_global": rr["global"], "resid_local": rr["local"], "resid_excess": rr["excess"],
            })
        print(parent, "done", flush=True)

    pen = np.array([r["penalty_512"] for r in records])
    exc = np.array([r["resid_excess"] for r in records])
    rho, p = spearmanr(exc, pen)
    within = []
    for parent in val["parents"]:
        sub = [r for r in records if r["parent"] == parent]
        rp, _ = spearmanr([r["resid_excess"] for r in sub], [r["penalty_512"] for r in sub])
        within.append(float(rp))
    q = np.quantile(exc, [0.25, 0.75])
    by_quart = {
        "low_quartile": float(np.mean(pen[exc <= q[0]])),
        "middle": float(np.mean(pen[(exc > q[0]) & (exc < q[1])])),
        "high_quartile": float(np.mean(pen[exc >= q[1]])),
    }
    by_pos = {k: {"n": int(sum(r["position"] == k for r in records)),
                  "penalty_512": float(np.mean([r["penalty_512"] for r in records if r["position"] == k])),
                  "penalty_256": float(np.mean([r["penalty_256"] for r in records if r["position"] == k]))}
              for k in ("centre", "interior", "edge")}
    rho_psnr, _ = spearmanr(exc, [r["psnr_penalty_512"] for r in records])
    summary = {
        "n_windows": len(records),
        "mean_penalty_512": float(pen.mean()),
        "median_penalty_512": float(np.median(pen)),
        "frac_windows_512_worse_than_bilinear": float(np.mean([r["penalty_512"] > r["gain_64"] for r in records])),
        "spearman_excess_vs_penalty_pooled": {"rho": float(rho), "p": float(p)},
        "spearman_excess_vs_psnr_penalty_pooled": float(rho_psnr),
        "spearman_within_parent": {"median": statistics.median(within), "min": min(within), "max": max(within),
                                   "n_positive": sum(x > 0 for x in within), "n": len(within)},
        "penalty_by_excess_quartile": by_quart,
        "penalty_by_position": by_pos,
    }
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
