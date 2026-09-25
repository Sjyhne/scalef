#!/usr/bin/env python3
"""Diagnose where and why ScaleF loses SSIM to bilinear at a named site.

For one seed-6 named-site run with float exports, the reference, evaluation mask, and LR stack
are rebuilt on CPU through the production loader. SSIM is decomposed into luminance, contrast,
and structure terms (torchmetrics' Gaussian window, sigma 1.5, 11 px; C1, C2 for data range 1)
and aggregated over the same valid pixels as ``masked_ssim``. Over 128x128 HR blocks (320 m)
that lie fully inside the mask we record SSIM/LPIPS differences, sub-pixel displacement
(phase correlation against the reference), gradient-energy and contrast ratios, mean bias, and
the LR stack's date-to-date change relative to the base frame. The block at the site's median
SSIM difference is exported as a representative crop.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import spearmanr
from skimage.registration import phase_cross_correlation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("tinycudann", types.ModuleType("tinycudann"))
import lpips  # noqa: E402

from data import get_dataset  # noqa: E402
from eval.masked_metrics import masked_ssim  # noqa: E402
from optimize import get_argparser  # noqa: E402

MANIFEST = ROOT / "paper" / "results" / "run_manifests" / "confirmatory_v5_boa__run_manifest.json"
OUT_DIR = ROOT / "paper" / "results" / "ssim_failure"
BLOCK = 128
C1, C2 = 0.01 ** 2, 0.03 ** 2


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.transpose(src.read((1, 2, 3)).astype(np.float32), (1, 2, 0))


def ssim_terms(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel luminance, contrast, structure, and SSIM maps averaged over RGB."""
    g = lambda a: gaussian_filter(a, 1.5, truncate=5 / 1.5, mode="reflect")  # noqa: E731
    out = {k: np.zeros(x.shape[:2], np.float64) for k in ("l", "c", "s", "ssim")}
    for b in range(3):
        xb, yb = x[..., b].astype(np.float64), y[..., b].astype(np.float64)
        mx, my = g(xb), g(yb)
        vx = np.maximum(g(xb * xb) - mx * mx, 0)
        vy = np.maximum(g(yb * yb) - my * my, 0)
        cxy = g(xb * yb) - mx * my
        sx, sy = np.sqrt(vx), np.sqrt(vy)
        lum = (2 * mx * my + C1) / (mx * mx + my * my + C1)
        con = (2 * sx * sy + C2) / (vx + vy + C2)
        struct = (cxy + C2 / 2) / (sx * sy + C2 / 2)
        out["l"] += lum / 3
        out["c"] += con / 3
        out["s"] += struct / 3
        out["ssim"] += lum * (2 * cxy + C2) / (vx + vy + C2) / 3
    return out


def grad_energy(a: np.ndarray) -> float:
    gy, gx = np.gradient(a.mean(axis=-1))
    return float(np.mean(gx * gx + gy * gy))


def to_bchw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="Run directory with qgis/sr_pred.tif")
    ap.add_argument("--seed", type=int, default=6)
    args_cli = ap.parse_args()

    job = next(j for j in json.loads(MANIFEST.read_text())["jobs"]
               if j["city"] == args_cli.city and int(j["seed"]) == args_cli.seed)
    args = get_argparser().parse_args(job["command"][2:])
    ds = get_dataset(args=args, name=args.dataset, training_device=torch.device("cpu"))
    off = float(ds.eval_reflectance_offset)
    mask = ds.get_hr_eval_mask().cpu().numpy().astype(bool)
    gt = np.clip(ds.get_original_hr().cpu().numpy().astype(np.float32), 0, 1)
    gt = np.where(gt.max(axis=-1, keepdims=True) > 0, gt - off, gt)
    q = args_cli.run_dir / "qgis"
    sr, bil = read(q / "sr_pred.tif"), read(q / "s2_bilinear.tif")

    fill = lambda p: np.where(mask[..., None], p, gt)  # noqa: E731
    touched = maximum_filter((~mask).astype(np.uint8), size=11, mode="constant", cval=0) > 0
    valid = ~touched
    terms = {"model": ssim_terms(fill(sr), gt), "bilinear": ssim_terms(fill(bil), gt)}
    agg = {who: {k: float(v[valid].mean()) for k, v in t.items()} for who, t in terms.items()}
    rows, cols = np.where(mask)
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    check = {who: masked_ssim(to_bchw(p[box]), to_bchw(gt[box]), torch.from_numpy(mask[box]))
             for who, p in (("model", sr), ("bilinear", bil))}
    check["model_clipped_at_0"] = masked_ssim(to_bchw(np.maximum(sr, 0)[box]), to_bchw(gt[box]),
                                              torch.from_numpy(mask[box]))

    local_ref = gaussian_filter(gt.mean(axis=-1), 1.5, truncate=5 / 1.5, mode="reflect")
    edges = [-np.inf, 0.02, 0.04, 0.08, np.inf]
    deficit = (terms["model"]["ssim"] - terms["bilinear"]["ssim"])[valid].sum()
    strata = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = valid & (local_ref >= lo) & (local_ref < hi)
        if not sel.any():
            continue
        row = {"ref_local_mean": [float(lo), float(hi)], "pixel_frac": float(sel.sum() / valid.sum()),
               "model_neg_frac": float((sr[sel] < 0).any(axis=-1).mean()),
               "share_of_ssim_difference": float((terms["model"]["ssim"] - terms["bilinear"]["ssim"])[sel].sum() / deficit)}
        for who, p in (("model", sr), ("bilinear", bil)):
            local_p = gaussian_filter(p.mean(axis=-1), 1.5, truncate=5 / 1.5, mode="reflect")
            err = (local_p - local_ref)[sel]
            row[who] = {k: float(terms[who][k][sel].mean()) for k in ("l", "c", "s", "ssim")}
            row[who].update(local_mean_bias=float(err.mean()), local_mean_abs_err=float(np.abs(err).mean()))
        strata.append(row)

    lr = ds.lr_rgb.cpu().numpy().astype(np.float32)
    clear = ds.lr_clear.cpu().numpy().astype(bool)
    if clear.ndim == 4:
        clear = clear[..., 0]
    df = gt.shape[0] // lr.shape[1]

    # LR-domain check of where ScaleF's low-frequency radiometry comes from: each date is mapped into the
    # base frame's gauge with its own per-band mean/std (as the fit's standardization does), and the
    # per-pixel median over clear dates is compared with ScaleF pooled onto the LR grid.
    lr_mean = ds.lr_mean.cpu().numpy().reshape(lr.shape[0], 1, 1, -1)
    lr_std = ds.lr_std.cpu().numpy().reshape(lr.shape[0], 1, 1, -1)
    in_base = (lr - lr_mean) / lr_std * lr_std[0] + lr_mean[0] - off
    consensus = np.nanmedian(np.where(clear[1:, ..., None], in_base[1:], np.nan), axis=0)
    base_lr = lr[0] - off
    pool = lambda a: a[: lr.shape[1] * df, : lr.shape[2] * df].reshape(lr.shape[1], df, lr.shape[2], df, -1).mean(axis=(1, 3))  # noqa: E731
    sr_lr, ref_lr = pool(sr), pool(gt)
    mask_lr = mask[: lr.shape[1] * df, : lr.shape[2] * df].reshape(lr.shape[1], df, lr.shape[2], df).all(axis=(1, 3))
    mask_lr &= clear[0] & np.isfinite(consensus).all(axis=-1)
    lr_domain = []
    ref_l = ref_lr.mean(-1)
    dark = mask_lr & (ref_l < 0.04)
    stamps = [str(f["stac_id"]).split("_")[2][:8] for f in ds.frames]
    dates = [f"{s[:4]}-{s[4:6]}-{s[6:]}" for s in stamps]
    per_date_dark = [{"date": dates[t], "clear_frac": float(clear[t][dark].mean()),
                      "minus_base": float((in_base[t] - base_lr).mean(-1)[dark & clear[t]].mean())}
                     for t in range(1, lr.shape[0]) if (dark & clear[t]).sum() >= 10]
    for lo, hi in ((-np.inf, 0.04), (0.04, 0.08), (0.08, np.inf)):
        sel = mask_lr & (ref_l >= lo) & (ref_l < hi)
        if sel.sum() < 10:
            continue
        d_sr = (sr_lr - base_lr).mean(-1)[sel]
        d_cons = (consensus - base_lr).mean(-1)[sel]
        lr_domain.append({
            "ref_mean": [float(lo), float(hi)], "n_lr_pixels": int(sel.sum()),
            "base_minus_ref": float((base_lr - ref_lr).mean(-1)[sel].mean()),
            "scalef_minus_base": float(d_sr.mean()), "consensus_minus_base": float(d_cons.mean()),
            "spearman_scalef_vs_consensus_departure": float(spearmanr(d_sr, d_cons).statistic),
        })
    lpips_fn = lpips.LPIPS(net="vgg", verbose=False).eval()

    blocks = []
    n = gt.shape[0] // BLOCK
    for by in range(n):
        for bx in range(n):
            ys, xs = slice(by * BLOCK, (by + 1) * BLOCK), slice(bx * BLOCK, (bx + 1) * BLOCK)
            if not mask[ys, xs].all():
                continue
            g_, s_, b_ = gt[ys, xs], sr[ys, xs], bil[ys, xs]
            v = valid[ys, xs]
            ly, lx = slice(ys.start // df, ys.stop // df), slice(xs.start // df, xs.stop // df)
            base = lr[0, ly, lx]
            dates = [lr[t, ly, lx] for t in range(1, lr.shape[0]) if clear[t, ly, lx].mean() > 0.95]
            change = float(np.median([np.abs(d - base).mean() for d in dates]) / max(base.mean(), 1e-6)) \
                if dates else float("nan")
            with torch.no_grad():
                lp = {who: float(lpips_fn(to_bchw(p) * 2 - 1, to_bchw(g_) * 2 - 1))
                      for who, p in (("model", s_), ("bilinear", b_))}
            shift = {who: phase_cross_correlation(g_.mean(-1), p.mean(-1), upsample_factor=10)[0].tolist()
                     for who, p in (("model", s_), ("bilinear", b_))}
            blocks.append({
                "by": by, "bx": bx,
                "d_ssim": float(terms["model"]["ssim"][ys, xs][v].mean() - terms["bilinear"]["ssim"][ys, xs][v].mean()),
                "d_l": float(terms["model"]["l"][ys, xs][v].mean() - terms["bilinear"]["l"][ys, xs][v].mean()),
                "d_c": float(terms["model"]["c"][ys, xs][v].mean() - terms["bilinear"]["c"][ys, xs][v].mean()),
                "d_s": float(terms["model"]["s"][ys, xs][v].mean() - terms["bilinear"]["s"][ys, xs][v].mean()),
                "d_lpips": lp["bilinear"] - lp["model"],
                "grad_ratio_model": grad_energy(s_) / max(grad_energy(g_), 1e-12),
                "grad_ratio_bilinear": grad_energy(b_) / max(grad_energy(g_), 1e-12),
                "std_ratio_model": float(s_.std() / max(g_.std(), 1e-9)),
                "std_ratio_bilinear": float(b_.std() / max(g_.std(), 1e-9)),
                "bias_model": float((s_ - g_).mean()), "bias_bilinear": float((b_ - g_).mean()),
                "neg_frac_model": float((s_ < 0).any(axis=-1).mean()),
                "ref_mean": float(g_.mean()),
                "shift_model_px": shift["model"], "shift_bilinear_px": shift["bilinear"],
                "lr_date_change": change,
            })
    d = np.array([b["d_ssim"] for b in blocks])
    order = np.argsort(d)
    median_block = blocks[int(order[len(order) // 2])]
    corr = {}
    for key in ("lr_date_change", "grad_ratio_model", "std_ratio_model", "neg_frac_model", "ref_mean", "d_lpips"):
        x = np.array([b[key] for b in blocks], dtype=float)
        ok = np.isfinite(x)
        corr[key] = float(spearmanr(x[ok], d[ok]).statistic)
    summary = {
        "city": args_cli.city, "run_dir": str(args_cli.run_dir.resolve().relative_to(ROOT)), "boa_offset": off,
        "base_date": ds.base_frame_date, "n_frames": int(lr.shape[0]),
        "ssim_check_masked_ssim": check, "ssim_terms": agg, "ssim_by_reference_brightness": strata,
        "lr_domain_departure_from_base": lr_domain, "dark_area_departure_per_date": per_date_dark,
        "neg_prediction_frac_in_mask": float((sr[mask] < 0).any(axis=-1).mean()),
        "n_blocks": len(blocks), "frac_blocks_ssim_worse": float((d < 0).mean()),
        "block_d_ssim_quantiles": dict(zip(("p10", "p50", "p90"), map(float, np.percentile(d, [10, 50, 90])))),
        "median_abs_shift_px": {who: float(np.median([np.hypot(*b[f"shift_{who}_px"]) for b in blocks]))
                                for who in ("model", "bilinear")},
        "median_grad_ratio": {who: float(np.median([b[f"grad_ratio_{who}"] for b in blocks])) for who in ("model", "bilinear")},
        "median_std_ratio": {who: float(np.median([b[f"std_ratio_{who}"] for b in blocks])) for who in ("model", "bilinear")},
        "spearman_vs_block_d_ssim": corr,
        "median_block": median_block,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args_cli.city}.json").write_text(json.dumps({**summary, "blocks": blocks}, indent=1) + "\n")
    ys = slice(median_block["by"] * BLOCK, (median_block["by"] + 1) * BLOCK)
    xs = slice(median_block["bx"] * BLOCK, (median_block["bx"] + 1) * BLOCK)
    ly, lx = slice(ys.start // df, ys.stop // df), slice(xs.start // df, xs.stop // df)
    np.savez_compressed(OUT_DIR / f"{args_cli.city}_median_block.npz", lr=lr[0, ly, lx], bilinear=bil[ys, xs],
                        model=sr[ys, xs], reference=gt[ys, xs],
                        ssim_model=terms["model"]["ssim"][ys, xs], ssim_bilinear=terms["bilinear"]["ssim"][ys, xs])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
