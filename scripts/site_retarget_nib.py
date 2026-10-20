#!/usr/bin/env python3
"""Named-site NIB check of affine output retargeting (see ``regional_retarget.py``).

For each seed-6 confirmatory v5 run, the site's own LR stack (the frames and clear masks the fit
saw, BOA offset removed) gives a regional reference C and confidence q on 80 m blocks, built as in
``regional_retarget.py``. The exported field F and the bilinear baseline B are each mapped per band
by a q-weighted least-squares affine at 80 m, fitted over the whole field without NIB:
  to_C     block means of F (or B) onto C;
  to_base  block means onto the base frame's clear block means (corrects only the contrast shrink
           relative to the frame the reference is histogram-matched to).
Scores use the paper's masked metrics against three references:
  current  NIB histogram-matched to the base frame (reported protocol);
  C_aff    raw NIB mapped once per band by a q-weighted 80 m affine onto C;
  C_qm     raw NIB mapped once per band by a monotone quantile map fitted on 80 m block means.
Retargeting never sees NIB; NIB is matched to C before and independently of any prediction.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules.setdefault("tinycudann", types.ModuleType("tinycudann"))
from data import get_dataset  # noqa: E402
from eval.masked_metrics import compute_masked_image_metrics  # noqa: E402
from optimize import get_argparser, get_lpips_model  # noqa: E402
from regional_retarget import affine_fit, robust_affine  # noqa: E402

MANIFEST = ROOT / "paper" / "results" / "run_manifests" / "confirmatory_v5_boa__run_manifest.json"
FLOAT_EXPORTS = ("figure_float_v5b", "figure_float_v5")
HR_PER_LR = 4


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.transpose(src.read((1, 2, 3)).astype(np.float32), (1, 2, 0))


def field_path(city: str, run_name: str) -> tuple[Path, str]:
    for tag in FLOAT_EXPORTS:
        p = ROOT / "single_samples" / city / tag / f"{tag}__{city}__seed6" / "qgis" / "sr_pred.tif"
        if p.is_file():
            return p, tag
    return ROOT / "single_samples" / city / "confirmatory_v5_boa" / run_name / "qgis" / "sr_pred.tif", "v5_export"


def block_means(x: np.ndarray, valid: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """x: H,W,3; valid: H,W. Returns (3,nb,nb) means over valid pixels and valid fraction."""
    h, w = valid.shape
    nby, nbx = h // block, w // block
    x, valid = x[:nby * block, :nbx * block], valid[:nby * block, :nbx * block]
    c = valid.reshape(nby, block, nbx, block).sum(axis=(1, 3)).astype(np.float64)
    s = (np.nan_to_num(x) * valid[..., None]).reshape(nby, block, nbx, block, 3).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.moveaxis(s / c[..., None], -1, 0)
    return m, c / float(block * block)


def site_reference(lr: np.ndarray, clear: np.ndarray, args) -> dict:
    means, fracs = [], []
    for t in range(lr.shape[0]):
        m, f = block_means(lr[t], clear[t], args.block)
        m[:, f < args.min_clear] = np.nan
        means.append(m)
        fracs.append(f)
    means = np.stack(means)
    med0 = np.nanmedian(means, axis=0)
    norm = np.full_like(means, np.nan)
    calib = []
    for t in range(means.shape[0]):
        row = []
        for b in range(3):
            y, x = means[t, b].ravel(), med0[b].ravel()
            if np.isfinite(y).sum() >= args.min_blocks:
                a, c, kept = robust_affine(y, x)
                norm[t, b] = (means[t, b] - c) / a
                row.append([a, c, kept])
            else:
                row.append([np.nan, np.nan, 0.0])
        calib.append(row)
    C = np.nanmedian(norm, axis=0)
    n_obs = np.isfinite(norm[:, 0]).sum(axis=0)
    spread = np.nanmedian(np.abs(norm - C[None]), axis=0).mean(axis=0) * 1.4826
    q = np.where(n_obs >= args.min_obs,
                 np.minimum(1.0, n_obs / 6.0) / (1.0 + (spread / args.spread_scale) ** 2), 0.0)
    return {"C": C, "q": np.nan_to_num(q), "n_obs": n_obs, "base_blocks": means[0], "calib": calib}


def fit_to(F_blocks: np.ndarray, target: np.ndarray, w: np.ndarray) -> list[tuple[float, float]]:
    return [affine_fit(F_blocks[b], target[b], w, False) for b in range(3)]


def apply_gh(x: np.ndarray, gh: list[tuple[float, float]]) -> np.ndarray:
    g = np.array([p[0] for p in gh], np.float32)
    h = np.array([p[1] for p in gh], np.float32)
    return x * g + h


def quantile_map(src_blocks: np.ndarray, dst_blocks: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Monotone map from src block-mean quantiles to dst quantiles, linearly extended at the ends."""
    qs = np.linspace(0.01, 0.99, 99)
    s, d = np.quantile(src_blocks, qs), np.quantile(dst_blocks, qs)
    s = np.maximum.accumulate(s + np.arange(s.size) * 1e-9)
    lo = (d[5] - d[0]) / max(s[5] - s[0], 1e-6)
    hi = (d[-1] - d[-6]) / max(s[-1] - s[-6], 1e-6)
    y = np.interp(x, s, d)
    y = np.where(x < s[0], d[0] + lo * (x - s[0]), y)
    return np.where(x > s[-1], d[-1] + hi * (x - s[-1]), y).astype(np.float32)


def hp_std(x: np.ndarray, mask: np.ndarray) -> float:
    lp = cv2.blur(x, (8 * HR_PER_LR, 8 * HR_PER_LR))
    return float((x - lp)[mask].std())


def to_bchw(a: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.transpose(a, (2, 0, 1))))[None].float().to(device)


def run_city(job: dict, args, lpips_fn, device) -> dict:
    city = job["city"]
    run_args = get_argparser().parse_args(job["command"][2:])
    run_args.device = args.device
    ds = get_dataset(args=run_args, name=run_args.dataset, training_device=torch.device("cpu"))
    raw_args = copy.deepcopy(run_args)
    raw_args.no_hr_harmonize = True
    ds_raw = get_dataset(args=raw_args, name=raw_args.dataset, training_device=torch.device("cpu"))

    from s2_dataset import L2A_REFLECTANCE_CAP

    off = float(ds.eval_reflectance_offset)
    mask = ds.get_hr_eval_mask().cpu().numpy().astype(bool)
    mask_disagree = float((mask != ds_raw.get_hr_eval_mask().cpu().numpy().astype(bool)).mean())
    gt_cur = np.clip(ds.get_original_hr().cpu().numpy().astype(np.float32), 0, L2A_REFLECTANCE_CAP) - off
    nib = ds_raw.get_original_hr().cpu().numpy().astype(np.float32)

    lr = ds.lr_rgb.cpu().numpy().astype(np.float32)
    if lr.shape[1] == 3 and lr.shape[-1] != 3:
        lr = np.transpose(lr, (0, 2, 3, 1))
    clear = ds.lr_clear.cpu().numpy().astype(bool) & np.all(lr > 0, axis=-1)
    lr = lr - off
    ref = site_reference(lr, clear, args)
    C, q = ref["C"], ref["q"]

    fpath, source = field_path(city, job["run_name"])
    F = read(fpath)
    hr_h, hr_w = mask.shape
    assert F.shape[:2] == (hr_h, hr_w), (F.shape, mask.shape)
    B = np.clip(
        cv2.resize(np.clip(lr[0] + off, 0, L2A_REFLECTANCE_CAP), (hr_w, hr_h), interpolation=cv2.INTER_LINEAR),
        0,
        L2A_REFLECTANCE_CAP,
    ) - off

    bhr = args.block * HR_PER_LR
    ones = np.ones((hr_h, hr_w), bool)
    F_b, _ = block_means(F, ones, bhr)
    B_b, _ = block_means(B, ones, bhr)
    base_b = ref["base_blocks"]
    w_base = np.isfinite(base_b[0]).astype(np.float64)
    fits = {
        "F_to_C": fit_to(F_b, C, q), "B_to_C": fit_to(B_b, C, q),
        "F_to_base": fit_to(F_b, base_b, w_base), "B_to_base": fit_to(B_b, base_b, w_base),
    }

    nib_b, nib_frac = block_means(nib, mask & np.all(nib > 0, axis=-1), bhr)
    w_nib = np.where(nib_frac >= 0.99, q, 0.0)
    nib_gh = fit_to(nib_b, C, w_nib)
    gt_caff = apply_gh(nib, nib_gh)
    sel = (w_nib > 0) & np.all(np.isfinite(C), axis=0)
    gt_cqm = np.stack([quantile_map(nib_b[b][sel], C[b][sel], nib[..., b]) for b in range(3)], axis=-1)

    preds = {
        "F_raw": F, "F_to_C": apply_gh(F, fits["F_to_C"]), "F_to_base": apply_gh(F, fits["F_to_base"]),
        "B_raw": B, "B_to_C": apply_gh(B, fits["B_to_C"]), "B_to_base": apply_gh(B, fits["B_to_base"]),
    }
    refs = {"current": gt_cur, "C_aff": gt_caff, "C_qm": gt_cqm}
    pairs = {"current": [("raw", "F_raw", "B_raw"), ("to_base", "F_to_base", "B_to_base"), ("to_C", "F_to_C", "B_to_C")],
             "C_aff": [("raw", "F_raw", "B_raw"), ("to_C", "F_to_C", "B_to_C")],
             "C_qm": [("raw", "F_raw", "B_raw"), ("to_C", "F_to_C", "B_to_C")]}
    mask_t = torch.from_numpy(mask)
    scores: dict = {}
    for rname, gt in refs.items():
        gt_t = to_bchw(gt, device)
        for arm, fk, bk in pairs[rname]:
            m = compute_masked_image_metrics(to_bchw(preds[fk], device), gt_t, to_bchw(preds[bk], device),
                                             mask_t, device=device, lpips_fn=lpips_fn)
            scores[f"{rname}|{arm}"] = {k: float(m[k]) for k in (
                "model_psnr", "bilinear_psnr", "model_ssim", "bilinear_ssim", "model_lpips", "bilinear_lpips")}
    diag = {k: {"hp_std_ratio_vs_current": hp_std(v, mask) / hp_std(gt_cur, mask),
                "hp_std_ratio_vs_C_qm": hp_std(v, mask) / hp_std(gt_cqm, mask)} for k, v in preds.items()}

    metrics_path = Path(job["expected_metrics_path"])
    logged = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    return {
        "city": city, "field_source": source, "field_path": str(fpath.relative_to(ROOT)),
        "field_zero_frac": float((F <= 0).any(axis=-1)[mask].mean()),
        "raw_nib_mask_disagree_frac": mask_disagree, "offset": off, "base_frame_date": ds.base_frame_date, "n_frames": int(lr.shape[0]),
        "q_median": float(np.median(q)), "n_obs_median": float(np.median(ref["n_obs"])),
        "nib_to_C_affine": nib_gh, "fits": fits, "scores": scores, "highpass": diag,
        "logged": {k: logged.get(k, {}).get("model") for k in ("psnr", "ssim", "lpips")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "results" / "site_retarget_v1")
    ap.add_argument("--device", default="0")
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--min_clear", type=float, default=0.75)
    ap.add_argument("--min_obs", type=int, default=3)
    ap.add_argument("--min_blocks", type=int, default=200)
    ap.add_argument("--spread_scale", type=float, default=0.01)
    args = ap.parse_args()

    jobs = [j for j in json.loads(MANIFEST.read_text())["jobs"] if int(j["seed"]) == 6]
    if args.cities:
        jobs = [j for j in jobs if j["city"] in args.cities]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    lpips_fn = get_lpips_model(device)
    args.out.mkdir(parents=True, exist_ok=True)
    for job in sorted(jobs, key=lambda j: j["city"]):
        res = run_city(job, args, lpips_fn, device)
        (args.out / f"{job['city']}.json").write_text(json.dumps(res, indent=1))
        s = res["scores"]
        print(f"{job['city']:12s} {res['field_source']:16s} "
              + "  ".join(f"{k}: dPSNR {v['model_psnr'] - v['bilinear_psnr']:+.2f} "
                          f"dLPIPS {v['bilinear_lpips'] - v['model_lpips']:+.3f}" for k, v in s.items()),
              flush=True)


if __name__ == "__main__":
    main()
