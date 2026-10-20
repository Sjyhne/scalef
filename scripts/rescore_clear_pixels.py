#!/usr/bin/env python3
"""Rescore the named-site fits with the base frame's clear pixels only, without refitting.

Two parts of the evaluation protocol use the base frame including whatever cloud and shadow the
loss mask missed: the exported field is destandardized with the frame's mean and standard
deviation over all pixels (``--lr_stats_pixels all``), and the NIB reference is histogram-matched
to the whole frame. With the full-window OmniCloudMask (``<stem>_lr512_ocm.tif``) both can be
redone over clear pixels: the field's gauge is one exact affine per band, and the NIB match is a
scoring step. Bilinear is left as the frame itself.

For each seed-6 confirmatory v5 run this scores ScaleF and bilinear against
  current  NIB matched to the whole base frame, field as exported (reported protocol);
  clear    NIB matched to the base frame's clear pixels, field re-expressed with clear-pixel stats;
and records the base frame's flagged fraction and the ratio of its clear-pixel to all-pixel
standard deviation, which is the contrast inflation the current protocol carries.
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
from s2_dataset import L2A_REFLECTANCE_CAP, _apply_hr_spatial_shift  # noqa: E402
from site_retarget_nib import MANIFEST, field_path, read, to_bchw  # noqa: E402

MASK_SUFFIX = "_lr512_ocm"


def match_to_pixels(hr: np.ndarray, ref_pixels: np.ndarray) -> np.ndarray:
    """Per-band histogram match of HR to a 1-D reference sample (skimage flattens the reference)."""
    from skimage.exposure import match_histograms
    out = np.zeros_like(hr)
    for c in range(3):
        src = hr[..., c]
        valid = src > 0
        matched = match_histograms(src, ref_pixels[:, c].reshape(1, -1), channel_axis=None)
        out[..., c] = np.where(valid, np.clip(matched, 0.0, L2A_REFLECTANCE_CAP), 0.0)
    return out


def run_city(job: dict, args, lpips_fn, device) -> dict:
    city = job["city"]
    run_args = get_argparser().parse_args(job["command"][2:])
    run_args.device = args.device
    ds = get_dataset(args=run_args, name=run_args.dataset, training_device=torch.device("cpu"))
    raw_args = copy.deepcopy(run_args)
    raw_args.no_hr_harmonize = True
    raw_args.no_hr_spatial_align = True  # match before the shift, as the loader does
    ds_raw = get_dataset(args=raw_args, name=raw_args.dataset, training_device=torch.device("cpu"))
    assert ds.lr_stats_pixels == "all"

    off = float(ds.eval_reflectance_offset)
    mask = ds.get_hr_eval_mask().cpu().numpy().astype(bool)
    gt_cur = np.clip(ds.get_original_hr().cpu().numpy().astype(np.float32), 0, L2A_REFLECTANCE_CAP) - off
    nib = ds_raw.get_original_hr().cpu().numpy().astype(np.float32)
    mask_raw = ds_raw.get_hr_eval_mask().cpu().numpy()
    base = ds.lr_rgb[0].cpu().numpy().astype(np.float32)  # H,W,3 incl. offset, as the loader saw it

    stem = Path(ds.frames[0]["path"]).stem
    with rasterio.open(ds.s2_dir / f"{stem}{MASK_SUFFIX}.tif") as s:
        classes = s.read(1)
    assert classes.shape == base.shape[:2], (classes.shape, base.shape)
    valid = np.all(base > 0, axis=-1)
    clear = (classes == 0) & valid
    old_clear = ds.lr_clear[0].cpu().numpy().astype(bool)

    mean_all, std_all = ds.lr_mean[0].cpu().numpy().reshape(3), ds.lr_std[0].cpu().numpy().reshape(3)
    px = base[clear]
    mean_clear, std_clear = px.mean(axis=0), np.maximum(px.std(axis=0, ddof=1), 1e-8)

    fpath, source = field_path(city, job["run_name"])
    F = read(fpath)  # offset already removed at export
    F_clear = ((F + off) - mean_all) / std_all * std_clear + mean_clear - off
    gt_clear = match_to_pixels(nib, px)
    if ds.hr_spatial_shift:
        gt_clear, mask_shift = _apply_hr_spatial_shift(gt_clear, mask_raw, ds.hr_spatial_shift["dy"],
                                                       ds.hr_spatial_shift["dx"])
        mask_disagree = float((mask_shift.astype(bool) != mask).mean())
    else:
        mask_disagree = 0.0
    gt_clear = np.clip(gt_clear, 0, L2A_REFLECTANCE_CAP).astype(np.float32) - off

    hr_h, hr_w = mask.shape
    B = np.clip(
        cv2.resize(np.clip(base, 0, L2A_REFLECTANCE_CAP), (hr_w, hr_h), interpolation=cv2.INTER_LINEAR),
        0,
        L2A_REFLECTANCE_CAP,
    ) - off

    mask_t = torch.from_numpy(mask)
    scores = {}
    for name, pred, gt in (("current", F, gt_cur), ("clear", F_clear, gt_clear),
                           ("clear_ref_only", F, gt_clear), ("clear_field_only", F_clear, gt_cur)):
        m = compute_masked_image_metrics(to_bchw(pred, device), to_bchw(gt, device), to_bchw(B, device),
                                         mask_t, device=device, lpips_fn=lpips_fn)
        scores[name] = {k: float(m[k]) for k in ("model_psnr", "bilinear_psnr", "model_ssim", "bilinear_ssim",
                                                  "model_lpips", "bilinear_lpips")}
    logged = json.loads(Path(job["expected_metrics_path"]).read_text())
    return {
        "city": city, "field_source": source, "base_frame_date": ds.base_frame_date,
        "base_flagged_old": float(1 - old_clear[valid].mean()), "base_flagged_new": float(1 - clear[valid].mean()),
        "std_clear_over_all": (std_clear / std_all).tolist(), "mean_shift": (mean_clear - mean_all).tolist(),
        "gt_std_current": float(gt_cur[mask].std()), "gt_std_clear": float(gt_clear[mask].std()),
        "eval_mask_disagree_frac": mask_disagree,
        "scores": scores, "logged_psnr": logged["psnr"]["model"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "results" / "clear_pixel_rescore_v1")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    jobs = [j for j in json.loads(MANIFEST.read_text())["jobs"] if int(j["seed"]) == 6]
    if args.cities:
        jobs = [j for j in jobs if j["city"] in args.cities]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    lpips_fn = get_lpips_model(device)
    args.out.mkdir(parents=True, exist_ok=True)
    for job in sorted(jobs, key=lambda j: j["city"]):
        r = run_city(job, args, lpips_fn, device)
        (args.out / f"{job['city']}.json").write_text(json.dumps(r, indent=1))
        c, k = r["scores"]["current"], r["scores"]["clear"]
        print(f"{r['city']:12s} flagged {r['base_flagged_old']*100:4.1f}%->{r['base_flagged_new']*100:4.1f}%  "
              f"std ratio {np.mean(r['std_clear_over_all']):.2f}  PSNR {c['model_psnr']:.2f}->{k['model_psnr']:.2f} "
              f"(bil {c['bilinear_psnr']:.2f}->{k['bilinear_psnr']:.2f})  SSIM {c['model_ssim']:.3f}->{k['model_ssim']:.3f} "
              f"(bil {c['bilinear_ssim']:.3f}->{k['bilinear_ssim']:.3f})  LPIPS {c['model_lpips']:.3f}->{k['model_lpips']:.3f}", flush=True)


if __name__ == "__main__":
    main()
