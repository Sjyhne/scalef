#!/usr/bin/env python3
"""Track where the HR error lives as self-supervised training proceeds.

Full-frame metrics show quality collapsing while the LR reconstruction loss
keeps improving. That is only diagnostic if we know *which* spatial frequencies
go wrong: content below the LR Nyquist is constrained by the data, content above
it is not. This splits the HR error at the degradation cutoff and reports both
bands against ground truth, on raw tensors rather than display PNGs (which are
stretched and resampled, and so cannot be compared across runs).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataset
from models.lr_alignment import align_prediction_hwc_to_target, default_lr_align_args
from models.s2_psf_forward import (
    DEFAULT_S2_PSF_SIGMA_M_BY_BAND,
    S2_RGB_BAND_ORDER,
    gaussian_blur_per_band,
)
from losses import laplace_nll_loss, resolve_recon_criterion
from optimize import build_model, build_projection_and_decoder, get_lpips_model
from torchmetrics.functional.image import structural_similarity_index_measure as _ssim

HETERO_LOSSES = {"gnll", "laplace"}
LOSS_CHOICES = ("mse", "mae", "charbonnier", "huber", "gnll", "laplace")


def _args(city: str, projection: str, model: str, hidden: int, degradation: str, mult: float,
          dev: int = 0, s2_dir: str | None = None, frames: int = 0,
          loss: str = "mse") -> Namespace:
    return Namespace(
        dataset=city,
        s2_dir=s2_dir or str(ROOT / "data" / "s2_revisits" / city),
        df=4,
        scale_factor=4,
        hr_gsd_m=0.0,
        s2_native_gsd_m=10.0,
        num_samples=frames,
        dataset_device=f"cuda:{dev}",
        device=str(dev),
        no_hr_harmonize=False,
        input_projection=projection,
        projection_dim=256,
        fourier_scale=10.0,
        model=model,
        network_depth=4,
        network_hidden_dim=hidden,
        hash_n_levels=16,
        hash_n_features_per_level=2,
        hash_log2_hashmap_size=21,
        hash_interpolation="linear",
        hash_max_resolution=0,
        hash_max_resolution_h=0,
        hash_max_resolution_w=0,
        hash_max_resolution_mult=mult,
        hash_base_resolution=0,
        hash_tcnn_output_dtype="fp32",
        lr_degradation=degradation,
        tcnn_mlp_dtype="fp16",
        recon_loss=loss if loss not in HETERO_LOSSES else "mse",
        charbonnier_eps=1e-3,
        huber_delta=0.05,
        use_gnll=loss == "gnll",
        use_laplace_nll=loss == "laplace",
        hetero_scale="pixel",
        hetero_region_size=4,
        lr_height=0,
        lr_width=0,
        lr_size=0,
    )


def _date_span(ds) -> int | None:
    """Days between the earliest and latest acquisition actually used."""
    days = [f["datetime"][:10] for f in getattr(ds, "frames", []) if f.get("datetime")]
    if not days:
        return None
    d = [_dt.date.fromisoformat(x) for x in days]
    return (max(d) - min(d)).days


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(10.0 * torch.log10(1.0 / F.mse_loss(a, b)))


def _holdout_mask(h: int, w: int, block: int, frac: float, device, seed: int = 0) -> torch.Tensor:
    """Boolean [1,H,W,1] mask, True where a pixel is used for training.

    Pixels are held out in blocks rather than singly, because the hashgrid's finest
    level is comparable to the LR pitch and isolated pixels would be interpolated
    from their trained neighbours. Each frame gets its own mask via ``seed``: the
    scene stays constrained everywhere by the other revisits, so the model is asked
    to predict an unseen *observation* rather than to invent an unsupervised region.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    bh, bw = (h + block - 1) // block, (w + block - 1) // block
    keep_block = torch.rand((bh, bw), generator=g) >= float(frac)
    mask = keep_block.repeat_interleave(block, 0).repeat_interleave(block, 1)[:h, :w]
    return mask.to(device).view(1, h, w, 1)


def _masked(elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of an elementwise loss over the selected pixels only."""
    m = mask.to(elem.dtype)
    return (elem * m).sum() / m.sum().clamp(min=1.0) / elem.shape[-1]


def _elementwise(name: str):
    """Per-pixel loss, so masking divides by the held-in count rather than the total."""
    if name == "mse":
        return lambda p, t: (p - t) ** 2
    if name == "mae":
        return lambda p, t: (p - t).abs()
    if name == "charbonnier":
        return lambda p, t: torch.sqrt((p - t) ** 2 + 1e-3 ** 2)
    if name == "huber":
        def _h(p, t, d=0.05):
            a = (p - t).abs()
            q = a.clamp(max=d)
            return 0.5 * q * q + d * (a - q)
        return _h
    raise ValueError(f"no elementwise form for {name!r}")


def _gauge_drift(model) -> dict[str, float]:
    """Mean geometric/radiometric offset of the trainable frames.

    Frame 0 pins the canonical frame only while it is in the training loop. Without
    it the whole set of per-frame transforms can slide together at no cost to the
    loss, so this tracks the collective drift that such a gauge freedom permits.
    """
    tx, ty, scale, bias = [], [], [], []
    for i, p in enumerate(model.affine_params):
        if i == 0:
            continue
        v = p.detach().reshape(-1)
        tx.append(float(v[2]))
        ty.append(float(v[5]))
    for i, chans in enumerate(model.color_transforms):
        if i == 0:
            continue
        for ch in chans:
            scale.append(float(ch.weight.detach().reshape(-1)[0]))
            bias.append(float(ch.bias.detach().reshape(-1)[0]))
    n = max(1, len(tx))
    return {
        "drift_tx": sum(tx) / n,
        "drift_ty": sum(ty) / n,
        "drift_scale": sum(scale) / max(1, len(scale)),
        "drift_bias": sum(bias) / max(1, len(bias)),
    }


def _bchw(x_hwc: torch.Tensor) -> torch.Tensor:
    return x_hwc.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)


def _perceptual(pred_hwc, gt_hwc, lpips_fn) -> tuple[float, float]:
    """LPIPS and SSIM: metrics that reward visible detail rather than penalising it."""
    p, g = _bchw(pred_hwc), _bchw(gt_hwc)
    with torch.no_grad():
        lp = float(lpips_fn(p * 2 - 1, g * 2 - 1).item())
        ss = float(_ssim(p, g, data_range=1.0).item())
    return lp, ss


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Correlation of two residual bands: how much of the detail is genuine."""
    x, y = a.flatten() - a.mean(), b.flatten() - b.mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def _split_bands(x_hwc: torch.Tensor, sigmas_px) -> tuple[torch.Tensor, torch.Tensor]:
    """Split [H,W,C] into (below-cutoff, above-cutoff) using the sensor PSF."""
    chw = x_hwc.permute(2, 0, 1).unsqueeze(0)
    low = gaussian_blur_per_band(chw, sigmas_px)
    return low[0].permute(1, 2, 0), (chw - low)[0].permute(1, 2, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="bergen")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--projection", default="hashgrid_tcnn")
    ap.add_argument("--model", default="mlp_tcnn")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--degradation", default="s2_psf_m")
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--s2-dir", dest="s2_dir", default=None,
                    help="Revisit directory (default: data/s2_revisits/<city>).")
    ap.add_argument("--frames", type=int, default=0,
                    help="Use only the first N frames (date-proximity order). 0 = all.")
    ap.add_argument("--json-out", dest="json_out", default=None,
                    help="Append a one-line JSON summary of the run to this file.")
    ap.add_argument("--loss", default="mse", choices=LOSS_CHOICES,
                    help="Training reconstruction criterion.")
    ap.add_argument("--holdout", action="store_true",
                    help="Exclude frame 0 from training and use it as a validation revisit. "
                         "Frame 0 is the anchored reference (identity affine, no colour "
                         "transform), so it needs no per-frame parameters and gives an "
                         "early-stopping signal that requires no HR ground truth.")
    ap.add_argument("--spatial-holdout", dest="spatial_holdout", type=float, default=0.0,
                    help="Fraction of LR pixel blocks held out of the training loss and used "
                         "for validation. Keeps every frame (including the anchor) in training.")
    ap.add_argument("--holdout-block", dest="holdout_block", type=int, default=8,
                    help="Side length in LR pixels of each held-out block.")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)
    args = _args(a.city, a.projection, a.model, a.hidden, a.degradation, a.mult, a.device,
                 s2_dir=a.s2_dir, frames=a.frames, loss=a.loss)
    ds = get_dataset(args)
    args.num_samples = ds.num_samples
    args.lr_height, args.lr_width = ds.lr_height, ds.lr_width

    proj, dec = build_projection_and_decoder(args, device)
    model = build_model(args, proj, dec, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    align_args = default_lr_align_args(lr_degradation=a.degradation)

    coords = ds.get_hr_coordinates().unsqueeze(0).to(device)
    gt = ds.get_original_hr().to(device)
    if gt.shape[0] in (1, 3) and gt.shape[0] != gt.shape[-1]:
        gt = gt.permute(1, 2, 0)
    gt = gt.float()

    gsd_hr = 10.0 / 4.0
    sigmas_px = [DEFAULT_S2_PSF_SIGMA_M_BY_BAND[b] / gsd_hr for b in S2_RGB_BAND_ORDER]

    lr_mean = ds.get_lr_mean(0).to(device)
    lr_std = ds.get_lr_std(0).to(device)

    gt_low, gt_high = _split_bands(gt, sigmas_px)
    lr0 = ds.get_lr_sample_hwc(0).to(device) * lr_std + lr_mean
    bil = F.interpolate(
        lr0.permute(2, 0, 1).unsqueeze(0),
        size=gt.shape[:2], mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0)
    bil_low, bil_high = _split_bands(bil, sigmas_px)
    lpips_fn = get_lpips_model(device)
    bil_lpips, bil_ssim = _perceptual(bil, gt, lpips_fn)

    marks = sorted({int(round(a.iters * f)) for f in
                    (0.03, 0.06, 0.12, 0.25, 0.5, 0.75, 1.0)} | {250, 500, 1000, 2000})
    marks = [m for m in marks if 0 < m <= a.iters]

    print(f"{a.city}: LR {ds.lr_height}x{ds.lr_width} -> HR {tuple(gt.shape[:2])}, "
          f"{ds.num_samples} frames, PSF sigma {[round(s,2) for s in sigmas_px]} HR px")
    print(f"bilinear: full {_psnr(bil, gt):.2f} dB | below-cutoff {_psnr(bil_low, gt_low):.2f} "
          f"| above-cutoff {_psnr(bil_high, gt_high):.2f}")
    print()
    print(f"bilinear perceptual: LPIPS {bil_lpips:.4f} | SSIM {bil_ssim:.4f}")
    print()
    print(f"{'iter':>6}{'LR loss':>11}{'HR full':>10}{'below':>10}{'above':>10}"
          f"{'hi ratio':>10}{'LPIPS':>9}{'SSIM':>8}")
    print("-" * 74)

    traj: list[dict] = []
    hetero = a.loss in HETERO_LOSSES
    spatial = a.spatial_holdout > 0.0
    if spatial:
        masks = [_holdout_mask(ds.lr_height, ds.lr_width, a.holdout_block,
                               a.spatial_holdout, device, seed=f)
                 for f in range(ds.num_samples)]
        held = 100.0 * float(torch.stack([(~m).float().mean() for m in masks]).mean())
        print(f"spatial holdout: {held:.1f}% of each frame's LR pixels held out in "
              f"{a.holdout_block}x{a.holdout_block} blocks, independently per frame")
    if hetero:
        criterion = (laplace_nll_loss if a.loss == "laplace"
                     else torch.nn.GaussianNLLLoss(reduction="none" if spatial else "mean"))
    else:
        criterion = _elementwise(a.loss) if spatial else resolve_recon_criterion(a.loss)

    val_target = ds.get_lr_sample_hwc(0).unsqueeze(0).to(device) if a.holdout else None
    val_frames = list(range(0, ds.num_samples, max(1, ds.num_samples // 4)))[:4]

    for it in range(1, a.iters + 1):
        if a.holdout:
            frame = 1 + (it % max(1, ds.num_samples - 1))
        else:
            frame = it % ds.num_samples
        sid = torch.tensor([frame], device=device)
        target = ds.get_lr_sample_hwc(int(sid.item())).unsqueeze(0).to(device)
        train_mask = masks[frame] if spatial else None
        if hetero:
            # The GNLL path degrades HR->LR inside the model so it can align the variance too.
            lr_pred, _, variance = model(coords, sid, lr_frames=target, lr_align_args=align_args)
            if spatial:
                elem = (criterion(lr_pred, target, variance) if a.loss == "gnll"
                        else laplace_nll_loss(lr_pred, target, variance, full=True))
                loss = _masked(elem, train_mask)
            else:
                loss = criterion(lr_pred, target, variance)
        else:
            out, _ = model(coords, sid, scale_factor=1, training=True)
            lr_pred = align_prediction_hwc_to_target(out[:, :, :, :3], target,
                                                     args=align_args, device=device)
            loss = (_masked(criterion(lr_pred, target), train_mask) if spatial
                    else criterion(lr_pred, target))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if hasattr(model, "clamp_reference_frame"):
            model.clamp_reference_frame()

        if it in marks:
            model.eval()
            with torch.no_grad():
                pred, _ = model(coords, torch.tensor([0], device=device), scale_factor=1, training=False)
                pred = pred[0, :, :, :3] * lr_std + lr_mean
                p_low, p_high = _split_bands(pred, sigmas_px)
                rec = {
                    "iter": it,
                    "lr_loss": float(loss),
                    "full": _psnr(pred, gt),
                    "below": _psnr(p_low, gt_low),
                    "above": _psnr(p_high, gt_high),
                    "hi_ratio": float(p_high.std() / gt_high.std()),
                    "hi_corr": _corr(p_high, gt_high),
                }
                rec["lpips"], rec["ssim"] = _perceptual(pred, gt, lpips_fn)
                rec.update(_gauge_drift(model))
                if val_target is not None:
                    out0, _ = model(coords, torch.tensor([0], device=device),
                                    scale_factor=1, training=False)
                    val_pred = align_prediction_hwc_to_target(
                        out0[:, :, :, :3], val_target, args=align_args, device=device)
                    rec["val_loss"] = float(F.mse_loss(val_pred, val_target))
                elif spatial:
                    # Held-out pixels, averaged over a few frames for a stabler signal.
                    vals = []
                    for f in val_frames:
                        vt = ds.get_lr_sample_hwc(f).unsqueeze(0).to(device)
                        vo, _ = model(coords, torch.tensor([f], device=device),
                                      scale_factor=1, training=True)
                        vp = align_prediction_hwc_to_target(
                            vo[:, :, :, :3], vt, args=align_args, device=device)
                        vals.append(_masked((vp - vt) ** 2, ~masks[f]))
                    rec["val_loss"] = float(torch.stack(vals).mean())
                traj.append(rec)
                print(f"{it:>6}{rec['lr_loss']:>11.3e}{rec['full']:>10.2f}"
                      f"{rec['below']:>10.2f}{rec['above']:>10.2f}"
                      f"{rec['hi_ratio']:>10.2f}{rec['lpips']:>9.4f}{rec['ssim']:>8.4f}")
            model.train()

    if a.json_out:
        best = max(traj, key=lambda r: r["above"])
        summary = {
            "city": a.city,
            "loss": a.loss,
            "frames": ds.num_samples,
            "s2_dir": args.s2_dir,
            "date_span_days": _date_span(ds),
            "bilinear_above": _psnr(bil_high, gt_high),
            "bilinear_full": _psnr(bil, gt),
            "bilinear_hi_corr": _corr(bil_high, gt_high),
            "bilinear_lpips": bil_lpips,
            "bilinear_ssim": bil_ssim,
            "best_lpips": min(r["lpips"] for r in traj),
            "best_lpips_iter": min(traj, key=lambda r: r["lpips"])["iter"],
            "hi_ratio_at_best": min(traj, key=lambda r: r["lpips"])["hi_ratio"],
            "hi_corr_at_best": min(traj, key=lambda r: r["lpips"])["hi_corr"],
            "mult": a.mult,
            "best_ssim": max(r["ssim"] for r in traj),
            "final_lpips": traj[-1]["lpips"],
            "holdout": a.holdout,
            "best_hi_corr": max(r["hi_corr"] for r in traj),
            "final_hi_corr": traj[-1]["hi_corr"],
            "final_hi_ratio": traj[-1]["hi_ratio"],
            "best_above": best["above"],
            "best_above_iter": best["iter"],
            "best_full": max(r["full"] for r in traj),
            "final_above": traj[-1]["above"],
            "trajectory": traj,
        }
        summary["above_vs_bilinear"] = summary["best_above"] - summary["bilinear_above"]
        summary["spatial_holdout"] = a.spatial_holdout
        if a.holdout or spatial:
            # Would stopping at the validation minimum have found the perceptual optimum?
            val_pick = min(traj, key=lambda r: r["val_loss"])
            summary["val_stop_iter"] = val_pick["iter"]
            summary["val_stop_lpips"] = val_pick["lpips"]
            summary["val_stop_regret"] = val_pick["lpips"] - summary["best_lpips"]
        with open(a.json_out, "a") as fh:
            fh.write(json.dumps(summary) + "\n")
        print(f"\nbest above-cutoff {summary['best_above']:.2f} dB @ iter {best['iter']} "
              f"| bilinear {summary['bilinear_above']:.2f} dB "
              f"| delta {summary['above_vs_bilinear']:+.2f} dB")


if __name__ == "__main__":
    main()
