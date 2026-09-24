#!/usr/bin/env python3
"""Render the SR imagery behind the 8/12/16-frame ladder.

The ladder experiment reported only scalars. This reconstructs the actual
pictures for the comparisons that mattered: bilinear against the model at 8 and
16 frames, and the model at its peak high-frequency correlation against the same
model left to train past it. All panels share one display stretch so differences
are real rather than artifacts of per-image normalization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataset
from eval.visualize import _draw_spot_rect, _shared_display_stretch, _upscale_spot
from models.lr_alignment import align_prediction_hwc_to_target, default_lr_align_args
from models.s2_psf_forward import (
    DEFAULT_S2_PSF_SIGMA_M_BY_BAND,
    S2_RGB_BAND_ORDER,
)
from losses import laplace_nll_loss, resolve_recon_criterion
from optimize import build_model, build_projection_and_decoder, get_lpips_model
from scripts.diagnose_divergence import (
    HETERO_LOSSES,
    _args,
    _corr,
    _perceptual,
    _psnr,
    _split_bands,
)


def _pick_spot(gt_high: torch.Tensor, size: int = 96) -> tuple[int, int, int, int]:
    """Crop the most textured region: where above-cutoff detail actually exists."""
    energy = gt_high.abs().mean(dim=2, keepdim=True).permute(2, 0, 1).unsqueeze(0)
    h, w = energy.shape[-2:]
    size = int(min(size, h, w))
    pooled = F.avg_pool2d(energy, kernel_size=size, stride=max(1, size // 4))
    idx = int(torch.argmax(pooled.flatten()))
    stride = max(1, size // 4)
    row, col = divmod(idx, pooled.shape[-1])
    y0 = min(row * stride, h - size)
    x0 = min(col * stride, w - size)
    return y0, y0 + size, x0, x0 + size


def _train(city: str, frames: int, iters: int, dev: int, s2_dir: str, loss: str = "mse",
           mult: float = 2.0):
    """Train one configuration; return the prediction at best LPIPS and at the final iteration."""
    device = torch.device(f"cuda:{dev}")
    torch.cuda.set_device(device)
    args = _args(city, "hashgrid_tcnn", "mlp_tcnn", 64, "s2_psf_m", mult, dev,
                 s2_dir=s2_dir, frames=frames, loss=loss)
    ds = get_dataset(args)
    args.num_samples = ds.num_samples
    args.lr_height, args.lr_width = ds.lr_height, ds.lr_width

    proj, dec = build_projection_and_decoder(args, device)
    model = build_model(args, proj, dec, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    align_args = default_lr_align_args(lr_degradation="s2_psf_m")

    coords = ds.get_hr_coordinates().unsqueeze(0).to(device)
    gt = ds.get_original_hr().to(device)
    if gt.shape[0] in (1, 3) and gt.shape[0] != gt.shape[-1]:
        gt = gt.permute(1, 2, 0)
    gt = gt.float()

    sigmas_px = [DEFAULT_S2_PSF_SIGMA_M_BY_BAND[b] / (10.0 / 4.0) for b in S2_RGB_BAND_ORDER]
    _, gt_high = _split_bands(gt, sigmas_px)
    lr_mean, lr_std = ds.get_lr_mean(0).to(device), ds.get_lr_std(0).to(device)

    hetero = loss in HETERO_LOSSES
    if hetero:
        criterion = laplace_nll_loss if loss == "laplace" else torch.nn.GaussianNLLLoss()
    else:
        criterion = resolve_recon_criterion(loss)

    lpips_fn = get_lpips_model(device)
    best = {"lpips": 1e9}
    marks = {int(round(iters * f)) for f in (0.03, 0.06, 0.12, 0.25, 0.5, 0.75, 1.0)}
    marks |= {250, 500, 750, 1000, 1500, 2000, 3000, 4500}
    marks = {m for m in marks if 0 < m <= iters}

    for it in range(1, iters + 1):
        sid = torch.tensor([it % ds.num_samples], device=device)
        target = ds.get_lr_sample_hwc(int(sid.item())).unsqueeze(0).to(device)
        if hetero:
            lr_pred, _, variance = model(coords, sid, lr_frames=target, lr_align_args=align_args)
            step_loss = criterion(lr_pred, target, variance)
        else:
            out, _ = model(coords, sid, scale_factor=1, training=True)
            lr_pred = align_prediction_hwc_to_target(out[:, :, :, :3], target,
                                                     args=align_args, device=device)
            step_loss = criterion(lr_pred, target)
        opt.zero_grad(set_to_none=True)
        step_loss.backward()
        opt.step()
        if hasattr(model, "clamp_reference_frame"):
            model.clamp_reference_frame()

        if it in marks:
            model.eval()
            with torch.no_grad():
                pred, _ = model(coords, torch.tensor([0], device=device),
                                scale_factor=1, training=False)
                pred = pred[0, :, :, :3] * lr_std + lr_mean
                lp, _ = _perceptual(pred, gt, lpips_fn)
                if lp < best["lpips"]:
                    best = {"lpips": lp, "iter": it, "img": pred.clone()}
            model.train()

    model.eval()
    with torch.no_grad():
        final, _ = model(coords, torch.tensor([0], device=device), scale_factor=1, training=False)
        final = final[0, :, :, :3] * lr_std + lr_mean

    return ds, gt, gt_high, sigmas_px, best, final, lpips_fn


def _score(img, gt, gt_high, sigmas_px, lpips_fn) -> str:
    lp, ss = _perceptual(img, gt, lpips_fn)
    return f"LPIPS {lp:.4f} | SSIM {ss:.3f} | PSNR {_psnr(img, gt):.2f} dB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="sandvika")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("figures/sr"))
    ap.add_argument("--mode", default="frames", choices=("frames", "losses", "capacity"),
                    help="Compare frame counts, training criteria, or hashgrid resolution.")
    a = ap.parse_args()

    CAPACITY = [1.0, 2.0, 4.0]
    s2_dir = str(ROOT / "data" / "s2_revisits_n16" / a.city)
    if a.mode == "capacity":
        runs = {}
        for m in CAPACITY:
            ds, gt, gt_high, sig, best, final, lpips_fn = _train(
                a.city, 16, a.iters, a.device, s2_dir, loss="laplace", mult=m)
            runs[m] = best
    elif a.mode == "losses":
        runs = {}
        for L in ("mse", "mae", "gnll", "laplace"):
            ds, gt, gt_high, sig, best, final, lpips_fn = _train(
                a.city, 16, a.iters, a.device, s2_dir, loss=L)
            runs[L] = best
    else:
        ds, gt, gt_high, sig, best16, final16, lpips_fn = _train(
            a.city, 16, a.iters, a.device, s2_dir)
        _, _, _, _, best8, _, _ = _train(a.city, 8, a.iters, a.device, s2_dir)

    device = gt.device
    lr0 = ds.get_lr_sample_hwc(0).to(device) * ds.get_lr_std(0).to(device) + ds.get_lr_mean(0).to(device)
    lr_chw = lr0.permute(2, 0, 1).unsqueeze(0)
    hw = tuple(gt.shape[:2])
    near = F.interpolate(lr_chw, size=hw, mode="nearest")[0].permute(1, 2, 0)
    bil = F.interpolate(lr_chw, size=hw, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

    if a.mode == "capacity":
        panels = [("bilinear upsample", bil, _score(bil, gt, gt_high, sig, lpips_fn))]
        panels += [
            (f"finest level = {m:g}x LR\nbest LPIPS @ iter {runs[m]['iter']}",
             runs[m]["img"], _score(runs[m]["img"], gt, gt_high, sig, lpips_fn))
            for m in CAPACITY
        ]
        panels.append(("HR ground truth", gt, None))
    elif a.mode == "losses":
        panels = [("bilinear upsample", bil, _score(bil, gt, gt_high, sig, lpips_fn))]
        panels += [
            (f"{L.upper()}\nbest LPIPS @ iter {runs[L]['iter']}", runs[L]["img"],
             _score(runs[L]["img"], gt, gt_high, sig, lpips_fn))
            for L in ("mse", "mae", "gnll", "laplace")
        ]
        panels.append(("HR ground truth", gt, None))
    else:
        panels = [
            ("LR input (nearest)", near, None),
            ("bilinear upsample", bil, _score(bil, gt, gt_high, sig, lpips_fn)),
            (f"model, 8 frames\nbest LPIPS @ iter {best8['iter']}", best8["img"],
             _score(best8["img"], gt, gt_high, sig, lpips_fn)),
            (f"model, 16 frames\nbest LPIPS @ iter {best16['iter']}", best16["img"],
             _score(best16["img"], gt, gt_high, sig, lpips_fn)),
            (f"model, 16 frames\n@ iter {a.iters}", final16,
             _score(final16, gt, gt_high, sig, lpips_fn)),
            ("HR ground truth", gt, None),
        ]

    arrays = [p[1].detach().cpu().numpy() for p in panels]
    stretched = _shared_display_stretch(*arrays)
    spot = _pick_spot(gt_high)

    n = len(panels)
    h, w = hw
    # One context view of the whole AOI, then the crops side by side for comparison.
    ctx_h = min(4.5, max(1.6, 13.0 * h / w))
    fig = plt.figure(figsize=(13.0, ctx_h + 4.2))
    gs = fig.add_gridspec(2, n, height_ratios=[ctx_h, 3.4])

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(stretched[-1])
    _draw_spot_rect(ax, spot)
    ax.set_title(f"HR ground truth, full AOI — yellow box marks the crop compared below",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    for i, ((title, _, score), img) in enumerate(zip(panels, stretched)):
        axz = fig.add_subplot(gs[1, i])
        axz.imshow(_upscale_spot(img[spot[0]:spot[1], spot[2]:spot[3]], 320))
        axz.set_title(title + ("\n" + score.replace(" | ", "\n") if score else ""),
                      fontsize=8)
        axz.set_xticks([]); axz.set_yticks([])

    fig.suptitle(
        f"{a.city}: 4x super-resolution, {ds.lr_height}x{ds.lr_width} LR -> {h}x{w} HR "
        f"(all panels share one display stretch)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    a.outdir.mkdir(parents=True, exist_ok=True)
    suffix = {"losses": "loss_comparison", "capacity": "capacity_comparison"}.get(
        a.mode, "sr_comparison")
    out = a.outdir / f"{a.city}_{suffix}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
