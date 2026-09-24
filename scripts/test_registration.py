#!/usr/bin/env python3
"""Is per-frame registration what caps high-frequency recovery?

Frame count and hashgrid capacity have both been ruled out: above-cutoff energy
sits near 47% of the truth no matter how many revisits the model gets or how
fine the encoder is allowed to be. The remaining candidate is the alignment.
Multi-frame super-resolution only recovers detail if the sub-pixel phase of each
revisit is known, and an error of a fraction of an LR pixel smears exactly the
frequencies the extra frames were supposed to supply.

The synthetic mode builds a problem whose registration is known exactly: the HR
ground truth is resampled at prescribed sub-pixel offsets and pushed through the
same PSF the model inverts, so the arms differ only in what the model is told
about the shifts.

  oracle    translations frozen at the true shifts
  init      translations started at the true shifts, then left trainable
  learned   translations learned from identity (the production setting)
  none      translations frozen at identity, though the frames really are shifted

``oracle`` bounds what perfect registration buys. ``learned`` is what we get.
``init`` separates "cannot find the shifts" from "finds them and then drifts",
and ``none`` calibrates the scale by showing what ignoring the shifts costs.

The real mode runs the production arms on the actual revisits and compares the
translations the model settles on against phase correlation between the LR
frames, which is an independent estimate of the same quantity.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataset
from models.lr_alignment import align_prediction_hwc_to_target, default_lr_align_args
from models.s2_psf_forward import DEFAULT_S2_PSF_SIGMA_M_BY_BAND, S2_RGB_BAND_ORDER
from losses import resolve_recon_criterion
from optimize import build_model, build_projection_and_decoder, get_lpips_model
from scripts.diagnose_divergence import _args, _corr, _perceptual, _psnr, _split_bands

ARMS = ("oracle", "init", "learned", "none")


def _fourier_shift(img_hwc: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Resample so the result at (x, y) holds the input at (x + dx, y + dy).

    A phase ramp rather than an interpolation kernel: bicubic or bilinear
    resampling would blur the very high frequencies this experiment is about,
    biasing the synthetic frames toward the answer we are testing for.
    """
    x = img_hwc.permute(2, 0, 1).to(torch.float64)
    _, h, w = x.shape
    fx = torch.fft.fftfreq(w, device=x.device, dtype=torch.float64).view(1, 1, w)
    fy = torch.fft.fftfreq(h, device=x.device, dtype=torch.float64).view(1, h, 1)
    ramp = torch.exp(2j * math.pi * (fx * float(dx) + fy * float(dy)))
    out = torch.fft.ifft2(torch.fft.fft2(x) * ramp).real
    return out.permute(1, 2, 0).to(img_hwc.dtype)


def _check_fourier_shift(img_hwc: torch.Tensor) -> None:
    """An integer shift must reproduce a roll, or the sign convention is wrong."""
    got = _fourier_shift(img_hwc, 1.0, 0.0)
    want = torch.roll(img_hwc, shifts=-1, dims=1)
    err = float((got[:, 2:-2] - want[:, 2:-2]).abs().max())
    if err > 1e-3:
        raise RuntimeError(f"Fourier shift disagrees with roll by {err:.2e}")


def _degrade(hr_hwc: torch.Tensor, lr_h: int, lr_w: int, align_args, device) -> torch.Tensor:
    """HR [H,W,C] -> LR [1,lr_h,lr_w,C] through the operator the model inverts."""
    dummy = torch.zeros(1, lr_h, lr_w, hr_hwc.shape[-1], device=hr_hwc.device)
    return align_prediction_hwc_to_target(
        hr_hwc.unsqueeze(0), dummy, args=align_args, device=device
    )


def _make_shifts(n: int, df: int, pattern: str, seed: int) -> list[tuple[float, float]]:
    """Sub-pixel offsets in HR pixels, frame 0 pinned to zero as the anchor."""
    if pattern == "lattice":
        # Ideal MFSR sampling: one revisit per sub-pixel position of the LR grid.
        grid = [(x - (df - 1) / 2.0, y - (df - 1) / 2.0)
                for y in range(df) for x in range(df)]
        shifts = [grid[i % len(grid)] for i in range(n)]
    else:
        g = torch.Generator().manual_seed(seed)
        r = (torch.rand((n, 2), generator=g) - 0.5) * float(df)
        shifts = [(float(a), float(b)) for a, b in r]
    shifts[0] = (0.0, 0.0)
    return shifts


def _set_affines(model, shifts_norm, trainable: bool) -> None:
    with torch.no_grad():
        for i, p in enumerate(model.affine_params):
            tx, ty = shifts_norm[i]
            p.copy_(torch.tensor([[1.0, 0.0, tx, 0.0, 1.0, ty]], device=p.device))
            p.requires_grad_(bool(trainable) and i != 0)


def _center_displacement(model, w_hr: int, h_hr: int) -> list[tuple[float, float]]:
    """Each frame's effective translation at the image centre, in HR pixels.

    Reading tx/ty alone would miss a shift expressed through the linear part of
    the affine, which is free to pick up scale or rotation during training.
    """
    out = []
    with torch.no_grad():
        for p in model.affine_params:
            a = p.detach().reshape(2, 3)
            c = torch.tensor([0.5, 0.5, 1.0], device=a.device, dtype=a.dtype)
            d = a @ c - c[:2]
            out.append((float(d[0]) * w_hr, float(d[1]) * h_hr))
    return out


def _registration_error(model, shifts, w_hr: int, h_hr: int, df: int) -> dict[str, float]:
    est = _center_displacement(model, w_hr, h_hr)
    err = [math.hypot(e[0] - t[0], e[1] - t[1]) for e, t in zip(est, shifts)]
    body = err[1:] or err
    return {
        "reg_err_hr_px": sum(body) / len(body),
        "reg_err_lr_px": sum(body) / len(body) / df,
        "reg_err_max_lr_px": max(body) / df,
    }


def _run_arm(arm, args, ds, targets, shifts, gt, gt_low, gt_high, sigmas_px,
             lpips_fn, align_args, device, iters, margin, lr_mask, lr=1e-3):
    """Train one registration regime and return its best-LPIPS state."""
    torch.manual_seed(0)
    proj, dec = build_projection_and_decoder(args, device)
    model = build_model(args, proj, dec, device)

    h_hr, w_hr = gt.shape[:2]
    true_norm = [(dx / w_hr, dy / h_hr) for dx, dy in shifts]
    zero_norm = [(0.0, 0.0)] * len(shifts)
    if arm == "oracle":
        _set_affines(model, true_norm, trainable=False)
    elif arm == "init":
        _set_affines(model, true_norm, trainable=True)
    elif arm == "none":
        _set_affines(model, zero_norm, trainable=False)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = resolve_recon_criterion(args.recon_loss)
    coords = ds.get_hr_coordinates().unsqueeze(0).to(device)
    n = len(targets)

    marks = sorted({int(round(iters * f)) for f in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)})
    best = None
    traj = []
    for it in range(1, iters + 1):
        frame = it % n
        sid = torch.tensor([frame], device=device)
        target = targets[frame]
        out, _ = model(coords, sid, scale_factor=1, training=True)
        pred = align_prediction_hwc_to_target(out[:, :, :, :3], target,
                                              args=align_args, device=device)
        loss = criterion(pred * lr_mask, target * lr_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if hasattr(model, "clamp_reference_frame"):
            model.clamp_reference_frame()

        if it in marks:
            model.eval()
            with torch.no_grad():
                p, _ = model(coords, torch.tensor([0], device=device),
                             scale_factor=1, training=False)
                p = p[0, :, :, :3].float()
                m = slice(margin, -margin) if margin else slice(None)
                pc, gc = p[m, m], gt[m, m]
                p_low, p_high = _split_bands(pc, sigmas_px)
                lp, ss = _perceptual(pc, gc, lpips_fn)
                rec = {
                    "iter": it,
                    "lr_loss": float(loss),
                    "lpips": lp,
                    "ssim": ss,
                    "full": _psnr(pc.clamp(0, 1), gc.clamp(0, 1)),
                    "above": _psnr(p_high, gt_high[m, m]),
                    "hi_ratio": float(p_high.std() / gt_high[m, m].std()),
                    "hi_corr": _corr(p_high, gt_high[m, m]),
                }
                rec.update(_registration_error(model, shifts, w_hr, h_hr, args.df))
                traj.append(rec)
                if best is None or rec["lpips"] < best["lpips"]:
                    best = dict(rec)
            model.train()
    best["trajectory"] = traj
    return best


def _phase_shifts(targets, df: int) -> list[tuple[float, float]]:
    """Independent sub-pixel offsets from phase correlation against frame 0."""
    from skimage.registration import phase_cross_correlation

    ref = targets[0][0].mean(dim=-1).detach().cpu().numpy()
    out = []
    for t in targets:
        mov = t[0].mean(dim=-1).detach().cpu().numpy()
        sh, _, _ = phase_cross_correlation(ref, mov, upsample_factor=100)
        # phase_cross_correlation returns (row, col) to apply to `mov`; the model's
        # convention is the coordinate offset at which the scene is sampled.
        out.append((-float(sh[1]) * df, -float(sh[0]) * df))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default="bergen")
    ap.add_argument("--mode", default="synthetic", choices=("synthetic", "real"))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--mult", type=float, default=2.0,
                    help="Finest hashgrid level as a multiple of the LR grid.")
    ap.add_argument("--hidden", type=int, default=64, help="Decoder width.")
    ap.add_argument("--mlp-dtype", dest="mlp_dtype", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--degradation", default="s2_psf_m")
    ap.add_argument("--shifts", default="random", choices=("random", "lattice"),
                    help="Sub-pixel offsets of the synthetic revisits.")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Gaussian noise std added to synthetic LR frames, in reflectance.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=int, default=16,
                    help="HR-pixel border excluded from metrics (the phase ramp wraps around).")
    ap.add_argument("--s2-dir", dest="s2_dir", default=None)
    ap.add_argument("--json-out", dest="json_out", default=None)
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)
    s2_dir = a.s2_dir or str(ROOT / "data" / "s2_revisits_n16" / a.city)
    args = _args(a.city, "hashgrid_tcnn", "mlp_tcnn", a.hidden, a.degradation, a.mult, a.device,
                 s2_dir=s2_dir, frames=a.frames, loss=a.loss)
    args.tcnn_mlp_dtype = a.mlp_dtype
    ds = get_dataset(args)
    args.lr_height, args.lr_width = ds.lr_height, ds.lr_width
    align_args = default_lr_align_args(lr_degradation=a.degradation)
    df = int(args.df)

    gt = ds.get_original_hr().to(device)
    if gt.shape[0] in (1, 3) and gt.shape[0] != gt.shape[-1]:
        gt = gt.permute(1, 2, 0)
    gt = gt.float()
    gt = gt[: ds.lr_height * df, : ds.lr_width * df]

    gsd_hr = 10.0 / df
    sigmas_px = [DEFAULT_S2_PSF_SIGMA_M_BY_BAND[b] / gsd_hr for b in S2_RGB_BAND_ORDER]
    m = slice(a.margin, -a.margin) if a.margin else slice(None)
    gt_low, gt_high = _split_bands(gt, sigmas_px)
    lpips_fn = get_lpips_model(device)

    if a.mode == "synthetic":
        _check_fourier_shift(gt)
        shifts = _make_shifts(a.frames, df, a.shifts, a.seed)
        g = torch.Generator(device="cpu").manual_seed(a.seed + 1)
        targets = []
        for dx, dy in shifts:
            src = gt if (dx == 0.0 and dy == 0.0) else _fourier_shift(gt, dx, dy)
            lr = _degrade(src, ds.lr_height, ds.lr_width, align_args, device)
            if a.noise > 0:
                lr = lr + torch.randn(lr.shape, generator=g).to(device) * a.noise
            targets.append(lr)
        args.num_samples = len(targets)
    else:
        targets = [ds.get_lr_sample_hwc(i).unsqueeze(0).to(device)
                   * ds.get_lr_std(i).to(device) + ds.get_lr_mean(i).to(device)
                   for i in range(ds.num_samples)]
        args.num_samples = len(targets)
        shifts = _phase_shifts(targets, df)

    # The phase ramp wraps, so the outermost LR ring sees content from the far edge.
    ring = max(1, a.margin // df // 2)
    lr_mask = torch.ones(1, ds.lr_height, ds.lr_width, 1, device=device)
    if a.mode == "synthetic" and ring:
        lr_mask[:, :ring], lr_mask[:, -ring:] = 0, 0
        lr_mask[:, :, :ring], lr_mask[:, :, -ring:] = 0, 0

    bil = F.interpolate(targets[0].permute(0, 3, 1, 2), size=gt.shape[:2],
                        mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    _, bil_high = _split_bands(bil[m, m], sigmas_px)
    bil_lpips, bil_ssim = _perceptual(bil[m, m], gt[m, m], lpips_fn)

    span = max(math.hypot(dx, dy) for dx, dy in shifts)
    print(f"{a.city} [{a.mode}]: LR {ds.lr_height}x{ds.lr_width} -> HR {tuple(gt.shape[:2])}, "
          f"{len(targets)} frames, df={df}, largest offset {span:.2f} HR px "
          f"({span / df:.2f} LR px)")
    print(f"bilinear: LPIPS {bil_lpips:.4f} | SSIM {bil_ssim:.4f} | "
          f"HF energy {float(bil_high.std() / gt_high[m, m].std()):.2f} of truth")
    print()
    print(f"{'arm':>9}{'LPIPS':>9}{'SSIM':>8}{'PSNR':>8}{'above':>8}"
          f"{'hi ratio':>10}{'hi corr':>9}{'reg err (LR px)':>17}{'iter':>7}")
    print("-" * 84)

    results = {}
    for arm in a.arms:
        r = _run_arm(arm, args, ds, targets, shifts, gt, gt_low, gt_high, sigmas_px,
                     lpips_fn, align_args, device, a.iters, a.margin, lr_mask, lr=a.lr)
        results[arm] = r
        print(f"{arm:>9}{r['lpips']:>9.4f}{r['ssim']:>8.4f}{r['full']:>8.2f}"
              f"{r['above']:>8.2f}{r['hi_ratio']:>10.2f}{r['hi_corr']:>9.3f}"
              f"{r['reg_err_lr_px']:>17.3f}{r['iter']:>7}")

    if "oracle" in results and "learned" in results:
        d = results["learned"]["lpips"] - results["oracle"]["lpips"]
        hf = results["oracle"]["hi_ratio"] - results["learned"]["hi_ratio"]
        print()
        print(f"knowing the true shifts is worth {d:+.4f} LPIPS and "
              f"{hf:+.2f} of the ground-truth HF energy")

    if a.json_out:
        summary = {
            "city": a.city,
            "mode": a.mode,
            "frames": len(targets),
            "shifts": a.shifts,
            "noise": a.noise,
            "mult": a.mult,
            "hidden": a.hidden,
            "mlp_dtype": a.mlp_dtype,
            "adam_lr": a.lr,
            "loss": a.loss,
            "iters": a.iters,
            "true_shifts": shifts,
            "bilinear_lpips": bil_lpips,
            "bilinear_ssim": bil_ssim,
            "arms": results,
        }
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json_out, "a") as fh:
            fh.write(json.dumps(summary) + "\n")
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
