#!/usr/bin/env python3
"""Check that a partial-update window predicts the same LR pixels as a full-field forward.

The frozen LR512 configuration is built through ``optimize.get_argparser`` /
``get_dataset`` / ``build_projection_and_decoder`` / ``build_model``. For a non-base
revisit with a non-trivial affine and colour transform, the LR prediction of an
LR128 window (the training crop: HR support = tile * df, no context) is compared with
the same crop of the full-field LR prediction. The comparison is repeated with a blur
halo of ``h`` LR pixels (HR support widened by ``h * df`` on every side that lies inside
the field, then cropped back to the window).

Field states: ``fit`` (a short full-field fit, realistic imagery) and ``stress`` (grid
parameters drawn with large amplitude, strong high-frequency content).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optimize as opt  # noqa: E402
from eval.lr_holdout import holdout_psf_pad_lr  # noqa: E402
from models.lr_tile_sampler import _dataset_train_coords, crop_lr_hr_tensors  # noqa: E402

FROZEN_FLAGS = [
    "--dataset", "asker", "--s2-dir", str(ROOT / "data/s2_revisits/asker_lr512"),
    "--df", "4", "--scale_factor", "4", "--num_samples", "16",
    "--input_projection", "hashgrid_tcnn", "--lr_degradation", "s2_psf_m",
    "--recon_loss", "charbonnier", "--charbonnier_eps", "0.01",
    "--lr_tile", "128", "--lr_tiles_per_step", "4", "--lr_tile_mix", "within",
    "--spatial_holdout", "0.1", "--seed", "6", "--device", "0",
    "--spatial_alignment_path", str(ROOT / "eval/confirmatory_eval_manifest.v2.json"),
    "--lr_stats_pixels", "all",
]
OUT = ROOT / "paper/results/partial_update_verification.json"


def lr_predict(model, coords, target_hw, sid, kwargs):
    target = torch.zeros(1, *target_hw, 3, device=coords.device)
    with torch.no_grad():
        out, _ = model(coords, sid, lr_frames=target, **kwargs)
    return out.float()


def compare(model, coords_full, lr_hw, sid, kwargs, origins, tile, halos, df):
    full = lr_predict(model, coords_full, lr_hw, sid, kwargs)
    H, W = lr_hw
    results = []
    for (r, c) in origins:
        ref = full[:, r:r + tile, c:c + tile]
        entry = {"origin": [r, c], "halo": {}}
        for h in halos:
            r0, c0 = max(0, r - h), max(0, c - h)
            r1, c1 = min(H, r + tile + h), min(W, c + tile + h)
            cc = coords_full[:, r0 * df:r1 * df, c0 * df:c1 * df].contiguous()
            pred = lr_predict(model, cc, (r1 - r0, c1 - c0), sid, kwargs)
            pred = pred[:, r - r0:r - r0 + tile, c - c0:c - c0 + tile]
            d = (pred - ref).abs().amax(dim=-1)[0]
            # distance (LR px) of each window pixel to the nearest window edge that is not a field edge
            yy = torch.arange(tile, device=d.device).view(-1, 1).expand(tile, tile)
            xx = torch.arange(tile, device=d.device).view(1, -1).expand(tile, tile)
            big = torch.full_like(yy, 10 ** 6)
            dist = torch.minimum(
                torch.minimum(torch.where(torch.tensor(r > 0), yy, big), torch.where(torch.tensor(r + tile < H), tile - 1 - yy, big)),
                torch.minimum(torch.where(torch.tensor(c > 0), xx, big), torch.where(torch.tensor(c + tile < W), tile - 1 - xx, big)),
            )
            ring = {}
            for k in range(0, 4):
                sel = dist == k
                if sel.any():
                    ring[str(k)] = float(d[sel].max())
            inner = dist >= 4
            entry["halo"][str(h)] = {
                "max_abs": float(d.max()),
                "rms": float(d.pow(2).mean().sqrt()),
                "max_abs_by_edge_distance_lr_px": ring,
                "max_abs_interior_ge4px": float(d[inner].max()) if inner.any() else None,
                "frac_pixels_gt_1e-4": float((d > 1e-4).float().mean()),
                "frac_pixels_gt_1e-3": float((d > 1e-3).float().mean()),
            }
        results.append(entry)
    return {"full_prediction_std": float(full.std()), "windows": results}


def training_path_check(model, coords, lr_hw, sid, kwargs, origins, tile, args) -> dict:
    """Feed the full-field prediction back as target through the K-window training path.

    With a consistent window forward every residual is zero, so the Charbonnier loss
    equals ``eps`` exactly; the excess over ``eps`` measures the window-edge error.
    """
    from models.lr_tile_sampler import stack_lr_hr_tiles

    target = lr_predict(model, coords, lr_hw, sid, kwargs)
    eps = float(args.charbonnier_eps)
    out = {"charbonnier_eps": eps}
    for label, halo in (("no_halo", 0), ("resolved_halo", opt.resolve_lr_tile_halo(args))):
        c, t, m, s, dx, dy = stack_lr_hr_tiles(
            coords, target, origins, tile, mask=None, sample_id=sid, gt_dx=torch.zeros(1, device=coords.device),
            gt_dy=torch.zeros(1, device=coords.device), halo=halo,
        )
        with torch.no_grad():
            losses = opt._train_forward_losses(
                model, c, t, s, None, dx, dy, lr_hw, args, kwargs, 0.0, 0.0, halo=halo,
            )
        out[label] = {"halo": halo, "recon_loss": float(losses["recon_loss"]),
                      "excess_over_eps": float(losses["recon_loss"]) - eps}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-steps", type=int, default=400)
    ap.add_argument("--frame", type=int, default=5)
    ap.add_argument("--out", type=Path, default=OUT)
    cli = ap.parse_args()

    args = opt.get_argparser().parse_args(FROZEN_FLAGS)
    args.halo_sr, args.halo_lr_px_west, args.halo_lr_px_north = None, 0, 0
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    args.dataset_device = opt.resolve_dataset_device(args, training_device=device)
    data = opt.get_dataset(args=args, name=args.dataset, training_device=device)
    args.lr_height, args.lr_width = int(data.lr_height), int(data.lr_width)
    proj, dec = opt.build_projection_and_decoder(args, device, output_dim=opt._hetero_output_dim(args))
    model = opt.build_model(args, proj, dec, device)
    kwargs = opt._step_schedule_kwargs(args, 10_000)

    coords = _dataset_train_coords(data).to(device)
    H, W = args.lr_height, args.lr_width
    df = int(coords.shape[1]) // H
    tile = int(args.lr_tile)

    # short full-field fit on all frames for realistic imagery
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(cli.fit_steps):
        fid = step % data.num_samples
        target = data.get_lr_sample_hwc(fid).unsqueeze(0).to(device)
        sid = torch.tensor([fid], device=device)
        out, _ = model(coords, sid, lr_frames=target, **kwargs)
        loss = torch.sqrt((out.float() - target) ** 2 + 1e-4).mean()
        optim.zero_grad()
        (loss * 1024).backward()
        optim.step()

    # non-trivial transform on the probed revisit
    with torch.no_grad():
        A = model.get_direct_affine(torch.tensor([cli.frame], device=device))
        # sub-pixel translation (0.37 / 0.61 LR px), 0.2 degree rotation, 0.1% scale
        th = torch.deg2rad(torch.tensor(0.2))
        s = 1.001
        new = torch.tensor([[s * torch.cos(th), -s * torch.sin(th), 0.37 / W],
                            [s * torch.sin(th), s * torch.cos(th), -0.61 / H]], device=device)
        model.affine_params[cli.frame].copy_(new.reshape(model.affine_params[cli.frame].shape))
        for ch, (g, b) in enumerate(((1.07, 0.03), (0.94, -0.02), (1.02, 0.05))):
            model.color_transforms[cli.frame][ch].weight.fill_(g)
            model.color_transforms[cli.frame][ch].bias.fill_(b)
        A_after = model.get_direct_affine(torch.tensor([cli.frame], device=device))

    sid = torch.tensor([cli.frame], device=device)
    corner, edge, interior = (0, 0), (0, 256), (256, 128)
    origins = [corner, edge, interior, (384, 384)]
    pad = holdout_psf_pad_lr(df, sigma_m=float(args.s2_psf_sigma_b04_m), truncate=float(args.s2_psf_truncate))
    halos = sorted({0, 1, pad, pad + 2})
    report = {
        "config": {"lr_side": H, "tile": tile, "df": df, "frame": cli.frame, "fit_steps": cli.fit_steps,
                   "psf_sigma_m": {b: float(getattr(args, f"s2_psf_sigma_{b.lower()}_m")) for b in ("B04", "B03", "B02")},
                   "psf_truncate": float(args.s2_psf_truncate), "holdout_psf_pad_lr": pad, "halos_tested": halos,
                   "affine_before": A.tolist(), "affine_after": A_after.tolist()},
        "fit": compare(model, coords, (H, W), sid, kwargs, origins, tile, halos, df),
    }
    report["training_path"] = training_path_check(model, coords, (H, W), sid, kwargs, origins, tile, args)
    with torch.no_grad():
        for p in model.input_projection.parameters():
            p.uniform_(-1.0, 1.0)
    report["stress"] = compare(model, coords, (H, W), sid, kwargs, origins, tile, halos, df)

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2))
    print("training path:", json.dumps(report["training_path"]))
    for state in ("fit", "stress"):
        print(f"== {state} (full-field LR std {report[state]['full_prediction_std']:.4f}, standardized units)")
        for w in report[state]["windows"]:
            for h, m in w["halo"].items():
                print(f"  origin {w['origin']} halo {h}: max {m['max_abs']:.2e} rms {m['rms']:.2e} "
                      f"by-edge-dist {({k: f'{v:.1e}' for k, v in m['max_abs_by_edge_distance_lr_px'].items()})} "
                      f"interior {m['max_abs_interior_ge4px']}")
    print(cli.out)


if __name__ == "__main__":
    main()
