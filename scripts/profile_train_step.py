#!/usr/bin/env python3
"""Break down where time goes in a single training iteration.

Times the hashgrid encode, MLP decode, HR→LR degradation, backward, and the
optimizer step separately, so it is clear what dominates the ~35 ms/iter.
"""
from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataset
from models.lr_alignment import align_prediction_hwc_to_target, default_lr_align_args
from optimize import build_model, build_projection_and_decoder


def _args(
    city: str,
    mult: float,
    log2: int,
    degradation: str,
    projection: str = "hashgrid",
    model: str = "mlp",
    hidden: int = 256,
) -> Namespace:
    return Namespace(
        dataset=city,
        s2_dir=str(ROOT / "data" / "s2_revisits" / city),
        df=4,
        scale_factor=4,
        hr_gsd_m=0.0,
        s2_native_gsd_m=10.0,
        num_samples=0,
        dataset_device="cuda:0",
        device="0",
        no_hr_harmonize=False,
        input_projection=projection,
        hash_tcnn_output_dtype="fp32",
        projection_dim=256,
        fourier_scale=10.0,
        model=model,
        network_depth=4,
        network_hidden_dim=hidden,
        hash_n_levels=16,
        hash_n_features_per_level=2,
        hash_log2_hashmap_size=log2,
        hash_interpolation="linear",
        hash_max_resolution=0,
        hash_max_resolution_h=0,
        hash_max_resolution_w=0,
        hash_max_resolution_mult=mult,
        hash_base_resolution=0,
        lr_degradation=degradation,
        tcnn_mlp_dtype="fp16",
        use_gnll=False,
        use_laplace_nll=False,
        hetero_scale="pixel",
        hetero_region_size=4,
        lr_height=0,
        lr_width=0,
        lr_size=0,
    )


class _Timer:
    def __init__(self):
        self.acc: dict[str, float] = {}

    def time(self, name, fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        self.acc[name] = self.acc.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", default="bergen")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--log2", type=int, default=21)
    ap.add_argument("--degradation", default="s2_psf")
    ap.add_argument("--tf32", action="store_true", help="Enable TF32 matmuls (H100)")
    ap.add_argument("--projection", default="hashgrid", help="hashgrid or hashgrid_tcnn")
    ap.add_argument("--model", default="mlp", help="mlp or mlp_tcnn")
    ap.add_argument("--hidden", type=int, default=256)
    a = ap.parse_args()

    if a.tf32:
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0")
    args = _args(a.city, a.mult, a.log2, a.degradation, a.projection, a.model, a.hidden)
    ds = get_dataset(args)
    args.num_samples = ds.num_samples
    args.lr_height, args.lr_width = ds.lr_height, ds.lr_width

    proj, dec = build_projection_and_decoder(args, device)
    model = build_model(args, proj, dec, device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    align_args = default_lr_align_args(lr_degradation=a.degradation)

    coords = ds.get_hr_coordinates().unsqueeze(0).to(device)
    n_hash = sum(p.numel() for p in model.input_projection.parameters())
    n_mlp = sum(p.numel() for p in model.decoder.parameters())
    hr_px = coords.shape[1] * coords.shape[2]
    print(
        f"{a.city}: LR {ds.lr_height}x{ds.lr_width}, HR {coords.shape[1]}x{coords.shape[2]} "
        f"({hr_px:,} queries/iter), mult={a.mult} log2={a.log2} deg={a.degradation}"
    )
    print(f"  hashgrid params {n_hash:,} ({n_hash*4/2**20:.0f} MiB)   MLP params {n_mlp:,}")
    print(
        f"  projection {a.projection}  finest level "
        f"{int(proj.resolutions_h[-1])}x{int(proj.resolutions_w[-1])}"
    )

    t = _Timer()
    total_ms = 0.0
    for it in range(a.iters + a.warmup):
        measuring = it >= a.warmup
        if measuring:
            torch.cuda.synchronize()
            t_start = time.perf_counter()
        sid = torch.tensor([it % ds.num_samples], device=device)
        target = ds.get_lr_sample_hwc(int(sid.item())).unsqueeze(0).to(device)

        timer = t if measuring else _Timer()

        A = timer.time("1_affine", lambda: model.get_direct_affine(sid))
        warped = timer.time("1_affine", lambda: model.apply_affine(coords, A))
        feats = timer.time("2_hashgrid_fwd", lambda: model.input_projection(warped))
        B, H, W, _ = warped.shape
        out = timer.time("3_mlp_fwd", lambda: model.decoder(feats.reshape(B * H * W, -1)).reshape(B, H, W, -1)[:, :, :, :3])
        out = timer.time("4_color", lambda: model.apply_color_transform(out, sid))
        lr_pred = timer.time(
            "5_degrade", lambda: align_prediction_hwc_to_target(out, target, args=align_args, device=device)
        )
        loss = timer.time("6_loss", lambda: torch.nn.functional.mse_loss(lr_pred, target))
        timer.time("7_backward", lambda: loss.backward())
        timer.time("8_optim_step", lambda: opt.step())
        timer.time("9_zero_grad", lambda: opt.zero_grad(set_to_none=False))

        if measuring:
            torch.cuda.synchronize()
            total_ms += (time.perf_counter() - t_start) * 1000.0

    print(f"\n  {'stage':<18}{'ms/iter':>10}{'share':>9}")
    print("  " + "-" * 37)
    per_iter = total_ms / a.iters
    for k in sorted(t.acc):
        ms = t.acc[k] / a.iters
        print(f"  {k:<18}{ms:>10.2f}{100*ms/per_iter:>8.1f}%")
    print("  " + "-" * 37)
    print(f"  {'TOTAL':<18}{per_iter:>10.2f}{'':>8}  ->  {1000/per_iter:.1f} it/s")


if __name__ == "__main__":
    main()
