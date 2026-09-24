#!/usr/bin/env python3
"""Profile within-frame k1 vs k2 train steps on a large scene (tile 512).

Breaks down crop/stack, forward, backward, optim — and reports peak VRAM —
so we can see whether the k2 slowdown is ~2× compute or something worse.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataset
from eval.lr_holdout import gather_train_masks, init_early_stop_state
from models.lr_tile_sampler import (
    LrTileSampler,
    stack_lr_hr_tiles,
)
from optimize import (
    _as_device_tensor,
    _step_schedule_kwargs,
    _train_forward_losses,
    build_model,
    build_projection_and_decoder,
    build_optimizer,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ms(device: torch.device, fn):
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, (time.perf_counter() - t0) * 1000.0


def profile_k(
    device: torch.device,
    ds,
    args,
    *,
    k: int,
    tile: int,
    iters: int,
    warmup: int,
    frame: int = 0,
    holdout: float = 0.0,
):
    args = SimpleNamespace(**vars(args))
    args.lr_tile = tile
    args.lr_tiles_per_step = k
    args.device = str(device.index if device.type == "cuda" else "cpu")

    proj, dec = build_projection_and_decoder(args, device)
    model = build_model(args, proj, dec, device)
    opt = build_optimizer(model.parameters(), args)
    sampler = LrTileSampler.from_shapes(
        ds.lr_height, ds.lr_width, tile, tiles_per_step=k, seed=0
    )

    # One frame sample like the DataLoader (batch dim present).
    sample = ds[int(frame)]
    coords0 = _as_device_tensor(sample["input"], device)
    if coords0.dim() == 3:
        coords0 = coords0.unsqueeze(0)
    lr0 = _as_device_tensor(sample["lr_target"], device)
    if lr0.dim() == 3:
        lr0 = lr0.unsqueeze(0)
    sid0 = _as_device_tensor(sample["sample_id"], device).reshape(-1)
    if sid0.dim() == 0:
        sid0 = sid0.unsqueeze(0)
    gt_dx = torch.zeros(1, device=device)
    gt_dy = torch.zeros(1, device=device)
    full_lr_hw = (int(ds.lr_height), int(ds.lr_width))

    mask0 = None
    if holdout > 0.0:
        state = init_early_stop_state(
            num_frames=int(ds.num_samples),
            lr_height=int(ds.lr_height),
            lr_width=int(ds.lr_width),
            spatial_holdout=float(holdout),
            holdout_block=0,
            patience=3,
            min_iters=1000,
            metric="holdout_mse",
            device=device,
        )
        mask0 = gather_train_masks(state.train_masks, sid0, device)

    torch.cuda.reset_peak_memory_stats(device)
    acc = {n: 0.0 for n in ("crop_stack", "forward", "backward", "optim", "total")}
    n_meas = 0

    for it in range(warmup + iters):
        measuring = it >= warmup
        origins = sampler.next_origins()

        def do_crop():
            return stack_lr_hr_tiles(
                coords0, lr0, origins, tile,
                mask=mask0,
                sample_id=sid0, gt_dx=gt_dx, gt_dy=gt_dy,
            )

        (coords, lr_t, mask_t, sid, dx, dy), crop_ms = _ms(device, do_crop)

        model.train()
        opt.zero_grad(set_to_none=True)
        model_kwargs = dict(_step_schedule_kwargs(args, it + 1))

        def do_fwd():
            return _train_forward_losses(
                model, coords, lr_t, sid, mask_t, dx, dy, full_lr_hw,
                args, model_kwargs, 0.0, 0.0,
            )

        losses, fwd_ms = _ms(device, do_fwd)
        _, bwd_ms = _ms(device, lambda: losses["total_loss"].backward())
        _, opt_ms = _ms(device, lambda: opt.step())
        total_ms = crop_ms + fwd_ms + bwd_ms + opt_ms

        if measuring:
            n_meas += 1
            acc["crop_stack"] += crop_ms
            acc["forward"] += fwd_ms
            acc["backward"] += bwd_ms
            acc["optim"] += opt_ms
            acc["total"] += total_ms

        # free graph
        del losses, coords, lr_t

    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    per = {k: v / n_meas for k, v in acc.items()}
    hr_side = tile * int(getattr(ds, "df", 4) or 4)
    return {
        "k": k,
        "tile": tile,
        "frame": int(frame),
        "holdout": float(holdout),
        "hr_window": hr_side,
        "hr_pixels_per_step": k * hr_side * hr_side,
        "ms": per,
        "it_s": 1000.0 / per["total"] if per["total"] > 0 else 0.0,
        "peak_gb": peak,
        "n_origins": len(sampler.origins),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-dir", default="data/s2_revisits/asker_lr2048")
    ap.add_argument("--device", type=int, default=2)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--hash_log2", type=int, default=19,
                    help="log2 hashmap size (optimize.py default is 21).")
    ap.add_argument("--hash_interpolation", default="smoothstep",
                    choices=["smoothstep", "linear"])
    ap.add_argument("--out", default="profile_k1_vs_k2_tile512.json")
    ap.add_argument("--frame", type=int, default=0,
                    help="Frame index to sample. Frame 0 skips apply_color_transform.")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="spatial_holdout fraction; >0 exercises the masked loss path.")
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)

    ds_args = SimpleNamespace(
        dataset="asker",
        s2_dir=str(ROOT / a.s2_dir),
        hr_path=None,
        df=4,
        scale_factor=4,
        hr_gsd_m=0.0,
        s2_native_gsd_m=10.0,
        num_samples=0,
        lr_size=0,
        dataset_device=str(device),
        no_hr_harmonize=False,
        spatial_alignment_path=str(ROOT / "eval" / "spatial_alignment.json"),
        input_projection="hashgrid_tcnn",
        model="mlp_tcnn",
        network_depth=4,
        network_hidden_dim=256,
        projection_dim=256,
        fourier_scale=10.0,
        hash_n_levels=16,
        hash_n_features_per_level=2,
        hash_log2_hashmap_size=a.hash_log2,
        hash_interpolation=a.hash_interpolation,
        hash_max_resolution=0,
        hash_max_resolution_h=0,
        hash_max_resolution_w=0,
        hash_max_resolution_mult=1.0,
        hash_base_resolution=0,
        hash_tcnn_output_dtype="fp32",
        tcnn_mlp_dtype="fp16",
        lr_degradation="s2_psf_m",
        recon_loss="charbonnier",
        charbonnier_eps=0.01,
        huber_delta=0.05,
        use_gnll=False,
        use_laplace_nll=False,
        hetero_scale="pixel",
        hetero_region_size=4,
        learning_rate=1e-3,
        weight_decay=0.05,
        optimizer="adamw",
        muon_momentum=0.95,
        no_muon_nesterov=False,
        muon_ns_steps=5,
        muon_eps=1e-8,
        seed=6,
        lr_height=0,
        lr_width=0,
        schedule_horizon_iters=3000,
        psf_curriculum="none",
        psf_sigma_schedule="none",
    )
    print(f"loading {ds_args.s2_dir} on {device} ...", flush=True)
    ds = get_dataset(ds_args)
    ds_args.num_samples = ds.num_samples
    ds_args.lr_height, ds_args.lr_width = ds.lr_height, ds.lr_width
    print(
        f"LR {ds.lr_height}×{ds.lr_width} HR {tuple(ds.get_original_hr().shape[:2])} "
        f"frames={ds.num_samples}",
        flush=True,
    )

    rows = []
    for k in a.ks:
        print(f"\n=== profile k={k} tile={a.tile} ===", flush=True)
        row = profile_k(
            device, ds, ds_args, k=k, tile=a.tile, iters=a.iters, warmup=a.warmup,
            frame=a.frame, holdout=a.holdout,
        )
        rows.append(row)
        ms = row["ms"]
        print(
            f"  crop={ms['crop_stack']:.1f}  fwd={ms['forward']:.1f}  "
            f"bwd={ms['backward']:.1f}  opt={ms['optim']:.1f}  "
            f"TOTAL={ms['total']:.1f} ms  ({row['it_s']:.2f} it/s)  "
            f"peak={row['peak_gb']:.2f} GB  "
            f"HR_px={row['hr_pixels_per_step']:,}",
            flush=True,
        )

    if len(rows) >= 2:
        a0, a1 = rows[0], rows[1]
        print("\n=== ratios (second / first) ===", flush=True)
        for key in ("crop_stack", "forward", "backward", "optim", "total"):
            r = a1["ms"][key] / max(a0["ms"][key], 1e-9)
            print(f"  {key}: {r:.2f}×", flush=True)
        print(
            f"  peak_gb: {a1['peak_gb']/max(a0['peak_gb'],1e-9):.2f}×  "
            f"hr_pixels: {a1['hr_pixels_per_step']/max(a0['hr_pixels_per_step'],1):.2f}×",
            flush=True,
        )

    out = ROOT / "single_samples" / "sweep_results" / a.out
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
