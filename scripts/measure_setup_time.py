#!/usr/bin/env python3
"""Measure per-AOI setup overhead for the production cost model.

Times dataset construction + model build + first train step (excluding the long
optimize loop). Setup is paid once per LR512 tile; at ~441 tiles/granule it can
rival the k2→k4 train gap if it is tens of seconds.

Example
-------
    python scripts/measure_setup_time.py --s2-dir data/s2_revisits/asker_t512_y00_x00 \\
        --allow-no-hr --device 0 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _time_once(args_ns, device: torch.device) -> dict:
    from data import get_dataset
    from models.lr_tile_sampler import build_cross_frame_tile_sampler, build_lr_tile_sampler
    from optimize import (
        _hetero_output_dim,
        build_grad_scaler,
        build_model,
        build_projection_and_decoder,
        build_train_dataloader,
        init_hetero_region_scales,
        train_one_iteration,
    )
    from optimizers import build_optimizer

    t0 = time.perf_counter()
    train_data = get_dataset(args=args_ns, name=args_ns.dataset, training_device=device)
    t_data = time.perf_counter()
    if not getattr(args_ns, "lr_height", 0):
        args_ns.lr_height = int(getattr(train_data, "lr_height", 0) or 0)
    if not getattr(args_ns, "lr_width", 0):
        args_ns.lr_width = int(getattr(train_data, "lr_width", 0) or 0)

    output_dim = _hetero_output_dim(args_ns)
    input_projection, decoder = build_projection_and_decoder(args_ns, device, output_dim=output_dim)
    model = build_model(args_ns, input_projection, decoder, device)
    init_hetero_region_scales(model, train_data, device)
    optimizer = build_optimizer(model.parameters(), args_ns)
    grad_scaler = build_grad_scaler(args_ns, device)
    train_dataloader = build_train_dataloader(train_data, args_ns)
    tile_sampler = build_lr_tile_sampler(train_data, args_ns)
    cross_sampler = build_cross_frame_tile_sampler(train_data, args_ns)
    t_model = time.perf_counter()

    train_sample = None if cross_sampler is not None else next(iter(train_dataloader))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_step0 = time.perf_counter()
    train_one_iteration(
        model,
        optimizer,
        train_sample,
        device,
        args_ns,
        variance_reg=float(getattr(args_ns, "variance_reg", 0.0) or 0.0),
        variance_smooth_reg=float(getattr(args_ns, "variance_smooth_reg", 0.0) or 0.0),
        holdout_state=None,
        iteration=1,
        tile_sampler=tile_sampler,
        cross_sampler=cross_sampler,
        dataset=train_data,
        grad_scaler=grad_scaler,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_step1 = time.perf_counter()

    return {
        "dataset_s": t_data - t0,
        "model_build_s": t_model - t_data,
        "first_step_s": t_step1 - t_step0,
        "total_setup_s": t_step1 - t0,
        "lr_hw": [int(train_data.lr_height), int(train_data.lr_width)],
        "n_frames": int(train_data.num_samples),
        "has_hr_gt": bool(getattr(train_data, "has_hr_gt", True)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--allow-no-hr", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "setup_time.json",
    )
    cli = ap.parse_args()

    from optimize import get_argparser

    s2 = cli.s2_dir if cli.s2_dir.is_absolute() else ROOT / cli.s2_dir
    dataset = cli.dataset or s2.name.split("_t")[0].split("_lr")[0].split("_g")[0]
    device = torch.device(f"cuda:{cli.device}" if torch.cuda.is_available() else "cpu")

    argv = [
        "--dataset",
        dataset,
        "--s2-dir",
        str(s2),
        "--lr_degradation",
        "s2_psf_m",
        "--recon_loss",
        "charbonnier",
        "--charbonnier_eps",
        "0.01",
        "--lr_tile",
        "128",
        "--lr_tiles_per_step",
        "4",
        "--lr_tile_mix",
        "within",
        "--device",
        str(cli.device),
        "--no_qgis_export",
        "--no_hr_spatial_align",
    ]
    if cli.allow_no_hr:
        argv.append("--allow_no_hr")

    parser = get_argparser()
    trials = []
    for i in range(int(cli.repeat)):
        ns = parser.parse_args(argv)
        ns.seed = 6 + i
        if device.type == "cuda":
            torch.cuda.empty_cache()
        row = _time_once(ns, device)
        row["trial"] = i
        trials.append(row)
        print(
            f"trial {i}: setup={row['total_setup_s']:.2f}s "
            f"(data {row['dataset_s']:.2f} + model {row['model_build_s']:.2f} "
            f"+ step {row['first_step_s']:.2f})",
            flush=True,
        )

    totals = [t["total_setup_s"] for t in trials]
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "s2_dir": str(s2.relative_to(ROOT)) if s2.is_relative_to(ROOT) else str(s2),
        "dataset": dataset,
        "allow_no_hr": bool(cli.allow_no_hr),
        "device": str(device),
        "mean_setup_s": sum(totals) / len(totals),
        "min_setup_s": min(totals),
        "max_setup_s": max(totals),
        "extrapolate_granule_441_h": (sum(totals) / len(totals) * 441) / 3600.0,
        "trials": trials,
    }
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"mean setup {out['mean_setup_s']:.2f}s → "
        f"~{out['extrapolate_granule_441_h']:.2f} h / granule (441 AOIs, serial)",
        flush=True,
    )
    print(f"Wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
