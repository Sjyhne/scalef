#!/usr/bin/env python3
"""Op-level CUDA breakdown of one train step (k tiles) via torch.profiler.

Used to explain why a non-zero frame index makes the backward pass much more
expensive: frame 0's affine is frozen, so coordinate gradients are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataset
from eval.lr_holdout import gather_train_masks, init_early_stop_state
from models.lr_tile_sampler import LrTileSampler, stack_lr_hr_tiles
from optimize import (
    _as_device_tensor,
    _step_schedule_kwargs,
    _train_forward_losses,
    build_model,
    build_optimizer,
    build_projection_and_decoder,
)
from scripts.profile_k1_vs_k2_step import _sync  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-dir", default="data/s2_revisits/asker_lr2048")
    ap.add_argument("--device", type=int, default=2)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--frame", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--rows", type=int, default=18)
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)

    ds_args = SimpleNamespace(
        dataset="asker", s2_dir=str(ROOT / a.s2_dir), hr_path=None, df=4,
        scale_factor=4, hr_gsd_m=0.0, s2_native_gsd_m=10.0, num_samples=0,
        lr_size=0, dataset_device=str(device), no_hr_harmonize=False,
        spatial_alignment_path=str(ROOT / "eval" / "spatial_alignment.json"),
        input_projection="hashgrid_tcnn", model="mlp_tcnn", network_depth=4,
        network_hidden_dim=256, projection_dim=256, fourier_scale=10.0,
        hash_n_levels=16, hash_n_features_per_level=2,
        hash_log2_hashmap_size=21, hash_interpolation="linear",
        hash_max_resolution=0, hash_max_resolution_h=0, hash_max_resolution_w=0,
        hash_max_resolution_mult=1.0, hash_base_resolution=0,
        hash_tcnn_output_dtype="fp32", tcnn_mlp_dtype="fp16",
        lr_degradation="s2_psf_m", recon_loss="charbonnier", charbonnier_eps=0.01,
        huber_delta=0.05, use_gnll=False, use_laplace_nll=False,
        hetero_scale="pixel", hetero_region_size=4, learning_rate=1e-3,
        weight_decay=0.05, optimizer="adamw", muon_momentum=0.95,
        no_muon_nesterov=False, muon_ns_steps=5, muon_eps=1e-8, seed=6,
        lr_height=0, lr_width=0, schedule_horizon_iters=3000,
        psf_curriculum="none", psf_sigma_schedule="none",
    )
    ds = get_dataset(ds_args)
    ds_args.num_samples = ds.num_samples
    ds_args.lr_height, ds_args.lr_width = ds.lr_height, ds.lr_width
    ds_args.lr_tile = a.tile
    ds_args.lr_tiles_per_step = a.k
    ds_args.device = str(a.device)

    proj, dec = build_projection_and_decoder(ds_args, device)
    model = build_model(ds_args, proj, dec, device)
    opt = build_optimizer(model.parameters(), ds_args)
    sampler = LrTileSampler.from_shapes(
        ds.lr_height, ds.lr_width, a.tile, tiles_per_step=a.k, seed=0
    )

    sample = ds[int(a.frame)]
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
    if a.holdout > 0:
        state = init_early_stop_state(
            num_frames=int(ds.num_samples), lr_height=int(ds.lr_height),
            lr_width=int(ds.lr_width), spatial_holdout=float(a.holdout),
            holdout_block=0, patience=3, min_iters=1000,
            metric="holdout_mse", device=device,
        )
        mask0 = gather_train_masks(state.train_masks, sid0, device)

    def one_step(it: int) -> None:
        origins = sampler.next_origins()
        coords, lr_t, mask_t, sid, dx, dy = stack_lr_hr_tiles(
            coords0, lr0, origins, a.tile, mask=mask0,
            sample_id=sid0, gt_dx=gt_dx, gt_dy=gt_dy,
        )
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = _train_forward_losses(
            model, coords, lr_t, sid, mask_t, dx, dy, full_lr_hw,
            ds_args, dict(_step_schedule_kwargs(ds_args, it + 1)), 0.0, 0.0,
        )
        losses["total_loss"].backward()
        opt.step()

    for it in range(3):
        one_step(it)
    _sync(device)

    print(f"\n=== op profile k={a.k} tile={a.tile} frame={a.frame} "
          f"holdout={a.holdout} coords_requires_grad_path ===", flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for it in range(a.iters):
            one_step(it)
        _sync(device)

    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=a.rows, max_name_column_width=55
    ))


if __name__ == "__main__":
    main()
