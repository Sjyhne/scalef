#!/usr/bin/env python3
"""Isolate apply_color_transform cost vs batch size (fwd + bwd).

The per-element in-place writes build a CopySlices autograd chain whose backward
touches the whole [B,H,W,3] tensor once per write, so cost grows ~O(B^2).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.inr import INRBase


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, x, device, iters=10, warmup=3):
    for _ in range(warmup + iters):
        pass
    _sync(device)
    t0 = time.perf_counter()
    for i in range(warmup + iters):
        if i == warmup:
            _sync(device)
            t0 = time.perf_counter()
        xi = x.detach().clone().requires_grad_(True)
        out = fn(xi)
        out.sum().backward()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=2)
    ap.add_argument("--hw", type=int, default=2048)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()

    device = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(device)

    module = INRBase(None, torch.nn.Identity(), num_samples=a.num_frames).to(device)
    # Give non-identity params so results are meaningful.
    with torch.no_grad():
        for sid in range(1, a.num_frames):
            for ch in range(3):
                module.color_transforms[sid][ch].weight.fill_(1.0 + 0.01 * sid)
                module.color_transforms[sid][ch].bias.fill_(0.001 * ch)

    print(f"HW={a.hw}x{a.hw} frames={a.num_frames} on {device}")
    for b in a.bs:
        x = torch.randn(b, a.hw, a.hw, 3, device=device)
        sid = torch.arange(1, b + 1, device=device) % a.num_frames
        sid = torch.where(sid == 0, torch.ones_like(sid), sid)

        old = _bench(lambda t: module.apply_color_transform_loop(t, sid), x, device)
        new = _bench(lambda t: module.apply_color_transform(t, sid), x, device)

        xi = x.detach().clone()
        with torch.no_grad():
            a_out = module.apply_color_transform_loop(xi, sid)
            b_out = module.apply_color_transform(xi, sid)
        max_diff = float((a_out - b_out).abs().max())
        print(
            f"  B={b}: loop={old:7.2f} ms   vectorized={new:6.2f} ms   "
            f"speedup={old / max(new, 1e-9):5.2f}x   max|Δ|={max_diff:.2e}"
        )


if __name__ == "__main__":
    main()
