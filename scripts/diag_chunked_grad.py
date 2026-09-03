#!/usr/bin/env python3
"""Sweep query-row counts through the real hashgrid + fused-MLP stack.

Fused k>=2 on LR2048/tile512 never trains the INR (holdout metric bit-frozen,
output collapses to a flat field), while k1 on the same AOI and k4/k8 on LR512
train fine. The broken configs are exactly the ones whose per-step row count
reaches 2**23:

    LR2048 k1 = 4.19M   ok        LR2048 k2 =  8.39M  broken
    LR512  k4 = 1.05M   ok        LR2048 k4 = 16.78M  broken
    LR512  k8 = 2.10M   ok

So walk the row count across that boundary and watch forward magnitude and
parameter gradients for a collapse.
"""

from __future__ import annotations

import argparse

import torch

from input_projections.hashgrid_tcnn import HashGridTcnn
from models.inr import INRBase
from models.mlp_tcnn import MLPTcnn

try:
    import tinycudann as tcnn
except ImportError:  # pragma: no cover
    tcnn = None


def _build(device: torch.device, log2: int, width: int, depth: int) -> INRBase:
    torch.manual_seed(0)
    proj = HashGridTcnn(
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=log2,
        base_resolution=16,
        max_resolution=8192,
        interpolation="smoothstep",
        device=device,
    )
    decoder = MLPTcnn(proj.output_dim, width, depth=depth, output_dim=3).to(device)
    print(f"decoder backend: {decoder._network.network_config['otype']}")
    return INRBase(proj, decoder, num_samples=4).to(device)


def _probe(module: INRBase, rows: int, max_rows: int,
           device: torch.device, loss_scale: float = 1.0) -> dict[str, float]:
    module.max_forward_rows = max_rows
    module.zero_grad(set_to_none=True)
    coords = torch.rand(1, rows, 1, 2, device=device,
                        generator=torch.Generator(device=device).manual_seed(3))
    sid = torch.zeros(1, dtype=torch.long, device=device)
    out, _ = module._encode_and_decode(coords, sid)
    # A mean loss makes dL/dout shrink as 1/rows. tcnn computes in fp16, so
    # past a few million rows the incoming grad lands in fp16 subnormals and
    # collapses to zero. loss_scale is the standard AMP remedy.
    ((out.float() - 0.5).square().mean() * loss_scale).backward()

    rep = {"out_absmean": float(out.detach().float().abs().mean()),
           "out_finite": float(torch.isfinite(out.detach().float()).all())}
    for name, p in module.named_parameters():
        key = name.split(".")[0]
        if key not in ("input_projection", "decoder"):
            continue
        g = 0.0 if p.grad is None else float(p.grad.float().abs().mean())
        rep[key] = rep.get(key, 0.0) + g
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--log2", type=int, default=21)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--no-chunk", action="store_true",
                    help="Disable row chunking so each size is one tcnn call.")
    args = ap.parse_args()

    if tcnn is None:
        raise SystemExit("tinycudann not available")
    device = torch.device(f"cuda:{args.device}")
    module = _build(device, args.log2, args.width, args.depth)

    sizes = [1 << 18, 1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24]
    print(f"width={args.width} depth={args.depth} log2={args.log2} "
          f"chunking={'off' if args.no_chunk else 'on'}, mean-reduced loss")
    print(f"{'rows':>12} {'scale':>8} {'hashgrid_grad':>14} {'decoder_grad':>13} "
          f"{'grads_alive':>12}")
    for scale in (1.0, float(1 << 15)):
        for rows in sizes:
            cap = 0 if args.no_chunk else INRBase.MAX_FORWARD_ROWS
            try:
                r = _probe(module, rows, cap, device, loss_scale=scale)
            except Exception as exc:  # noqa: BLE001
                print(f"{rows:>12,} {scale:>8.0f}  ERROR: "
                      f"{type(exc).__name__}: {str(exc)[:60]}")
                continue
            hg = r.get("input_projection", 0.0)
            dec = r.get("decoder", 0.0)
            print(f"{rows:>12,} {scale:>8.0f} {hg:>14.6g} {dec:>13.6g} "
                  f"{'yes' if hg > 0 and dec > 0 else 'NO':>12}")
            del r
            torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
