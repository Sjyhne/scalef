#!/usr/bin/env python3
"""Retrain one east/south cell with a frozen west/north neighbor halo.

Does not overwrite the independent run. Default neighbor is ``prod_k4_base2_*``.
Writes ``<out-prefix>_<tile_id>``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_production import _cmd, _sr_geotiff_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=str, default="32VNM")
    ap.add_argument("--tile", type=str, default="32VNM_t512_y04_x18")
    ap.add_argument("--neighbor", type=str, default="32VNM_t512_y04_x17")
    ap.add_argument("--s2-dir", type=Path, default=None)
    ap.add_argument("--neighbor-prefix", type=str, default="prod_k4_base2")
    ap.add_argument("--out-prefix", type=str, default="prod_k4_base2_halo_full")
    ap.add_argument("--halo-mode", type=str, default="full", choices=["full", "lowfreq", "mean"])
    ap.add_argument("--halo-weight", type=float, default=1.0)
    ap.add_argument("--halo-lf-hr-px", type=int, default=16)
    ap.add_argument("--device", type=int, required=True, help="CUDA index in 0-6 (not 7).")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--halo-lr-px", type=int, default=32)
    ap.add_argument("--halo-sides", type=str, default="west")
    args = ap.parse_args()
    if int(args.device) == 7:
        raise SystemExit("GPU 7 is reserved; use 0-6")

    default_s2 = ROOT / "data" / "s2_revisits" / "national_2025_v2" / args.tile
    s2_dir = args.s2_dir or default_s2
    neighbor_tif = _sr_geotiff_path(args.parent, args.neighbor, args.neighbor_prefix)
    if not neighbor_tif.is_file():
        raise SystemExit(f"missing neighbor SR {neighbor_tif}")
    if not s2_dir.is_dir():
        raise SystemExit(f"missing s2 dir {s2_dir}")
    tile = {
        "parent": args.parent,
        "tile_id": args.tile,
        "s2_dir": str(s2_dir.relative_to(ROOT)),
    }
    cmd = _cmd(tile, int(args.device), int(args.iters), export_geotiff=True)
    run_name = f"{args.out_prefix}_{args.tile}"
    for i, tok in enumerate(cmd):
        if tok == "--run_name":
            cmd[i + 1] = run_name
            break
    cmd += [
        "--halo_sr",
        str(neighbor_tif),
        "--halo_lr_px",
        str(int(args.halo_lr_px)),
        "--halo_sides",
        str(args.halo_sides),
        "--halo_weight",
        str(float(args.halo_weight)),
        "--halo_mode",
        str(args.halo_mode),
        "--halo_lf_hr_px",
        str(int(args.halo_lf_hr_px)),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
