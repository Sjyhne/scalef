#!/usr/bin/env python3
"""Compare learned affines on the 2048 / 1024 / 512 tiling of asker.

For each S2 frame, print the 2048 centre shift and the 2x2 / 4x4 maps of
the same quantity in metres. If a single global affine is enough, the
small-tile arrows agree with each other and with the 2048 value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "single_samples" / "asker" / "sample"


def _load(run: str) -> dict | None:
    p = RUNS / run / "affines.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _shift(dump: dict, frame: int) -> tuple[float, float]:
    fr = dump["frames"][frame]
    return fr["center_shift_east_m"], fr["center_shift_north_m"]


def _fmt(dx: float, dy: float) -> str:
    return f"({dx:+5.2f},{dy:+5.2f})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    del args

    big = _load("aff_g2048")
    if big is None:
        raise SystemExit("missing aff_g2048/affines.json — wait for the sweep")

    n = len(big["frames"])
    rows_1024: list[list[dict | None]] = [
        [_load(f"aff_g1024_y{iy}_x{ix}") for ix in range(2)] for iy in range(2)
    ]
    rows_512: list[list[dict | None]] = [
        [_load(f"aff_g512_y{iy}_x{ix}") for ix in range(4)] for iy in range(4)
    ]

    print("centre-point shift in metres: (east, north). Frame 0 is frozen identity.\n")
    for f in range(n):
        frozen = " [frozen]" if big["frames"][f]["frozen"] else ""
        bdx, bdy = _shift(big, f)
        print(f"frame {f:02d}{frozen}")
        print(f"  2048     {_fmt(bdx, bdy)}")
        print("  1024")
        for iy in range(2):
            cells = []
            for ix in range(2):
                d = rows_1024[iy][ix]
                cells.append(_fmt(*_shift(d, f)) if d else "  (miss)  ")
            print("         " + "  ".join(cells))
        print("  512")
        for iy in range(4):
            cells = []
            for ix in range(4):
                d = rows_512[iy][ix]
                cells.append(_fmt(*_shift(d, f)) if d else "  (miss)  ")
            print("         " + "  ".join(cells))
        # spread among 512 tiles that exist
        pts = [_shift(rows_512[iy][ix], f)
               for iy in range(4) for ix in range(4) if rows_512[iy][ix]]
        if pts:
            e = [p[0] for p in pts]
            n_ = [p[1] for p in pts]
            print(f"  512 spread  east {max(e)-min(e):.2f} m  north {max(n_)-min(n_):.2f} m"
                  f"  vs 2048  Δe={sum(e)/len(e)-bdx:+.2f}  Δn={sum(n_)/len(n_)-bdy:+.2f}")
        print()


if __name__ == "__main__":
    main()
