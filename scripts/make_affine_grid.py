#!/usr/bin/env python3
"""Cut a 1 / 2x2 / 4x4 AOI grid that tiles the existing asker LR2048 window.

Writes ``data/s2_revisits/asker_g2048``, ``asker_g1024_y{i}_x{j}``,
``asker_g512_y{i}_x{j}`` with symlinked frames and a new ``aoi_window``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _write_variant(parent_meta: dict, parent_dir: Path, dest: Path, row: int, col: int, side: int) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    meta = dict(parent_meta)
    meta["aoi_window"] = {
        "col_off": int(col),
        "row_off": int(row),
        "width": int(side),
        "height": int(side),
    }
    meta["parent_s2_dir"] = str(parent_dir)
    meta["lr_size_request"] = int(side)
    meta["grid_parent"] = "asker_lr2048"
    for fr in parent_meta.get("frames") or []:
        for key in ("path", "cloud_mask"):
            name = fr.get(key)
            if name:
                _symlink(parent_dir / name, dest / name)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{dest.name}: {side}x{side} row={row} col={col}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "s2_revisits" / "asker_lr2048")
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "s2_revisits")
    args = ap.parse_args()

    meta = json.loads((args.src / "meta.json").read_text())
    aoi = meta["aoi_window"]
    row0, col0 = int(aoi["row_off"]), int(aoi["col_off"])
    parent = Path(meta.get("parent_s2_dir") or args.src)
    if not parent.is_dir():
        parent = ROOT / "data" / "s2_revisits" / "asker"

    _write_variant(meta, parent, args.out_root / "asker_g2048", row0, col0, 2048)
    for iy in range(2):
        for ix in range(2):
            _write_variant(
                meta, parent,
                args.out_root / f"asker_g1024_y{iy}_x{ix}",
                row0 + iy * 1024, col0 + ix * 1024, 1024,
            )
    for iy in range(4):
        for ix in range(4):
            _write_variant(
                meta, parent,
                args.out_root / f"asker_g512_y{iy}_x{ix}",
                row0 + iy * 512, col0 + ix * 512, 512,
            )


if __name__ == "__main__":
    main()
