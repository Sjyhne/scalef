#!/usr/bin/env python3
"""Create LR-size AOI variants that share full MGRS tiles via symlinks.

Example
-------
python scripts/make_lr_size_variants.py --city asker --sizes 256 512 1024

Writes ``data/s2_revisits/asker_lr256/`` etc. with:
  - symlinks to the parent city's ``*.tif`` frames / cloud masks
  - a new ``meta.json`` whose ``aoi_window`` is a centered ``size×size`` crop
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_xy
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_aois() -> dict:
    return json.loads((ROOT / "data" / "s2_revisits" / "aois.json").read_text())


def _center_px_from_lonlat(src, lon: float, lat: float) -> tuple[int, int]:
    xs, ys = warp_xy("EPSG:4326", src.crs, [lon], [lat])
    inv = ~src.transform
    col, row = inv * (xs[0], ys[0])
    return int(round(row)), int(round(col))


def _window_centered(tile_h: int, tile_w: int, row_c: int, col_c: int, side: int) -> Window:
    side = int(side)
    row0 = int(np.clip(row_c - side // 2, 0, max(0, tile_h - side)))
    col0 = int(np.clip(col_c - side // 2, 0, max(0, tile_w - side)))
    # If tile smaller than side, take full tile.
    h = min(side, tile_h)
    w = min(side, tile_w)
    return Window(col0, row0, w, h)


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def make_variant(city: str, side: int, *, out_root: Path, force: bool = False) -> Path:
    src_dir = out_root / city
    meta_path = src_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path} — fetch revisits first")

    meta = json.loads(meta_path.read_text())
    frames = meta.get("frames") or []
    if not frames:
        raise ValueError(f"{meta_path} has no frames")

    aois = _load_aois()
    area = next((a for a in aois.get("areas", []) if a["id"] == city), None)
    lon = float(area["center_lon"]) if area else None
    lat = float(area["center_lat"]) if area else None
    if lon is None:
        # Fall back to current AOI center.
        aoi = meta["aoi_window"]
        lon = lat = None

    first = src_dir / frames[0]["path"]
    with rasterio.open(first) as src:
        tile_h, tile_w = int(src.height), int(src.width)
        if lon is not None:
            row_c, col_c = _center_px_from_lonlat(src, lon, lat)
        else:
            aoi = meta["aoi_window"]
            row_c = int(aoi["row_off"]) + int(aoi["height"]) // 2
            col_c = int(aoi["col_off"]) + int(aoi["width"]) // 2
        win = _window_centered(tile_h, tile_w, row_c, col_c, side)

    dest = out_root / f"{city}_lr{side}"
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    new_meta = dict(meta)
    new_meta["aoi_window"] = {
        "col_off": int(win.col_off),
        "row_off": int(win.row_off),
        "width": int(win.width),
        "height": int(win.height),
    }
    new_meta["parent_s2_dir"] = str(src_dir)
    new_meta["lr_size_request"] = int(side)
    new_meta["nib_acquisition_date"] = (
        (area or {}).get("date") or meta.get("center_date")
    )
    if area:
        new_meta["focus_project_folder"] = area.get("project_folder")
        new_meta["center_lon"] = area.get("center_lon")
        new_meta["center_lat"] = area.get("center_lat")

    # Symlink frame products (not meta/preview).
    for fr in frames:
        for key in ("path", "cloud_mask"):
            name = fr.get(key)
            if not name:
                continue
            _symlink(src_dir / name, dest / name)
    if (src_dir / "preview.png").is_file():
        _symlink(src_dir / "preview.png", dest / "preview.png")

    (dest / "meta.json").write_text(json.dumps(new_meta, indent=2) + "\n")
    print(
        f"{dest.name}: aoi {int(win.width)}×{int(win.height)} "
        f"at row={int(win.row_off)} col={int(win.col_off)} "
        f"({len(frames)} frames)"
    )
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default="asker")
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument("--out-root", type=Path, default=ROOT / "data" / "s2_revisits")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    for side in args.sizes:
        make_variant(args.city, side, out_root=args.out_root, force=args.force)


if __name__ == "__main__":
    main()
