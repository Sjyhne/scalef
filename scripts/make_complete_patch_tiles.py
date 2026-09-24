#!/usr/bin/env python3
"""Materialize LR512 S2 dirs for every complete NIB patch-grid tile.

Reads ``data/s2_revisits/map/patch_grid_lr512/complete_patches_lr512.geojson``,
places a 512×512 window on the covering MGRS revisit stack (centred on the
patch in S2 CRS), and writes symlink variants:

    data/s2_revisits/{city}_p{row:02d}_{col:02d}/

Also writes a manifest JSON used by ``bench_complete_patches.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.warp import transform as warp_xy

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2_dataset import FOCUS_PROJECT_BY_CITY, resolve_focus_project_dir  # noqa: E402

FORMER_ID_RENAME = {
    "nib01_32VLL": "algard",
    "nib01_32VKL": "naerbo",
    "nib02_32VLK": "flekkefjord",
    "nib03_34WEC": "rafsbotn",
    "nib04_32VNM": "nittedal",
    "nib05_32VNR": "melhus",
    "nib06_34WEC": "alta",
    "nib08_35WMT": "karasjok",
    "nib09_34WEB": "kautokeino",
}

DEFAULT_GEOJSON = (
    ROOT / "data" / "s2_revisits" / "map" / "patch_grid_lr512" / "complete_patches_lr512.geojson"
)
DEFAULT_MANIFEST = (
    ROOT / "data" / "s2_revisits" / "map" / "patch_grid_lr512" / "complete_patch_tiles_manifest.json"
)
MASK_CRS = "EPSG:25833"
SIDE = 512


def default_paths_for_side(side: int) -> tuple[Path, Path]:
    """GeoJSON + manifest paths under ``data/s2_revisits/map/patch_grid_lr{side}/``."""
    base = ROOT / "data" / "s2_revisits" / "map" / f"patch_grid_lr{int(side)}"
    return (
        base / f"complete_patches_lr{int(side)}.geojson",
        base / f"complete_patch_tiles_manifest.json",
    )


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _proj_to_cities() -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for city, folder in FOCUS_PROJECT_BY_CITY.items():
        proj = resolve_focus_project_dir(city)
        if proj is not None:
            out[proj.name].append(city)
    return dict(out)


def _patch_center_xy(feat: dict) -> tuple[float, float]:
    xs = [p[0] for ring in feat["geometry"]["coordinates"] for p in ring]
    ys = [p[1] for ring in feat["geometry"]["coordinates"] for p in ring]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _place_window(
    city: str, cx: float, cy: float, *, side: int = SIDE
) -> tuple[int, int, int, int] | None:
    """Return (row0, col0, row_c, col_c) in S2 pixels, or None if 512² does not fit."""
    s2 = ROOT / "data" / "s2_revisits" / city
    meta = json.loads((s2 / "meta.json").read_text())
    frames = meta.get("frames") or []
    if not frames:
        return None
    first = s2 / frames[0]["path"]
    with rasterio.open(first) as src:
        xs, ys = warp_xy(MASK_CRS, src.crs, [cx], [cy])
        col, row = (~src.transform) * (xs[0], ys[0])
        col_c, row_c = int(round(col)), int(round(row))
        row0 = row_c - side // 2
        col0 = col_c - side // 2
        if row0 < 0 or col0 < 0 or row0 + side > src.height or col0 + side > src.width:
            return None
        return row0, col0, row_c, col_c


def _write_tile_dir(
    parent_city: str,
    dest: Path,
    row0: int,
    col0: int,
    *,
    side: int,
    patch_row: int,
    patch_col: int,
    force: bool,
) -> dict:
    parent = ROOT / "data" / "s2_revisits" / parent_city
    meta = json.loads((parent / "meta.json").read_text())
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    area = next(
        (a for a in json.loads((ROOT / "data" / "s2_revisits" / "aois.json").read_text())["areas"]
         if a["id"] == parent_city),
        None,
    )
    new_meta = dict(meta)
    new_meta["aoi_window"] = {
        "col_off": int(col0),
        "row_off": int(row0),
        "width": int(side),
        "height": int(side),
    }
    new_meta["parent_s2_dir"] = str(parent)
    new_meta["lr_size_request"] = int(side)
    new_meta["patch_grid"] = {
        "city": parent_city,
        "row": int(patch_row),
        "col": int(patch_col),
        "side": int(side),
    }
    new_meta["focus_project_folder"] = FOCUS_PROJECT_BY_CITY.get(parent_city)
    if area:
        new_meta["nib_acquisition_date"] = area.get("date") or meta.get("center_date")

    for fr in meta.get("frames") or []:
        for key in ("path", "cloud_mask"):
            name = fr.get(key)
            if name:
                _symlink(parent / name, dest / name)
    (dest / "meta.json").write_text(json.dumps(new_meta, indent=2) + "\n")
    return {
        "tile_id": dest.name,
        "s2_dir": str(dest.relative_to(ROOT)),
        "parent_city": parent_city,
        "patch_row": int(patch_row),
        "patch_col": int(patch_col),
        "row_off": int(row0),
        "col_off": int(col0),
        "side": int(side),
        "n_frames": len(meta.get("frames") or []),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "s2_revisits")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--side", type=int, default=SIDE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fix-geojson-names", action="store_true", default=True)
    args = ap.parse_args()

    geo = json.loads(args.geojson.read_text())
    renamed = 0
    for feat in geo["features"]:
        old = feat["properties"]["city"]
        new = FORMER_ID_RENAME.get(old, old)
        if new != old:
            feat["properties"]["city"] = new
            renamed += 1
    if args.fix_geojson_names and renamed:
        args.geojson.write_text(json.dumps(geo))
        print(f"Updated {renamed} geojson city names → {args.geojson}")

    proj_cities = _proj_to_cities()
    seen: set[tuple[str, int, int]] = set()
    tiles: list[dict] = []
    skipped: list[dict] = []

    for feat in geo["features"]:
        city0 = feat["properties"]["city"]
        if city0 not in FOCUS_PROJECT_BY_CITY:
            skipped.append({"city": city0, "reason": "unknown_city"})
            continue
        folder = FOCUS_PROJECT_BY_CITY[city0]
        prow = int(feat["properties"]["row"])
        pcol = int(feat["properties"]["col"])
        key = (folder, prow, pcol)
        if key in seen:
            continue
        seen.add(key)

        cx, cy = _patch_center_xy(feat)
        placed = None
        for cand in proj_cities.get(folder, [city0]):
            win = _place_window(cand, cx, cy, side=args.side)
            if win is not None:
                placed = (cand, *win)
                break
        if placed is None:
            skipped.append(
                {
                    "city": city0,
                    "patch_row": prow,
                    "patch_col": pcol,
                    "reason": "outside_fetched_mgrs",
                }
            )
            print(f"SKIP {city0} r{prow} c{pcol}: outside fetched MGRS")
            continue

        parent_city, row0, col0, row_c, col_c = placed
        # Keep legacy LR512 names so existing prod_k4_* runs remain skippable.
        if int(args.side) == 512:
            dest = args.out_root / f"{parent_city}_p{prow:02d}_{pcol:02d}"
        else:
            dest = args.out_root / f"{parent_city}_lr{int(args.side)}_p{prow:03d}_{pcol:03d}"
        row = _write_tile_dir(
            parent_city,
            dest,
            row0,
            col0,
            side=args.side,
            patch_row=prow,
            patch_col=pcol,
            force=args.force,
        )
        row["mask_center_xy"] = [cx, cy]
        row["s2_center_rc"] = [row_c, col_c]
        tiles.append(row)
        print(
            f"{dest.name}: {args.side}² @ row={row0} col={col0} "
            f"(from {parent_city}, patch r{prow} c{pcol})"
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geojson": str(args.geojson.relative_to(ROOT)),
        "side": args.side,
        "n_complete_in_geojson": len(geo["features"]),
        "n_unique_project_patches": len(seen),
        "n_tiles": len(tiles),
        "n_skipped": len(skipped),
        "tiles": tiles,
        "skipped": skipped,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {k: payload[k] for k in ("n_tiles", "n_skipped", "n_unique_project_patches", "side")},
            indent=2,
        )
    )
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
