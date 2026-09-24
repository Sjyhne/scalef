#!/usr/bin/env python3
"""Filter LR512 production tiles by mainland land overlap (and optional clear counts).

National Norway SR should train only on non-overlapping 512² cells that intersect
mainland (+ nearshore islands), not every ocean cell inside an MGRS granule.

Used by ``make_granule_tiles.py`` / ``run_production.py`` / ``run_granule_batch.py``,
and as a standalone manifest filter:

    python scripts/land_mask_lr512.py \\
      --manifest data/s2_revisits/32VNM/granule_tiles_lr512_manifest.json \\
      --land-mask production/cloud_availability/norway_outline_ne50m.geojson \\
      --min-clear 6 \\
      --clear-counts-dir production/cloud_availability/lr512_norway_july2025_pm45
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.path import Path as MplPath
from rasterio.transform import xy
from rasterio.warp import transform as warp_xy

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAND_MASK = (
    ROOT / "production" / "cloud_availability" / "norway_outline_ne50m.geojson"
)


def load_land_paths(
    geojson_path: Path,
    *,
    exclude_svalbard: bool = True,
    exclude_jan_mayen: bool = True,
) -> list[MplPath]:
    """Load WGS84 land rings from a MultiPolygon/Polygon GeoJSON FeatureCollection."""
    data = json.loads(Path(geojson_path).read_text())
    feats = data["features"] if "features" in data else [data]
    paths: list[MplPath] = []
    for feat in feats:
        geom = feat.get("geometry") or feat
        rings: list = []
        if geom["type"] == "Polygon":
            rings = [geom["coordinates"][0]]
        elif geom["type"] == "MultiPolygon":
            rings = [poly[0] for poly in geom["coordinates"]]
        else:
            continue
        for ring in rings:
            arr = np.asarray(ring, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            lon_min = float(arr[:, 0].min())
            lat_min = float(arr[:, 1].min())
            lat_max = float(arr[:, 1].max())
            if exclude_svalbard and lat_min >= 72.0:
                continue
            if exclude_jan_mayen and lon_min < 0.0:
                continue
            # Close ring for Path.
            if not np.allclose(arr[0], arr[-1]):
                arr = np.vstack([arr, arr[0]])
            paths.append(MplPath(arr))
    if not paths:
        raise ValueError(f"no land rings loaded from {geojson_path}")
    return paths


def window_lonlat_samples(
    transform,
    crs,
    *,
    row0: int,
    col0: int,
    side: int,
    n: int = 9,
) -> np.ndarray:
    """Return (N,2) lon/lat samples on an n×n grid inside the LR window (pixel centers)."""
    n = max(2, int(n))
    # Sample at pixel centers spanning the window.
    rows = row0 + (np.arange(n) + 0.5) * (side / n)
    cols = col0 + (np.arange(n) + 0.5) * (side / n)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xs, ys = xy(transform, rr.ravel(), cc.ravel())
    if crs is None:
        raise ValueError("raster CRS is required to map LR windows to lon/lat")
    lons, lats = warp_xy(crs, "EPSG:4326", xs, ys)
    return np.column_stack([np.asarray(lons, float), np.asarray(lats, float)])


def land_hit_frac(paths: list[MplPath], lonlat: np.ndarray) -> float:
    """Fraction of lon/lat samples that fall on any land ring."""
    if lonlat.size == 0:
        return 0.0
    hits = 0
    for lon, lat in lonlat:
        pt = (float(lon), float(lat))
        if any(p.contains_point(pt) for p in paths):
            hits += 1
    return hits / float(len(lonlat))


def clear_count_at(
    clear_tif: Path,
    *,
    lon: float,
    lat: float,
) -> int | None:
    """Sample a clear-count GeoTIFF (one pixel per LR512 cell) at lon/lat."""
    if not clear_tif.is_file():
        return None
    with rasterio.open(clear_tif) as src:
        xs, ys = warp_xy("EPSG:4326", src.crs, [lon], [lat])
        x, y = float(xs[0]), float(ys[0])
        if not (src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top):
            return None
        val = next(src.sample([(x, y)]))[0]
        nodata = src.nodata
        if nodata is not None and val == nodata:
            return None
        return int(val)


def resolve_clear_tif(clear_counts_dir: Path, mgrs: str, side: int) -> Path:
    return Path(clear_counts_dir) / f"{mgrs}_lr{int(side)}_clear_counts.tif"


def filter_tile_specs(
    tiles: list[dict],
    *,
    transform,
    crs,
    side: int,
    land_paths: list[MplPath] | None = None,
    min_land_frac: float = 0.0,
    sample_n: int = 9,
    clear_counts_dir: Path | None = None,
    min_clear: int | None = None,
    mgrs: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Return (kept, dropped, stats). Mutates nothing; adds filter fields on kept copies."""
    kept: list[dict] = []
    dropped: list[dict] = []
    clear_tif = None
    if clear_counts_dir is not None and min_clear is not None:
        if not mgrs:
            raise ValueError("mgrs tile id required when filtering by clear counts")
        clear_tif = resolve_clear_tif(clear_counts_dir, mgrs, side)

    for tile in tiles:
        row0 = int(tile["row_off"])
        col0 = int(tile["col_off"])
        tile_side = int(tile.get("side", side))
        samples = window_lonlat_samples(
            transform, crs, row0=row0, col0=col0, side=tile_side, n=sample_n
        )
        frac = land_hit_frac(land_paths, samples) if land_paths is not None else 1.0
        # min_land_frac == 0 → any positive overlap (at least one sample).
        land_ok = True if land_paths is None else (
            frac > 0.0 if min_land_frac <= 0.0 else frac >= min_land_frac
        )

        clear_n = None
        clear_ok = True
        if clear_tif is not None and min_clear is not None:
            # Sample cell center.
            mid = samples[len(samples) // 2]
            clear_n = clear_count_at(clear_tif, lon=float(mid[0]), lat=float(mid[1]))
            clear_ok = clear_n is not None and clear_n >= int(min_clear)

        row = dict(tile)
        row["land_frac"] = round(float(frac), 4)
        if clear_n is not None:
            row["clear_count"] = int(clear_n)

        if land_ok and clear_ok:
            kept.append(row)
        else:
            reason = []
            if not land_ok:
                reason.append(f"land_frac={frac:.3f}<{max(min_land_frac, 1e-12):g}")
            if not clear_ok:
                reason.append(f"clear={clear_n}<{min_clear}")
            row["drop_reason"] = ",".join(reason)
            dropped.append(row)

    stats = {
        "n_in": len(tiles),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "min_land_frac": float(min_land_frac),
        "sample_n": int(sample_n),
        "min_clear": None if min_clear is None else int(min_clear),
        "clear_tif": None if clear_tif is None else str(clear_tif),
        "mgrs": mgrs,
    }
    return kept, dropped, stats


def raster_georef(first_tif: Path) -> tuple[object, object]:
    with rasterio.open(first_tif) as src:
        return src.transform, src.crs


def filter_manifest_dict(
    man: dict,
    *,
    src_dir: Path,
    land_mask: Path | None,
    min_land_frac: float,
    sample_n: int,
    clear_counts_dir: Path | None,
    min_clear: int | None,
) -> tuple[dict, dict]:
    """Filter ``man['tiles']`` in place-ish; returns (new_manifest, stats)."""
    meta = json.loads((src_dir / "meta.json").read_text())
    frames = meta.get("frames") or []
    if not frames:
        raise SystemExit(f"{src_dir}/meta.json has no frames")
    first = src_dir / frames[0]["path"]
    transform, crs = raster_georef(first)
    side = int(man.get("side") or meta.get("lr_size_request") or 512)
    mgrs = meta.get("mgrs_tile") or frames[0].get("mgrs_tile")

    land_paths = load_land_paths(land_mask) if land_mask is not None else None
    kept, dropped, stats = filter_tile_specs(
        list(man.get("tiles") or []),
        transform=transform,
        crs=crs,
        side=side,
        land_paths=land_paths,
        min_land_frac=min_land_frac,
        sample_n=sample_n,
        clear_counts_dir=clear_counts_dir,
        min_clear=min_clear,
        mgrs=mgrs,
    )
    out = dict(man)
    out["tiles"] = kept
    out["grid"] = dict(man.get("grid") or {})
    out["grid"]["n_tiles"] = len(kept)
    out["grid"]["n_tiles_before_filter"] = stats["n_in"]
    out["land_filter"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "land_mask": None if land_mask is None else str(land_mask),
        "dropped": [
            {
                "tile_id": t.get("tile_id"),
                "iy": t.get("iy"),
                "ix": t.get("ix"),
                "land_frac": t.get("land_frac"),
                "clear_count": t.get("clear_count"),
                "reason": t.get("drop_reason"),
            }
            for t in dropped
        ],
        **stats,
    }
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Filtered manifest path (default: overwrite --manifest).",
    )
    ap.add_argument(
        "--land-mask",
        type=Path,
        default=DEFAULT_LAND_MASK,
        help="GeoJSON outline (NE50 mainland+islands). Pass empty to disable.",
    )
    ap.add_argument(
        "--no-land-mask",
        action="store_true",
        help="Disable land overlap filter (clear-count only).",
    )
    ap.add_argument(
        "--min-land-frac",
        type=float,
        default=0.0,
        help="Min fraction of window samples on land (0 = any overlap).",
    )
    ap.add_argument("--sample-n", type=int, default=9, help="n×n samples per LR window.")
    ap.add_argument(
        "--clear-counts-dir",
        type=Path,
        default=None,
        help="Dir with {MGRS}_lr{side}_clear_counts.tif (optional).",
    )
    ap.add_argument(
        "--min-clear",
        type=int,
        default=None,
        help="Drop cells with clear-count < this (requires --clear-counts-dir).",
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Parent revisit dir (default: manifest['src']).",
    )
    args = ap.parse_args()

    man_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    man = json.loads(man_path.read_text())
    src = args.src
    if src is None:
        src = Path(man["src"])
    src = src if src.is_absolute() else ROOT / src

    land_mask = None if args.no_land_mask else (
        args.land_mask if args.land_mask.is_absolute() else ROOT / args.land_mask
    )
    clear_dir = None
    if args.clear_counts_dir is not None:
        clear_dir = (
            args.clear_counts_dir
            if args.clear_counts_dir.is_absolute()
            else ROOT / args.clear_counts_dir
        )
    if args.min_clear is not None and clear_dir is None:
        raise SystemExit("--min-clear requires --clear-counts-dir")

    out_man, stats = filter_manifest_dict(
        man,
        src_dir=src,
        land_mask=land_mask,
        min_land_frac=float(args.min_land_frac),
        sample_n=int(args.sample_n),
        clear_counts_dir=clear_dir,
        min_clear=args.min_clear,
    )
    out_path = args.out or man_path
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_man, indent=2) + "\n")
    print(
        f"land-mask filter: kept {stats['n_kept']}/{stats['n_in']} "
        f"(dropped {stats['n_dropped']}) → {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
