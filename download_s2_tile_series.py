#!/usr/bin/env python3
"""Download a Sentinel-2 L2A time series for one MGRS tile.

Example (dry-run)::

    python download_s2_tile_series.py \\
        --mgrs-tile 19KDQ --center-lon -69.3 --center-lat -23.2 \\
        --start-date 2025-06-01 --end-date 2025-09-30 \\
        --dry-run

Example (smoke download)::

    python download_s2_tile_series.py \\
        --mgrs-tile 19KDQ --center-lon -69.3 --center-lat -23.2 \\
        --start-date 2025-06-01 --end-date 2025-06-30 \\
        --output data/s2_atacama_smoke \\
        --auto-crop-size 512 --max-scenes 2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from download_s2_earth_search import (
    DEFAULT_COLLECTION,
    EARTH_SEARCH_STAC_URL,
    _bbox_from_center,
    _iso_range,
    _item_mgrs_tile,
    _time_to_filename,
    _utm_epsg_from_lon_lat,
    best_valid_window,
    reference_item_for_tile,
    stack_extent_metadata,
    stack_items,
)
from s2_bands import band_manifest_dict, resolve_band_preset
from s2_preview import (
    DEFAULT_PREVIEW_DOWNSAMPLE,
    scene_stretch_limits_from_arrays,
    select_rgb_hwc,
    write_reflectance_preview_png,
)

SCL_INVALID: frozenset[int] = frozenset({0, 1, 3, 8, 9, 10, 11})


def _normalize_mgrs_tile(tile: str) -> str:
    text = str(tile).strip().upper().replace("MGRS-", "")
    if not text:
        raise ValueError("MGRS tile must not be empty")
    return text


def search_tile_items(
    mgrs_tile: str,
    *,
    datetime_range: str,
    max_cloud_cover: float,
    bbox: tuple[float, float, float, float] | None = None,
    max_items: int | None = None,
):
    import pystac_client

    tile = _normalize_mgrs_tile(mgrs_tile)
    catalog = pystac_client.Client.open(EARTH_SEARCH_STAC_URL)
    kwargs: dict = {
        "collections": [DEFAULT_COLLECTION],
        "datetime": datetime_range,
        "query": {"eo:cloud_cover": {"lt": float(max_cloud_cover)}},
        "max_items": 500,
    }
    if bbox is not None:
        kwargs["bbox"] = list(bbox)
    items = [
        it
        for it in catalog.search(**kwargs).items()
        if (_item_mgrs_tile(it) or "").upper().replace("MGRS-", "") == tile
    ]
    items.sort(key=lambda it: it.properties.get("datetime", ""))
    if max_items is not None and max_items > 0:
        items = items[: int(max_items)]
    return items


def _day_number(dt_text: str) -> float:
    text = str(dt_text).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = datetime.fromisoformat(text.split(".")[0])
    return dt.timestamp() / 86400.0


def _asset_order_indices(band_coord: list[str], assets: tuple[str, ...]) -> list[int]:
    lookup = {str(name): i for i, name in enumerate(band_coord)}
    missing = [a for a in assets if a not in lookup]
    if missing:
        raise RuntimeError(f"Stack missing assets {missing}; have {band_coord}")
    return [lookup[a] for a in assets]


def arrays_from_stack(computed, reflectance_assets: tuple[str, ...]):
    band_coord = [str(b) for b in computed.coords["band"].values]
    refl_idx = _asset_order_indices(band_coord, reflectance_assets)
    scl_idx = band_coord.index("scl")
    values = np.asarray(computed.values, dtype=np.float32)
    refl_tchw = values[:, refl_idx, :, :]
    scl_plane = values[:, scl_idx, :, :]
    scl_thw = np.asarray(scl_plane, dtype=np.float32)
    refl_thwc = np.transpose(refl_tchw, (0, 2, 3, 1))
    return refl_thwc, scl_thw


def choose_auto_crop(
    nodata_thw: np.ndarray,
    cloud_thw: np.ndarray,
    *,
    crop_size: int,
    coverage_gate_pct: float,
) -> dict | None:
    if crop_size <= 0:
        return None
    worst = np.maximum(nodata_thw.astype(np.float32), cloud_thw.astype(np.float32)).max(axis=0)
    h, w = worst.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"auto-crop-size {crop_size} exceeds stack shape {(h, w)}")
    row0, col0, score = best_valid_window(worst, crop_size, crop_size)
    invalid_pct = 100.0 * float(score) / float(crop_size * crop_size)
    if invalid_pct > float(coverage_gate_pct):
        raise RuntimeError(
            f"Best {crop_size}px window still has {invalid_pct:.2f}% invalid "
            f"(gate {coverage_gate_pct:.1f}%)"
        )
    return {
        "applied": True,
        "crop_size_px": int(crop_size),
        "row0": int(row0),
        "col0": int(col0),
        "height": int(crop_size),
        "width": int(crop_size),
        "worst_invalid_pct": float(invalid_pct),
    }


def _crop_hw(arr: np.ndarray, row0: int, col0: int, size: int) -> np.ndarray:
    ys = slice(row0, row0 + size)
    xs = slice(col0, col0 + size)
    if arr.ndim == 2:
        return arr[ys, xs]
    if arr.ndim == 3:
        return arr[:, ys, xs]
    if arr.ndim == 4:
        return arr[:, ys, xs, :]
    raise ValueError(f"Unsupported array rank {arr.ndim}")


def fill_nodata_temporal(
    reflectance_thwc: np.ndarray,
    nodata_thw: np.ndarray,
    datetimes: list[str],
    *,
    max_gap_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    filled = reflectance_thwc.copy()
    provenance = np.full(nodata_thw.shape, -1, dtype=np.int16)
    days = np.asarray([_day_number(dt) for dt in datetimes], dtype=np.float64)
    t_count = int(reflectance_thwc.shape[0])
    for t in range(t_count):
        need = nodata_thw[t].copy()
        if not need.any():
            continue
        for dt in range(1, t_count):
            for sign in (1, -1):
                t2 = t + sign * dt
                if t2 < 0 or t2 >= t_count:
                    continue
                if abs(days[t2] - days[t]) > float(max_gap_days):
                    continue
                src_ok = need & ~nodata_thw[t2]
                if not src_ok.any():
                    continue
                filled[t, src_ok] = reflectance_thwc[t2, src_ok]
                provenance[t, src_ok] = np.int16(t2)
                need &= ~src_ok
            if not need.any():
                break
    return filled, provenance


def export_scene(
    out_dir: Path,
    *,
    index: int,
    item,
    reflectance_hwc: np.ndarray,
    valid_hw: np.ndarray,
    fill_provenance_hw: np.ndarray,
    preview_path: Path,
    preview_lo: np.ndarray | None,
    preview_hi: np.ndarray | None,
    band_names: tuple[str, ...],
    preview_downsample: int,
) -> dict:
    dt = str(item.properties.get("datetime", ""))
    stem = f"{_time_to_filename(dt)}_{item.id}"
    npz_name = f"{stem}_reflectance.npz"
    valid_name = f"{stem}_valid.npy"
    prov_name = f"{stem}_fill_provenance.npy"

    np.savez_compressed(
        out_dir / npz_name,
        reflectance_hwc=reflectance_hwc.astype(np.float32),
        band_names=np.asarray(band_names, dtype="U4"),
    )
    np.save(out_dir / valid_name, valid_hw.astype(np.uint8))
    np.save(out_dir / prov_name, fill_provenance_hw.astype(np.int16))
    write_reflectance_preview_png(
        reflectance_hwc,
        preview_path,
        band_names=band_names,
        valid_hw=valid_hw,
        downsample=preview_downsample,
        lo=preview_lo,
        hi=preview_hi,
    )

    valid_pct = 100.0 * float(np.mean(valid_hw))
    return {
        "index": int(index),
        "stac_id": item.id,
        "datetime": dt,
        "eo_cloud_cover": float(item.properties.get("eo:cloud_cover", np.nan)),
        "mgrs_tile": f"MGRS-{_normalize_mgrs_tile(_item_mgrs_tile(item) or '')}",
        "reflectance_npz": npz_name,
        "valid_npy": valid_name,
        "fill_provenance_npy": prov_name,
        "preview_png": preview_path.name,
        "shape_hwc": [int(reflectance_hwc.shape[0]), int(reflectance_hwc.shape[1]), int(reflectance_hwc.shape[2])],
        "valid_pct": valid_pct,
    }


def run_download(args: argparse.Namespace) -> Path:
    preset = resolve_band_preset(args.band_preset)
    reflectance_assets = preset.earth_search_assets()
    datetime_range = _iso_range(args.start_date, args.end_date)

    ref_item, tile_bbox, detected_tile = reference_item_for_tile(
        args.center_lon,
        args.center_lat,
        datetime_range=datetime_range,
        collection=DEFAULT_COLLECTION,
    )
    mgrs_tile = _normalize_mgrs_tile(args.mgrs_tile or detected_tile or _item_mgrs_tile(ref_item) or "")
    if not mgrs_tile:
        raise RuntimeError("Could not determine MGRS tile; pass --mgrs-tile")

    crop_size = int(args.auto_crop_size)
    if crop_size > 0:
        half_km = (crop_size * float(args.resolution_m)) / 1000.0 * float(args.bbox_margin)
        stack_bbox = _bbox_from_center(args.center_lon, args.center_lat, half_km)
        aoi_mode = f"center_bbox_crop{crop_size}"
    else:
        stack_bbox = tile_bbox
        aoi_mode = "full_mgrs_tile"

    items = search_tile_items(
        mgrs_tile,
        datetime_range=datetime_range,
        max_cloud_cover=args.max_cloud_cover,
        bbox=stack_bbox,
        max_items=args.max_scenes,
    )
    if not items:
        raise RuntimeError(
            f"No {DEFAULT_COLLECTION} items for tile {mgrs_tile} in {datetime_range}"
        )

    epsg = int(args.epsg) if args.epsg else _utm_epsg_from_lon_lat(args.center_lon, args.center_lat)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tile {mgrs_tile} | EPSG:{epsg} | {len(items)} scene(s) | aoi={aoi_mode}")

    stack = stack_items(
        items,
        assets=tuple(reflectance_assets) + ("scl",),
        epsg=epsg,
        resolution_m=float(args.resolution_m),
        bbox=stack_bbox,
    )
    computed = stack.compute()
    refl_thwc, scl_thw = arrays_from_stack(computed, reflectance_assets)

    datetimes = [str(it.properties.get("datetime", "")) for it in items]
    nodata_thw = np.any(np.isnan(refl_thwc), axis=-1)
    cloud_thw = np.isnan(scl_thw) | np.isin(
        np.nan_to_num(scl_thw, nan=-1).astype(np.int16), list(SCL_INVALID)
    )

    crop_meta = choose_auto_crop(
        nodata_thw,
        cloud_thw,
        crop_size=crop_size,
        coverage_gate_pct=float(args.coverage_gate_pct),
    )
    if crop_meta is not None:
        size = int(crop_meta["crop_size_px"])
        row0, col0 = int(crop_meta["row0"]), int(crop_meta["col0"])
        refl_thwc = _crop_hw(refl_thwc, row0, col0, size)
        scl_thw = _crop_hw(scl_thw, row0, col0, size)
        nodata_thw = _crop_hw(nodata_thw, row0, col0, size)
        cloud_thw = _crop_hw(cloud_thw, row0, col0, size)

    if args.max_aoi_nodata_pct is not None:
        scene_nodata = [100.0 * float(np.mean(nodata_thw[t])) for t in range(nodata_thw.shape[0])]
        keep = [t for t, frac in enumerate(scene_nodata) if frac <= float(args.max_aoi_nodata_pct)]
        if not keep:
            raise RuntimeError(
                f"No scenes with AOI nodata <= {args.max_aoi_nodata_pct:.3f}% "
                f"(best was {min(scene_nodata):.3f}%)."
            )
        items = [items[t] for t in keep]
        datetimes = [datetimes[t] for t in keep]
        refl_thwc = refl_thwc[keep]
        nodata_thw = nodata_thw[keep]
        cloud_thw = cloud_thw[keep]
        computed = computed.isel(time=keep)
        print(f"  nodata filter kept {len(items)} / {len(scene_nodata)} scenes")

    filled, provenance = fill_nodata_temporal(
        refl_thwc,
        nodata_thw,
        datetimes,
        max_gap_days=float(args.fill_max_days),
    )
    still_nodata = np.any(np.isnan(filled), axis=-1)
    valid_thw = ~(cloud_thw | still_nodata)

    georef_stack = computed
    if crop_meta is not None:
        size = int(crop_meta["crop_size_px"])
        row0, col0 = int(crop_meta["row0"]), int(crop_meta["col0"])
        georef_stack = computed.isel(y=slice(row0, row0 + size), x=slice(col0, col0 + size))
    georef = stack_extent_metadata(georef_stack, epsg=epsg)

    rgb_list = [select_rgb_hwc(filled[t], preset.band_names) for t in range(filled.shape[0])]
    valid_list = [valid_thw[t] for t in range(filled.shape[0])]
    preview_lo, preview_hi = scene_stretch_limits_from_arrays(
        rgb_list,
        valid_list,
        downsample=int(args.preview_downsample),
    )

    scene_entries: list[dict] = []
    for t, item in enumerate(items):
        stem = f"{_time_to_filename(datetimes[t])}_{item.id}"
        entry = export_scene(
            out_dir,
            index=t,
            item=item,
            reflectance_hwc=filled[t],
            valid_hw=valid_thw[t],
            fill_provenance_hw=provenance[t],
            preview_path=out_dir / f"{stem}_preview.png",
            preview_lo=preview_lo,
            preview_hi=preview_hi,
            band_names=preset.band_names,
            preview_downsample=int(args.preview_downsample),
        )
        entry["aoi_nodata_pct"] = float(100.0 * np.mean(nodata_thw[t]))
        entry["aoi_cloud_pct"] = float(100.0 * np.mean(cloud_thw[t]))
        scene_entries.append(entry)
        print(f"  [{t + 1}/{len(items)}] {entry['preview_png']}")

    manifest = {
        "source": "element84-earth-search",
        "stac_url": EARTH_SEARCH_STAC_URL,
        "collection": DEFAULT_COLLECTION,
        "mgrs_tile": mgrs_tile,
        "center_wgs84": [float(args.center_lon), float(args.center_lat)],
        "bbox_wgs84": list(stack_bbox),
        "tile_bbox_wgs84": list(tile_bbox),
        "aoi_mode": aoi_mode,
        "epsg": epsg,
        "resolution_m": float(args.resolution_m),
        "datetime_range": datetime_range,
        "band_preset": preset.name,
        "band_names": list(preset.band_names),
        "assets": list(reflectance_assets) + ["scl"],
        "max_cloud_cover": float(args.max_cloud_cover),
        "max_aoi_nodata_pct": args.max_aoi_nodata_pct,
        "auto_valid_crop": crop_meta,
        "num_scenes": len(scene_entries),
        "georef": georef,
        "scenes": scene_entries,
    }
    compositing = {
        "scl_invalid_classes": sorted(SCL_INVALID),
        "nodata_fill": {
            "method": "nearest_clear_date",
            "max_gap_days": float(args.fill_max_days),
            "fills_cloud": False,
        },
        "preview": {
            "downsample": int(args.preview_downsample),
            "stretch": "scene_per_channel_percentile",
            "percentile_low": 2.0,
            "percentile_high": 98.0,
            "masked": True,
        },
        "fill_pixels_per_scene": [
            int(np.sum(provenance[t] >= 0)) for t in range(provenance.shape[0])
        ],
    }
    with (out_dir / "stac_download_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    with (out_dir / "compositing_meta.json").open("w") as f:
        json.dump(compositing, f, indent=2)
    with (out_dir / "band_manifest.json").open("w") as f:
        json.dump(band_manifest_dict(preset.band_names), f, indent=2)

    print(f"Done → {out_dir} ({len(scene_entries)} scenes)")
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mgrs-tile", type=str, default=None)
    p.add_argument("--center-lon", type=float, required=True)
    p.add_argument("--center-lat", type=float, required=True)
    p.add_argument("--start-date", type=str, required=True)
    p.add_argument("--end-date", type=str, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--band-preset", type=str, default="rgb_nir")
    p.add_argument("--epsg", type=int, default=None)
    p.add_argument("--resolution-m", type=float, default=10.0)
    p.add_argument("--max-cloud-cover", type=float, default=10.0)
    p.add_argument("--max-aoi-nodata-pct", type=float, default=0.1)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--auto-crop-size", type=int, default=0)
    p.add_argument("--coverage-gate-pct", type=float, default=20.0)
    p.add_argument("--bbox-margin", type=float, default=1.25)
    p.add_argument("--fill-max-days", type=float, default=21.0)
    p.add_argument("--preview-downsample", type=int, default=DEFAULT_PREVIEW_DOWNSAMPLE)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        datetime_range = _iso_range(args.start_date, args.end_date)
        _, tile_bbox, detected_tile = reference_item_for_tile(
            args.center_lon,
            args.center_lat,
            datetime_range=datetime_range,
        )
        tile = _normalize_mgrs_tile(args.mgrs_tile or detected_tile or "")
        items = search_tile_items(
            tile,
            datetime_range=datetime_range,
            max_cloud_cover=args.max_cloud_cover,
            bbox=tile_bbox,
            max_items=args.max_scenes,
        )
        print(f"Found {len(items)} items for {tile} in {datetime_range}")
        for it in items:
            print(
                f"  {it.id}  {it.properties.get('datetime')}  "
                f"cloud={it.properties.get('eo:cloud_cover')}"
            )
        return
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run")
    run_download(args)


if __name__ == "__main__":
    main()
