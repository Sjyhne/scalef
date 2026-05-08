#!/usr/bin/env python3
"""
Download Sentinel-2 L2A imagery for a large area (bbox) as a timeseries.

Uses the Element84 Earth Search STAC API and streams COG data (no full product
download). Saves RGB (TCI) and/or multiband GeoTIFFs per scene, with optional
cloud/snow filtering via the Scene Classification Layer (SCL). With
``--write-raw-b432``, also writes ``geotiff_raw/<same_basename>.tif`` as three
float32 bands (B4, B3, B2) BOA reflectance in ~0–1 for SR / post-hoc RGB stretch.

SCL cloud/snow percentages are computed over **valid** SCL pixels (class != 0),
not over nodata edges. Optional ``--min-scl-coverage`` rejects windows that are
mostly nodata. TCI PNGs are written as explicit uint8 RGB with dtype-aware scaling.
STAC items are sorted by acquisition time before ``--max-scenes`` is applied.

Requirements (install if needed):
  pip install rasterio pystac-client pillow

Usage:
  python download_s2_timeseries.py --bbox 10.0 59.0 11.0 60.0 --start 2024-01-01 --end 2024-06-30 --out ./s2_timeseries
  python download_s2_timeseries.py --center 10.5 59.5 --size-deg 0.5 --start 2024-01-01 --end 2024-12-31 --out ./s2 --format geotiff --max-scenes 20
  python download_s2_timeseries.py --center 10.5 59.5 --size-deg 0.25 --tile 32VNM --start 2024-06-01 --end 2024-09-01 -o ./s2_single_tile   # one tile only (square-ish crop)
  python download_s2_timeseries.py ... --stac-url https://planetarycomputer.microsoft.com/api/stac/v1   # alternate STAC source
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

# GDAL/rasterio env for COG streaming (Element84, AWS)
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif,.JP2,.jp2")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")

import rasterio
from PIL import Image
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# SCL class values (Sentinel-2 L2A Scene Classification Layer)
SCL_SNOW = 11
SCL_CLOUD_SHADOW = 2
SCL_CLOUD_LOW = 7
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_CIRRUS = 10
SCL_CLOUD_CLASSES = {SCL_CLOUD_SHADOW, SCL_CLOUD_LOW, SCL_CLOUD_MEDIUM, SCL_CLOUD_HIGH, SCL_CIRRUS}
# ESA SCL: 0 = NO_DATA — exclude from cloud/snow denominators so edge nodata does not dilute scores.
SCL_NO_DATA = 0

DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"


def bbox_from_center(
    center_lon: float,
    center_lat: float,
    size_deg: float,
    square_km: bool = False,
) -> tuple[float, float, float, float]:
    """
    Return (min_lon, min_lat, max_lon, max_lat) around center.

    If square_km is False, size_deg is the half-size in degrees for both axes (box is
    square in degrees, but not in ground distance at high latitudes).
    If square_km is True, size_deg is the half-size in latitude degrees; longitude
    half-size is set so that at center_lat the ground distance is equal (square in km).
    """
    half_lat = size_deg / 2.0
    if square_km:
        # At center_lat, 1° lat ≈ 111 km, 1° lon ≈ 111*cos(lat) km. For equal N-S and E-W
        # distance we need half_lon_deg = half_lat_deg / cos(center_lat).
        lat_rad = math.radians(center_lat)
        cos_lat = max(math.cos(lat_rad), 0.01)  # avoid div by zero near poles
        half_lon = half_lat / cos_lat
    else:
        half_lon = half_lat
    return (
        center_lon - half_lon,
        center_lat - half_lat,
        center_lon + half_lon,
        center_lat + half_lat,
    )


def search_sentinel2(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    limit: int | None = None,
    stac_url: str = DEFAULT_STAC_URL,
    collection: str = DEFAULT_COLLECTION,
    tile: str | None = None,
):
    """Search STAC for Sentinel-2 L2A items overlapping bbox (min_lon, min_lat, max_lon, max_lat)."""
    catalog = Client.open(stac_url)
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
    )
    items = list(search.items())
    if tile is not None:
        tile_upper = tile.strip().upper()

        def get_item_tile(item) -> str | None:
            # STAC extension properties
            t = item.properties.get("s2:mgrs_tile")
            if t and isinstance(t, str):
                return t.upper()
            # MGRS extension: utm_zone + latitude_band + grid_square (e.g. 32 + V + NM -> 32VNM)
            uz = item.properties.get("mgrs:utm_zone")
            lb = item.properties.get("mgrs:latitude_band")
            gs = item.properties.get("mgrs:grid_square")
            if uz is not None and lb is not None and gs is not None:
                return f"{uz}{lb}{gs}".upper()
            # Parse from item id: S2A_30TVT_... or S2B_MSIL2A_..._T32VNM_... or ..._32VNM_...
            parts = item.id.replace("-", "_").split("_")
            for p in parts:
                if not p:
                    continue
                p = p.upper()
                if p == tile_upper:
                    return p
                if p.startswith("T") and len(p) == 6 and p[1:].isalnum():
                    return p[1:]
                # 5-char MGRS tile: 2 digits + 3 letters (e.g. 32VNM, 30TVT)
                if len(p) == 5 and p[:2].isdigit() and p[2:].isalpha():
                    return p
            return None

        original_items = items
        filtered = [it for it in items if get_item_tile(it) == tile_upper]
        items = filtered
        if not items and limit is None and original_items:
            logger.warning("No items found for tile %s; try without --tile", tile)
            sample = original_items[0]
            logger.info("Sample STAC item id: %s", sample.id)
            logger.info("Extracted tile from sample: %s", get_item_tile(sample))
            # Log any property that might contain tile info
            tile_keys = [k for k in sample.properties if "tile" in k.lower() or "mgrs" in k.lower() or "s2:" in k.lower()]
            if tile_keys:
                logger.info("Relevant properties: %s", {k: sample.properties[k] for k in tile_keys})

    def _item_datetime_key(it) -> str:
        return (it.properties.get("datetime") or "")[:32]

    items = sorted(items, key=_item_datetime_key)
    if limit is not None:
        items = items[:limit]
    return items


def resolve_b432_href_urls(assets) -> tuple[str, str, str] | None:
    """Return (B04, B03, B02) COG hrefs if present on the STAC item."""

    def href(key: str) -> str | None:
        a = assets.get(key)
        return a.href if a is not None else None

    for k4, k3, k2 in (
        ("B04", "B03", "B02"),
        ("b04", "b03", "b02"),
        # Element84 Earth Search Sentinel-2 L2A COG assets use color names.
        ("red", "green", "blue"),
        ("red-jp2", "green-jp2", "blue-jp2"),
    ):
        u4, u3, u2 = href(k4), href(k3), href(k2)
        if u4 and u3 and u2:
            return u4, u3, u2
    return None


def _chw_plane_to_2d(plane: np.ndarray) -> np.ndarray:
    a = np.asarray(plane)
    if a.ndim == 3:
        return a[0]
    return a


def read_b432_reflectance_chw(
    b04_url: str,
    b03_url: str,
    b02_url: str,
    bbox_wgs84: tuple[float, float, float, float],
) -> tuple[np.ndarray, object, object]:
    """
    Read B4,B3,B2 for the bbox window, return (CHW float32 reflectance ~0–1, transform, crs).

    Uses the same windowing as ``read_cog_bbox``; all three bands must match in shape.
    """
    d4, tf4, crs4 = read_cog_bbox(b04_url, bbox_wgs84)
    d3, tf3, crs3 = read_cog_bbox(b03_url, bbox_wgs84)
    d2, tf2, crs2 = read_cog_bbox(b02_url, bbox_wgs84)
    b4 = _chw_plane_to_2d(d4).astype(np.float32)
    b3 = _chw_plane_to_2d(d3).astype(np.float32)
    b2 = _chw_plane_to_2d(d2).astype(np.float32)
    if b4.shape != b3.shape or b4.shape != b2.shape:
        raise ValueError(
            f"B04/B03/B02 shape mismatch in window: {b4.shape}, {b3.shape}, {b2.shape}"
        )
    chw = np.stack([b4, b3, b2], axis=0)
    chw = np.nan_to_num(chw, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(np.nanmax(chw)) if chw.size else 0.0
    if mx > 1.5:
        chw = chw / 10000.0
    chw = np.clip(chw, 0.0, None)
    # Prefer B04 georeference as reference
    return chw, tf4, crs4


def read_cog_bbox(cog_url: str, bbox_wgs84: tuple[float, float, float, float]):
    """
    Read the raster window that covers bbox_wgs84 (min_lon, min_lat, max_lon, max_lat).
    Returns (data array, transform for the window, crs).
    """
    with rasterio.open(cog_url) as src:
        src_crs = src.crs
        # Transform bbox from WGS84 to source CRS
        bounds_src = transform_bounds(
            "EPSG:4326",
            src_crs,
            bbox_wgs84[0],
            bbox_wgs84[1],
            bbox_wgs84[2],
            bbox_wgs84[3],
        )
        try:
            window = from_bounds(*bounds_src, src.transform)
        except Exception as e:
            raise ValueError(
                f"from_bounds failed for {cog_url!r} bounds_src={bounds_src!r} crs={src_crs!r}: {e}"
            ) from e
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width <= 0 or window.height <= 0:
            raise ValueError("bbox does not intersect the raster")
        data = src.read(window=window)
        transform = rasterio.windows.transform(window, src.transform)
        return data, transform, src_crs


def scl_cloud_snow_ratio(scl_band: np.ndarray) -> tuple[float, float, float]:
    """
    Cloud / snow / valid coverage from SCL (2D single band or first plane of CHW).

    Cloud and snow percentages are among **valid** SCL pixels only (class != NO_DATA),
    so nodata at swath edges does not deflate cloud scores.

    Returns:
        cloud_pct: % of valid pixels that are cloud (any SCL_CLOUD_CLASSES)
        snow_pct: % of valid pixels that are snow
        valid_coverage_pct: % of all raster pixels in the window with SCL != NO_DATA
    """
    scl = np.asarray(scl_band)
    if scl.ndim == 3:
        scl = scl[0]
    flat = scl.ravel()
    total = int(flat.size)
    if total == 0:
        return 0.0, 0.0, 0.0
    valid_mask = flat != SCL_NO_DATA
    valid = int(np.count_nonzero(valid_mask))
    valid_coverage_pct = 100.0 * valid / float(total)
    if valid == 0:
        return 0.0, 0.0, 0.0
    vv = flat[valid_mask]
    cloud = int(np.sum(np.isin(vv, list(SCL_CLOUD_CLASSES))))
    snow = int(np.sum(vv == SCL_SNOW))
    return 100.0 * cloud / valid, 100.0 * snow / valid, valid_coverage_pct


def visual_chw_to_uint8_hwc(visual_data: np.ndarray) -> np.ndarray:
    """
    Convert (C,H,W) RGB (+ optional alpha) to (H,W,3) uint8 for PNG.

    Handles typical STAC ``visual`` assets (uint8) and integer/float reflectance-style
    rasters without relying on implicit Pillow behavior.
    """
    if visual_data.ndim != 3 or visual_data.shape[0] < 3:
        raise ValueError(f"expected CHW with C>=3, got shape {visual_data.shape}")
    raw = np.ascontiguousarray(visual_data[:3])

    if np.issubdtype(raw.dtype, np.floating):
        rgb = np.nan_to_num(raw.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        mx = float(np.max(rgb)) if rgb.size else 0.0
        if mx <= 1.01:
            out_chw = (np.clip(rgb, 0.0, 1.0) * 255.0).round()
        elif mx <= 350.0:
            out_chw = np.clip(rgb, 0.0, 255.0).round()
        else:
            out_chw = (np.clip(rgb / 10000.0, 0.0, 1.0) * 255.0).round()
        out_chw = np.clip(out_chw, 0, 255).astype(np.uint8)
        logger.debug("PNG conversion: floating dtype=%s max_in=%.4f -> uint8", raw.dtype, mx)
    else:
        rgb = raw.astype(np.float64)
        mx = float(np.max(rgb)) if rgb.size else 0.0
        if mx <= 255.0:
            out_chw = np.clip(rgb, 0, 255).astype(np.uint8)
            logger.debug("PNG conversion: integer dtype=%s max=%.1f -> direct uint8", raw.dtype, mx)
        else:
            p995 = float(np.quantile(rgb, 0.995)) if rgb.size else 255.0
            denom = max(p995, 1.0)
            out_chw = (np.clip(rgb / denom, 0, 1) * 255.0).round().astype(np.uint8)
            logger.debug(
                "PNG conversion: integer dtype=%s max=%.1f p99.5=%.1f -> linear stretch to uint8",
                raw.dtype,
                mx,
                p995,
            )

    return np.transpose(out_chw, (1, 2, 0))


def process_scene(
    scene,
    bbox_wgs84: tuple[float, float, float, float],
    output_dir: Path,
    cloud_max: float,
    snow_max: float,
    out_format: str,
    skip_filter: bool,
    min_scl_coverage: float,
    write_raw_b432: bool = False,
) -> dict:
    """
    Download one scene for the given bbox: read SCL + visual (and bands if geotiff),
    apply cloud/snow filter, save TCI and optionally multiband GeoTIFF.

    ``min_scl_coverage`` is a fraction in [0, 1]: require at least this share of SCL
    pixels to be non-NO_DATA in the bbox window (set to 0 to disable).
    """
    scene_date = (scene.properties.get("datetime") or "").split("T")[0]
    scene_id = scene.id
    assets = scene.assets

    def href(key: str, alt_keys: list[str] | None = None):
        if key in assets:
            return assets[key].href
        for k in alt_keys or []:
            if k in assets:
                return assets[k].href
        return None

    # SCL for filtering
    scl_url = href("scl")
    if not scl_url:
        return {
            "scene_id": scene_id,
            "date": scene_date,
            "status": "error",
            "error": "No SCL asset",
        }

    try:
        scl_data, _, _ = read_cog_bbox(scl_url, bbox_wgs84)
    except Exception as e:
        return {"scene_id": scene_id, "date": scene_date, "status": "error", "error": str(e)}

    if scl_data.size == 0:
        return {"scene_id": scene_id, "date": scene_date, "status": "error", "error": "Empty SCL"}

    if scl_data.ndim == 3:
        scl_band = scl_data[0]
    elif scl_data.ndim == 2:
        scl_band = scl_data
    else:
        return {
            "scene_id": scene_id,
            "date": scene_date,
            "status": "error",
            "error": f"Unexpected SCL shape {scl_data.shape}",
        }
    cloud_pct, snow_pct, scl_valid_coverage_pct = scl_cloud_snow_ratio(scl_band)

    if min_scl_coverage > 0 and scl_valid_coverage_pct < 100.0 * min_scl_coverage:
        return {
            "scene_id": scene_id,
            "date": scene_date,
            "status": "skipped_low_scl_coverage",
            "reason": (
                f"SCL valid coverage {scl_valid_coverage_pct:.1f}% < "
                f"{100.0 * min_scl_coverage:.1f}% (nodata / edge-heavy window)"
            ),
            "scl_valid_coverage_pct": round(scl_valid_coverage_pct, 1),
        }

    if not skip_filter:
        if snow_pct > snow_max:
            return {
                "scene_id": scene_id,
                "date": scene_date,
                "status": "skipped_snow",
                "reason": f"snow {snow_pct:.1f}% > {snow_max:.1f}%",
            }
        if cloud_pct > cloud_max:
            return {
                "scene_id": scene_id,
                "date": scene_date,
                "status": "skipped_cloud",
                "reason": f"cloud {cloud_pct:.1f}% > {cloud_max:.1f}%",
            }

    # Visual (TCI) RGB
    visual_url = href("visual")
    if not visual_url:
        return {
            "scene_id": scene_id,
            "date": scene_date,
            "status": "error",
            "error": "No visual asset",
        }

    try:
        visual_data, transform, crs = read_cog_bbox(visual_url, bbox_wgs84)
    except Exception as e:
        return {"scene_id": scene_id, "date": scene_date, "status": "error", "error": str(e)}

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{scene_date}_{scene_id}"

    if out_format in ("png", "both"):
        tci_dir = output_dir / "TCI"
        tci_dir.mkdir(exist_ok=True)
        tci_path = tci_dir / f"{base_name}.png"
        try:
            img_uint8 = visual_chw_to_uint8_hwc(visual_data)
        except ValueError as e:
            return {
                "scene_id": scene_id,
                "date": scene_date,
                "status": "error",
                "error": f"PNG conversion: {e}",
            }
        Image.fromarray(img_uint8, mode="RGB").save(tci_path)
        logger.info("Wrote %s", tci_path)

    if out_format in ("geotiff", "both"):
        gt_dir = output_dir / "geotiff"
        gt_dir.mkdir(exist_ok=True)
        gt_path = gt_dir / f"{base_name}.tif"
        profile = {
            "driver": "GTiff",
            "width": visual_data.shape[2],
            "height": visual_data.shape[1],
            "count": visual_data.shape[0],
            "dtype": visual_data.dtype,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }
        with rasterio.open(gt_path, "w", **profile) as dst:
            dst.write(visual_data)
        logger.info("Wrote %s", gt_path)

    if write_raw_b432:
        b432 = resolve_b432_href_urls(assets)
        if not b432:
            logger.warning("No B04/B03/B02 STAC assets for %s; skipping geotiff_raw", scene_id)
        else:
            u4, u3, u2 = b432
            try:
                chw, tf_raw, crs_raw = read_b432_reflectance_chw(u4, u3, u2, bbox_wgs84)
                raw_dir = output_dir / "geotiff_raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"{base_name}.tif"
                raw_profile = {
                    "driver": "GTiff",
                    "width": chw.shape[2],
                    "height": chw.shape[1],
                    "count": 3,
                    "dtype": "float32",
                    "crs": crs_raw,
                    "transform": tf_raw,
                    "compress": "lzw",
                }
                with rasterio.open(raw_path, "w", **raw_profile) as dst:
                    dst.write(chw.astype(np.float32))
                    dst.set_band_description(1, "B4 BOA reflectance (0–1)")
                    dst.set_band_description(2, "B3 BOA reflectance (0–1)")
                    dst.set_band_description(3, "B2 BOA reflectance (0–1)")
                logger.info("Wrote %s", raw_path)
            except Exception as e:
                logger.warning("geotiff_raw export failed for %s: %s", scene_id, e)

    return {
        "scene_id": scene_id,
        "date": scene_date,
        "status": "processed",
        "cloud_pct": round(cloud_pct, 1),
        "snow_pct": round(snow_pct, 1),
        "scl_valid_coverage_pct": round(scl_valid_coverage_pct, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L2A timeseries for a bounding box via STAC (COG streaming).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bbox_group = parser.add_mutually_exclusive_group(required=True)
    bbox_group.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box in WGS84: min_lon min_lat max_lon max_lat",
    )
    bbox_group.add_argument(
        "--center",
        type=float,
        nargs=2,
        metavar=("LON", "LAT"),
        help="Center (lon, lat); use with --size-deg",
    )
    parser.add_argument(
        "--size-deg",
        type=float,
        default=0.1,
        help="Half-size in degrees (for --center). Total side = 2 * size_deg",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="With --center: make the box square in ground distance (km). Uses size-deg for latitude; longitude is adjusted for the center latitude so the image is square.",
    )
    parser.add_argument(
        "--start", "--start-date", dest="start_date", required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", "--end-date", dest="end_date", required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "-o",
        "--out",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("s2_timeseries"),
        help="Output directory",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process (default: all)",
    )
    parser.add_argument(
        "--cloud-max",
        type=float,
        default=20.0,
        help="Skip scene if SCL cloud %% (among valid / non-NO_DATA SCL pixels) exceeds this (0–100)",
    )
    parser.add_argument(
        "--snow-max",
        type=float,
        default=5.0,
        help="Skip scene if SCL snow %% (among valid / non-NO_DATA SCL pixels) exceeds this (0–100)",
    )
    parser.add_argument(
        "--min-scl-coverage",
        type=float,
        default=0.10,
        metavar="FRACTION",
        help=(
            "Skip scene if fewer than this fraction of SCL pixels in the bbox are non-NO_DATA "
            "(0–1; default 0.10). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Do not filter by cloud/snow; download all scenes",
    )
    parser.add_argument(
        "--format",
        choices=("png", "geotiff", "both"),
        default="both",
        help="Output format: png (TCI only), geotiff, or both",
    )
    parser.add_argument(
        "--write-raw-b432",
        action="store_true",
        help=(
            "Also write geotiff_raw/<basename>.tif: float32 CHW B4,B3,B2 BOA reflectance (~0–1), "
            "same grid as the visual GeoTIFF / TCI crop (for SR with RGB stretch after upsampling)."
        ),
    )
    parser.add_argument(
        "--stac-url",
        type=str,
        default=DEFAULT_STAC_URL,
        help="STAC API URL (e.g. https://earth-search.aws.element84.com/v1 or https://planetarycomputer.microsoft.com/api/stac/v1)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="STAC collection id (e.g. sentinel-2-l2a)",
    )
    parser.add_argument(
        "--tile",
        type=str,
        default=None,
        metavar="MGRS",
        help="Restrict to one MGRS tile (e.g. 32VNM) so all scenes have the same extent",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        bbox = bbox_from_center(
            args.center[0],
            args.center[1],
            args.size_deg,
            square_km=args.square,
        )

    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        logger.error("Invalid bbox: min must be < max (got %s)", bbox)
        sys.exit(1)

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Bbox: %.6f, %.6f, %.6f, %.6f (min_lon, min_lat, max_lon, max_lat)", *bbox)
    logger.info("Date range: %s to %s", args.start_date, args.end_date)
    logger.info("Output: %s", args.output_dir)

    items = search_sentinel2(
        bbox,
        args.start_date,
        args.end_date,
        limit=args.max_scenes,
        stac_url=args.stac_url,
        collection=args.collection,
        tile=args.tile,
    )
    if not items:
        logger.warning("No Sentinel-2 scenes found for this bbox and date range.")
        sys.exit(0)

    logger.info("Found %d scene(s), processing...", len(items))
    results = []
    for i, item in enumerate(items):
        result = process_scene(
            item,
            bbox,
            args.output_dir,
            cloud_max=args.cloud_max,
            snow_max=args.snow_max,
            out_format=args.format,
            skip_filter=args.no_filter,
            min_scl_coverage=min(max(0.0, float(args.min_scl_coverage)), 1.0),
            write_raw_b432=bool(args.write_raw_b432),
        )
        results.append(result)
        if result.get("status") == "processed":
            logger.info(
                "[%d/%d] %s %s OK (cloud=%.1f%% of valid SCL, snow=%.1f%%, SCL valid=%.1f%% of window)",
                i + 1,
                len(items),
                result.get("date"),
                result.get("scene_id"),
                result.get("cloud_pct", 0),
                result.get("snow_pct", 0),
                result.get("scl_valid_coverage_pct", 0),
            )
        else:
            logger.info(
                "[%d/%d] %s %s %s",
                i + 1,
                len(items),
                result.get("date"),
                result.get("scene_id"),
                result.get("status"),
            )

    processed = sum(1 for r in results if r.get("status") == "processed")
    logger.info("Done. Processed %d / %d scenes.", processed, len(results))


if __name__ == "__main__":
    main()
