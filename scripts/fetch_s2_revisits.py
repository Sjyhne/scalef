#!/usr/bin/env python3
"""Download Sentinel-2 L2A revisits as full MGRS 10 m tiles.

lon/lat (or bbox) + --size-km define the study AOI: STAC query, cloud filter, and
the default crop window stored in meta.json. The GeoTIFF itself is the native
MGRS granule (typically 10980×10980 at 10 m) so later training can window any
patch from that tile.

Cloud filtering uses the AOI window, not scene-level STAC cloud cover:

  omnicloudmask  L2A Red/Green/NIR  (default)
  s2cloudless    L1C 10-band detector (B10 requires L1C; L2A is still what we save)

  pip install numpy rasterio matplotlib pystac-client planetary-computer
  pip install omnicloudmask          # --cloud-method omnicloudmask
  pip install s2cloudless            # --cloud-method s2cloudless

  python scripts/fetch_s2_revisits.py \\
    --date 2019-07-15 --lon -76.53 --lat 37.41 --size-km 1.28 \\
    --num-samples 8 --cloud-method omnicloudmask --out data/s2_revisits/demo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import transform_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
L2A = "sentinel-2-l2a"
L1C = "sentinel-2-l1c"

# Saved product: RGBNIR at 10 m.
BANDS_L2A = ["B04", "B03", "B02", "B08"]
# OmniCloudMask: Red, Green, NIR.
OCM_BANDS = ["B04", "B03", "B08"]
# s2cloudless all_bands=False order (reflectance 0–1). B10 is L1C-only.
S2CL_BANDS = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]

S2_ID_RE = re.compile(
    r"(S2[ABC]).*?_(\d{8}T\d{6})_R(\d{3})_T([0-9A-Z]{5})",
    re.IGNORECASE,
)

# OmniCloudMask class ids.
OCM_CLEAR, OCM_THICK, OCM_THIN, OCM_SHADOW = 0, 1, 2, 3


def _yyyy_mm_dd(value: str, *, label: str = "date") -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise argparse.ArgumentTypeError(f"Invalid {label} {value!r}; expected YYYY-MM-DD")
    datetime.strptime(text, "%Y-%m-%d")
    return text


def _bbox_from_center(lon: float, lat: float, half_size_km: float) -> list[float]:
    half = float(half_size_km)
    lat_rad = math.radians(float(lat))
    dlat = half / 111.32
    dlon = half / (111.32 * max(math.cos(lat_rad), 1e-6))
    return [float(lon) - dlon, float(lat) - dlat, float(lon) + dlon, float(lat) + dlat]


def date_window(center_iso: str, days_before: int, days_after: int) -> str:
    center = datetime.fromisoformat(str(center_iso)[:10]).replace(tzinfo=timezone.utc)
    start = (center - timedelta(days=int(days_before))).date().isoformat()
    end = (center + timedelta(days=int(days_after))).date().isoformat()
    return f"{start}/{end}"


def s2_key(item_id: str) -> tuple[str, str, str, str] | None:
    m = S2_ID_RE.search(item_id or "")
    if not m:
        return None
    return m.group(1).upper(), m.group(2), m.group(3), m.group(4).upper()


def item_mgrs(item) -> str | None:
    props = getattr(item, "properties", None) or {}
    tile = props.get("s2:mgrs_tile")
    if isinstance(tile, str) and tile.strip():
        return tile.strip().upper().lstrip("T")
    key = s2_key(getattr(item, "id", "") or "")
    return None if key is None else key[3]


def _gtiff_profile(src) -> dict:
    return {
        "driver": "GTiff",
        "width": int(src.width),
        "height": int(src.height),
        "count": len(BANDS_L2A),
        "dtype": "uint16",
        "crs": src.crs,
        "transform": src.transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }


def aoi_window(transform, crs, height: int, width: int, bbox_wgs84: list[float]) -> Window:
    west, south, east, north = (float(v) for v in bbox_wgs84)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, west, south, east, north)
    win = from_bounds(minx, miny, maxx, maxy, transform)
    win = win.intersection(Window(0, 0, width, height))
    if win.width <= 0 or win.height <= 0:
        raise ValueError("AOI does not intersect the MGRS tile")
    return win.round_offsets().round_lengths()


def window_to_meta(win: Window) -> dict:
    return {
        "col_off": int(win.col_off),
        "row_off": int(win.row_off),
        "width": int(win.width),
        "height": int(win.height),
    }


def crop_window(stack: np.ndarray, win: Window) -> np.ndarray:
    r0 = int(win.row_off)
    c0 = int(win.col_off)
    r1 = r0 + int(win.height)
    c1 = c0 + int(win.width)
    return stack[:, r0:r1, c0:c1]


def catalog():
    import planetary_computer as pc
    from pystac_client import Client

    return Client.open(PC_STAC, modifier=pc.sign_inplace)


def query_items(
    collection: str,
    bbox,
    datetime_range: str,
    *,
    max_cloud: float,
    max_items: int,
    mgrs_tile: str | None = None,
):
    kwargs = {
        "collections": [collection],
        "datetime": datetime_range,
        "max_items": int(max_items),
    }
    mgrs = None if not mgrs_tile else str(mgrs_tile).strip().upper().lstrip("T")
    # MGRS filter is complete for a granule; a full-tile bbox search on PC
    # silently under-returns (national 32VNM: 48 vs 77 items).
    if mgrs:
        pass
    elif bbox is not None:
        kwargs["bbox"] = list(bbox)
    else:
        raise ValueError("query_items needs bbox or mgrs_tile")
    query: dict = {}
    if max_cloud < 100:
        query["eo:cloud_cover"] = {"lt": float(max_cloud)}
    if mgrs:
        query["s2:mgrs_tile"] = {"eq": mgrs}
    if query:
        kwargs["query"] = query
    last_err = None
    for attempt in range(5):
        try:
            items = list(catalog().search(**kwargs).items())
            if mgrs:
                items = [it for it in items if item_mgrs(it) == mgrs]
            return items
        except Exception as exc:  # noqa: BLE001 - retry any transient STAC/API failure
            last_err = exc
            wait = 5 * (attempt + 1)
            print(f"STAC query failed ({exc}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"STAC query failed after 5 attempts: {last_err}")


def rank_items(items, center_iso: str):
    center = datetime.fromisoformat(str(center_iso)[:10]).replace(tzinfo=timezone.utc)

    def key(it):
        dt = it.datetime
        if dt is None:
            delta = 1e18
        else:
            delta = abs((dt.astimezone(timezone.utc) - center).total_seconds())
        return (delta, float(it.properties.get("eo:cloud_cover", 100.0)))

    return sorted(items, key=key)


def _sign(item):
    import planetary_computer as pc

    return pc.sign(item)


def warp_band(href: str, dst_transform, dst_crs, height: int, width: int, resampling: Resampling) -> np.ndarray:
    with rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=dst_crs,
            transform=dst_transform,
            height=height,
            width=width,
            resampling=resampling,
        ) as vrt:
            return vrt.read(1)


def warp_bands(item, bands: list[str], dst_transform, dst_crs, height: int, width: int) -> np.ndarray:
    import planetary_computer as pc

    signed = _sign(item)

    def one(band: str) -> np.ndarray:
        href = pc.sign(signed.assets[band].href)
        return warp_band(href, dst_transform, dst_crs, height, width, Resampling.bilinear)

    if len(bands) == 1:
        return one(bands[0])[None]
    with ThreadPoolExecutor(max_workers=len(bands)) as pool:
        arrays = list(pool.map(one, bands))
    return np.stack(arrays, axis=0)


def read_native_stack(item, bands: list[str]) -> tuple[np.ndarray, dict]:
    """Read 10 m COGs on the granule's native MGRS grid (full tile)."""
    import planetary_computer as pc

    signed = _sign(item)
    href0 = pc.sign(signed.assets[bands[0]].href)
    with rasterio.open(href0) as src0:
        profile = _gtiff_profile(src0)
        dst_transform = src0.transform
        dst_crs = src0.crs
        height, width = int(src0.height), int(src0.width)

    def one(band: str) -> np.ndarray:
        href = pc.sign(signed.assets[band].href)
        with rasterio.open(href) as src:
            if (
                int(src.width) == width
                and int(src.height) == height
                and src.transform == dst_transform
                and src.crs == dst_crs
            ):
                return src.read(1)
        return warp_band(href, dst_transform, dst_crs, height, width, Resampling.bilinear)

    with ThreadPoolExecutor(max_workers=len(bands)) as pool:
        arrays = list(pool.map(one, bands))
    return np.stack(arrays, axis=0), profile


def to_reflectance(stack: np.ndarray) -> np.ndarray:
    arr = stack.astype(np.float32)
    peak = float(np.nanmax(arr)) if arr.size else 0.0
    if peak > 1.5:
        arr = arr / 10000.0
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.5)


def valid_mask(stack: np.ndarray) -> np.ndarray:
    """True where any RGBNIR band has signal (L2A nodata is typically 0)."""
    return np.any(stack > 0, axis=0)


def cloud_frac_and_mask(
    *,
    method: str,
    l2a_stack: np.ndarray,
    l1c_stack: np.ndarray | None,
    include_shadow: bool,
    device: str,
) -> tuple[float, np.ndarray, dict]:
    valid = valid_mask(l2a_stack)
    extra: dict = {"valid_frac": float(valid.mean())}

    if method == "omnicloudmask":
        from omnicloudmask import predict_from_array

        # OmniCloudMask wants (3, H, W) Red, Green, NIR.
        red = to_reflectance(l2a_stack[0:1])[0]
        green = to_reflectance(l2a_stack[1:2])[0]
        nir = to_reflectance(l2a_stack[3:4])[0]
        rg_nir = np.stack([red, green, nir], axis=0)
        pred = predict_from_array(rg_nir, inference_device=device)
        pred = np.asarray(pred).squeeze().astype(np.uint8)
        cloudy_ids = {OCM_THICK, OCM_THIN}
        if include_shadow:
            cloudy_ids.add(OCM_SHADOW)
        cloudy = np.isin(pred, list(cloudy_ids))
        extra["class_counts"] = {
            "clear": int((pred == OCM_CLEAR).sum()),
            "thick": int((pred == OCM_THICK).sum()),
            "thin": int((pred == OCM_THIN).sum()),
            "shadow": int((pred == OCM_SHADOW).sum()),
        }
        mask = pred
    elif method == "none":
        mask = np.zeros(l2a_stack.shape[1:], dtype=np.uint8)
        cloudy = np.zeros(l2a_stack.shape[1:], dtype=bool)
        extra["skipped_aoi_cloud"] = True
    elif method == "s2cloudless":
        from s2cloudless import S2PixelCloudDetector

        if l1c_stack is None:
            raise ValueError("s2cloudless needs a matching L1C patch")
        refl = to_reflectance(l1c_stack)
        cube = np.transpose(refl, (1, 2, 0))[None]
        detector = S2PixelCloudDetector(
            threshold=0.4,
            average_over=4,
            dilation_size=2,
            all_bands=False,
        )
        probs = detector.get_cloud_probability_maps(cube)
        mask = detector.get_mask_from_probs(probs)[0].astype(np.uint8)
        cloudy = mask > 0
        extra["cloud_prob_mean"] = float(np.asarray(probs)[0].mean())
    else:
        raise ValueError(f"Unknown cloud method {method!r}")

    if valid.any():
        frac = float(cloudy[valid].mean())
    else:
        frac = 1.0
    extra["cloud_frac"] = frac
    return frac, mask.astype(np.uint8), extra


def index_l1c(items_l1c) -> dict:
    out = {}
    for it in items_l1c:
        key = s2_key(it.id)
        if key is None:
            continue
        out.setdefault(key, it)
    return out


def write_rgbnir(path: Path, stack: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if stack.dtype == np.uint16:
        out = stack
    else:
        out = np.nan_to_num(stack.astype(np.float32), nan=0.0)
        out = np.clip(np.rint(out), 0, 65535).astype(np.uint16)
    prof = profile.copy()
    prof.update(
        count=out.shape[0],
        dtype="uint16",
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=int(profile.get("blockxsize", 256)),
        blockysize=int(profile.get("blockysize", 256)),
    )
    prof.pop("photometric", None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out)
        for i, name in enumerate(BANDS_L2A, start=1):
            dst.set_band_description(i, name)


def write_mask(path: Path, mask: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    prof.update(count=1, dtype="uint8", compress="deflate", nodata=None)
    prof.pop("photometric", None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(mask.astype(np.uint8), 1)
        dst.set_band_description(1, "cloud")


def _rgb01(stack: np.ndarray) -> np.ndarray:
    rgb = np.stack([stack[i].astype(np.float32) for i in range(3)], axis=-1)
    peak = float(np.percentile(rgb, 99.5)) if rgb.size else 1.0
    scale = 3000.0 if peak > 1.5 else max(peak, 1e-6)
    return np.clip(rgb / scale, 0, 1)


def save_preview(out_dir: Path, meta: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = list(meta.get("frames") or [])
    if not frames:
        raise FileNotFoundError(f"no frames to preview in {out_dir}")

    n = len(frames)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(np.array(axes)).ravel()

    for i, fr in enumerate(frames):
        path = out_dir / fr["path"]
        win_meta = meta.get("aoi_window") or {}
        with rasterio.open(path) as ds:
            if win_meta:
                win = Window(
                    int(win_meta["col_off"]),
                    int(win_meta["row_off"]),
                    int(win_meta["width"]),
                    int(win_meta["height"]),
                )
                rgb = ds.read(window=win)
            else:
                rgb = ds.read(
                    out_shape=(ds.count, min(512, ds.height), min(512, ds.width)),
                    resampling=Resampling.bilinear,
                )
        ax = axes[i]
        ax.imshow(_rgb01(rgb))
        date = (fr.get("datetime") or "")[:10] or path.stem
        cloud = fr.get("cloud_frac")
        title = path.stem if cloud is None else f"{date}  cloud={float(cloud):.2f}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        mask_rel = fr.get("cloud_mask")
        if mask_rel:
            mpath = out_dir / mask_rel
            if mpath.is_file():
                with rasterio.open(mpath) as ds:
                    mask = ds.read(1)
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[mask > 0] = (1.0, 0.2, 0.1, 0.35)
                ax.imshow(overlay)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{meta.get('out')} | {meta.get('cloud_method')} | "
        f"{n} frames | {meta.get('date_range')}",
        fontsize=11,
    )
    fig.tight_layout()
    dest = out_dir / "preview.png"
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dest


def resolve_bbox(args) -> list[float]:
    if args.bbox is not None:
        west, south, east, north = (float(v) for v in args.bbox)
        if west >= east or south >= north:
            raise SystemExit("Invalid --bbox: need west south east north with west<east, south<north")
        return [west, south, east, north]
    if args.lon is None or args.lat is None:
        raise SystemExit("Provide --bbox WEST SOUTH EAST NORTH or --lon/--lat (optional --size-km for the AOI window)")
    if args.size_km is None or float(args.size_km) <= 0:
        raise SystemExit("--size-km must be > 0 when using --lon/--lat")
    return _bbox_from_center(args.lon, args.lat, 0.5 * float(args.size_km))


def resolve_dates(args) -> tuple[str, str]:
    if args.start_date and args.end_date:
        start = _yyyy_mm_dd(args.start_date, label="start-date")
        end = _yyyy_mm_dd(args.end_date, label="end-date")
        if start > end:
            raise SystemExit("--start-date must be ≤ --end-date")
        center = args.date or start
        return _yyyy_mm_dd(center) if args.date else start, f"{start}/{end}"
    if not args.date:
        raise SystemExit("Provide --date (with optional --days-before/--days-after) or --start-date/--end-date")
    center = _yyyy_mm_dd(args.date)
    return center, date_window(center, args.days_before, args.days_after)


def fetch(args) -> dict:
    bbox = resolve_bbox(args)
    center, window = resolve_dates(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates_file = getattr(args, "dates_file", None)
    wanted_dates = None
    wanted_stac_ids: dict[str, str] = {}
    if dates_file is not None:
        from scripts.national_cell_queue import load_dates_file, load_plan_stac_ids

        df_path = Path(dates_file)
        wanted_dates = set(load_dates_file(df_path))
        wanted_stac_ids = load_plan_stac_ids(df_path)
        if not wanted_dates:
            raise SystemExit(f"--dates-file has no dates: {dates_file}")

    skip_aoi_cloud = str(args.cloud_method) == "none" or bool(
        getattr(args, "skip_aoi_cloud_filter", False)
    )
    cap = int(args.num_samples)
    unlimited = cap <= 0 or wanted_dates is not None

    meta = {
        "bbox_wgs84": bbox,
        "aoi_size_km": None if args.bbox is not None else float(args.size_km),
        "center_date": center,
        "date_range": window,
        "bands": BANDS_L2A,
        "cloud_method": args.cloud_method,
        "skip_aoi_cloud_filter": bool(skip_aoi_cloud),
        "max_cloud_frac": float(args.max_cloud_frac),
        "min_valid_frac": float(args.min_valid_frac),
        "include_shadow": bool(args.include_shadow),
        "max_stac_cloud": float(args.max_stac_cloud),
        "dates_file": None if dates_file is None else str(dates_file),
        "wanted_dates": None if wanted_dates is None else sorted(wanted_dates),
        "wanted_stac_ids": wanted_stac_ids or None,
        "out": str(out_dir),
        "product": "mgrs_tile_10m",
        "frames": [],
        "skipped": [],
    }

    existing_days: set[str] = set()
    written = 0
    if (out_dir / "meta.json").is_file() and not args.dry_run:
        from scripts.national_cell_queue import frame_day as _frame_day

        old = json.loads((out_dir / "meta.json").read_text())
        meta["frames"] = list(old.get("frames") or [])
        meta["skipped"] = list(old.get("skipped") or [])
        for fr in meta["frames"]:
            try:
                existing_days.add(_frame_day(fr))
            except ValueError:
                continue
            written = max(written, int(fr.get("index") or 0))
        if meta["frames"]:
            print(
                f"resume {len(meta['frames'])} existing frames in {out_dir}",
                flush=True,
            )

    print(f"AOI {bbox}  STAC {window}  method={args.cloud_method}", flush=True)
    items = query_items(
        L2A,
        bbox,
        window,
        max_cloud=args.max_stac_cloud,
        max_items=args.max_stac_items,
        mgrs_tile=args.mgrs_tile,
    )
    items = rank_items(items, center)
    if not items:
        raise SystemExit(f"No L2A scenes for AOI {bbox} in {window}")

    if args.mgrs_tile:
        mgrs = str(args.mgrs_tile).strip().upper().lstrip("T")
        items = [it for it in items if item_mgrs(it) == mgrs]
        if not items:
            raise SystemExit(f"No L2A scenes on MGRS {mgrs} for AOI {bbox} in {window}")
    else:
        mgrs = item_mgrs(items[0])
        if mgrs is None:
            raise SystemExit(f"Could not parse MGRS tile from {items[0].id}")
        items = [it for it in items if item_mgrs(it) == mgrs]
    meta["mgrs_tile"] = mgrs
    print(f"{len(items)} L2A scenes on MGRS {mgrs} after STAC query")

    if wanted_dates is not None:
        from scripts.national_cell_queue import filter_stac_items_for_plan

        filtered, missing_ids = filter_stac_items_for_plan(
            items, wanted_dates, wanted_stac_ids
        )
        for day, sid in missing_ids:
            try:
                got = catalog().get_collection(L2A).get_item(str(sid))
            except Exception:
                got = None
            if got is None:
                print(f"WARN: planned STAC miss {day} {sid}", flush=True)
                continue
            filtered.append(got)
        have = {
            it.datetime.astimezone(timezone.utc).strftime("%Y-%m-%d")
            for it in filtered
            if it.datetime is not None
        }
        missing = sorted(wanted_dates - have)
        if missing:
            print(
                f"WARN: {len(missing)} planned dates missing from STAC: {missing}",
                flush=True,
            )
        items = filtered
        pinned = sum(1 for it in items if it.id in set(wanted_stac_ids.values()))
        print(
            f"{len(items)} scenes match dates-file ({len(wanted_dates)} requested, "
            f"{pinned} pinned STAC ids)",
            flush=True,
        )
        if not items:
            raise SystemExit(f"No L2A scenes match --dates-file on MGRS {mgrs}")

    l1c_index = {}
    if args.cloud_method == "s2cloudless" and not args.dry_run:
        l1c_items = query_items(
            L1C,
            bbox,
            window,
            max_cloud=100.0,
            max_items=args.max_stac_items,
            mgrs_tile=args.mgrs_tile,
        )
        l1c_index = index_l1c(l1c_items)
        print(f"{len(l1c_index)} L1C scenes indexed for s2cloudless")

    if args.dry_run:
        dry_items = items if unlimited else items[: max(cap, 0)]
        for it in dry_items:
            meta["frames"].append(
                {
                    "id": it.id,
                    "datetime": None if it.datetime is None else it.datetime.isoformat(),
                    "eo:cloud_cover": it.properties.get("eo:cloud_cover"),
                    "mgrs_tile": item_mgrs(it),
                    "source": "stac_dry_run",
                }
            )
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"dry-run wrote {out_dir / 'meta.json'}")
        return meta

    profile = None
    aoi_win = None
    seen_days: set[str] = set(existing_days)
    for it in items:
        if not unlimited and written >= cap:
            break
        day = "" if it.datetime is None else it.datetime.strftime("%Y-%m-%d")
        if day and day in existing_days:
            continue
        if args.one_per_day and day and day in seen_days:
            meta["skipped"].append({"id": it.id, "reason": "duplicate_day", "datetime": day})
            continue
        signed = _sign(it)
        missing = [b for b in BANDS_L2A if b not in signed.assets]
        if missing:
            meta["skipped"].append({"id": it.id, "reason": f"missing_assets:{missing}"})
            continue

        l1c_item = None
        l1c_stack = None
        if args.cloud_method == "s2cloudless" and not skip_aoi_cloud:
            key = s2_key(it.id)
            l1c_item = None if key is None else l1c_index.get(key)
            if l1c_item is None:
                meta["skipped"].append({"id": it.id, "reason": "no_matching_l1c"})
                print(f"  skip {it.id}: no matching L1C")
                continue
            missing_l1c = [b for b in S2CL_BANDS if b not in _sign(l1c_item).assets]
            if missing_l1c:
                meta["skipped"].append({"id": it.id, "reason": f"missing_l1c:{missing_l1c}"})
                continue

        try:
            stack, item_profile = read_native_stack(it, BANDS_L2A)
            if profile is None:
                profile = item_profile
                aoi_win = aoi_window(
                    profile["transform"],
                    profile["crs"],
                    int(profile["height"]),
                    int(profile["width"]),
                    bbox,
                )
                meta["crs"] = str(profile["crs"])
                meta["resolution_m"] = float(profile["transform"].a)
                meta["width"] = int(profile["width"])
                meta["height"] = int(profile["height"])
                meta["transform"] = list(profile["transform"])[:6]
                meta["aoi_window"] = window_to_meta(aoi_win)
                print(
                    f"tile {mgrs}  {profile['width']}x{profile['height']}px @ "
                    f"{meta['resolution_m']}m  AOI window "
                    f"{int(aoi_win.width)}x{int(aoi_win.height)}px"
                )
            frac = 0.0
            extra = {"valid_frac": 1.0, "cloud_frac": 0.0}
            aoi_mask = None
            if skip_aoi_cloud:
                extra["skipped_aoi_cloud"] = True
            else:
                aoi_stack = crop_window(stack, aoi_win)
                if args.cloud_method == "s2cloudless":
                    win_transform = window_transform(aoi_win, profile["transform"])
                    l1c_stack = warp_bands(
                        l1c_item,
                        S2CL_BANDS,
                        win_transform,
                        profile["crs"],
                        int(aoi_win.height),
                        int(aoi_win.width),
                    )
                frac, aoi_mask, extra = cloud_frac_and_mask(
                    method=args.cloud_method,
                    l2a_stack=aoi_stack,
                    l1c_stack=l1c_stack,
                    include_shadow=args.include_shadow,
                    device=args.device,
                )
        except Exception as exc:  # noqa: BLE001
            meta["skipped"].append({"id": it.id, "reason": str(exc)})
            print(f"  skip {it.id}: {exc}")
            continue

        if not skip_aoi_cloud and extra.get("valid_frac", 1.0) < float(args.min_valid_frac):
            meta["skipped"].append(
                {
                    "id": it.id,
                    "reason": "low_valid_frac",
                    "valid_frac": extra.get("valid_frac"),
                }
            )
            print(f"  skip {it.id} valid={extra.get('valid_frac'):.3f}")
            continue
        if not skip_aoi_cloud and frac > float(args.max_cloud_frac):
            meta["skipped"].append(
                {
                    "id": it.id,
                    "reason": "cloudy_patch",
                    "cloud_frac": frac,
                    "eo:cloud_cover": it.properties.get("eo:cloud_cover"),
                    "datetime": None if it.datetime is None else it.datetime.isoformat(),
                }
            )
            print(f"  skip {it.id} patch_cloud={frac:.3f}")
            continue

        dt = it.datetime.strftime("%Y%m%d") if it.datetime else f"unk{written}"
        stem = f"{written + 1:03d}_{dt}"
        tif_name = f"{stem}.tif"
        write_rgbnir(out_dir / tif_name, stack, profile)
        rec = {
            "index": written + 1,
            "path": tif_name,
            "cloud_mask": None,
            "stac_id": it.id,
            "mgrs_tile": mgrs,
            "datetime": None if it.datetime is None else it.datetime.isoformat(),
            "eo:cloud_cover": it.properties.get("eo:cloud_cover"),
            "cloud_frac": frac,
            **{k: v for k, v in extra.items() if k != "cloud_frac"},
        }
        if aoi_mask is not None:
            mask_name = f"{stem}_aoi_cloud.tif"
            aoi_profile = profile.copy()
            aoi_profile.update(
                width=int(aoi_win.width),
                height=int(aoi_win.height),
                transform=window_transform(aoi_win, profile["transform"]),
            )
            write_mask(out_dir / mask_name, aoi_mask, aoi_profile)
            rec["cloud_mask"] = mask_name
        else:
            try:
                from eval.s2_cloud_mask import cloudy_from_scl, read_scl_10m, reproject_scl_to_grid

                scl, scl_profile = read_scl_10m(it)
                scl = reproject_scl_to_grid(
                    scl,
                    scl_profile,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    height=int(profile["height"]),
                    width=int(profile["width"]),
                )
                cloudy = cloudy_from_scl(scl, include_shadow=bool(args.include_shadow))
                mask_name = f"{stem}_aoi_cloud.tif"
                scl_name = f"{stem}_scl.tif"
                write_mask(out_dir / mask_name, cloudy.astype(np.uint8), profile)
                write_mask(out_dir / scl_name, scl, profile)
                rec["cloud_mask"] = mask_name
                rec["scl_path"] = scl_name
                rec["mask_source"] = "SCL"
                rec["cloud_frac"] = float(cloudy.mean()) if cloudy.size else 1.0
            except Exception as exc:  # noqa: BLE001
                print(f"  SCL mask skip {it.id}: {exc}", flush=True)
        if l1c_item is not None:
            rec["l1c_id"] = l1c_item.id
        meta["frames"].append(rec)
        if day:
            seen_days.add(day)
            existing_days.add(day)
        written += 1
        print(
            f"  wrote {tif_name}  {profile['width']}x{profile['height']}  "
            f"scene_cloud={it.properties.get('eo:cloud_cover')}  "
            f"aoi_cloud={frac:.3f}",
            flush=True,
        )
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if meta["frames"] and not args.no_preview:
        dest = save_preview(out_dir, meta)
        meta["preview"] = str(dest)
        print(f"preview={dest}")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    n = len(meta["frames"])
    print(f"frames={n} skipped={len(meta['skipped'])} → {out_dir}")
    if not unlimited and n < cap:
        print(f"WARN: only {n}/{cap} clear revisits")
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Sentinel-2 revisits for a date and AOI")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--date", type=str, default=None, help="Center date YYYY-MM-DD")
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--days-before", type=int, default=90)
    p.add_argument("--days-after", type=int, default=90)
    p.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument(
        "--mgrs-tile",
        type=str,
        default=None,
        help="Lock STAC filtering to this MGRS tile (e.g. 32VLL). "
        "Default: tile of the first ranked scene.",
    )
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--lat", type=float, default=None)
    p.add_argument(
        "--size-km",
        type=float,
        default=1.28,
        help="Study AOI side length (km) for cloud filtering and the crop window in meta.json. "
        "The saved GeoTIFF is the full MGRS 10 m tile covering this AOI.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Max frames to keep (0 = no cap). Ignored when --dates-file is set.",
    )
    p.add_argument(
        "--dates-file",
        type=Path,
        default=None,
        help="JSON list or plan.json with union_dates. Downloads those days only. "
        "If the plan has days_scored_items, fetch those STAC ids (not first-hit). "
        "Does not skip AOI cloud filtering unless --cloud-method none.",
    )
    p.add_argument(
        "--skip-aoi-cloud-filter",
        action="store_true",
        help="Do not skip scenes based on the study-AOI cloud fraction.",
    )
    p.add_argument(
        "--cloud-method",
        choices=["omnicloudmask", "s2cloudless", "none"],
        default="omnicloudmask",
    )
    p.add_argument(
        "--max-cloud-frac",
        type=float,
        default=0.15,
        help="Skip a scene if this fraction of valid AOI pixels is cloudy",
    )
    p.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.85,
        help="Skip if too much of the AOI is nodata",
    )
    p.add_argument(
        "--include-shadow",
        action="store_true",
        help="Count OmniCloudMask shadow (class 3) as cloudy (default: thick+thin only)",
    )
    p.add_argument(
        "--max-stac-cloud",
        type=float,
        default=100.0,
        help="Optional STAC eo:cloud_cover prefilter; 100 disables it",
    )
    p.add_argument("--max-stac-items", type=int, default=200)
    p.add_argument("--one-per-day", action="store_true", default=True)
    p.add_argument("--allow-same-day", dest="one_per_day", action="store_false")
    p.add_argument("--device", type=str, default="cpu", help="OmniCloudMask device (cpu or cuda)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    return p


def main() -> None:
    fetch(build_parser().parse_args())


if __name__ == "__main__":
    main()
