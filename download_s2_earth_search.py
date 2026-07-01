"""Shared STAC/stackstac helpers for Sentinel-2 download scripts."""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from typing import Iterable

import numpy as np

EARTH_SEARCH_STAC_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"
DEFAULT_ASSETS = ("red", "green", "blue")


def _parse_bbox(bbox: Iterable[float]) -> tuple[float, float, float, float]:
    west, south, east, north = (float(v) for v in bbox)
    if west >= east:
        raise ValueError(f"Invalid bbox: west ({west}) must be < east ({east})")
    if south >= north:
        raise ValueError(f"Invalid bbox: south ({south}) must be < north ({north})")
    return west, south, east, north


def _bbox_from_center(lon: float, lat: float, half_size_km: float) -> tuple[float, float, float, float]:
    half = float(half_size_km)
    lat_rad = math.radians(float(lat))
    dlat = half / 111.32
    dlon = half / (111.32 * max(math.cos(lat_rad), 1e-6))
    return float(lon) - dlon, float(lat) - dlat, float(lon) + dlon, float(lat) + dlat


def _utm_epsg_from_lon_lat(lon: float, lat: float) -> int:
    zone = int((float(lon) + 180.0) // 6.0) + 1
    return (32600 if float(lat) >= 0.0 else 32700) + zone


def _iso_range(start: str, end: str) -> str:
    return f"{_validate_yyyy_mm_dd(start, label='start')}/{_validate_yyyy_mm_dd(end, label='end')}"


def _validate_yyyy_mm_dd(value: str, *, label: str = "date") -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise argparse.ArgumentTypeError(f"Invalid {label} {value!r}; expected YYYY-MM-DD")
    datetime.strptime(text, "%Y-%m-%d")
    return text


def _time_to_filename(dt_value: str) -> str:
    text = str(dt_value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = datetime.fromisoformat(text.split(".")[0])
    return dt.strftime("%Y%m%d_%H%M%S")


def _coord_values(coord) -> np.ndarray:
    return np.asarray(getattr(coord, "values", coord), dtype=np.float64)


def _north_is_low_y_index(y) -> bool:
    yv = _coord_values(y)
    return len(yv) > 1 and yv[0] > yv[-1]


def _west_is_low_x_index(x) -> bool:
    xv = _coord_values(x)
    return len(xv) > 1 and xv[0] < xv[-1]


def clip_stack_mgrs_subtile(stack, subtile: str):
    subtile = str(subtile).lower()
    valid = {"full", "nw", "ne", "sw", "se", "n", "s", "e", "w"}
    if subtile not in valid:
        raise ValueError(f"Unknown mgrs subtile: {subtile!r}")
    if subtile == "full":
        return stack

    if hasattr(stack, "coords"):
        y = stack.coords["y"].values
        x = stack.coords["x"].values
    else:
        y = stack.y.values
        x = stack.x.values
    h, w = len(y), len(x)
    north = slice(0, h // 2) if _north_is_low_y_index(y) else slice(h // 2, h)
    south = slice(h // 2, h) if _north_is_low_y_index(y) else slice(0, h // 2)
    west = slice(0, w // 2) if _west_is_low_x_index(x) else slice(w // 2, w)
    east = slice(w // 2, w) if _west_is_low_x_index(x) else slice(0, w // 2)

    cuts = {
        "nw": (north, west),
        "ne": (north, east),
        "sw": (south, west),
        "se": (south, east),
        "n": (north, slice(None)),
        "s": (south, slice(None)),
        "w": (slice(None), west),
        "e": (slice(None), east),
    }
    y_sl, x_sl = cuts[subtile]
    return stack.isel(y=y_sl, x=x_sl)


def aoi_nodata_fractions(stack, assets: tuple[str, ...]) -> list[float]:
    bands = list(stack.coords.get("band", []))
    use_bands = [b for b in assets if b in bands] or bands
    fracs: list[float] = []
    for i in range(int(stack.sizes["time"])):
        frame = stack.isel(time=i)
        if use_bands:
            data = frame.sel(band=use_bands).transpose("y", "x", "band").values
        else:
            data = frame.transpose("y", "x", "band").values
        nodata = np.any(np.isnan(data), axis=-1)
        fracs.append(100.0 * float(nodata.mean()))
    return fracs


def filter_stack_by_aoi_nodata(
    stack,
    *,
    max_aoi_nodata_pct: float,
    assets: tuple[str, ...],
):
    fracs = aoi_nodata_fractions(stack, assets)
    keep = [i for i, frac in enumerate(fracs) if frac <= float(max_aoi_nodata_pct)]
    if not keep:
        raise RuntimeError(
            f"No scenes with AOI nodata <= {max_aoi_nodata_pct:.3f}% "
            f"(best was {min(fracs):.3f}%)."
        )
    kept_fracs = [fracs[i] for i in keep]
    return stack.isel(time=keep), kept_fracs


def best_valid_window(invalid: np.ndarray, win_h: int, win_w: int) -> tuple[int, int, float]:
    arr = np.asarray(invalid, dtype=np.float64)
    h, w = arr.shape
    best_score: float | None = None
    best_rc = (0, 0)
    for row in range(0, h - int(win_h) + 1):
        for col in range(0, w - int(win_w) + 1):
            score = float(arr[row : row + int(win_h), col : col + int(win_w)].sum())
            if best_score is None or score < best_score:
                best_score = score
                best_rc = (row, col)
    assert best_score is not None
    return best_rc[0], best_rc[1], best_score


def stack_items(
    items,
    *,
    assets: tuple[str, ...],
    epsg: int,
    resolution_m: float,
    bbox: tuple[float, float, float, float],
):
    import stackstac

    return stackstac.stack(
        items,
        assets=list(assets),
        epsg=int(epsg),
        resolution=float(resolution_m),
        bounds_latlon=bbox,
        snap_bounds=True,
        rescale=True,
    )


def stack_extent_metadata(stack, *, epsg: int) -> dict:
    y = _coord_values(stack.coords["y"])
    x = _coord_values(stack.coords["x"])
    h, w = int(stack.sizes["y"]), int(stack.sizes["x"])
    north_low = _north_is_low_y_index(y)
    west_low = _west_is_low_x_index(x)
    if north_low:
        y_max, y_min = float(y[0]), float(y[-1])
    else:
        y_min, y_max = float(y[0]), float(y[-1])
    if west_low:
        x_min, x_max = float(x[0]), float(x[-1])
    else:
        x_max, x_min = float(x[0]), float(x[-1])
    resolution_m = min(
        abs((x_max - x_min) / max(w - 1, 1)),
        abs((y_max - y_min) / max(h - 1, 1)),
    )
    return {
        "epsg": int(epsg),
        "height": h,
        "width": w,
        "resolution_m": float(resolution_m),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "north_is_low_row_index": bool(north_low),
        "west_is_low_col_index": bool(west_low),
    }


def _item_mgrs_tile(item) -> str | None:
    props = item.properties
    square = props.get("mgrs:square") or props.get("mgrs:grid_square")
    tile = props.get("mgrs:utm_zone"), props.get("mgrs:latitude_band"), square
    if all(v is not None and v != "" for v in tile):
        return "".join(str(t) for t in tile)
    grid_code = props.get("grid:code")
    if grid_code:
        return str(grid_code).replace("MGRS-", "")
    for key in ("s2:mgrs_tile", "mgrs_tile"):
        if props.get(key):
            return str(props[key]).replace("MGRS-", "")
    return None


def detect_main_tile(items) -> str | None:
    from collections import Counter

    counts = Counter(filter(None, (_item_mgrs_tile(it) for it in items)))
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def reference_item_for_tile(
    lon: float,
    lat: float,
    *,
    datetime_range: str,
    collection: str = DEFAULT_COLLECTION,
):
    import pystac_client

    catalog = pystac_client.Client.open(EARTH_SEARCH_STAC_URL)
    bbox = _bbox_from_center(lon, lat, 0.5)
    items = list(
        catalog.search(
            collections=[collection],
            bbox=list(bbox),
            datetime=datetime_range,
            max_items=50,
        ).items()
    )
    if not items:
        raise RuntimeError(f"No {collection} items found near ({lon}, {lat}) for {datetime_range}")
    items.sort(key=lambda it: it.properties.get("datetime", ""))
    item = items[0]
    geom = item.geometry
    coords = geom["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    tile_bbox = (min(lons), min(lats), max(lons), max(lats))
    return item, tile_bbox, _item_mgrs_tile(item)
