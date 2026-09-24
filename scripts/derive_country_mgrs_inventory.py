#!/usr/bin/env python3
"""Derive an offline MGRS inventory from a supplied WGS84 country land outline.

The utility performs no network access. It clips each country polygon to local
100 km MGRS squares in the relevant UTM zones and writes a deterministic tile
list plus per-tile WGS84 land-intersection bboxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.country_config import (  # noqa: E402
    CountryConfigError,
    configured_path,
    load_country_config,
    resolve_root_path,
)

SQUARE_M = 100_000.0


class InventoryInputError(ValueError):
    """Raised when an inventory cannot be safely derived from local input."""


def _load_runtime():
    try:
        import mgrs
        from pyproj import Transformer
    except ImportError as exc:
        raise InventoryInputError(
            "offline inventory derivation requires the Python packages 'mgrs' and 'pyproj'"
        ) from exc
    return mgrs, Transformer


def _iter_polygons(geojson: dict) -> list[list[list[list[float]]]]:
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features")
        if not isinstance(features, list) or not features:
            raise InventoryInputError("GeoJSON FeatureCollection must contain features")
        geometries = [feature.get("geometry") for feature in features if isinstance(feature, dict)]
    elif geojson.get("type") == "Feature":
        geometries = [geojson.get("geometry")]
    else:
        geometries = [geojson]

    polygons = []
    for geometry in geometries:
        if not isinstance(geometry, dict):
            continue
        kind = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if kind == "Polygon":
            candidates = [coordinates]
        elif kind == "MultiPolygon":
            candidates = coordinates
        else:
            raise InventoryInputError(f"unsupported GeoJSON geometry type: {kind!r}")
        if not isinstance(candidates, list):
            raise InventoryInputError("GeoJSON geometry coordinates must be arrays")
        for polygon in candidates:
            if not isinstance(polygon, list) or not polygon:
                raise InventoryInputError("each polygon must contain an exterior ring")
            clean_polygon = []
            for ring in polygon:
                if not isinstance(ring, list) or len(ring) < 4:
                    raise InventoryInputError("polygon rings must contain at least four positions")
                clean_ring = []
                for position in ring:
                    if (
                        not isinstance(position, list)
                        or len(position) < 2
                        or not all(isinstance(v, (int, float)) for v in position[:2])
                    ):
                        raise InventoryInputError("GeoJSON positions must contain numeric lon/lat")
                    lon, lat = float(position[0]), float(position[1])
                    if not (-180 <= lon <= 180 and -80 <= lat <= 84):
                        raise InventoryInputError(
                            f"position outside MGRS-supported WGS84 bounds: {[lon, lat]}"
                        )
                    clean_ring.append([lon, lat])
                if clean_ring[0] != clean_ring[-1]:
                    raise InventoryInputError("polygon rings must be closed")
                clean_polygon.append(clean_ring)
            polygons.append(clean_polygon)
    if not polygons:
        raise InventoryInputError("GeoJSON contains no Polygon or MultiPolygon geometry")
    return polygons


def load_outline(path: Path) -> tuple[list[list[list[list[float]]]], str]:
    if not path.is_file():
        raise InventoryInputError(f"required country land-outline GeoJSON is missing: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InventoryInputError(f"invalid GeoJSON JSON in {path}: {exc}") from exc
    crs = payload.get("crs")
    if crs:
        crs_text = json.dumps(crs).upper()
        if "4326" not in crs_text and "CRS84" not in crs_text:
            raise InventoryInputError("country outline must use WGS84 lon/lat (EPSG:4326/CRS84)")
    return _iter_polygons(payload), hashlib.sha256(raw).hexdigest()


def _clip_ring(
    ring: list[tuple[float, float]], west: float, south: float, east: float, north: float
) -> list[tuple[float, float]]:
    points = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else list(ring)

    def clip(points, inside, intersect):
        if not points:
            return []
        output = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside != previous_inside:
                output.append(intersect(previous, current))
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
        return output

    def vertical(x_bound):
        def intersect(a, b):
            ratio = (x_bound - a[0]) / (b[0] - a[0])
            return x_bound, a[1] + ratio * (b[1] - a[1])

        return intersect

    def horizontal(y_bound):
        def intersect(a, b):
            ratio = (y_bound - a[1]) / (b[1] - a[1])
            return a[0] + ratio * (b[0] - a[0]), y_bound

        return intersect

    points = clip(points, lambda p: p[0] >= west, vertical(west))
    points = clip(points, lambda p: p[0] <= east, vertical(east))
    points = clip(points, lambda p: p[1] >= south, horizontal(south))
    return clip(points, lambda p: p[1] <= north, horizontal(north))


def _ring_area(ring: list[tuple[float, float]]) -> float:
    if len(ring) < 3:
        return 0.0
    return abs(
        sum(
            ring[i][0] * ring[(i + 1) % len(ring)][1]
            - ring[(i + 1) % len(ring)][0] * ring[i][1]
            for i in range(len(ring))
        )
    ) / 2.0


def _zone_range(west: float, east: float) -> range:
    first = max(1, min(60, int(math.floor((west + 180.0) / 6.0)) + 1))
    last = max(1, min(60, int(math.floor((min(east, 179.999999) + 180.0) / 6.0)) + 1))
    return range(first, last + 1)


def derive_inventory(
    polygons: list[list[list[list[float]]]],
    *,
    min_tile_land_fraction: float = 0.0,
    include_bbox: list[float] | None = None,
) -> tuple[list[str], dict[str, list[float]], dict[str, float]]:
    """Return sorted MGRS ids, land-intersection bboxes, and land fractions."""
    mgrs_module, Transformer = _load_runtime()
    converter = mgrs_module.MGRS()
    all_points = [position for polygon in polygons for ring in polygon for position in ring]
    west = min(point[0] for point in all_points)
    south = min(point[1] for point in all_points)
    east = max(point[0] for point in all_points)
    north = max(point[1] for point in all_points)
    if include_bbox is not None:
        if len(include_bbox) != 4 or include_bbox[0] >= include_bbox[2] or include_bbox[1] >= include_bbox[3]:
            raise InventoryInputError("inventory.include_bbox must be [west,south,east,north]")
        west, south = max(west, include_bbox[0]), max(south, include_bbox[1])
        east, north = min(east, include_bbox[2]), min(north, include_bbox[3])
        if west >= east or south >= north:
            raise InventoryInputError("inventory.include_bbox does not overlap the country outline")

    results: dict[str, tuple[list[float], float]] = {}
    for zone in _zone_range(west, east):
        forward = Transformer.from_crs("EPSG:4326", f"EPSG:326{zone:02d}", always_xy=True)
        inverse = Transformer.from_crs(f"EPSG:326{zone:02d}", "EPSG:4326", always_xy=True)
        projected_polygons = []
        for polygon in polygons:
            projected = []
            for ring in polygon:
                projected.append([forward.transform(point[0], point[1]) for point in ring])
            projected_polygons.append(projected)

        relevant = [
            point
            for polygon, projected in zip(polygons, projected_polygons)
            for ring, projected_ring in zip(polygon, projected)
            for point, projected_point in zip(ring, projected_ring)
            if west <= point[0] <= east and south <= point[1] <= north
        ]
        # Corners ensure candidate coverage when no source vertex lies in this zone.
        relevant.extend([[west, south], [west, north], [east, south], [east, north]])
        projected_bounds = [forward.transform(point[0], point[1]) for point in relevant]
        min_e = math.floor(min(point[0] for point in projected_bounds) / SQUARE_M) * SQUARE_M
        max_e = math.floor(max(point[0] for point in projected_bounds) / SQUARE_M) * SQUARE_M
        min_n = math.floor(min(point[1] for point in projected_bounds) / SQUARE_M) * SQUARE_M
        max_n = math.floor(max(point[1] for point in projected_bounds) / SQUARE_M) * SQUARE_M

        easting = min_e
        while easting <= max_e:
            northing = min_n
            while northing <= max_n:
                center_lon, center_lat = inverse.transform(
                    easting + SQUARE_M / 2, northing + SQUARE_M / 2
                )
                if not (west - 2 <= center_lon <= east + 2 and south - 2 <= center_lat <= north + 2):
                    northing += SQUARE_M
                    continue
                tile_id = converter.toMGRS(center_lat, center_lon, MGRSPrecision=0)
                tile_id = tile_id.decode() if isinstance(tile_id, bytes) else str(tile_id)
                if int(tile_id[:2]) != zone:
                    northing += SQUARE_M
                    continue
                try:
                    _, _, tile_e, tile_n = converter.MGRSToUTM(tile_id)
                except Exception:  # pragma: no cover - defensive around third-party parser
                    northing += SQUARE_M
                    continue
                if abs(tile_e - easting) > 1 or abs(tile_n - northing) > 1:
                    northing += SQUARE_M
                    continue

                clipped_points: list[tuple[float, float]] = []
                land_area = 0.0
                for projected in projected_polygons:
                    exterior = _clip_ring(
                        projected[0], easting, northing, easting + SQUARE_M, northing + SQUARE_M
                    )
                    land_area += _ring_area(exterior)
                    clipped_points.extend(exterior)
                    for hole in projected[1:]:
                        clipped_hole = _clip_ring(
                            hole, easting, northing, easting + SQUARE_M, northing + SQUARE_M
                        )
                        land_area -= _ring_area(clipped_hole)
                fraction = max(0.0, land_area) / (SQUARE_M * SQUARE_M)
                if clipped_points and fraction > 0 and fraction + 1e-12 >= min_tile_land_fraction:
                    lonlat = [inverse.transform(x, y) for x, y in clipped_points]
                    bbox = [
                        round(min(point[0] for point in lonlat), 7),
                        round(min(point[1] for point in lonlat), 7),
                        round(max(point[0] for point in lonlat), 7),
                        round(max(point[1] for point in lonlat), 7),
                    ]
                    previous = results.get(tile_id)
                    if previous is None:
                        results[tile_id] = (bbox, fraction)
                    else:
                        old_bbox, old_fraction = previous
                        results[tile_id] = (
                            [
                                min(old_bbox[0], bbox[0]),
                                min(old_bbox[1], bbox[1]),
                                max(old_bbox[2], bbox[2]),
                                max(old_bbox[3], bbox[3]),
                            ],
                            max(old_fraction, fraction),
                        )
                northing += SQUARE_M
            easting += SQUARE_M

    ids = sorted(results)
    return (
        ids,
        {tile_id: results[tile_id][0] for tile_id in ids},
        {tile_id: round(results[tile_id][1], 6) for tile_id in ids},
    )


def blocker_payload(*, country: dict, required_path: Path, config_path: Path) -> dict:
    return {
        "schema_version": 1,
        "status": "blocked",
        "country": country,
        "blocker": "missing_authoritative_country_land_outline_geojson",
        "required_input": {
            "path": str(required_path),
            "format": "GeoJSON FeatureCollection, Feature, Polygon, or MultiPolygon",
            "crs": "WGS84 longitude/latitude (EPSG:4326 or CRS84)",
            "geometry_requirement": (
                "authoritative country land outline with closed polygon rings; "
                "include the islands intended for national processing"
            ),
        },
        "resume_command": (
            f"python scripts/derive_country_mgrs_inventory.py --config {config_path}"
        ),
    }


def write_blocker(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run(config_path: Path, *, outline_override: Path | None = None, blocker_out: Path | None = None) -> dict:
    config = load_country_config(config_path)
    outline = (
        resolve_root_path(outline_override)
        if outline_override is not None
        else configured_path(config, "inputs", "land_outline_geojson")
    )
    if not outline.is_file():
        blocker = blocker_payload(
            country=config["country"],
            required_path=outline,
            config_path=Path(config["_config_path"]),
        )
        if blocker_out is not None:
            write_blocker(resolve_root_path(blocker_out), blocker)
        raise InventoryInputError(blocker["required_input"]["geometry_requirement"] + f": {outline}")

    polygons, source_sha256 = load_outline(outline)
    inventory_config = config["inventory"]
    ids, bboxes, fractions = derive_inventory(
        polygons,
        min_tile_land_fraction=float(inventory_config.get("min_tile_land_fraction", 0.0)),
        include_bbox=inventory_config.get("include_bbox"),
    )
    if not ids:
        raise InventoryInputError("country outline produced no MGRS tiles")
    provenance = {
        "method": "offline_utm_100km_polygon_clip_v1",
        "source_outline": str(outline),
        "source_sha256": source_sha256,
        "min_tile_land_fraction": float(inventory_config.get("min_tile_land_fraction", 0.0)),
        "include_bbox": inventory_config.get("include_bbox"),
    }
    list_payload = {
        "schema_version": 1,
        "country": config["country"],
        "mgrs": ids,
        "n": len(ids),
        "per_tile_land_fraction": fractions,
        "provenance": provenance,
    }
    list_out = configured_path(config, "inventory", "mgrs_list_json")
    bboxes_out = configured_path(config, "inventory", "mgrs_bboxes_json")
    for path, payload in ((list_out, list_payload), (bboxes_out, bboxes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"mgrs_list": str(list_out), "mgrs_bboxes": str(bboxes_out), "n": len(ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outline", type=Path, default=None, help="Override configured local outline.")
    parser.add_argument(
        "--blocker-out",
        type=Path,
        default=None,
        help="Write a machine-readable blocker if the authoritative outline is absent.",
    )
    args = parser.parse_args()
    try:
        result = run(args.config, outline_override=args.outline, blocker_out=args.blocker_out)
    except (CountryConfigError, InventoryInputError) as exc:
        raise SystemExit(f"inventory preflight failed: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
