#!/usr/bin/env python3
"""Overlay LR 512 / HR 2048 training patches on NIB focus valid masks.

Each complete 512×512 @ 10 m S2 patch covers the same ground footprint as a
2048×2048 @ 2.5 m HR training tile (5.12 km on a side).

Writes per-city PNGs, a GeoJSON of complete patches, and a summary JSON/PNG.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2_dataset import FOCUS_PROJECT_BY_CITY, resolve_focus_project_dir  # noqa: E402

FOCUS_ROOT = ROOT / "data" / "nib_focus_1m_worldcover"  # default map output root

LR_PATCH = 512
HR_SCALE = 4  # df=4 → HR 2048
LR_GSD_M = 10.0
HR_GSD_M = LR_GSD_M / HR_SCALE


@dataclass
class PatchCell:
    city: str
    row: int
    col: int
    row_off: int
    col_off: int
    width: int
    height: int
    valid_frac: float
    complete: bool
    west: float
    south: float
    east: float
    north: float

    @property
    def hr_width(self) -> int:
        return int(self.width * HR_SCALE)

    @property
    def hr_height(self) -> int:
        return int(self.height * HR_SCALE)


def _load_aoi_bbox_25833(city: str) -> tuple[float, float, float, float] | None:
    aois_path = ROOT / "data" / "s2_revisits" / "aois.json"
    if not aois_path.is_file():
        return None
    areas = {a["id"]: a for a in json.loads(aois_path.read_text())["areas"]}
    area = areas.get(city)
    if not area:
        return None
    w, s, e, n = area["bbox_wgs84"]
    tf = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    xs, ys = zip(*(tf.transform(x, y) for x, y in [(w, s), (w, n), (e, s), (e, n)]))
    return min(xs), min(ys), max(xs), max(ys)


def _enumerate_patches(
    city: str,
    mask: np.ndarray,
    transform: rasterio.Affine,
    patch_lr: int,
    min_valid_frac: float,
) -> list[PatchCell]:
    h, w = mask.shape
    cells: list[PatchCell] = []
    for row, row_off in enumerate(range(0, h, patch_lr)):
        for col, col_off in enumerate(range(0, w, patch_lr)):
            height = min(patch_lr, h - row_off)
            width = min(patch_lr, w - col_off)
            patch = mask[row_off : row_off + height, col_off : col_off + width]
            valid_frac = float(patch.mean()) if patch.size else 0.0
            # Pixel corners in CRS (north-up, row increases southward for typical geo transforms)
            x0, y0 = transform * (col_off, row_off)
            x1, y1 = transform * (col_off + width, row_off + height)
            west, east = sorted((float(x0), float(x1)))
            south, north = sorted((float(y0), float(y1)))
            complete = (
                height == patch_lr
                and width == patch_lr
                and valid_frac >= min_valid_frac
            )
            cells.append(
                PatchCell(
                    city=city,
                    row=row,
                    col=col,
                    row_off=row_off,
                    col_off=col_off,
                    width=width,
                    height=height,
                    valid_frac=valid_frac,
                    complete=complete,
                    west=west,
                    south=south,
                    east=east,
                    north=north,
                )
            )
    return cells


def _plot_city(
    city: str,
    mask: np.ndarray,
    transform: rasterio.Affine,
    cells: list[PatchCell],
    out_png: Path,
    aoi_bbox: tuple[float, float, float, float] | None,
    *,
    patch_lr: int,
) -> None:
    h, w = mask.shape
    # Downsample for display if large
    max_side = 1800
    stride = max(1, int(np.ceil(max(h, w) / max_side)))
    view = mask[::stride, ::stride]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        view,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0, w, h, 0),
        interpolation="nearest",
        origin="upper",
    )

    for cell in cells:
        color = "#2ca02c" if cell.complete else ("#ff7f0e" if cell.valid_frac > 0.05 else "#bbbbbb")
        lw = 1.4 if cell.complete else 0.6
        alpha = 0.95 if cell.complete else 0.55
        rect = mpatches.Rectangle(
            (cell.col_off, cell.row_off),
            cell.width,
            cell.height,
            fill=False,
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
        )
        ax.add_patch(rect)

    if aoi_bbox is not None:
        west, south, east, north = aoi_bbox
        # Convert CRS meters → pixel coords via inverse transform
        inv = ~transform
        c0, r0 = inv * (west, north)
        c1, r1 = inv * (east, south)
        px = min(c0, c1)
        py = min(r0, r1)
        pw = abs(c1 - c0)
        ph = abs(r1 - r0)
        ax.add_patch(
            mpatches.Rectangle(
                (px, py),
                pw,
                ph,
                fill=False,
                edgecolor="#d62728",
                linewidth=2.2,
                linestyle="--",
                label="current study AOI (~2.5 km)",
            )
        )

    n_complete = sum(1 for c in cells if c.complete)
    n_partial = sum(1 for c in cells if (not c.complete) and c.valid_frac > 0.05)
    n_empty = len(cells) - n_complete - n_partial
    span_km = patch_lr * LR_GSD_M / 1000.0
    ax.set_title(
        f"{city}: LR {patch_lr}×{patch_lr} @ {LR_GSD_M:g} m  →  "
        f"HR {patch_lr * HR_SCALE}×{patch_lr * HR_SCALE} @ {HR_GSD_M:g} m\n"
        f"patch span {span_km:.2f} km  |  complete {n_complete}  |  "
        f"partial {n_partial}  |  empty/edge {n_empty}",
        fontsize=11,
    )
    ax.set_xlabel("LR column (10 m px)")
    ax.set_ylabel("LR row (10 m px)")
    legend = [
        mpatches.Patch(edgecolor="#2ca02c", facecolor="none", label=f"complete ({n_complete})"),
        mpatches.Patch(edgecolor="#ff7f0e", facecolor="none", label=f"partial ({n_partial})"),
        mpatches.Patch(edgecolor="#bbbbbb", facecolor="none", label=f"empty/edge ({n_empty})"),
        mpatches.Patch(edgecolor="#d62728", facecolor="none", linestyle="--", label="current AOI"),
    ]
    ax.legend(handles=legend, loc="upper right", framealpha=0.9)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _cells_to_geojson(cells: list[PatchCell], crs: str, *, patch_lr: int) -> dict:
    feats = []
    for c in cells:
        if not c.complete:
            continue
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "city": c.city,
                    "row": c.row,
                    "col": c.col,
                    "lr_w": c.width,
                    "lr_h": c.height,
                    "hr_w": c.hr_width,
                    "hr_h": c.hr_height,
                    "valid_frac": round(c.valid_frac, 6),
                    "span_m": patch_lr * LR_GSD_M,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [c.west, c.south],
                            [c.east, c.south],
                            [c.east, c.north],
                            [c.west, c.north],
                            [c.west, c.south],
                        ]
                    ],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": feats,
    }


def _plot_summary(summary_rows: list[dict], out_png: Path, *, patch_lr: int) -> None:
    cities = [r["city"] for r in summary_rows]
    complete = [r["n_complete"] for r in summary_rows]
    partial = [r["n_partial"] for r in summary_rows]
    x = np.arange(len(cities))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, complete, color="#2ca02c", label=f"complete {patch_lr}→{patch_lr * HR_SCALE} patches")
    ax.bar(x, partial, bottom=complete, color="#ff7f0e", alpha=0.85, label="partial (>5% valid)")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right")
    ax.set_ylabel("patches")
    ax.set_title(
        f"Complete LR {patch_lr}×{patch_lr} / HR {patch_lr * HR_SCALE}×{patch_lr * HR_SCALE} "
        f"patches over NIB focus valid masks"
    )
    for i, r in enumerate(summary_rows):
        ax.text(i, complete[i] + partial[i] + 0.5, str(complete[i]), ha="center", va="bottom", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cities", nargs="+", default=list(FOCUS_PROJECT_BY_CITY))
    ap.add_argument("--patch-lr", type=int, default=LR_PATCH)
    ap.add_argument(
        "--min-valid-frac",
        type=float,
        default=1.0,
        help="Minimum valid-mask fraction for a complete patch (default: 1.0 = fully inside).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=FOCUS_ROOT / "map" / "patch_grid_lr512",
    )
    args = ap.parse_args()
    patch_lr = int(args.patch_lr)
    patch_hr = patch_lr * HR_SCALE
    patch_span_m = patch_lr * LR_GSD_M

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_complete: list[PatchCell] = []
    summary_rows: list[dict] = []
    crs_name = "EPSG:25833"

    for city in args.cities:
        proj = resolve_focus_project_dir(city)
        if proj is None:
            raise FileNotFoundError(f"No NIB project dir for {city}")
        mask_path = proj / "valid_mask_10m.tif"
        if not mask_path.is_file():
            print(f"SKIP {city}: missing {mask_path}")
            continue
        with rasterio.open(mask_path) as ds:
            mask = (ds.read(1) > 0).astype(np.float32)
            transform = ds.transform
            crs_name = str(ds.crs) if ds.crs else crs_name
            height, width = ds.height, ds.width

        cells = _enumerate_patches(city, mask, transform, patch_lr, args.min_valid_frac)
        aoi = _load_aoi_bbox_25833(city)
        png = out_dir / f"{city}_patch_grid_lr{patch_lr}.png"
        _plot_city(city, mask, transform, cells, png, aoi, patch_lr=patch_lr)

        n_complete = sum(1 for c in cells if c.complete)
        n_partial = sum(1 for c in cells if (not c.complete) and c.valid_frac > 0.05)
        row = {
            "city": city,
            "project_folder": proj.name,
            "nib_package": proj.parent.parent.name,
            "mask_shape_lr": [height, width],
            "mask_span_km": [width * LR_GSD_M / 1000.0, height * LR_GSD_M / 1000.0],
            "patch_lr": patch_lr,
            "patch_hr": patch_hr,
            "patch_span_km": patch_span_m / 1000.0,
            "n_grid_cells": len(cells),
            "n_complete": n_complete,
            "n_partial": n_partial,
            "n_empty_or_edge": len(cells) - n_complete - n_partial,
            "complete_area_km2": n_complete * (patch_span_m / 1000.0) ** 2,
            "png": str(png.resolve().relative_to(ROOT)),
        }
        summary_rows.append(row)
        all_complete.extend([c for c in cells if c.complete])
        print(
            f"{city:16s} complete={n_complete:4d}  partial={n_partial:4d}  "
            f"grid={len(cells):4d}  → {png.resolve().relative_to(ROOT)}"
        )

    # Shared project folders (e.g. algard / naerbo) must not be summed twice.
    seen_projects: set[str] = set()
    unique_rows: list[dict] = []
    for row in summary_rows:
        key = f"{row['nib_package']}/{row['project_folder']}"
        if key in seen_projects:
            continue
        seen_projects.add(key)
        unique_rows.append(row)

    keep_cities = {r["city"] for r in unique_rows}
    geojson = _cells_to_geojson(
        [c for c in all_complete if c.city in keep_cities],
        crs_name,
        patch_lr=patch_lr,
    )
    geo_path = (out_dir / f"complete_patches_lr{patch_lr}.geojson").resolve()
    geo_path.write_text(json.dumps(geojson))

    summary_png = (out_dir / f"complete_patches_summary_lr{patch_lr}.png").resolve()
    _plot_summary(unique_rows, summary_png, patch_lr=patch_lr)

    total_complete = sum(r["n_complete"] for r in unique_rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lr_patch_px": patch_lr,
        "hr_patch_px": patch_hr,
        "lr_gsd_m": LR_GSD_M,
        "hr_gsd_m": HR_GSD_M,
        "patch_span_m": patch_span_m,
        "min_valid_frac": args.min_valid_frac,
        "crs": crs_name,
        "n_area_ids": len(summary_rows),
        "n_unique_projects": len(unique_rows),
        "total_complete_patches": total_complete,
        "cities": summary_rows,
        "unique_projects": unique_rows,
        "geojson": str(geo_path.relative_to(ROOT)),
        "summary_png": str(summary_png.relative_to(ROOT)),
    }
    out_json = out_dir.resolve() / f"patch_grid_summary_lr{patch_lr}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "total_complete_patches": total_complete,
                "n_unique_projects": len(unique_rows),
                "n_area_ids": len(summary_rows),
                "lr_patch_px": patch_lr,
                "hr_patch_px": patch_hr,
                "patch_span_m": patch_span_m,
            },
            indent=2,
        )
    )
    print(f"Wrote {out_json}")
    print(f"Wrote {geo_path} ({len(geojson['features'])} complete polygons)")


if __name__ == "__main__":
    main()
