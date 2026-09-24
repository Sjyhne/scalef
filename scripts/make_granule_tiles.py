#!/usr/bin/env python3
"""Tile a full Sentinel-2 MGRS revisit stack into LR512 AOIs.

Production path (no NIB HR). Reads a parent ``data/s2_revisits/<id>/`` with
frames covering the granule (or a large AOI), writes symlink tile dirs::

    data/s2_revisits/{parent}_t{side}_y{iy:02d}_x{ix:02d}/
    # with --overlap-frac 0.1:
    data/s2_revisits/{parent}_t{side}_ovl10_y{iy:02d}_x{ix:02d}/

and a manifest JSON consumed by ``scripts/run_production.py``.

Only full ``side×side`` windows are emitted; incomplete edge strips are
skipped unless overlap strides cover them (same 5.12 km AOI recipe as METHOD.md).

National Norway runs should pass ``--mainland-only`` (or ``--land-mask``) so
ocean LR512 cells inside coastal MGRS granules are not written or trained.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.land_mask_lr512 import (  # noqa: E402
    DEFAULT_LAND_MASK,
    filter_tile_specs,
    load_land_paths,
    raster_georef,
)
from scripts.national_cell_queue import cell_lookup, filter_frames_for_cell  # noqa: E402

DEFAULT_SIDE = 512


def _plan_record_for_window(
    cells: list[dict],
    *,
    row_off: int,
    col_off: int,
    side: int,
) -> dict | None:
    """Combine planned dates from every base cell intersected by a tile window."""
    hits = []
    dates: set[str] = set()
    for rec in cells:
        rec_side = int(rec.get("side") or DEFAULT_SIDE)
        rr = int(rec.get("row_off") or 0)
        cc = int(rec.get("col_off") or 0)
        if (
            rr + rec_side <= row_off
            or row_off + side <= rr
            or cc + rec_side <= col_off
            or col_off + side <= cc
        ):
            continue
        hits.append(rec)
        dates.update(str(day) for day in (rec.get("dates") or []))
    if not hits or not dates:
        return None
    return {
        "dates": sorted(dates),
        "n_frames": len(dates),
        "source_plan_cells": [
            {
                "iy": int(rec["iy"]),
                "ix": int(rec["ix"]),
                "row_off": int(rec.get("row_off") or 0),
                "col_off": int(rec.get("col_off") or 0),
            }
            for rec in hits
        ],
    }


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _raster_extent(first_tif: Path) -> tuple[int, int, int, int]:
    with rasterio.open(first_tif) as src:
        return 0, 0, int(src.height), int(src.width)


def _aoi_extent(meta: dict, first_tif: Path, *, use_aoi_window: bool) -> tuple[int, int, int, int]:
    """Return (row0, col0, height, width) in parent raster pixels."""
    if use_aoi_window:
        aoi = meta.get("aoi_window")
        if aoi:
            return (
                int(aoi["row_off"]),
                int(aoi["col_off"]),
                int(aoi["height"]),
                int(aoi["width"]),
            )
    return _raster_extent(first_tif)


def _write_tile(
    parent_dir: Path,
    parent_meta: dict,
    dest: Path,
    *,
    row0: int,
    col0: int,
    side: int,
    iy: int,
    ix: int,
    force: bool,
    stride: int | None = None,
    overlap_frac: float = 0.0,
    frames: list[dict] | None = None,
    cell_rec: dict | None = None,
) -> dict:
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    stride_i = int(side if stride is None else stride)
    tile_frames = (
        list(frames) if frames is not None else list(parent_meta.get("frames") or [])
    )
    meta = dict(parent_meta)
    meta.pop("_tile_stride", None)
    meta.pop("_tile_overlap_frac", None)
    meta["frames"] = tile_frames
    meta["aoi_window"] = {
        "col_off": int(col0),
        "row_off": int(row0),
        "width": int(side),
        "height": int(side),
    }
    meta["parent_s2_dir"] = str(parent_dir)
    meta["lr_size_request"] = int(side)
    meta["granule_tile"] = {
        "parent": parent_dir.name,
        "iy": int(iy),
        "ix": int(ix),
        "side": int(side),
        "row_off": int(row0),
        "col_off": int(col0),
        "stride": stride_i,
        "overlap_frac": float(overlap_frac),
    }
    # Production tiles must not inherit focus/NIB markers.
    meta.pop("focus_project_folder", None)
    meta.pop("patch_grid", None)
    if cell_rec is not None:
        meta["national_cell"] = {
            k: cell_rec[k]
            for k in (
                "iy",
                "ix",
                "key",
                "dates",
                "n_frames",
                "date_span_days",
                "mean_snow_used",
                "mean_cloud_used",
                "source_plan_cells",
            )
            if k in cell_rec
        }

    for fr in tile_frames:
        for key in ("path", "cloud_mask", "scl_path"):
            name = fr.get(key)
            if name:
                _symlink(parent_dir / name, dest / name)
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    row = {
        "tile_id": dest.name,
        "s2_dir": str(dest.relative_to(ROOT)),
        "parent": parent_dir.name,
        "iy": int(iy),
        "ix": int(ix),
        "row_off": int(row0),
        "col_off": int(col0),
        "side": int(side),
        "stride": stride_i,
        "overlap_frac": float(overlap_frac),
        "n_frames": len(tile_frames),
    }
    if cell_rec is not None:
        row["dates"] = list(cell_rec.get("dates") or [])
        if cell_rec.get("source_plan_cells") is not None:
            row["source_plan_cells"] = cell_rec["source_plan_cells"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Parent S2 revisit directory with meta.json + frames (full MGRS or large AOI).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "data" / "s2_revisits",
        help="Where tile dirs are written.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path (default: <src>/granule_tiles_lr{side}_manifest.json).",
    )
    ap.add_argument("--side", type=int, default=DEFAULT_SIDE)
    ap.add_argument(
        "--overlap-frac",
        type=float,
        default=0.0,
        help="Fractional overlap between adjacent tiles (e.g. 0.1 = 10%%). "
        "Stride = side * (1 - overlap). Default 0 = non-overlapping partition.",
    )
    ap.add_argument("--force", action="store_true", help="Rebuild existing tile dirs.")
    ap.add_argument(
        "--use-aoi-window",
        action="store_true",
        help=(
            "Tile only meta.json aoi_window (dev crops). Default tiles the full "
            "MGRS raster so a national/granule run covers the whole stack."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tile count only; do not write dirs.",
    )
    ap.add_argument(
        "--mainland-only",
        action="store_true",
        help=(
            "Keep only LR windows that overlap mainland Norway (+ nearshore "
            f"islands) using {DEFAULT_LAND_MASK.relative_to(ROOT)}."
        ),
    )
    ap.add_argument(
        "--land-mask",
        type=Path,
        default=None,
        help="GeoJSON land outline for overlap filtering (implies land filter).",
    )
    ap.add_argument(
        "--min-land-frac",
        type=float,
        default=0.0,
        help="Min fraction of n×n window samples on land (0 = any overlap).",
    )
    ap.add_argument(
        "--land-sample-n",
        type=int,
        default=9,
        help="n×n lon/lat samples per LR window for land overlap.",
    )
    ap.add_argument(
        "--clear-counts-dir",
        type=Path,
        default=None,
        help="Optional dir with {MGRS}_lr{side}_clear_counts.tif.",
    )
    ap.add_argument(
        "--min-clear",
        type=int,
        default=None,
        help="Drop LR cells with clear-count < this (needs --clear-counts-dir). "
        "Do not use for the national 2025 recipe (use --cell-plan instead).",
    )
    ap.add_argument(
        "--cell-plan",
        type=Path,
        default=None,
        help="plan.json from plan_national_mgrs.py. Each tile keeps only that "
        "cell's dates; cells with zero SCL-pass days are omitted. No min-clear floor.",
    )
    args = ap.parse_args()

    src = args.src if args.src.is_absolute() else ROOT / args.src
    if not src.is_dir():
        raise SystemExit(f"missing --src {src}")
    meta_path = src / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    frames = meta.get("frames") or []
    if not frames:
        raise SystemExit(f"{meta_path} has no frames")

    first = src / frames[0]["path"]
    row0, col0, height, width = _aoi_extent(
        meta, first, use_aoi_window=bool(args.use_aoi_window)
    )
    side = int(args.side)
    overlap_frac = float(args.overlap_frac)
    if overlap_frac < 0.0 or overlap_frac >= 1.0:
        raise SystemExit("--overlap-frac must be in [0, 1)")
    stride = side if overlap_frac == 0.0 else max(1, int(round(side * (1.0 - overlap_frac))))
    if stride > side:
        raise SystemExit(f"stride {stride} > side {side}")
    overlap_px = side - stride
    ovl_tag = f"_ovl{int(round(overlap_frac * 100))}" if overlap_frac > 0 else ""

    if height < side or width < side:
        raise SystemExit(
            f"AOI {height}×{width} cannot fit any {side}×{side} tile "
            f"(row0={row0}, col0={col0})"
        )

    # Origins: 0, stride, 2*stride, ... while origin+side <= extent; always
    # include a last origin that flushes to the far edge when leftover remains.
    def _origins(extent: int) -> list[int]:
        if extent == side:
            return [0]
        last = extent - side
        out = list(range(0, last + 1, stride))
        if out[-1] != last:
            out.append(last)
        return out

    row_offs = [row0 + o for o in _origins(height)]
    col_offs = [col0 + o for o in _origins(width)]
    n_y, n_x = len(row_offs), len(col_offs)

    land_mask_path = args.land_mask
    if args.mainland_only and land_mask_path is None:
        land_mask_path = DEFAULT_LAND_MASK
    if land_mask_path is not None and not Path(land_mask_path).is_absolute():
        land_mask_path = ROOT / land_mask_path
    clear_dir = args.clear_counts_dir
    if clear_dir is not None and not Path(clear_dir).is_absolute():
        clear_dir = ROOT / clear_dir
    if args.min_clear is not None and clear_dir is None:
        raise SystemExit("--min-clear requires --clear-counts-dir")
    cell_plan = None
    plan_by_cell: dict[tuple[int, int], dict] = {}
    plan_cells: list[dict] = []
    if args.cell_plan is not None:
        if args.min_clear is not None:
            raise SystemExit("use --cell-plan or --min-clear, not both")
        plan_path = args.cell_plan if args.cell_plan.is_absolute() else ROOT / args.cell_plan
        cell_plan = json.loads(plan_path.read_text())
        plan_cells = list(cell_plan.get("cells") or [])
        plan_by_cell = cell_lookup(cell_plan)

    candidates: list[dict] = []
    for iy, r in enumerate(row_offs):
        for ix, c in enumerate(col_offs):
            dest = args.out_root / f"{src.name}_t{side}{ovl_tag}_y{iy:02d}_x{ix:02d}"
            candidates.append(
                {
                    "tile_id": dest.name,
                    "dest": dest,
                    "iy": iy,
                    "ix": ix,
                    "row_off": r,
                    "col_off": c,
                    "side": side,
                    "stride": stride,
                    "overlap_frac": overlap_frac,
                }
            )

    land_filter_meta = None
    if land_mask_path is not None or args.min_clear is not None:
        transform, crs = raster_georef(first)
        land_paths = load_land_paths(land_mask_path) if land_mask_path is not None else None
        mgrs = meta.get("mgrs_tile") or frames[0].get("mgrs_tile")
        kept, dropped, stats = filter_tile_specs(
            candidates,
            transform=transform,
            crs=crs,
            side=side,
            land_paths=land_paths,
            min_land_frac=float(args.min_land_frac),
            sample_n=int(args.land_sample_n),
            clear_counts_dir=clear_dir,
            min_clear=args.min_clear,
            mgrs=mgrs,
        )
        # Preserve dest Path objects (filter copies dicts).
        kept_ids = {t["tile_id"] for t in kept}
        land_frac_by_id = {t["tile_id"]: t.get("land_frac") for t in kept}
        clear_by_id = {t["tile_id"]: t.get("clear_count") for t in kept}
        candidates = [c for c in candidates if c["tile_id"] in kept_ids]
        for c in candidates:
            if land_frac_by_id.get(c["tile_id"]) is not None:
                c["land_frac"] = land_frac_by_id[c["tile_id"]]
            if clear_by_id.get(c["tile_id"]) is not None:
                c["clear_count"] = clear_by_id[c["tile_id"]]
        land_filter_meta = {
            "land_mask": None if land_mask_path is None else str(
                land_mask_path.relative_to(ROOT)
                if land_mask_path.is_relative_to(ROOT)
                else land_mask_path
            ),
            "min_land_frac": float(args.min_land_frac),
            "sample_n": int(args.land_sample_n),
            "min_clear": args.min_clear,
            "clear_counts_dir": None
            if clear_dir is None
            else str(
                clear_dir.relative_to(ROOT) if clear_dir.is_relative_to(ROOT) else clear_dir
            ),
            **{k: stats[k] for k in ("n_in", "n_kept", "n_dropped", "mgrs", "clear_tif")},
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
        }
        print(
            f"land/clear filter: kept {stats['n_kept']}/{stats['n_in']} "
            f"(dropped {stats['n_dropped']})",
            flush=True,
        )

    if cell_plan is not None:
        kept_plan = []
        dropped_plan = 0
        for c in candidates:
            if stride == side:
                rec = plan_by_cell.get((int(c["iy"]), int(c["ix"])))
            else:
                rec = _plan_record_for_window(
                    plan_cells,
                    row_off=int(c["row_off"]),
                    col_off=int(c["col_off"]),
                    side=side,
                )
            if rec is None or int(rec.get("n_frames") or 0) < 1:
                dropped_plan += 1
                continue
            c["cell_rec"] = rec
            kept_plan.append(c)
        candidates = kept_plan
        print(
            f"cell-plan filter: kept {len(candidates)} "
            f"(dropped {dropped_plan} with no selected dates)",
            flush=True,
        )

    tiles: list[dict] = []
    for cand in candidates:
        dest = cand["dest"]
        r, c = int(cand["row_off"]), int(cand["col_off"])
        iy, ix = int(cand["iy"]), int(cand["ix"])
        if args.dry_run:
            row = {
                "tile_id": dest.name,
                "iy": iy,
                "ix": ix,
                "row_off": r,
                "col_off": c,
                "side": side,
                "stride": stride,
                "overlap_frac": overlap_frac,
            }
            if "land_frac" in cand:
                row["land_frac"] = cand["land_frac"]
            if "clear_count" in cand:
                row["clear_count"] = cand["clear_count"]
            if "cell_rec" in cand:
                row["n_frames"] = cand["cell_rec"]["n_frames"]
                row["date_span_days"] = cand["cell_rec"].get("date_span_days")
            tiles.append(row)
            continue
        cell_rec = cand.get("cell_rec")
        tile_frames = meta.get("frames") or []
        if cell_rec is not None:
            tile_frames = filter_frames_for_cell(tile_frames, cell_rec.get("dates") or [])
            if not tile_frames:
                print(
                    f"skip {dest.name}: planned dates not present in parent stack",
                    flush=True,
                )
                continue
        row = _write_tile(
            src,
            meta,
            dest,
            row0=r,
            col0=c,
            side=side,
            iy=iy,
            ix=ix,
            force=args.force,
            stride=stride,
            overlap_frac=overlap_frac,
            frames=tile_frames,
            cell_rec=cell_rec,
        )
        if "land_frac" in cand:
            row["land_frac"] = cand["land_frac"]
        if "clear_count" in cand:
            row["clear_count"] = cand["clear_count"]
        if cell_rec is not None:
            row["date_span_days"] = cell_rec.get("date_span_days")
        tiles.append(row)
        print(
            f"{dest.name}: row={r} col={c} {side}×{side} stride={stride}",
            flush=True,
        )

    rem_h = height - (row_offs[-1] - row0 + side) if row_offs else height
    rem_w = width - (col_offs[-1] - col0 + side) if col_offs else width
    # With edge-flush origins leftover should be 0.
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "src": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "parent": src.name,
        "side": side,
        "stride": stride,
        "overlap_frac": overlap_frac,
        "overlap_px": overlap_px,
        "aoi": {"row_off": row0, "col_off": col0, "height": height, "width": width},
        "use_aoi_window": bool(args.use_aoi_window),
        "grid": {
            "n_y": n_y,
            "n_x": n_x,
            "n_tiles_full_grid": n_y * n_x,
            "n_tiles": len(tiles),
        },
        "skipped_edge_px": {"height": rem_h, "width": rem_w},
        "allow_no_hr": True,
        "merge_method_hint": "feather" if overlap_frac > 0 else "first",
        "tiles": tiles,
    }
    if cell_plan is not None:
        manifest["cell_plan"] = str(
            args.cell_plan if args.cell_plan.is_absolute() else ROOT / args.cell_plan
        )
        manifest["keep_thin_stacks"] = True
    if land_filter_meta is not None:
        manifest["land_filter"] = land_filter_meta
    man_path = args.manifest
    if man_path is None:
        suffix = f"_ovl{int(round(overlap_frac * 100))}" if overlap_frac > 0 else ""
        man_path = src / f"granule_tiles_lr{side}{suffix}_manifest.json"
    if not args.dry_run:
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {len(tiles)} tiles → {man_path}", flush=True)
    else:
        print(
            f"dry-run: full grid {n_y}×{n_x} = {n_y * n_x}; kept {len(tiles)} tiles of {side}² "
            f"stride={stride} overlap={overlap_frac:.0%} ({overlap_px} px) "
            f"on AOI {height}×{width} (edge leftover {rem_h}×{rem_w} px)",
            flush=True,
        )


if __name__ == "__main__":
    main()
