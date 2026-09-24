#!/usr/bin/env python3
"""End-to-end production for one or more full MGRS revisit stacks.

For each parent under ``data/s2_revisits/<name>/``:

1. ``make_granule_tiles.py`` — LR512 grid + manifest (optionally mainland-only)
2. ``run_production.py`` — train all AOIs, write ``qgis/sr_pred.tif``
3. ``mosaic_granule_sr.py`` — stitch into ``production/mosaics/<name>_sr_2p5m.tif``

Example
-------
    python scripts/run_granule_batch.py \\
      --parents asker bergen rana tromso amli stavanger \\
      --gpus 8

National Norway (skip ocean LR512 cells)::

    python scripts/run_granule_batch.py \\
      --parents 32VNM 32VKL ... \\
      --mainland-only --min-clear 6 \\
      --clear-counts-dir production/cloud_availability/lr512_norway_july2025_pm45 \\
      --gpus 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.land_mask_lr512 import DEFAULT_LAND_MASK  # noqa: E402


def _manifest_path(parent: str, side: int, overlap_frac: float = 0.0) -> Path:
    suffix = f"_ovl{int(round(overlap_frac * 100))}" if overlap_frac > 0 else ""
    return ROOT / "data" / "s2_revisits" / parent / f"granule_tiles_lr{side}{suffix}_manifest.json"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def process_parent(
    parent: str,
    *,
    side: int,
    overlap_frac: float,
    iters: int,
    gpus: int,
    gpu_offset: int,
    skip_existing: bool,
    skip_train: bool,
    skip_mosaic: bool,
    force_tiles: bool,
    mainland_only: bool,
    land_mask: Path | None,
    min_land_frac: float,
    clear_counts_dir: Path | None,
    min_clear: int | None,
) -> dict:
    src = ROOT / "data" / "s2_revisits" / parent
    if not (src / "meta.json").is_file():
        raise FileNotFoundError(f"missing {src}/meta.json")

    man = _manifest_path(parent, side, overlap_frac)
    tile_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_granule_tiles.py"),
        "--src",
        str(src),
        "--side",
        str(side),
        "--overlap-frac",
        str(overlap_frac),
        "--manifest",
        str(man),
    ]
    if force_tiles:
        tile_cmd.append("--force")
    if mainland_only:
        tile_cmd.append("--mainland-only")
    if land_mask is not None:
        tile_cmd.extend(["--land-mask", str(land_mask)])
    if mainland_only or land_mask is not None:
        tile_cmd.extend(["--min-land-frac", str(min_land_frac)])
    if clear_counts_dir is not None:
        tile_cmd.extend(["--clear-counts-dir", str(clear_counts_dir)])
    if min_clear is not None:
        tile_cmd.extend(["--min-clear", str(min_clear)])

    if not man.is_file() or force_tiles:
        _run(tile_cmd)
    else:
        print(f"reuse manifest {man.relative_to(ROOT)}", flush=True)

    ovl_tag = f"_ovl{int(round(overlap_frac * 100))}" if overlap_frac > 0 else ""
    prod_summary = (
        ROOT
        / "single_samples"
        / "sweep_results"
        / f"production_{parent}_lr{side}{ovl_tag}.json"
    )
    if not skip_train:
        train_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_production.py"),
            "--manifest",
            str(man),
            "--iters",
            str(iters),
            "--gpus",
            str(gpus),
            "--gpu-offset",
            str(gpu_offset),
            "--out",
            str(prod_summary),
        ]
        if skip_existing:
            train_cmd.append("--skip-existing")
        # Re-apply filter at train time only if manifest was reused from an
        # unfiltered earlier run.
        if mainland_only or land_mask is not None or min_clear is not None:
            man_obj = json.loads(man.read_text())
            if "land_filter" not in man_obj:
                if mainland_only:
                    train_cmd.append("--mainland-only")
                if land_mask is not None:
                    train_cmd.extend(["--land-mask", str(land_mask)])
                train_cmd.extend(["--min-land-frac", str(min_land_frac)])
                if clear_counts_dir is not None:
                    train_cmd.extend(["--clear-counts-dir", str(clear_counts_dir)])
                if min_clear is not None:
                    train_cmd.extend(["--min-clear", str(min_clear)])
        _run(train_cmd)

    mosaic_out = ROOT / "production" / "mosaics" / f"{parent}{ovl_tag}_sr_2p5m.tif"
    mosaic_meta = None
    if not skip_mosaic:
        mos_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "mosaic_granule_sr.py"),
            "--manifest",
            str(man),
            "--out",
            str(mosaic_out),
            "--allow-missing",
        ]
        if overlap_frac > 0:
            mos_cmd.extend(["--method", "feather"])
        _run(mos_cmd)
        side_json = mosaic_out.with_suffix(mosaic_out.suffix + ".json")
        if side_json.is_file():
            mosaic_meta = json.loads(side_json.read_text())

    man_obj = json.loads(man.read_text())
    n_tiles = len(man_obj["tiles"])
    return {
        "parent": parent,
        "manifest": str(man.relative_to(ROOT)),
        "n_tiles": n_tiles,
        "land_filter": man_obj.get("land_filter"),
        "overlap_frac": overlap_frac,
        "production_summary": str(prod_summary.relative_to(ROOT)) if prod_summary.is_file() else None,
        "mosaic": str(mosaic_out.relative_to(ROOT)) if mosaic_out.is_file() else None,
        "mosaic_meta": mosaic_meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parents",
        nargs="+",
        required=True,
        help="Parent revisit folder names under data/s2_revisits/ (one MGRS stack each).",
    )
    ap.add_argument("--side", type=int, default=512)
    ap.add_argument(
        "--overlap-frac",
        type=float,
        default=0.0,
        help="Tile overlap fraction (e.g. 0.1). Mosaic uses mean blend when >0.",
    )
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-mosaic", action="store_true")
    ap.add_argument("--force-tiles", action="store_true")
    ap.add_argument(
        "--mainland-only",
        action="store_true",
        help=f"Skip ocean LR512 cells using {DEFAULT_LAND_MASK.relative_to(ROOT)}.",
    )
    ap.add_argument(
        "--land-mask",
        type=Path,
        default=None,
        help="Custom GeoJSON land outline (overrides default when set).",
    )
    ap.add_argument(
        "--min-land-frac",
        type=float,
        default=0.0,
        help="Min land-sample fraction (0 = any overlap).",
    )
    ap.add_argument(
        "--clear-counts-dir",
        type=Path,
        default=None,
        help="Dir with {MGRS}_lr512_clear_counts.tif for --min-clear.",
    )
    ap.add_argument(
        "--min-clear",
        type=int,
        default=None,
        help="Drop LR cells below this clear-count.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "granule_batch.json",
    )
    args = ap.parse_args()

    land_mask = args.land_mask
    if land_mask is not None and not land_mask.is_absolute():
        land_mask = ROOT / land_mask
    clear_dir = args.clear_counts_dir
    if clear_dir is not None and not clear_dir.is_absolute():
        clear_dir = ROOT / clear_dir
    if args.min_clear is not None and clear_dir is None:
        raise SystemExit("--min-clear requires --clear-counts-dir")

    # Deduplicate by MGRS id when possible (same granule, different city labels).
    selected: list[str] = []
    seen_mgrs: set[str] = set()
    for parent in args.parents:
        meta_path = ROOT / "data" / "s2_revisits" / parent / "meta.json"
        mgrs = None
        if meta_path.is_file():
            frames = json.loads(meta_path.read_text()).get("frames") or []
            if frames:
                mgrs = frames[0].get("mgrs_tile")
        if mgrs and mgrs in seen_mgrs:
            print(f"skip {parent}: MGRS {mgrs} already in batch", flush=True)
            continue
        if mgrs:
            seen_mgrs.add(mgrs)
        selected.append(parent)

    results = []
    for parent in selected:
        print(f"\n===== {parent} =====", flush=True)
        try:
            results.append(
                process_parent(
                    parent,
                    side=args.side,
                    overlap_frac=float(args.overlap_frac),
                    iters=args.iters,
                    gpus=args.gpus,
                    gpu_offset=args.gpu_offset,
                    skip_existing=args.skip_existing,
                    skip_train=args.skip_train,
                    skip_mosaic=args.skip_mosaic,
                    force_tiles=args.force_tiles,
                    mainland_only=bool(args.mainland_only),
                    land_mask=land_mask,
                    min_land_frac=float(args.min_land_frac),
                    clear_counts_dir=clear_dir,
                    min_clear=args.min_clear,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"parent": parent, "error": str(exc)})
            print(f"FAIL {parent}: {exc}", flush=True)

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parents_requested": list(args.parents),
        "parents_run": selected,
        "mgrs_seen": sorted(seen_mgrs),
        "overlap_frac": float(args.overlap_frac),
        "mainland_only": bool(args.mainland_only),
        "min_clear": args.min_clear,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nBatch summary → {args.out}", flush=True)


if __name__ == "__main__":
    main()
