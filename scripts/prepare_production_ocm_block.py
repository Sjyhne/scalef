#!/usr/bin/env python3
"""Prepare one production block to the benchmark radiometry.

Symlinks full-granule OmniCloudMask maps into the cell stacks, drops frames that
fail the benchmark admission test on the cell (thick+thin cloud above 15%, or
less than 85% valid), and writes an ICM identity plan scored with the same
mask the fit will use (thick, thin, and shadow). The existing delivery is not
modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plan_identity_icm import CENTER, assign_icm  # noqa: E402

S2 = ROOT / "data/s2_revisits/national_2025_v2"
ADMIT_CLOUD = 0.15
ADMIT_VALID = 0.85
HARD_CLOUD = 0.02
SLACK_CLOUD = 0.05


def _cells(parent: str, y0: int, x0: int, n: int) -> list[tuple[int, int, Path]]:
    out = []
    for iy in range(y0, y0 + n):
        for ix in range(x0, x0 + n):
            d = S2 / f"{parent}_t512_y{iy:02d}_x{ix:02d}"
            if not (d / "meta.json").is_file():
                raise SystemExit(f"missing cell stack {d}")
            out.append((iy, ix, d))
    return out


def _link_masks(cells: list[tuple[int, int, Path]], parent_dir: Path) -> None:
    for _iy, _ix, d in cells:
        meta = json.loads((d / "meta.json").read_text())
        for fr in meta["frames"]:
            stem = Path(fr["path"]).stem
            src = parent_dir / f"{stem}_ocm.tif"
            dst = d / f"{stem}_ocm.tif"
            if not src.is_file():
                continue
            if dst.is_symlink() or dst.exists():
                continue
            dst.symlink_to(src)


def _score(cells: list[tuple[int, int, Path]], parent_dir: Path) -> dict:
    """Per cell, per frame: admission cloud (thick+thin), validity, and masked fraction."""
    by_frame: dict[str, list[tuple[int, int, dict]]] = {}
    windows: dict[tuple[int, int], dict] = {}
    for iy, ix, d in cells:
        meta = json.loads((d / "meta.json").read_text())
        win = meta.get("aoi_window") or meta["granule_tile"]
        windows[(iy, ix)] = win
        for fr in meta["frames"]:
            by_frame.setdefault(fr["path"], []).append((iy, ix, fr))

    scores: dict[tuple[int, int], dict[str, dict]] = {(iy, ix): {} for iy, ix, _ in cells}
    for frame_path, recs in by_frame.items():
        stem = Path(frame_path).stem
        ocm_path = parent_dir / f"{stem}_ocm.tif"
        rgb_path = parent_dir / frame_path
        if not ocm_path.is_file() or not rgb_path.is_file():
            continue
        with rasterio.open(ocm_path) as ocm_src, rasterio.open(rgb_path) as rgb_src:
            for iy, ix, fr in recs:
                win = windows[(iy, ix)]
                window = Window(int(win["col_off"]), int(win["row_off"]), int(win["width"]), int(win["height"]))
                classes = ocm_src.read(1, window=window)
                rgb = rgb_src.read((1, 2, 3), window=window)
                valid = np.any(rgb > 0, axis=0)
                day = str(fr["datetime"])[:10]
                scores[(iy, ix)][fr["path"]] = {
                    "date": day,
                    "cloud": float(np.isin(classes, (1, 2)).mean()),
                    "valid": float(valid.mean()),
                    "masked": float(np.mean(classes > 0)),
                }
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", default="32VNM")
    ap.add_argument("--y0", type=int, default=2)
    ap.add_argument("--x0", type=int, default=15)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--center", default=CENTER.isoformat())
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Tile manifest to score instead of a square y0/x0/n block. "
            "Cells with no admitted frame are dropped."
        ),
    )
    args = ap.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest is not None:
        manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
        full = json.loads(manifest_path.read_text())
        parent_dir = Path(full["src"])
        parent_dir = parent_dir if parent_dir.is_absolute() else ROOT / parent_dir
        tiles = list(full["tiles"])
        cells = []
        for t in tiles:
            d = Path(t["s2_dir"])
            d = d if d.is_absolute() else ROOT / d
            cells.append((int(t["iy"]), int(t["ix"]), d))
        drop_empty = True
    else:
        parent_dir = S2 / args.parent
        manifest_path = parent_dir / "granule_tiles_lr512_manifest.json"
        full = json.loads(manifest_path.read_text())
        wanted = {
            f"{args.parent}_t512_y{iy:02d}_x{ix:02d}"
            for iy in range(args.y0, args.y0 + args.n)
            for ix in range(args.x0, args.x0 + args.n)
        }
        tiles = [t for t in full["tiles"] if t["tile_id"] in wanted]
        if len(tiles) != args.n * args.n:
            raise SystemExit(f"manifest has {len(tiles)} of {args.n * args.n} cells")
        cells = _cells(args.parent, args.y0, args.x0, args.n)
        drop_empty = False
    _link_masks(cells, parent_dir)
    scores = _score(cells, parent_dir)

    stacks = {}
    date_clouds: dict[tuple[int, int], dict[str, float]] = {}
    tile_ids: dict[tuple[int, int], str] = {}
    kept_tiles = []
    dropped: list[dict] = []
    n_excluded = 0
    tiles_by_id = {t["tile_id"]: t for t in tiles}
    for iy, ix, d in cells:
        meta = json.loads((d / "meta.json").read_text())
        exclude = []
        clouds: dict[str, float] = {}
        for fr in meta["frames"]:
            rec = scores[(iy, ix)].get(fr["path"])
            if rec is None or rec["cloud"] > ADMIT_CLOUD or rec["valid"] < ADMIT_VALID:
                exclude.append(fr["path"])
                continue
            clouds[rec["date"]] = rec["masked"]
        if not clouds:
            if drop_empty:
                dropped.append({"tile_id": d.name, "iy": iy, "ix": ix, "n_frames": len(meta["frames"])})
                continue
            raise SystemExit(f"{d.name} has no frame under the admission rule")
        n_excluded += len(exclude)
        stacks[d.name] = {
            "stack": str(d.relative_to(ROOT)),
            "n_frames": len(meta["frames"]),
            "n_excluded": len(exclude),
            "exclude": sorted(exclude),
            "flagged": {},
            "cloud_mask_suffix": "_ocm",
        }
        date_clouds[(iy, ix)] = clouds
        tile_ids[(iy, ix)] = d.name
        kept_tiles.append(tiles_by_id[d.name])
    if not date_clouds:
        raise SystemExit("no cell has a frame under the admission rule")

    screen = {
        "schema": "frame_screen.v1",
        "rule": (
            "production block: admit when OmniCloudMask thick+thin cloud is at most "
            "15% of the cell and at least 85% of the cell is valid; loss mask is "
            "full-cell thick+thin+shadow"
        ),
        "source": "scripts/prepare_production_ocm_block.py",
        "stacks": stacks,
    }
    screen_path = out_dir / "frame_screen.json"
    screen_path.write_text(json.dumps(screen, indent=1) + "\n")

    plan = assign_icm(
        date_clouds,
        center=date.fromisoformat(args.center),
        hard_cloud=HARD_CLOUD,
        slack_cloud=SLACK_CLOUD,
        parent=str(full.get("parent") or args.parent),
        tile_ids=tile_ids,
    )
    plan["mask"] = "omnicloudmask thick+thin+shadow on the full cell"
    plan["admission"] = {"max_thick_thin": ADMIT_CLOUD, "min_valid": ADMIT_VALID}
    plan_path = out_dir / "identity_plan.json"
    plan_path.write_text(json.dumps(plan, indent=1) + "\n")

    block = dict(full)
    block["tiles"] = kept_tiles
    if args.manifest is None:
        block["grid"] = {"n_y": args.n, "n_x": args.n, "n_tiles": len(kept_tiles)}
    else:
        grid = dict(full.get("grid") or {})
        grid["n_tiles"] = len(kept_tiles)
        grid["n_dropped_admission"] = len(dropped)
        block["grid"] = grid
    block["protocol"] = "benchmark-aligned production block"
    block["admission_dropped"] = dropped
    man_path = out_dir / "manifest.json"
    man_path.write_text(json.dumps(block, indent=1) + "\n")
    print(
        f"cells {len(kept_tiles)}  dropped {len(dropped)}  excluded_frames {n_excluded}  "
        f"primary {plan['primary']}  date_cuts {plan['n_boundaries_independent']}->{plan['n_boundaries_icm']}"
    )
    print(f"screen {screen_path}")
    print(f"plan {plan_path}")
    print(f"manifest {man_path}")


if __name__ == "__main__":
    main()
