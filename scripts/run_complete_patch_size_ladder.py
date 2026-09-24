#!/usr/bin/env python3
"""Nested complete-HR size ladder (same footprints, subdivided).

Locks the complete LR512 tiles (full NIB HR coverage), then subdivides each
into non-overlapping LR 256 / 128 / 64 children covering the **exact same**
ground footprint. Train production recipe (size-aware fused-k) and report:

  1. Per parent-512: mean LPIPS over its children at that size
  2. Per project: mean of those parent scores
  3. Across projects: unweighted mean of project means

This is the fair size comparison (not independent re-grids per LR size).

Example
-------
    python scripts/run_complete_patch_size_ladder.py \\
      --sizes 512 256 128 64 --gpus 8 --skip-existing
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2_dataset import FOCUS_PROJECT_BY_CITY  # noqa: E402

BASE_MANIFEST = (
    ROOT
    / "data"
    / "s2_revisits"
    / "map"
    / "patch_grid_lr512"
    / "complete_patch_tiles_manifest.json"
)
NEST_ROOT = ROOT / "data" / "s2_revisits" / "map" / "patch_grid_nested"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _train_knobs(side: int, run_tag: str = "") -> dict:
    side = int(side)
    if side >= 256:
        knobs = {
            "lr_tile": 128,
            "lr_tiles_per_step": 4,
            "run_prefix": "prod_k4" if side == 512 else f"prod_k4_nest{side}",
        }
    else:
        knobs = {
            "lr_tile": 0,
            "lr_tiles_per_step": 1,
            "run_prefix": f"prod_full_nest{side}",
        }
    if run_tag:
        knobs["run_prefix"] = f"{knobs['run_prefix']}_{run_tag}"
    return knobs


def _project_of_city(city: str) -> str:
    return FOCUS_PROJECT_BY_CITY.get(city, city)


def _write_child_dir(
    parent_tile: dict,
    *,
    side: int,
    iy: int,
    ix: int,
    force: bool,
) -> dict:
    parent_city = parent_tile["parent_city"]
    parent_dir = ROOT / "data" / "s2_revisits" / parent_city
    meta = json.loads((parent_dir / "meta.json").read_text())
    row0 = int(parent_tile["row_off"]) + iy * side
    col0 = int(parent_tile["col_off"]) + ix * side
    prow = int(parent_tile["patch_row"])
    pcol = int(parent_tile["patch_col"])
    tile_id = f"{parent_city}_nest{side}_p{prow:02d}_{pcol:02d}_y{iy:02d}_x{ix:02d}"
    dest = ROOT / "data" / "s2_revisits" / tile_id
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    new_meta = dict(meta)
    new_meta["aoi_window"] = {
        "col_off": int(col0),
        "row_off": int(row0),
        "width": int(side),
        "height": int(side),
    }
    new_meta["parent_s2_dir"] = str(parent_dir)
    new_meta["lr_size_request"] = int(side)
    new_meta["patch_grid"] = {
        "city": parent_city,
        "row": prow,
        "col": pcol,
        "side": int(side),
        "parent_side": 512,
        "nest_iy": int(iy),
        "nest_ix": int(ix),
        "parent_tile_id": parent_tile["tile_id"],
    }
    new_meta["focus_project_folder"] = FOCUS_PROJECT_BY_CITY.get(parent_city)
    for fr in meta.get("frames") or []:
        for key in ("path", "cloud_mask"):
            name = fr.get(key)
            if name:
                _symlink(parent_dir / name, dest / name)
    (dest / "meta.json").write_text(json.dumps(new_meta, indent=2) + "\n")
    return {
        "tile_id": tile_id,
        "s2_dir": str(dest.relative_to(ROOT)),
        "parent_city": parent_city,
        "parent_tile_id": parent_tile["tile_id"],
        "project_folder": FOCUS_PROJECT_BY_CITY.get(parent_city),
        "patch_row": prow,
        "patch_col": pcol,
        "nest_iy": int(iy),
        "nest_ix": int(ix),
        "row_off": int(row0),
        "col_off": int(col0),
        "side": int(side),
        "n_frames": len(meta.get("frames") or []),
    }


def build_nested_manifest(
    base_tiles: list[dict],
    side: int,
    *,
    force: bool,
    max_children_per_parent: int,
    seed: int,
) -> Path:
    """Subdivide each LR512 parent into (512/side)² children; optional subsample."""
    side = int(side)
    if 512 % side != 0:
        raise SystemExit(f"side {side} must divide 512")
    n = 512 // side
    NEST_ROOT.mkdir(parents=True, exist_ok=True)
    man_path = NEST_ROOT / f"nested_lr{side}_manifest.json"
    if man_path.is_file() and not force:
        payload = json.loads(man_path.read_text())
        if int(payload.get("side") or 0) == side and payload.get("tiles"):
            print(f"reuse nested manifest {man_path.relative_to(ROOT)} n={len(payload['tiles'])}", flush=True)
            return man_path

    rng = random.Random(seed + side)
    tiles: list[dict] = []
    for parent in base_tiles:
        coords = [(iy, ix) for iy in range(n) for ix in range(n)]
        if max_children_per_parent > 0 and len(coords) > max_children_per_parent:
            coords = rng.sample(coords, max_children_per_parent)
        for iy, ix in coords:
            if side == 512:
                # Reuse existing parent dirs / ids for skip-existing on prior bench.
                tiles.append(
                    {
                        **parent,
                        "parent_tile_id": parent["tile_id"],
                        "project_folder": FOCUS_PROJECT_BY_CITY.get(parent["parent_city"]),
                        "nest_iy": 0,
                        "nest_ix": 0,
                    }
                )
            else:
                tiles.append(
                    _write_child_dir(parent, side=side, iy=iy, ix=ix, force=force)
                )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "nested_from_lr512",
        "base_manifest": str(BASE_MANIFEST.relative_to(ROOT)),
        "side": side,
        "n_per_parent_full": n * n,
        "max_children_per_parent": max_children_per_parent,
        "n_parents": len(base_tiles),
        "n_tiles": len(tiles),
        "tiles": tiles,
    }
    man_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"nested LR{side}: {len(base_tiles)} parents → {len(tiles)} tiles → {man_path.relative_to(ROOT)}",
        flush=True,
    )
    return man_path


def bench(
    side: int,
    man: Path,
    *,
    gpus: int,
    gpu_offset: int,
    iters: int,
    skip_existing: bool,
    limit: int,
    run_tag: str,
) -> Path:
    knobs = _train_knobs(side, run_tag)
    suffix = f"_{run_tag}" if run_tag else ""
    out = (
        ROOT
        / "single_samples"
        / "sweep_results"
        / f"bench_complete_patches_nest_lr{side}{suffix}.json"
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bench_complete_patches.py"),
        "--manifest",
        str(man),
        "--gpus",
        str(gpus),
        "--gpu-offset",
        str(gpu_offset),
        "--iters",
        str(iters),
        "--out",
        str(out),
        "--lr-tile",
        str(knobs["lr_tile"]),
        "--lr-tiles-per-step",
        str(knobs["lr_tiles_per_step"]),
        "--run-prefix",
        knobs["run_prefix"],
        "--side",
        str(side),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    _run(cmd)
    return out


def summarize_ladder(bench_paths: dict[int, Path], *, run_tag: str = "") -> dict:
    """Primary score = mean over parent-512 footprints (children averaged first)."""
    by_size: dict[str, dict] = {}
    for side, path in sorted(bench_paths.items()):
        payload = json.loads(path.read_text())
        rows = [r for r in (payload.get("rows") or []) if r.get("lpips") is not None]
        # Map tile_id → parent via nest manifest if needed
        man = json.loads((NEST_ROOT / f"nested_lr{side}_manifest.json").read_text())
        parent_of = {
            t["tile_id"]: t.get("parent_tile_id") or t["tile_id"] for t in man["tiles"]
        }
        city_of = {t["tile_id"]: t["parent_city"] for t in man["tiles"]}

        by_parent: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            pid = parent_of.get(r["tile_id"], r["tile_id"])
            by_parent[pid].append(r)

        parent_scores = []
        by_proj: dict[str, list[float]] = defaultdict(list)
        by_proj_bil: dict[str, list[float]] = defaultdict(list)
        for pid, rs in by_parent.items():
            lp = sum(r["lpips"] for r in rs) / len(rs)
            bil_vals = [r["lpips_bilinear"] for r in rs if r.get("lpips_bilinear") is not None]
            bil = (sum(bil_vals) / len(bil_vals)) if bil_vals else None
            city = rs[0].get("parent_city") or city_of.get(pid, "")
            proj = _project_of_city(city)
            parent_scores.append(
                {
                    "parent_tile_id": pid,
                    "parent_city": city,
                    "project_folder": proj,
                    "n_children": len(rs),
                    "mean_lpips": lp,
                    "mean_lpips_bilinear": bil,
                }
            )
            by_proj[proj].append(lp)
            if bil is not None:
                by_proj_bil[proj].append(bil)

        per_project = {
            proj: {
                "n_parents": len(vals),
                "mean_lpips": sum(vals) / len(vals),
                "mean_lpips_bilinear": (
                    sum(by_proj_bil[proj]) / len(by_proj_bil[proj])
                    if by_proj_bil.get(proj)
                    else None
                ),
            }
            for proj, vals in sorted(by_proj.items())
        }
        proj_means = [v["mean_lpips"] for v in per_project.values()]
        proj_bil = [
            v["mean_lpips_bilinear"]
            for v in per_project.values()
            if v.get("mean_lpips_bilinear") is not None
        ]
        by_size[str(side)] = {
            "bench_json": str(path.relative_to(ROOT)),
            "n_tiles_ok": len(rows),
            "n_parents": len(parent_scores),
            "n_projects": len(per_project),
            "mean_lpips_across_parents": (
                sum(p["mean_lpips"] for p in parent_scores) / len(parent_scores)
                if parent_scores
                else None
            ),
            "mean_lpips_across_projects": (
                sum(proj_means) / len(proj_means) if proj_means else None
            ),
            "mean_bilinear_across_projects": (
                sum(proj_bil) / len(proj_bil) if proj_bil else None
            ),
            "per_project": per_project,
            "train_knobs": _train_knobs(side, run_tag),
        }

    table = []
    for side, block in by_size.items():
        table.append(
            {
                "lr_side": int(side),
                "n_projects": block["n_projects"],
                "n_parents": block["n_parents"],
                "n_tiles": block["n_tiles_ok"],
                "mean_lpips_project": block["mean_lpips_across_projects"],
                "mean_lpips_parent": block["mean_lpips_across_parents"],
                "mean_lpips_bilinear_project": block["mean_bilinear_across_projects"],
            }
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "nested_from_lr512",
        "run_tag": run_tag,
        "metric": (
            "lpips (lower better). Primary = unweighted mean of per-project means; "
            "each project mean averages parent-512 footprints; each parent score is "
            "the mean of its nested children at that LR size."
        ),
        "table": table,
        "by_size": by_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", type=int, default=[512, 256, 128, 64])
    ap.add_argument(
        "--base-manifest",
        type=Path,
        default=BASE_MANIFEST,
        help="LR512 complete-patch manifest (defines locked footprints).",
    )
    ap.add_argument(
        "--max-children-per-parent",
        type=int,
        default=0,
        help="Optional subsample of nested children per parent (0 = all). "
        "Useful to cap LR64 (64 children/parent).",
    )
    ap.add_argument(
        "--max-children-64",
        type=int,
        default=0,
        help="If >0, only applied when side==64 (e.g. 8 keeps overnight feasible).",
    )
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-offset", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    ap.add_argument("--force-tiles", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--exclude-tile",
        action="append",
        default=[],
        help="LR512 parent tile id to exclude as non-evaluable (repeatable).",
    )
    ap.add_argument(
        "--run-tag",
        default="",
        help="Suffix run names and bench JSONs (for example, lr512align_v2).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "single_samples"
        / "sweep_results"
        / "bench_complete_patches_size_ladder_nested.json",
    )
    args = ap.parse_args()

    base_path = args.base_manifest if args.base_manifest.is_absolute() else ROOT / args.base_manifest
    base = json.loads(base_path.read_text())
    excluded = set(args.exclude_tile)
    base_tiles = [tile for tile in base["tiles"] if tile["tile_id"] not in excluded]
    if excluded:
        missing = excluded - {tile["tile_id"] for tile in base["tiles"]}
        if missing:
            raise SystemExit(f"excluded tile ids not in base manifest: {sorted(missing)}")
        print(f"excluded non-evaluable LR512 parents: {sorted(excluded)}", flush=True)
    print(
        f"locked footprints: {len(base_tiles)} LR512 complete tiles from {base_path.relative_to(ROOT)}",
        flush=True,
    )

    bench_paths: dict[int, Path] = {}
    for side in args.sizes:
        print(f"\n===== nested LR {side} =====", flush=True)
        cap = int(args.max_children_per_parent)
        if cap <= 0 and int(side) == 64 and int(args.max_children_64) > 0:
            cap = int(args.max_children_64)
        man = build_nested_manifest(
            base_tiles,
            int(side),
            force=args.force_tiles,
            max_children_per_parent=cap,
            seed=args.seed,
        )
        if args.skip_train:
            continue
        bench_paths[int(side)] = bench(
            int(side),
            man,
            gpus=args.gpus,
            gpu_offset=args.gpu_offset,
            iters=args.iters,
            skip_existing=args.skip_existing,
            limit=args.limit,
            run_tag=args.run_tag,
        )

    if bench_paths:
        # Ensure nest manifests exist for summarize (512 may use base tile ids).
        if 512 in bench_paths and not (NEST_ROOT / "nested_lr512_manifest.json").is_file():
            build_nested_manifest(
                base_tiles, 512, force=True, max_children_per_parent=0, seed=args.seed
            )
        summary = summarize_ladder(bench_paths, run_tag=args.run_tag)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print("\n=== nested size ladder (mean LPIPS across projects) ===", flush=True)
        for row in summary["table"]:
            print(
                f"  LR{row['lr_side']:>4}: projects={row['n_projects']} "
                f"parents={row['n_parents']} tiles={row['n_tiles']}  "
                f"LPIPS={row['mean_lpips_project']:.4f}  "
                f"bilinear={row['mean_lpips_bilinear_project']}",
                flush=True,
            )
        print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
