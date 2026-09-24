#!/usr/bin/env python3
"""Rescore nested size-ladder runs on the shared LR64 geographic window.

Each LR64 child defines the comparison window. Predictions from the covering
LR128/256/512 parent fields are cropped to that window and scored against the
child's saved ground truth and bilinear baseline. Bilinear LPIPS is therefore
identical across sizes by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.common_footprint import crop_hwc_frac, nest_window_frac  # noqa: E402
from eval.masked_metrics import mask_bbox_slices, masked_psnr  # noqa: E402
from optimize import get_lpips_model  # noqa: E402
from s2_dataset import _l2a_to_reflectance  # noqa: E402
from scripts.bench_complete_patches import _run_name  # noqa: E402
from scripts.run_complete_patch_size_ladder import _project_of_city, _train_knobs  # noqa: E402

NEST_ROOT = ROOT / "data" / "s2_revisits" / "map" / "patch_grid_nested"
SIDES = (64, 128, 256, 512)
TARGET_HW = 256


def _load_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(image, 0.0, 1.0)


def _hwc_to_bchw(image: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(image, (TARGET_HW, TARGET_HW), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))
    return tensor.unsqueeze(0)


def _raw_base_lr(child: dict, run_dir: Path) -> np.ndarray:
    """Read the LR64 run's designated base frame without display stretching."""
    s2_dir = ROOT / child["s2_dir"]
    meta = json.loads((s2_dir / "meta.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    base_date = metrics["base_frame"]["date"]
    matches = [
        frame
        for frame in meta["frames"]
        if str(frame.get("datetime") or "")[:10] == base_date
        or base_date.replace("-", "") in str(frame.get("path") or "")
    ]
    if len(matches) != 1:
        raise ValueError(f"{child['tile_id']}: expected one frame on {base_date}, got {len(matches)}")
    aoi = meta["aoi_window"]
    window = Window(aoi["col_off"], aoi["row_off"], aoi["width"], aoi["height"])
    with rasterio.open(s2_dir / matches[0]["path"]) as src:
        stack = src.read(window=window)
    return np.transpose(_l2a_to_reflectance(stack)[0:3], (1, 2, 0))


def _infer_display_vmax(display_lr: np.ndarray, raw_lr: np.ndarray) -> tuple[float, float]:
    """Recover the scalar used by ``_shared_display_stretch`` from the LR panel."""
    display_native = cv2.resize(
        display_lr,
        (int(raw_lr.shape[1]), int(raw_lr.shape[0])),
        interpolation=cv2.INTER_AREA,
    )
    valid = (display_native > 0.05) & (display_native < 0.95) & (raw_lr > 0)
    if int(valid.sum()) < 100:
        raise ValueError("insufficient unclipped LR pixels to infer display stretch")
    vmax = float(np.median(raw_lr[valid] / display_native[valid]))
    mae = float(np.abs(display_native * vmax - raw_lr).mean())
    return vmax, mae


def _base_date(run_dir: Path) -> str:
    return str(json.loads((run_dir / "metrics.json").read_text())["base_frame"]["date"])


def _run_dir(parent_city: str, tile_id: str, side: int, run_tag: str) -> Path:
    prefix = _train_knobs(side, run_tag)["run_prefix"]
    return ROOT / "single_samples" / parent_city / "sample" / _run_name(tile_id, prefix)


def _covering_tile(child: dict, parent_side: int, index: dict[tuple, dict]) -> dict | None:
    if parent_side == 64:
        return child
    stride = parent_side // 64
    key = (
        child["parent_city"],
        int(child["patch_row"]),
        int(child["patch_col"]),
        int(child["nest_iy"]) // stride,
        int(child["nest_ix"]) // stride,
        parent_side,
    )
    if parent_side == 512:
        key = (child["parent_city"], int(child["patch_row"]), int(child["patch_col"]), 512)
    return index.get(key)


def _index_manifests() -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for side in SIDES:
        payload = json.loads((NEST_ROOT / f"nested_lr{side}_manifest.json").read_text())
        for tile in payload["tiles"]:
            city = tile["parent_city"]
            prow, pcol = int(tile["patch_row"]), int(tile["patch_col"])
            if side == 512:
                out[(city, prow, pcol, 512)] = tile
            else:
                out[(city, prow, pcol, int(tile["nest_iy"]), int(tile["nest_ix"]), side)] = tile
    return out


def _aggregate(rows: list[dict]) -> dict:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_size[int(row["side"])].append(row)
    table = []
    by_size_out = {}
    for side in SIDES:
        group = by_size.get(side, [])
        by_parent: dict[str, list[dict]] = defaultdict(list)
        for row in group:
            by_parent[row["parent_tile_id"]].append(row)
        parent_scores = []
        by_proj: dict[str, list[float]] = defaultdict(list)
        by_proj_bil: dict[str, list[float]] = defaultdict(list)
        for pid, rs in by_parent.items():
            lp = sum(r["lpips"] for r in rs) / len(rs)
            bil = sum(r["lpips_bilinear"] for r in rs) / len(rs)
            proj = rs[0]["project_folder"]
            parent_scores.append(
                {
                    "parent_tile_id": pid,
                    "parent_city": rs[0]["parent_city"],
                    "project_folder": proj,
                    "n_children": len(rs),
                    "mean_lpips": lp,
                    "mean_lpips_bilinear": bil,
                }
            )
            by_proj[proj].append(lp)
            by_proj_bil[proj].append(bil)
        per_project = {
            proj: {
                "n_parents": len(vals),
                "mean_lpips": sum(vals) / len(vals),
                "mean_lpips_bilinear": sum(by_proj_bil[proj]) / len(by_proj_bil[proj]),
            }
            for proj, vals in sorted(by_proj.items())
        }
        proj_means = [v["mean_lpips"] for v in per_project.values()]
        proj_bil = [v["mean_lpips_bilinear"] for v in per_project.values()]
        block = {
            "n_tiles_ok": len(group),
            "n_parents": len(parent_scores),
            "n_projects": len(per_project),
            "mean_lpips_across_parents": (
                sum(p["mean_lpips"] for p in parent_scores) / len(parent_scores) if parent_scores else None
            ),
            "mean_lpips_across_projects": (
                sum(proj_means) / len(proj_means) if proj_means else None
            ),
            "mean_bilinear_across_projects": (
                sum(proj_bil) / len(proj_bil) if proj_bil else None
            ),
            "per_project": per_project,
        }
        by_size_out[str(side)] = block
        table.append(
            {
                "lr_side": side,
                "n_projects": block["n_projects"],
                "n_parents": block["n_parents"],
                "n_tiles": block["n_tiles_ok"],
                "mean_lpips_project": block["mean_lpips_across_projects"],
                "mean_lpips_parent": block["mean_lpips_across_parents"],
                "mean_lpips_bilinear_project": block["mean_bilinear_across_projects"],
            }
        )
    return {"table": table, "by_size": by_size_out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="lr512align_v2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "single_samples"
        / "sweep_results"
        / "bench_complete_patches_size_ladder_nested_common_footprint.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lpips_fn = get_lpips_model(device)
    index = _index_manifests()
    children = json.loads((NEST_ROOT / "nested_lr64_manifest.json").read_text())["tiles"]
    if args.limit > 0:
        children = children[: args.limit]

    image_cache: dict[Path, np.ndarray] = {}
    rows: list[dict] = []
    missing = 0
    excluded_base_frame = 0
    stretch_recovery_mae: list[float] = []
    baseline_match_mae: list[float] = []
    for i, child in enumerate(children, start=1):
        child_dir = _run_dir(child["parent_city"], child["tile_id"], 64, args.run_tag)
        gt_path = child_dir / "ground_truth.png"
        bil_path = child_dir / "bilinear_baseline.png"
        pred64_path = child_dir / "model_output_aligned.png"
        if not gt_path.is_file() or not bil_path.is_file() or not pred64_path.is_file():
            missing += 1
            continue
        covering = {side: _covering_tile(child, side, index) for side in SIDES}
        if any(tile is None for tile in covering.values()):
            missing += 1
            continue
        run_dirs = {
            side: _run_dir(tile["parent_city"], tile["tile_id"], side, args.run_tag)
            for side, tile in covering.items()
        }
        dates = {_base_date(run_dir) for run_dir in run_dirs.values()}
        if len(dates) != 1:
            excluded_base_frame += 1
            continue

        display_gt = _load_rgb(gt_path)
        display_bilinear = _load_rgb(bil_path)
        display_lr = _load_rgb(child_dir / "lr_original.png")
        vmax64, recovery_mae = _infer_display_vmax(
            display_lr, _raw_base_lr(child, child_dir)
        )
        stretch_recovery_mae.append(recovery_mae)
        gt_t = (_hwc_to_bchw(display_gt) * vmax64).to(device)
        bil_t = (_hwc_to_bchw(display_bilinear) * vmax64).to(device)
        mask = torch.all(gt_t[0] > 1e-6, dim=0)
        y0, y1, x0, x1 = mask_bbox_slices(mask)
        bilinear_lpips = float(
            lpips_fn(
                bil_t[:, :, y0:y1, x0:x1] * 2 - 1,
                gt_t[:, :, y0:y1, x0:x1] * 2 - 1,
            ).item()
        )
        bilinear_psnr = masked_psnr(bil_t, gt_t, mask)
        if i % 200 == 0 or i == 1:
            print(f"{i}/{len(children)} {child['tile_id']}", flush=True)
        for side in SIDES:
            tile = covering[side]
            run_dir = run_dirs[side]
            pred_path = run_dir / "model_output_aligned.png"
            if not pred_path.is_file():
                missing += 1
                continue
            if pred_path not in image_cache:
                image_cache[pred_path] = _load_rgb(pred_path)
            if side == 64:
                pred_display = image_cache[pred_path]
                relative_vmax = 1.0
            else:
                n = side // 64
                fy, fx, fh, fw = nest_window_frac(
                    int(child["nest_iy"]) % n,
                    int(child["nest_ix"]) % n,
                    64,
                    side,
                )
                pred_display = crop_hwc_frac(image_cache[pred_path], fy, fx, fh, fw)
                source_bil_path = run_dir / "bilinear_baseline.png"
                if source_bil_path not in image_cache:
                    image_cache[source_bil_path] = _load_rgb(source_bil_path)
                source_bil = _hwc_to_bchw(
                    crop_hwc_frac(image_cache[source_bil_path], fy, fx, fh, fw)
                )
                target_bil = _hwc_to_bchw(display_bilinear)
                valid = (
                    (source_bil > 0.05)
                    & (source_bil < 0.95)
                    & (target_bil > 0.05)
                    & (target_bil < 0.95)
                )
                if int(valid.sum()) < 100:
                    raise ValueError(f"{child['tile_id']} LR{side}: baseline overlap too small")
                relative_vmax = float(torch.median(target_bil[valid] / source_bil[valid]))
                baseline_match_mae.append(
                    float(torch.mean(torch.abs(source_bil * relative_vmax - target_bil)))
                )
            pred_t = (_hwc_to_bchw(pred_display) * vmax64 * relative_vmax).to(device)
            model_lpips = float(
                lpips_fn(
                    pred_t[:, :, y0:y1, x0:x1] * 2 - 1,
                    gt_t[:, :, y0:y1, x0:x1] * 2 - 1,
                ).item()
            )
            model_psnr = masked_psnr(pred_t, gt_t, mask)
            rows.append(
                {
                    "child_tile_id": child["tile_id"],
                    "parent_tile_id": child["parent_tile_id"],
                    "parent_city": child["parent_city"],
                    "project_folder": _project_of_city(child["parent_city"]),
                    "side": side,
                    "source_tile_id": tile["tile_id"],
                    "lpips": model_lpips,
                    "lpips_bilinear": bilinear_lpips,
                    "lpips_vs_bilinear": bilinear_lpips - model_lpips,
                    "psnr": model_psnr,
                    "psnr_bilinear": bilinear_psnr,
                    "hr_hw": [TARGET_HW, TARGET_HW],
                    "common_footprint": True,
                    "display_vmax_recovered": vmax64 * relative_vmax,
                }
            )
        if len(image_cache) > 192:
            image_cache.clear()

    if not rows:
        raise SystemExit("no common-footprint rows scored")
    summary = _aggregate(rows)
    payload = {
        "schema": "scalef.nested_common_footprint.v2",
        "mode": "nested_common_footprint_lr64",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "window": "lr64_child",
        "target_hw": [TARGET_HW, TARGET_HW],
        "n_children": len(children),
        "n_scored_rows": len(rows),
        "n_missing": missing,
        "n_excluded_base_frame_mismatch": excluded_base_frame,
        "recovery_validation": {
            "mean_lr_stretch_recovery_mae": float(np.mean(stretch_recovery_mae)),
            "max_lr_stretch_recovery_mae": float(np.max(stretch_recovery_mae)),
            "mean_scaled_baseline_match_mae": float(np.mean(baseline_match_mae)),
            "max_scaled_baseline_match_mae": float(np.max(baseline_match_mae)),
        },
        "table": summary["table"],
        "by_size": summary["by_size"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)
    for row in summary["table"]:
        print(
            f"  LR{row['lr_side']:>4}: parents={row['n_parents']} tiles={row['n_tiles']}  "
            f"LPIPS={row['mean_lpips_project']:.4f}  "
            f"bilinear={row['mean_lpips_bilinear_project']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
