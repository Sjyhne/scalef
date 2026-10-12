#!/usr/bin/env python3
"""Score sampling geometry on the fixed-budget, no-hold-out named-site fits.

This is a post-hoc diagnostic. It reads the affines exported at the end of
each fit and does not resume training, restore a checkpoint, or change the
reported images.

The orthophoto shifts in eval/spatial_alignment.json are not inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.sampling_geometry import (  # noqa: E402
    circular_deviation,
    sampling_geometry_score,
    sampling_offsets_lr,
)

FACTORS = (2, 3, 4)
PATCH = 64
PROBES = (8, 24, 40, 56)
CENTER_PROBE = 24
SEEDS = (6, 7, 8)
RUN_PREFIX = "confirmatory_nocut__b0_17__lr512_b0__"
PERTURBATION_DRAWS = 30
PERTURBATION_SEED = 20261011
ABSOLUTE_PERTURBATION_LR_PX = 0.05


def _run_dir(city: str, seed: int) -> Path:
    name = f"{RUN_PREFIX}{city}__seed{seed}"
    return ROOT / "single_samples" / city / "confirmatory_nocut" / name


def _discover() -> list[tuple[str, int, Path]]:
    found = []
    root = ROOT / "single_samples"
    for city_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seed in SEEDS:
            path = _run_dir(city_dir.name, seed)
            if path.is_dir():
                found.append((city_dir.name, seed, path))
    return found


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _compatible(metrics: dict) -> tuple[bool, str]:
    args = metrics.get("provenance", {}).get("args", {})
    holdout = float(args.get("spatial_holdout", -1))
    iters = int(metrics.get("iterations") or args.get("iters") or 0)
    completed = int(metrics.get("completed_iters") or 0)
    if holdout != 0.0:
        return False, f"spatial_holdout={holdout}"
    if metrics.get("early_stop") is not None:
        return False, "early_stop is set"
    if int(args.get("early_stop_patience") or 0) != 0:
        return False, "early_stop_patience is not 0"
    if completed != iters or completed <= 0:
        return False, f"completed_iters={completed}, iterations={iters}"
    return True, "final parameters after a fixed budget"


def _metric_gain(block: dict, higher_is_better: bool) -> float:
    model = float(block["model"])
    bilinear = float(block["bilinear"])
    return model - bilinear if higher_is_better else bilinear - model


def _frames_from_dataset(args: dict, cache_path: Path) -> tuple[np.ndarray, list[dict]]:
    if cache_path.is_file():
        payload = np.load(cache_path, allow_pickle=False)
        identities = json.loads(cache_path.with_suffix(".json").read_text())
        return payload["clear"], identities
    from s2_dataset import S2NIBRevisitDataset, _frame_acquisition_date

    namespace = Namespace(**args)
    namespace.dataset_device = "cpu"
    namespace.device = "cpu"
    dataset = S2NIBRevisitDataset(namespace, training_device="cpu")
    clear = dataset.get_lr_clear().detach().cpu().numpy().astype(np.bool_)
    identities = []
    for frame in dataset.frames:
        identities.append(
            {
                "path": frame.get("path"),
                "date": _frame_acquisition_date(frame).isoformat(),
                "stac_id": frame.get("stac_id") or frame.get("id"),
            }
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, clear=clear)
    cache_path.with_suffix(".json").write_text(json.dumps(identities, indent=2))
    return clear, identities


def _probe_coordinates(row0: int, col0: int) -> tuple[np.ndarray, np.ndarray]:
    local_r, local_c = np.meshgrid(PROBES, PROBES, indexing="ij")
    return (row0 + local_r).reshape(-1), (col0 + local_c).reshape(-1)


def _score_location(offsets_tn2: np.ndarray, usable: np.ndarray) -> dict[int, dict]:
    return {factor: sampling_geometry_score(offsets_tn2, usable, factor) for factor in FACTORS}


def _summarize_factor(results: list[dict]) -> dict:
    usable = [item for item in results if item["status"] == "ok"]
    if not usable:
        return {
            "status": "no_data",
            "score_median": None,
            "score_min": None,
            "score_max": None,
            "rank_median": None,
            "n_usable_median": None,
        }
    scores = np.array([item["score"] for item in usable], dtype=np.float64)
    ranks = np.array([item["rank"] for item in usable], dtype=np.float64)
    counts = np.array([item["n_usable"] for item in usable], dtype=np.float64)
    return {
        "status": "ok",
        "score_median": float(np.median(scores)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "rank_median": float(np.median(ranks)),
        "n_usable_median": float(np.median(counts)),
    }


def _patch_variation(displacement: np.ndarray, clear: np.ndarray) -> dict:
    """Variation of the full affine across one patch. One value per revisit."""
    ranges = []
    circular = []
    for frame in range(displacement.shape[0]):
        mask = clear[frame].reshape(-1)
        if not np.any(mask):
            continue
        raw = displacement[frame].reshape(-1, 2)[mask]
        ranges.append(np.max(raw, axis=0) - np.min(raw, axis=0))
        circular.append(circular_deviation(np.mod(raw, 1.0)))
    if not ranges:
        return {"raw_range_lr_px": None, "circular_deviation_lr_px": None}
    raw_range = np.max(np.stack(ranges, axis=0), axis=0)
    circ = np.max(np.stack(circular, axis=0), axis=0)
    return {
        "raw_range_lr_px": float(np.max(raw_range)),
        "circular_deviation_lr_px": float(np.max(circ)),
    }


def _score_run(affines: np.ndarray, clear: np.ndarray) -> list[dict]:
    height, width = clear.shape[1:]
    if height % PATCH or width % PATCH:
        raise ValueError(f"LR field {height}×{width} is not a multiple of {PATCH}")
    rows_n, cols_n = height // PATCH, width // PATCH
    patches = []
    yy, xx = np.meshgrid(np.arange(PATCH), np.arange(PATCH), indexing="ij")
    for iy in range(rows_n):
        for ix in range(cols_n):
            row0, col0 = iy * PATCH, ix * PATCH
            local_rows = (row0 + yy).reshape(-1)
            local_cols = (col0 + xx).reshape(-1)
            displacement, _ = sampling_offsets_lr(affines, local_rows, local_cols, height, width)
            variation = _patch_variation(displacement, clear[:, row0 : row0 + PATCH, col0 : col0 + PATCH])
            probe_rows, probe_cols = _probe_coordinates(row0, col0)
            _, fractional = sampling_offsets_lr(affines, probe_rows, probe_cols, height, width)
            per_factor = {factor: [] for factor in FACTORS}
            center_local = PROBES.index(CENTER_PROBE)
            center_index = center_local * len(PROBES) + center_local
            center = {}
            for point in range(fractional.shape[1]):
                usable = clear[:, probe_rows[point], probe_cols[point]]
                scored = _score_location(fractional[:, point, :], usable)
                for factor, result in scored.items():
                    per_factor[factor].append(result)
                if point == center_index:
                    center = {
                        f"s{factor}_center": result["score"] if result["status"] == "ok" else None
                        for factor, result in scored.items()
                    }
                    center["n_usable_center"] = int(scored[2]["n_usable"])
                    center["rank4_center"] = int(scored[4]["rank"])
                    center["center_offsets"] = fractional[:, point, :].tolist()
                    center["center_usable"] = usable.astype(bool).tolist()
            summary = {factor: _summarize_factor(per_factor[factor]) for factor in FACTORS}
            n_no_data = sum(item["status"] != "ok" for item in per_factor[2])
            clear_fraction = float(clear[:, row0 : row0 + PATCH, col0 : col0 + PATCH].mean())
            patches.append(
                {
                    "patch_iy": iy,
                    "patch_ix": ix,
                    "patch_row0": row0,
                    "patch_col0": col0,
                    "n_probe": int(fractional.shape[1]),
                    "n_probe_no_data": int(n_no_data),
                    "clear_fraction_labelled": clear_fraction,
                    "factors": {str(factor): summary[factor] for factor in FACTORS},
                    **center,
                    **variation,
                }
            )
    return patches


def _finite(values: list[float | None]) -> np.ndarray:
    return np.array([float(value) for value in values if value is not None], dtype=np.float64)


def _median_or_none(values: list[float | None]) -> float | None:
    kept = _finite(values)
    if kept.size == 0:
        return None
    return float(np.median(kept))


def _spearman(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy.stats import spearmanr

    if len(x) < 3 or np.ptp(x) < 1e-4 or np.ptp(y) < 1e-4:
        return {"rho": None, "pvalue": None, "n": int(len(x)), "reason": "no meaningful variation"}
    result = spearmanr(x, y)
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue), "n": int(len(x))}


def _partial_spearman(x: np.ndarray, y: np.ndarray, covariate: np.ndarray) -> dict:
    """Spearman correlation after linear residualizing average ranks on one covariate."""
    from scipy.stats import rankdata

    if min(np.ptp(x), np.ptp(y), np.ptp(covariate)) < 1e-4:
        return {"rho": None, "pvalue": None, "n": int(len(x)), "reason": "no meaningful variation"}
    rx, ry, rc = rankdata(x, method="average"), rankdata(y, method="average"), rankdata(covariate, method="average")
    design = np.column_stack([np.ones(len(rc)), rc])
    bx, _, _, _ = np.linalg.lstsq(design, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(design, ry, rcond=None)
    return _spearman(rx - design @ bx, ry - design @ by)


def _perturb(offsets: np.ndarray, usable: np.ndarray, magnitude: float, rng: np.random.Generator) -> dict:
    baselines = {}
    for factor in FACTORS:
        result = sampling_geometry_score(offsets, usable, factor)
        baselines[factor] = result["score"] if result["status"] == "ok" else None
    samples = {factor: [] for factor in FACTORS}
    for _ in range(PERTURBATION_DRAWS):
        noise = rng.uniform(-magnitude, magnitude, size=offsets.shape)
        noise[0] = 0.0
        moved = np.mod(offsets + noise, 1.0)
        for factor in FACTORS:
            result = sampling_geometry_score(moved, usable, factor)
            if baselines[factor] is not None and result["status"] == "ok":
                samples[factor].append(result["score"])
    out = {}
    for factor in FACTORS:
        values = np.asarray(samples[factor], dtype=np.float64)
        if baselines[factor] is None or values.size == 0:
            out[f"s{factor}_mean_absolute_change"] = None
            continue
        out[f"s{factor}_mean_absolute_change"] = float(np.mean(np.abs(values - baselines[factor])))
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _phase_figure(path: Path, offsets: np.ndarray, usable: np.ndarray, title: str, scores: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    grid = np.arange(4) / 4.0
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    ax.scatter(gx.ravel(), gy.ravel(), s=28, c="#d0d0d0", marker="s", label="ideal 4× lattice", zorder=1)
    shown = offsets[usable]
    ax.scatter(shown[:, 0], shown[:, 1], s=36, c="#1f4e79", label="usable revisits", zorder=2)
    if np.any(~usable):
        hidden = offsets[~usable]
        ax.scatter(
            hidden[:, 0],
            hidden[:, 1],
            s=36,
            facecolors="none",
            edgecolors="#b85c38",
            label="masked at this pixel",
            zorder=3,
        )
    ax.scatter([offsets[0, 0]], [offsets[0, 1]], s=64, c="#c45c26", marker="X", label="base frame", zorder=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("fractional LR-pixel offset  x")
    ax.set_ylabel("fractional LR-pixel offset  y")
    ax.set_title(title)
    note = (
        f"S2={scores[2]:.1f}   S3={scores[3]:.1f}   S4={scores[4]:.1f}\n"
        f"usable={int(usable.sum())}   rank4={scores['rank4']}"
    )
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _comparison_figure(path: Path, sites: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True)
    x_score = np.array([row["s2"] for row in sites], dtype=np.float64)
    x_count = np.array([row["n_usable"] for row in sites], dtype=np.float64)
    y = np.array([row["lpips_gain"] for row in sites], dtype=np.float64)
    for ax, xs, label in (
        (axes[0], x_score, "median sampling-geometry score at 2×"),
        (axes[1], x_count, "median usable revisits"),
    ):
        ax.scatter(xs, y, s=28, c="#1f4e79")
        for row, x_value in zip(sites, xs):
            ax.annotate(row["site"], (x_value, row["lpips_gain"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(label)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel("LPIPS improvement over bilinear")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper" / "results" / "sampling_geometry",
    )
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    discovered = _discover()
    missing = []
    rejected = []
    field_rows = []
    patch_rows = []
    site_masks: dict[str, tuple[np.ndarray, list[dict]]] = {}
    illustration = None
    typical_lengths = []
    center_samples = []

    for city, seed, path in discovered:
        affine_path = path / "affines.json"
        metrics_path = path / "metrics.json"
        if not affine_path.is_file() or not metrics_path.is_file():
            missing.append({"site": city, "seed": seed, "run": str(path), "reason": "affines.json or metrics.json missing"})
            continue
        metrics = _load_json(metrics_path)
        ok, reason = _compatible(metrics)
        if not ok:
            rejected.append({"site": city, "seed": seed, "run": path.name, "reason": reason})
            continue
        affine_payload = _load_json(affine_path)
        matrices = np.asarray([frame["matrix_2x3"] for frame in affine_payload["frames"]], dtype=np.float64)
        geo = affine_payload
        if city not in site_masks:
            cache = out / "masks" / f"{city}.npz"
            clear, identities = _frames_from_dataset(metrics["provenance"]["args"], cache)
            site_masks[city] = (clear, identities)
        clear, identities = site_masks[city]
        if matrices.shape[0] != clear.shape[0]:
            missing.append(
                {
                    "site": city,
                    "seed": seed,
                    "run": path.name,
                    "reason": f"affine frames {matrices.shape[0]} != clear frames {clear.shape[0]}",
                }
            )
            continue
        if tuple(clear.shape[1:]) != (int(geo["lr_height"]), int(geo["lr_width"])):
            missing.append(
                {
                    "site": city,
                    "seed": seed,
                    "run": path.name,
                    "reason": f"mask {clear.shape[1:]} != affine geo {(geo['lr_height'], geo['lr_width'])}",
                }
            )
            continue
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        if not np.allclose(matrices[0], identity, atol=1e-6):
            missing.append({"site": city, "seed": seed, "run": path.name, "reason": "frame 0 is not the frozen identity"})
            continue

        patches = _score_run(matrices, clear)
        center_patch = next(item for item in patches if item["patch_iy"] == 3 and item["patch_ix"] == 3)
        center_offsets = np.asarray(center_patch["center_offsets"], dtype=np.float64)
        center_usable = np.asarray(center_patch["center_usable"], dtype=bool)
        lengths = np.linalg.norm(np.minimum(center_offsets, 1.0 - center_offsets), axis=1)
        typical_lengths.extend(float(value) for value, keep in zip(lengths, center_usable) if keep and value > 0)
        center_samples.append(
            {"site": city, "seed": seed, "offsets": center_offsets, "usable": center_usable}
        )

        gains = {
            "lpips_gain": _metric_gain(metrics["lpips"], higher_is_better=False),
            "psnr_gain": _metric_gain(metrics["psnr"], higher_is_better=True),
            "ssim_gain": _metric_gain(metrics["ssim"], higher_is_better=True),
        }
        field_rows.append(
            {
                "site": city,
                "seed": seed,
                "parent_tile": f"{city}_lr512_b0",
                "configuration": "confirmatory_nocut_b0_17",
                "run_name": path.name,
                "n_frames": int(matrices.shape[0]),
                "base_date": metrics.get("base_frame", {}).get("date"),
                "lpips_model": float(metrics["lpips"]["model"]),
                "lpips_bilinear": float(metrics["lpips"]["bilinear"]),
                **gains,
                "s2": _median_or_none([item["factors"]["2"]["score_median"] for item in patches]),
                "s3": _median_or_none([item["factors"]["3"]["score_median"] for item in patches]),
                "s4": _median_or_none([item["factors"]["4"]["score_median"] for item in patches]),
                "rank4": _median_or_none([item["factors"]["4"]["rank_median"] for item in patches]),
                "n_usable": _median_or_none([item["factors"]["4"]["n_usable_median"] for item in patches]),
                "s4_within_patch_range": _median_or_none(
                    [
                        None
                        if item["factors"]["4"]["score_min"] is None
                        else item["factors"]["4"]["score_max"] - item["factors"]["4"]["score_min"]
                        for item in patches
                    ]
                ),
                "raw_offset_range_lr_px": _median_or_none([item["raw_range_lr_px"] for item in patches]),
                "clear_fraction_labelled": float(clear.mean()),
                "affine_path": str(affine_path.relative_to(ROOT)),
                "metrics_path": str(metrics_path.relative_to(ROOT)),
            }
        )
        for item in patches:
            patch_rows.append(
                {
                    "site": city,
                    "seed": seed,
                    "parent_tile": f"{city}_lr512_b0",
                    "patch": f"y{item['patch_iy']:02d}_x{item['patch_ix']:02d}",
                    "configuration": "confirmatory_nocut_b0_17",
                    "run_name": path.name,
                    "n_frames": int(matrices.shape[0]),
                    "patch_row0": item["patch_row0"],
                    "patch_col0": item["patch_col0"],
                    "n_probe": item["n_probe"],
                    "n_probe_no_data": item["n_probe_no_data"],
                    "clear_fraction_labelled": item["clear_fraction_labelled"],
                    "s2": item["factors"]["2"]["score_median"],
                    "s2_min": item["factors"]["2"]["score_min"],
                    "s2_max": item["factors"]["2"]["score_max"],
                    "rank2": item["factors"]["2"]["rank_median"],
                    "s3": item["factors"]["3"]["score_median"],
                    "s3_min": item["factors"]["3"]["score_min"],
                    "s3_max": item["factors"]["3"]["score_max"],
                    "rank3": item["factors"]["3"]["rank_median"],
                    "s4": item["factors"]["4"]["score_median"],
                    "s4_min": item["factors"]["4"]["score_min"],
                    "s4_max": item["factors"]["4"]["score_max"],
                    "rank4": item["factors"]["4"]["rank_median"],
                    "n_usable": item["factors"]["4"]["n_usable_median"],
                    "s4_center": item["s4_center"],
                    "n_usable_center": item["n_usable_center"],
                    "rank4_center": item["rank4_center"],
                    "raw_offset_range_lr_px": item["raw_range_lr_px"],
                    "circular_deviation_lr_px": item["circular_deviation_lr_px"],
                }
            )
        center_s2 = center_patch["s2_center"]
        eligible = center_s2 is not None and int(center_patch["n_usable_center"]) == int(matrices.shape[0])
        if eligible and (illustration is None or center_s2 > illustration["score"]):
            center_scores = {}
            for factor in FACTORS:
                result = sampling_geometry_score(center_offsets, center_usable, factor)
                center_scores[factor] = result["score"] if result["status"] == "ok" else float("nan")
                if factor == 4:
                    center_scores["rank4"] = result["rank"]
            illustration = {
                "site": city,
                "seed": seed,
                "patch": "y03_x03",
                "offsets": center_offsets,
                "usable": center_usable,
                "scores": center_scores,
                "score": center_s2,
            }

    median_shift = float(np.median(typical_lengths)) if typical_lengths else None
    rng = np.random.default_rng(PERTURBATION_SEED)
    perturbations = []
    magnitudes = [(ABSOLUTE_PERTURBATION_LR_PX, "0.05_lr_px")]
    if median_shift is not None and median_shift > 0:
        magnitudes.append((median_shift, "median_fractional_shift"))
    for sample in center_samples:
        for magnitude, label in magnitudes:
            summary = _perturb(sample["offsets"], sample["usable"], magnitude, rng)
            summary.update(
                {
                    "site": sample["site"],
                    "seed": sample["seed"],
                    "magnitude_lr_px": magnitude,
                    "label": label,
                }
            )
            perturbations.append(summary)

    by_site: dict[str, list[dict]] = {}
    for row in field_rows:
        by_site.setdefault(row["site"], []).append(row)
    site_rows = []
    for site, rows in sorted(by_site.items()):
        site_rows.append(
            {
                "site": site,
                "n_seeds": len(rows),
                "s2": float(np.mean([row["s2"] for row in rows])),
                "s3": float(np.mean([row["s3"] for row in rows])),
                "s4": float(np.mean([row["s4"] for row in rows])),
                "n_usable": float(np.mean([row["n_usable"] for row in rows])),
                "lpips_gain": float(np.mean([row["lpips_gain"] for row in rows])),
                "psnr_gain": float(np.mean([row["psnr_gain"] for row in rows])),
                "ssim_gain": float(np.mean([row["ssim_gain"] for row in rows])),
                "clear_fraction_labelled": float(np.mean([row["clear_fraction_labelled"] for row in rows])),
                "raw_offset_range_lr_px": float(np.mean([row["raw_offset_range_lr_px"] for row in rows])),
            }
        )

    def _pair(key: str, gain: str) -> dict:
        xs = np.array([row[key] for row in site_rows], dtype=np.float64)
        ys = np.array([row[gain] for row in site_rows], dtype=np.float64)
        return _spearman(xs, ys)

    associations = {}
    for gain in ("lpips_gain", "psnr_gain", "ssim_gain"):
        block = {"n_usable": _pair("n_usable", gain)}
        for key in ("s2", "s3", "s4"):
            block[key] = _pair(key, gain)
            block[f"{key}_partial_n_usable"] = _partial_spearman(
                np.array([row[key] for row in site_rows]),
                np.array([row[gain] for row in site_rows]),
                np.array([row["n_usable"] for row in site_rows]),
            )
        associations[gain] = block
    perturb_summary = []
    for label in sorted({item["label"] for item in perturbations}):
        group = [item for item in perturbations if item["label"] == label]
        if not group:
            continue
        entry = {
            "label": label,
            "magnitude_lr_px": group[0]["magnitude_lr_px"],
            "n_runs": len(group),
        }
        for factor in FACTORS:
            values = [
                item[f"s{factor}_mean_absolute_change"]
                for item in group
                if item[f"s{factor}_mean_absolute_change"] is not None
            ]
            entry[f"s{factor}_median_absolute_change"] = None if not values else float(np.median(values))
        perturb_summary.append(entry)

    if illustration is not None:
        _phase_figure(
            out / "sampling_phase.png",
            illustration["offsets"],
            illustration["usable"],
            f"{illustration['site']} seed {illustration['seed']}  LR64 y03_x03 centre",
            illustration["scores"],
        )
    if site_rows:
        _comparison_figure(out / "score_vs_lpips.png", site_rows)

    patch_csv = [
        {key: value for key, value in row.items()}
        for row in patch_rows
    ]
    _write_csv(out / "patches.csv", patch_csv)
    _write_csv(out / "fields.csv", field_rows)
    _write_csv(out / "sites.csv", site_rows)

    ranges = _finite([row["raw_offset_range_lr_px"] for row in patch_rows])
    circulars = _finite([row["circular_deviation_lr_px"] for row in patch_rows])
    scores4 = _finite([row["s4"] for row in patch_rows])
    summary = {
        "role": "post-hoc sampling geometry diagnostic",
        "not": [
            "achieved spatial resolution",
            "probability of correct reconstruction",
            "checkpoint selection or validation stopping",
        ],
        "formula": {
            "factors": list(FACTORS),
            "score": "100 * lambda_min(P^H diag(w) P / sum(w))",
            "offsets": "learned inter-revisit affine, fractional LR pixels, modulo 1",
            "weights": "per-probe clear mask; patch clear fraction is labelled and not used as w",
            "locations_are_not_extra_revisits": True,
        },
        "coordinates": {
            "normalized": "pixel centres (col+0.5)/width and (row+0.5)/height in [0, 1)",
            "affine": "models.inr.apply_affine maps normalized coordinates before encoding",
            "frame0": "frozen identity; offsets are relative to the base frame",
            "lr_pixel": "1/width in x and 1/height in y",
            "not_used": "eval/spatial_alignment.json",
        },
        "inputs": {
            "runs": "single_samples/*/confirmatory_nocut/confirmatory_nocut__b0_17__lr512_b0__*__seed{6,7,8}",
            "requirement": "spatial_holdout 0, early_stop null, patience 0, completed_iters == iterations",
            "affine_export": "optimize.py writes affines.json from the in-memory model after metrics.json",
            "n_discovered": len(discovered),
            "n_scored_runs": len(field_rows),
            "n_patches": len(patch_rows),
            "missing": missing,
            "rejected": rejected,
        },
        "spatial": {
            "footprint": "non-overlapping LR64 windows inside each LR512 field",
            "probes_per_patch": "4 by 4 pixels at local indices 8, 24, 40, 56",
            "reported_score": "median of probe scores that have usable observations",
            "median_raw_offset_range_lr_px": None if ranges.size == 0 else float(np.median(ranges)),
            "max_raw_offset_range_lr_px": None if ranges.size == 0 else float(np.max(ranges)),
            "median_circular_deviation_lr_px": None if circulars.size == 0 else float(np.median(circulars)),
            "max_circular_deviation_lr_px": None if circulars.size == 0 else float(np.max(circulars)),
            "approximation": (
                "The phase matrix uses one offset per revisit at each probe. "
                "Raw displacement range across the 64 by 64 window measures rotation, shear, and scale. "
                "A range far below one LR pixel means the local translation approximation is adequate inside that window."
            ),
        },
        "scores": {
            "patch_s2_median": None if _finite([row["s2"] for row in patch_rows]).size == 0 else float(np.median(_finite([row["s2"] for row in patch_rows]))),
            "patch_s3_median": None if _finite([row["s3"] for row in patch_rows]).size == 0 else float(np.median(_finite([row["s3"] for row in patch_rows]))),
            "patch_s4_median": None if scores4.size == 0 else float(np.median(scores4)),
            "patch_s4_min": None if scores4.size == 0 else float(np.min(scores4)),
            "patch_s4_max": None if scores4.size == 0 else float(np.max(scores4)),
            "fraction_patches_s4_zero": None if scores4.size == 0 else float(np.mean(scores4 == 0.0)),
            "fraction_patches_n_usable_below_16": None
            if not patch_rows
            else float(np.mean([float(row["n_usable"]) < 16 for row in patch_rows])),
            "sites_raw_range_above_one_lr_pixel": sorted(
                {row["site"] for row in site_rows if row["raw_offset_range_lr_px"] > 1.0}
            ),
            "sites": site_rows,
        },
        "association_site_means": associations,
        "association_note": (
            "Correlations use one point per site, the mean of seeds 6, 7, and 8. "
            "Patches and seeds are not treated as independent. "
            "Scene content and cloud coverage also differ. A correlation is not a causal claim."
        ),
        "perturbation": {
            "how_magnitudes_were_chosen": (
                "0.05 LR pixel is 0.5 m at the 10 m Sentinel-2 grid, a shift smaller than "
                "the roughly 1 m centre shifts visible in individual affine dumps. "
                "The second magnitude is the median length of the non-base fractional offsets "
                "at the centre of LR64 window y03_x03, so it matches the size of a typical estimated sampling offset."
            ),
            "median_fractional_offset_lr_px": median_shift,
            "draws": PERTURBATION_DRAWS,
            "seed": PERTURBATION_SEED,
            "base_frame_unperturbed": True,
            "location": "centre pixel of LR64 window y03_x03",
            "summary": perturb_summary,
        },
        "illustration": None
        if illustration is None
        else {
            "site": illustration["site"],
            "seed": illustration["seed"],
            "patch": illustration["patch"],
            "why": "centre window with the highest 2× score among runs whose centre pixel can use every revisit",
        },
        "quality_metrics": (
            "LPIPS, PSNR, and SSIM gains are the full-field values stored in each run's metrics.json. "
            "These fits did not save patch-level reconstruction scores, so the comparison is at site level."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "scored_runs": len(field_rows),
        "patches": len(patch_rows),
        "missing": missing,
        "rejected": rejected,
        "s4_median": summary["scores"]["patch_s4_median"],
        "offset_range_max": summary["spatial"]["max_raw_offset_range_lr_px"],
        "associations": associations,
        "perturbation": perturb_summary,
        "illustration": summary["illustration"],
    }, indent=2))


if __name__ == "__main__":
    main()
