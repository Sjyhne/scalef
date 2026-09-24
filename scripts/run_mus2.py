#!/usr/bin/env python3
"""Plan or run ScaleF over a prepared MuS2 manifest, then evaluate outputs.

The command is dry-run by default. ``--execute`` is required to launch ScaleF
or write evaluation results. External predictions can be evaluated with
``--evaluate-only --predictions-root DIR --execute``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.mus2 import (  # noqa: E402
    balanced_score,
    bicubic_baseline,
    evaluate_arrays,
    read_grayscale,
)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def make_jobs(
    manifest: dict,
    *,
    run_prefix: str,
    iters: int,
    device: str,
    predictions_root: Path | None,
    extra: list[str],
) -> list[dict]:
    """Build deterministic training/evaluation jobs from a MuS2 manifest."""
    jobs = []
    for scene in manifest["scenes"]:
        for band, entry in scene["bands"].items():
            dataset = _slug(f"mus2_{scene['id']}_{band}")
            run_name = _slug(f"{run_prefix}_{band}")
            if predictions_root is None:
                prediction = (
                    ROOT
                    / "single_samples"
                    / dataset
                    / "sample"
                    / run_name
                    / "qgis"
                    / "sr_pred.tif"
                )
            else:
                prediction = predictions_root / scene["id"] / f"{band}.tif"
            command = [
                sys.executable,
                str(ROOT / "optimize.py"),
                "--dataset",
                dataset,
                "--s2-dir",
                entry["prepared_dir"],
                "--hr-path",
                entry["prepared_hr"],
                "--df",
                "3",
                "--scale_factor",
                "3",
                "--num_samples",
                str(len(entry["lr_files"])),
                "--iters",
                str(iters),
                "--device",
                device,
                "--run_name",
                run_name,
                "--no_hr_harmonize",
                "--no_hr_spatial_align",
                *extra,
            ]
            jobs.append(
                {
                    "scene": scene["id"],
                    "band": band,
                    "command": command,
                    "prediction": str(prediction),
                    "reference": entry["prepared_hr"],
                    "mask": entry.get("mask"),
                    "lr_files": entry["lr_files"],
                }
            )
    return jobs


def _crop_to_prediction_grid(
    prediction: Path,
    reference: Path,
    reference_array,
    excluded_mask,
):
    """Crop full MuS2 reference/mask to ScaleF's valid-data output window."""
    import rasterio
    from rasterio.windows import Window

    prediction_array = read_grayscale(prediction)
    if prediction_array.shape == reference_array.shape:
        return prediction_array, reference_array, excluded_mask, (slice(None), slice(None))

    with rasterio.open(prediction) as pred_src, rasterio.open(reference) as ref_src:
        if pred_src.crs != ref_src.crs:
            raise ValueError(
                f"prediction/reference CRS mismatch: {pred_src.crs} != {ref_src.crs}"
            )
        col_off = round((pred_src.transform.c - ref_src.transform.c) / ref_src.transform.a)
        row_off = round((pred_src.transform.f - ref_src.transform.f) / ref_src.transform.e)
        window = Window(col_off, row_off, pred_src.width, pred_src.height)
        cropped_reference = ref_src.read(1, window=window)

    if cropped_reference.shape != prediction_array.shape:
        raise ValueError(
            f"could not align prediction {prediction_array.shape} to reference "
            f"{reference_array.shape} with window {window}"
        )
    rows = slice(row_off, row_off + prediction_array.shape[0])
    cols = slice(col_off, col_off + prediction_array.shape[1])
    cropped_mask = excluded_mask[rows, cols] if excluded_mask is not None else None
    return prediction_array, cropped_reference, cropped_mask, (rows, cols)


def evaluate_jobs(jobs: list[dict], *, max_shift: int, use_lpips: bool) -> dict:
    """Evaluate completed jobs and aggregate with the official bicubic baseline."""
    lpips_model = None
    if use_lpips:
        import lpips
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for job in jobs:
        prediction = Path(job["prediction"])
        reference = Path(job["reference"])
        mask = Path(job["mask"]) if job.get("mask") else None
        if not prediction.is_file():
            raise FileNotFoundError(f"missing prediction {prediction}")
        hr_full = read_grayscale(reference)
        excluded_full = read_grayscale(mask) if mask is not None else None
        prediction_array, hr, excluded, crop = _crop_to_prediction_grid(
            prediction,
            reference,
            hr_full,
            excluded_full,
        )
        candidate = evaluate_arrays(
            prediction_array,
            hr,
            excluded_mask=excluded,
            max_shift=max_shift,
            lpips_model=lpips_model,
        )
        baseline_full = bicubic_baseline(
            [Path(path) for path in job["lr_files"]],
            hr_full.shape,
        )
        baseline_image = baseline_full[crop]
        baseline = evaluate_arrays(
            baseline_image,
            hr,
            excluded_mask=excluded,
            max_shift=max_shift,
            lpips_model=lpips_model,
        )
        candidate["balanced_score"] = balanced_score(candidate, baseline)
        rows.append(
            {
                "scene": job["scene"],
                "band": job["band"],
                "candidate": candidate,
                "bicubic": baseline,
            }
        )

    per_band = {}
    metric_names = ["cPSNR", "cSSIM"] + (["LPIPS"] if use_lpips else [])
    for band in sorted({row["band"] for row in rows}):
        selected = [row for row in rows if row["band"] == band]
        candidate_mean = {
            metric: sum(row["candidate"][metric] for row in selected) / len(selected)
            for metric in metric_names
        }
        bicubic_mean = {
            metric: sum(row["bicubic"][metric] for row in selected) / len(selected)
            for metric in metric_names
        }
        per_band[band] = {
            "scenes": len(selected),
            "candidate_mean": candidate_mean,
            "bicubic_mean": bicubic_mean,
            "balanced_score_from_means": balanced_score(candidate_mean, bicubic_mean),
        }
    return {
        "protocol": "MuS2 public evaluator compatible",
        "max_shift": max_shift,
        "lpips_enabled": use_lpips,
        "rows": rows,
        "per_band": per_band,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-prefix", default="external_eval")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=None,
        help="External layout DIR/SCENE/BAND.tif; also implies no ScaleF output lookup.",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate predictions after training (evaluation-only always evaluates).",
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Enable official LPIPS-Alex metric (may download model weights).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "mus2_external_eval.json",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch commands/write results. Omit for a side-effect-free dry run.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra optimize.py arguments after '--'.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "scalef.mus2-manifest.v1":
        raise SystemExit(f"Unsupported manifest schema: {manifest.get('schema')!r}")
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    jobs = make_jobs(
        manifest,
        run_prefix=args.run_prefix,
        iters=args.iters,
        device=args.device,
        predictions_root=args.predictions_root,
        extra=extra,
    )
    plan = {
        "dry_run": not args.execute,
        "job_count": len(jobs),
        "evaluate": args.evaluate or args.evaluate_only,
        "jobs": jobs,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print("DRY RUN: pass --execute to launch ScaleF and/or evaluate.")
        return

    if not args.evaluate_only:
        for job in jobs:
            print("+", " ".join(job["command"]), flush=True)
            subprocess.run(job["command"], cwd=ROOT, check=True)

    if args.evaluate or args.evaluate_only:
        report = evaluate_jobs(jobs, max_shift=args.max_shift, use_lpips=args.lpips)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
        print(f"Evaluation written to {args.results}")


if __name__ == "__main__":
    main()
