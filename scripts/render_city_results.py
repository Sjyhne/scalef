#!/usr/bin/env python3
"""Regenerate result figures from saved PNGs + metrics.json (no retraining)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.visualize import (
    regenerate_visualizations_from_dir,
    save_all_cities_metrics_summary,
)

DEFAULT_CITIES = [
    "bergen",
    "kristiansand",
    "rana",
    "sandvika",
    "stavanger",
    "tromso",
    "trondheim",
]


def _result_dir(city: str, run_name: str, sample_id: str) -> Path:
    return ROOT / "single_samples" / city / sample_id / run_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", default=DEFAULT_CITIES)
    parser.add_argument("--run-name", default="sweep_best")
    parser.add_argument("--sample-id", default="sample")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "single_samples" / "sweep_results" / "all_cities_metrics.png",
    )
    args = parser.parse_args()

    city_metrics: list[tuple[str, dict]] = []
    for city in args.cities:
        result_dir = _result_dir(city, args.run_name, args.sample_id)
        if not (result_dir / "metrics.json").exists():
            print(f"skip {city}: no metrics at {result_dir}")
            continue
        regenerate_visualizations_from_dir(result_dir, sample_label=city)
        metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
        city_metrics.append((city, metrics))
        print(f"{city}: updated figures in {result_dir}")

    if city_metrics:
        save_all_cities_metrics_summary(city_metrics, args.summary_out)
        print(f"All-cities table: {args.summary_out}")


if __name__ == "__main__":
    main()
