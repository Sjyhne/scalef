#!/usr/bin/env python3
"""Render harmonized HR + all S2 LR revisits for each city."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataset
from eval.visualize import save_hr_lr_revisit_panel

CITIES = [
    "bergen",
    "kristiansand",
    "rana",
    "sandvika",
    "stavanger",
    "tromso",
    "trondheim",
]


def _dataset_args(city: str) -> Namespace:
    return Namespace(
        dataset="s2",
        s2_dir=str(ROOT / "data" / "s2_revisits" / city),
        df=4,
        scale_factor=4,
        hr_gsd_m=0.0,
        s2_native_gsd_m=10.0,
        num_samples=0,
        dataset_device="cpu",
        no_hr_harmonize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cities",
        nargs="+",
        default=CITIES,
        help="Cities to render (default: all 7)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "single_samples" / "harmonization_audit",
        help="Directory for per-city panels",
    )
    args = parser.parse_args()

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    for city in args.cities:
        ds = get_dataset(_dataset_args(city))
        out_path = out_root / f"{city}_hr_lr_revisits.png"
        save_hr_lr_revisit_panel(ds, out_path, city_name=city)
        print(
            f"{city}: {out_path} "
            f"({ds.num_samples} LR frames, base={ds.base_frame_index})"
        )

    print(f"Done — panels in {out_root}")


if __name__ == "__main__":
    main()
