#!/usr/bin/env python3
"""Render HR+LR revisit panels for LR-size ladder variants."""

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default="asker")
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    p.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "single_samples" / "harmonization_audit",
    )
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    for side in args.sizes:
        s2_dir = ROOT / "data" / "s2_revisits" / f"{args.city}_lr{side}"
        if not (s2_dir / "meta.json").is_file():
            print(f"skip {s2_dir}: missing meta — run make_lr_size_variants.py first")
            continue
        ds_args = Namespace(
            dataset="s2",
            s2_dir=str(s2_dir),
            hr_path=None,
            df=4,
            scale_factor=4,
            hr_gsd_m=0.0,
            s2_native_gsd_m=10.0,
            num_samples=0,
            lr_size=0,
            dataset_device="cpu",
            no_hr_harmonize=False,
        )
        print(f"loading {s2_dir} ...")
        ds = get_dataset(ds_args)
        out = args.out_root / f"{args.city}_lr{side}_hr_lr_revisits.png"
        save_hr_lr_revisit_panel(ds, out, city_name=f"{args.city} lr{side}")
        print(
            f"  wrote {out}  LR {ds.lr_height}×{ds.lr_width}  "
            f"frames={ds.num_samples} base={ds.base_frame_index}"
        )


if __name__ == "__main__":
    main()
