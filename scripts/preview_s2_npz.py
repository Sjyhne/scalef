#!/usr/bin/env python3
"""Regenerate downsampled RGB preview PNGs from *_reflectance.npz exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from s2_bands import resolve_band_preset
from s2_preview import (
    DEFAULT_PREVIEW_DOWNSAMPLE,
    scene_stretch_limits_from_arrays,
    select_rgb_hwc,
    write_reflectance_preview_png,
)


def _load_scene(npz_path: Path) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...]]:
    valid_path = npz_path.with_name(npz_path.name.replace("_reflectance.npz", "_valid.npy"))
    valid = np.load(valid_path).astype(bool) if valid_path.is_file() else None
    with np.load(npz_path) as z:
        refl = z["reflectance_hwc"]
        if "band_names" in z:
            band_names = tuple(str(x) for x in z["band_names"])
        else:
            band_names = resolve_band_preset("rgb_nir").band_names
    return refl, valid, band_names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="NPZ file or directory")
    p.add_argument("-o", "--output", type=Path, help="Output PNG path (single file mode)")
    p.add_argument("--manifest", action="store_true", help="Batch from stac_download_manifest.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--downsample", type=int, default=DEFAULT_PREVIEW_DOWNSAMPLE)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--stretch", choices=("scalar", "per_channel"), default="scalar")
    args = p.parse_args()

    if args.input.is_file():
        npz_paths = [args.input]
        out_dir = args.input.parent
    else:
        out_dir = args.input
        if args.manifest:
            manifest = json.loads((out_dir / "stac_download_manifest.json").read_text())
            npz_paths = [out_dir / s["reflectance_npz"] for s in manifest["scenes"]]
        else:
            npz_paths = sorted(out_dir.glob("*_reflectance.npz"))

    rgb_list: list[np.ndarray] = []
    valid_list: list[np.ndarray | None] = []
    band_names: tuple[str, ...] | None = None
    for npz_path in npz_paths:
        refl, valid, names = _load_scene(npz_path)
        band_names = names
        rgb_list.append(select_rgb_hwc(refl, names))
        valid_list.append(valid)

    lo, hi = scene_stretch_limits_from_arrays(
        rgb_list,
        valid_list,
        downsample=args.downsample,
        stretch=args.stretch,
    )

    for npz_path, valid in zip(npz_paths, valid_list):
        out_path = (
            args.output
            if args.output and len(npz_paths) == 1
            else npz_path.with_name(npz_path.name.replace("_reflectance.npz", "_preview.png"))
        )
        if out_path.is_file() and not args.overwrite:
            continue
        with np.load(npz_path) as z:
            refl = z["reflectance_hwc"]
        write_reflectance_preview_png(
            refl,
            out_path,
            band_names=band_names,
            valid_hw=valid,
            downsample=args.downsample,
            lo=lo,
            hi=hi,
            gamma=args.gamma,
            stretch=args.stretch,
        )
        print(out_path)


if __name__ == "__main__":
    main()
