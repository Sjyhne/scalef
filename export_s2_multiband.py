#!/usr/bin/env python3
"""Export multi-band Sentinel-2 scenes in satburst-compatible layout.

See ``data_multiband/README.txt`` for layout, band presets, and usage.

Modes:

1. **GeoSR parquet** (12-band ``lr_stack`` on a common grid)::

    python export_s2_multiband.py geosr \\
        --input geosr_misr_sample20 \\
        --output data_multiband_geosr \\
        --scene-id 165 \\
        --band-preset rgb_nir

2. **Synthetic demo** (no external data; layout / loader smoke test)::

    python export_s2_multiband.py synthetic \\
        --output data_multiband_synthetic \\
        --scene-id demo \\
        --band-preset rgb_nir \\
        --lr-size 32 --num-frames 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from export_geosr_misr_to_satburst import export_scene, load_manifest
from s2_bands import BAND_PRESETS, BandPreset, band_manifest_dict, resolve_band_preset
from s2_reflectance import (
    S2_RGB_DISPLAY_BANDS,
    reflectance_hwc_rgb_to_uint8,
    reflectance_list_to_uint8_per_frame_percentile,
    stack_bands_hwc,
)


def write_band_manifest(scene_dir: Path, band_names: tuple[str, ...]) -> Path:
    scene_dir = Path(scene_dir)
    out = scene_dir / "band_manifest.json"
    with out.open("w") as f:
        json.dump(band_manifest_dict(band_names), f, indent=2)
    return out


def _zero_shift_entry(path: str, shape: list[int], *, is_reference: bool = False) -> dict:
    entry = {
        "dx_pixels_hr": 0,
        "dy_pixels_hr": 0,
        "dx_pixels_lr": 0.0,
        "dy_pixels_lr": 0.0,
        "dx_percent": 0.0,
        "dy_percent": 0.0,
        "magnitude_pixels_hr": 0.0,
        "magnitude_pixels_lr": 0.0,
        "shape": shape,
        "path": path,
        "augmentation": "none",
    }
    if is_reference:
        entry["is_reference"] = True
    return entry


def export_geosr_multiband(
    *,
    input_root: Path,
    output_root: Path,
    scene_id: str,
    preset: BandPreset,
    df: int,
    lr_shift: float,
    aug: str,
    overwrite: bool,
) -> Path:
    rows = load_manifest(input_root)
    rows = [r for r in rows if str(r["id"]) == str(scene_id)]
    if not rows:
        raise SystemExit(f"scene id {scene_id!r} not found in {input_root / 'manifest.csv'}")

    rgb_indices = preset.geosr_stack_indices()[:3]
    if len(rgb_indices) < 3:
        raise ValueError(f"preset {preset.name} has fewer than 3 bands for RGB PNG preview")

    scene_dir = export_scene(
        rows[0],
        input_root=input_root,
        output_root=output_root,
        df=df,
        lr_shift=lr_shift,
        aug=aug,
        rgb_bands=(rgb_indices[0], rgb_indices[1], rgb_indices[2]),
        supervision_bands=preset.geosr_stack_indices(),
        reflectance_scale=10000.0,
        lr_normalize="per_frame_percentile",
        percentile_low=1.0,
        percentile_high=99.0,
        lr_gamma=1.0,
        export_raw_npz=True,
        hr_crop="resize",
        overwrite=overwrite,
    )
    write_band_manifest(scene_dir, preset.band_names)

    meta_path = scene_dir / "multiband_export_meta.json"
    with meta_path.open("w") as f:
        json.dump(
            {
                "source": "geosr_misr",
                "band_preset": preset.name,
                "band_names": list(preset.band_names),
                "supervision_bands_stack_indices": list(preset.geosr_stack_indices()),
            },
            f,
            indent=2,
        )
    return scene_dir


def export_synthetic_multiband(
    *,
    output_scene_dir: Path,
    preset: BandPreset,
    df: int,
    lr_shift: float,
    aug: str,
    lr_size: int,
    num_frames: int,
    seed: int,
) -> Path:
    """Create a tiny synthetic burst for pipeline smoke tests."""
    rng = np.random.default_rng(seed)
    scene_dir = (
        Path(output_scene_dir)
        / f"scale_{df}_shift_{lr_shift:.1f}px_aug_{aug}"
    )
    scene_dir.mkdir(parents=True, exist_ok=True)

    c = preset.num_channels
    hr_side = int(df) * int(lr_size)
    # Shared HR latent field per channel (smooth + noise).
    yy, xx = np.mgrid[0:hr_side, 0:hr_side]
    base_fields = []
    for ch in range(c):
        freq = 0.04 + 0.01 * ch
        field = 0.35 + 0.25 * np.sin(freq * xx + 0.3 * ch) * np.cos(freq * yy - 0.2 * ch)
        field += 0.05 * rng.standard_normal((hr_side, hr_side))
        base_fields.append(np.clip(field, 0.05, 0.95).astype(np.float32))
    hr_stack = np.stack(base_fields, axis=-1)

    transform_log: dict[str, dict] = {}
    reflectance_frames: list[np.ndarray] = []
    for i in range(int(num_frames)):
        shift_y = int(rng.integers(-2, 3))
        shift_x = int(rng.integers(-2, 3))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        warped = cv2.warpAffine(
            hr_stack,
            M,
            (hr_side, hr_side),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        lr = cv2.resize(warped, (lr_size, lr_size), interpolation=cv2.INTER_AREA)
        gain = 0.9 + 0.1 * rng.random(c)
        bias = -0.02 + 0.04 * rng.random(c)
        lr = np.clip(lr * gain + bias, 0.0, 1.0).astype(np.float32)
        reflectance_frames.append(lr)

        name = f"sample_{i:02d}.png"
        key = name.replace(".png", "")
        npz_name = f"{key}_reflectance.npz"
        np.savez_compressed(scene_dir / npz_name, reflectance_supervision=lr)
        entry = _zero_shift_entry(name, [lr_size, lr_size, c], is_reference=(i == 0))
        entry["raw_reflectance_npz"] = npz_name
        transform_log[key] = entry

    for i, frame in enumerate(reflectance_frames):
        rgb = stack_bands_hwc(frame, preset.band_names, S2_RGB_DISPLAY_BANDS)
        if rgb is not None:
            png = reflectance_hwc_rgb_to_uint8(rgb)
        else:
            png = reflectance_list_to_uint8_per_frame_percentile([frame[..., :3]])[0]
        cv2.imwrite(
            str(scene_dir / f"sample_{i:02d}.png"),
            cv2.cvtColor(png, cv2.COLOR_RGB2BGR),
        )

    hr_rgb = stack_bands_hwc(hr_stack, preset.band_names, S2_RGB_DISPLAY_BANDS)
    if hr_rgb is not None:
        hr_png = reflectance_hwc_rgb_to_uint8(hr_rgb)
    else:
        hr_png = reflectance_list_to_uint8_per_frame_percentile([hr_stack[..., :3]])[0]
    cv2.imwrite(
        str(scene_dir / "hr_ground_truth.png"),
        cv2.cvtColor(hr_png, cv2.COLOR_RGB2BGR),
    )

    with (scene_dir / "transform_log.json").open("w") as f:
        json.dump(transform_log, f, indent=2)
    write_band_manifest(scene_dir, preset.band_names)
    with (scene_dir / "multiband_export_meta.json").open("w") as f:
        json.dump(
            {
                "source": "synthetic",
                "band_preset": preset.name,
                "band_names": list(preset.band_names),
                "lr_size": lr_size,
                "num_frames": num_frames,
                "df": df,
            },
            f,
            indent=2,
        )
    return scene_dir


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--df", type=int, default=4)
    parser.add_argument("--lr-shift", type=float, default=1.0)
    parser.add_argument("--aug", type=str, default="none")
    parser.add_argument(
        "--band-preset",
        type=str,
        default="rgb_nir",
        choices=sorted(BAND_PRESETS),
        help=f"Band group (default rgb_nir). Choices: {', '.join(sorted(BAND_PRESETS))}",
    )
    parser.add_argument("--overwrite", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_geosr = sub.add_parser("geosr", help="Export from GeoSR MISR parquet pack")
    p_geosr.add_argument("--input", type=Path, default=Path("geosr_misr_sample20"))
    p_geosr.add_argument("--scene-id", type=str, required=True)
    _add_common_args(p_geosr)

    p_syn = sub.add_parser("synthetic", help="Write a synthetic multi-band demo scene")
    p_syn.add_argument(
        "--scene-id",
        type=str,
        default="demo",
        help="Scene folder name under --output parent",
    )
    p_syn.add_argument("--lr-size", type=int, default=32)
    p_syn.add_argument("--num-frames", type=int, default=6)
    p_syn.add_argument("--seed", type=int, default=0)
    _add_common_args(p_syn)

    args = parser.parse_args()
    preset = resolve_band_preset(args.band_preset)

    if args.command == "geosr":
        scene_dir = export_geosr_multiband(
            input_root=args.input.resolve(),
            output_root=args.output.resolve(),
            scene_id=args.scene_id,
            preset=preset,
            df=args.df,
            lr_shift=args.lr_shift,
            aug=args.aug,
            overwrite=args.overwrite,
        )
        print(f"Exported GeoSR scene {args.scene_id} -> {scene_dir}")
        print(f"See data_multiband/README.txt for layout and band presets ({preset.name}).")
        return

    if args.command == "synthetic":
        scene_dir = export_synthetic_multiband(
            output_scene_dir=args.output.resolve() / args.scene_id,
            preset=preset,
            df=args.df,
            lr_shift=args.lr_shift,
            aug=args.aug,
            lr_size=args.lr_size,
            num_frames=args.num_frames,
            seed=args.seed,
        )
        print(f"Wrote synthetic scene -> {scene_dir}")
        print(f"See data_multiband/README.txt for layout and band presets ({preset.name}).")


if __name__ == "__main__":
    main()
