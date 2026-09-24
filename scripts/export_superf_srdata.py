#!/usr/bin/env python3
"""Export a processed ScaleF S2/NIB sample in original SuperF SRData layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2_dataset import S2NIBRevisitDataset

SOURCE_COMMIT = "dff884c9ed61a6227737537c834a2604792af478"


def _save_rgb(path: Path, image: np.ndarray) -> None:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB, got {array.shape}")
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), "RGB").save(path)


def export_arrays(
    lr_frames: np.ndarray,
    hr: np.ndarray,
    output_dir: Path,
    *,
    frame_sources: list[str] | None = None,
    provenance: dict | None = None,
) -> dict:
    """Write arrays and zero-shift transform metadata accepted by SRData loaders."""
    lr_frames = np.asarray(lr_frames)
    hr = np.asarray(hr)
    if lr_frames.ndim != 4 or lr_frames.shape[-1] != 3:
        raise ValueError(f"expected THWC LR frames, got {lr_frames.shape}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(output_dir / "hr_ground_truth.png", hr)
    transform_log = {}
    sources = frame_sources or ["unknown"] * len(lr_frames)
    for i, frame in enumerate(lr_frames):
        name = f"sample_{i:02d}"
        filename = f"{name}.png"
        _save_rgb(output_dir / filename, frame)
        transform_log[name] = {
            "dx_pixels_hr": 0.0,
            "dy_pixels_hr": 0.0,
            "dx_pixels_lr": 0.0,
            "dy_pixels_lr": 0.0,
            "dx_percent": 0.0,
            "dy_percent": 0.0,
            "magnitude_pixels_hr": 0.0,
            "magnitude_pixels_lr": 0.0,
            "shape": list(frame.shape),
            "path": filename,
            "augmentation": "none",
            "source_file": sources[i] if i < len(sources) else "unknown",
        }
    (output_dir / "transform_log.json").write_text(json.dumps(transform_log, indent=2) + "\n")
    payload = {
        "format": "SuperF SRData",
        "source_commit": SOURCE_COMMIT,
        "preprocessing": provenance or {},
        "frame_count": int(len(lr_frames)),
        "lr_shape": list(lr_frames.shape[1:]),
        "hr_shape": list(hr.shape),
        "quantization": "RGB uint8 PNG, round(clip(reflectance, 0, 1) * 255)",
    }
    (output_dir / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def export_dataset(dataset: S2NIBRevisitDataset, output_dir: Path) -> dict:
    if not dataset.has_hr_gt:
        raise ValueError("SuperF export requires NIB HR ground truth")
    lr = dataset.lr_rgb.detach().cpu().numpy()
    hr = dataset.get_original_hr().detach().cpu().numpy()
    geo = dataset.get_geo_meta()
    geo_json = {
        key: list(value)[:6] if key.endswith("_transform") and value is not None else value
        for key, value in geo.items()
    }
    return export_arrays(
        lr,
        hr,
        output_dir,
        frame_sources=[str(dataset.s2_dir / frame["path"]) for frame in dataset.frames],
        provenance={
            "source_s2_dir": str(dataset.s2_dir.resolve()),
            "source_meta": str((dataset.s2_dir / "meta.json").resolve()),
            "base_frame_index": dataset.base_frame_index,
            "base_frame_date": dataset.base_frame_date,
            "nib_acquisition_date": dataset.nib_acquisition_date,
            "hr_source": str(dataset.hr_path),
            "hr_harmonization": dataset.hr_harmonize_method,
            "hr_spatial_shift": dataset.hr_spatial_shift,
            "georeferencing": geo_json,
            "note": "GeoTIFFs are rendered to PNG; source CRS/affines are retained here.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("s2_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", default="s2")
    parser.add_argument("--hr-path")
    parser.add_argument("--df", type=int, default=4)
    parser.add_argument("--hr-gsd-m", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--no-hr-harmonize", action="store_true")
    parser.add_argument("--no-hr-spatial-align", action="store_true")
    parser.add_argument("--spatial-alignment-path")
    args = parser.parse_args()
    dataset_args = SimpleNamespace(
        s2_dir=str(args.s2_dir),
        hr_path=args.hr_path,
        df=args.df,
        scale_factor=args.df,
        hr_gsd_m=args.hr_gsd_m,
        s2_native_gsd_m=10.0,
        num_samples=args.num_samples,
        lr_size=0,
        allow_no_hr=False,
        no_hr_harmonize=args.no_hr_harmonize,
        no_hr_spatial_align=args.no_hr_spatial_align,
        spatial_alignment_path=args.spatial_alignment_path,
        dataset_device="cpu",
    )
    dataset = S2NIBRevisitDataset(dataset_args, name=args.dataset, training_device="cpu")
    payload = export_dataset(dataset, args.output)
    print(f"Wrote {args.output} ({payload['frame_count']} LR frames)")


if __name__ == "__main__":
    main()
