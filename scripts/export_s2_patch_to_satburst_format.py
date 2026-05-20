#!/usr/bin/env python3
"""
Export a fixed Sentinel-2 patch stack into satburst_synth-compatible folder format.

Output folder structure matches what SRData expects:
  <output_dir>/
    hr_ground_truth.png
    sample_00.png
    sample_01.png
    ...
    transform_log.json

Notes:
- We do not have true HR ground truth in S2 TCI time-series patches.
- For compatibility, hr_ground_truth.png is set to bicubic upsampled sample_00.

Example (satburst layout, v2 folder; GeoTIFF stack from download_s2_timeseries):

  python scripts/export_s2_patch_to_satburst_format.py \\
    --tci_folder s2_norway/geotiff \\
    --output_dir data/s2_patch_center_v2/scale_4_shift_1.0px_aug_none \\
    --patch_size 64 --df 4 --convert_raw_tif_to_rgb

PNG-only TCI folder works without rasterio (uses Pillow only for non-TIFF).

Optional ``--raw_b432_folder`` (GeoTIFFs from ``download_s2_timeseries.py --write-raw-b432``):
writes ``sample_XX_raw.npz`` with float32 ``reflectance_b432`` (H,W,3) B4–B3–2 BOA ~0–1.
Apply ``s2_reflectance_utils.post_sr_reflectance_to_rgb_u8`` on SR outputs (CHW) for display RGB.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from s2_reflectance_utils import s2_to_rgb

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Pillow 9.1+ uses Image.Resampling; older versions use Image.BICUBIC.
try:
    _RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    _RESAMPLE_BICUBIC = Image.BICUBIC


def _list_images(folder: Path) -> list[Path]:
    exts = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.tif", "*.tiff")
    paths: list[Path] = []
    for ext in exts:
        paths.extend(sorted(folder.glob(ext)))
    return sorted(paths)


def _read_rgb_float01(
    path: Path,
    convert_raw_tif_to_rgb: bool = False,
    raw_reflectance_scale: float = 10000.0,
    raw_clip_percentile: float = 99.5,
    raw_gamma: float = 0.7,
) -> np.ndarray | None:
    suffix = path.suffix.lower()
    arr = None
    if suffix in {".tif", ".tiff"} and HAS_RASTERIO:
        try:
            with rasterio.open(path) as ds:
                if convert_raw_tif_to_rgb:
                    n = ds.count
                else:
                    n = min(3, ds.count)
                if n > 0:
                    arr = ds.read(list(range(1, n + 1)))  # [C,H,W]
                    arr = np.moveaxis(arr, 0, -1)         # [H,W,C]
        except Exception:
            arr = None

    if arr is None:
        try:
            im = Image.open(path)
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        except Exception:
            return None
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]

    orig_dtype = arr.dtype
    arr = arr.astype(np.float32)
    is_tif = suffix in {".tif", ".tiff"}
    looks_raw = arr.max() > 255.0 or orig_dtype != np.uint8
    if convert_raw_tif_to_rgb and is_tif and looks_raw:
        chw = np.moveaxis(arr, -1, 0)
        if chw.shape[0] >= 3:
            if chw.shape[0] > 13:
                chw = chw[:13]
            if chw.shape[0] == 3:
                if float(np.nanmax(chw)) <= 1.5 and raw_reflectance_scale > 0:
                    chw = chw * (10000.0 / float(raw_reflectance_scale))
            elif raw_reflectance_scale > 0:
                chw = chw * (10000.0 / float(raw_reflectance_scale))
            rgb8 = s2_to_rgb(
                chw,
                smooth_quantiles=bool(raw_clip_percentile > 0),
                gamma=raw_gamma,
            )
            return (rgb8.astype(np.float32) / 255.0).clip(0.0, 1.0)

    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _get_common_hw(paths: list[Path]) -> tuple[int, int]:
    h_min = None
    w_min = None
    for p in paths:
        img = _read_rgb_float01(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        h_min = h if h_min is None else min(h_min, h)
        w_min = w if w_min is None else min(w_min, w)
    if h_min is None or w_min is None:
        raise RuntimeError("Could not read any images from input folder.")
    return int(h_min), int(w_min)


def _center_crop(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    y0 = (ih - h) // 2
    x0 = (iw - w) // 2
    return img[y0 : y0 + h, x0 : x0 + w]


def mosaic_center_origin(
    h_min: int,
    w_min: int,
    *,
    reference_patch_size: int,
    stride: int,
    patch_row: int | None = None,
    patch_col: int | None = None,
) -> tuple[int, int, int, int]:
    """Top-left (y0, x0) of the center patch on a h_min×w_min mosaic for a reference patch size."""
    ref = int(reference_patch_size)
    st = int(stride)
    if ref <= 0 or st <= 0:
        raise ValueError("reference_patch_size and stride must be positive")
    n_rows = max(1, (int(h_min) - ref) // st + 1)
    n_cols = max(1, (int(w_min) - ref) // st + 1)
    pr = n_rows // 2 if patch_row is None else max(0, min(int(patch_row), n_rows - 1))
    pc = n_cols // 2 if patch_col is None else max(0, min(int(patch_col), n_cols - 1))
    return pr * st, pc * st, int(h_min), int(w_min)


def _resolve_mosaic_origin(
    h_min: int,
    w_min: int,
    *,
    patch: int,
    stride: int,
    patch_row: int | None,
    patch_col: int | None,
    mosaic_y0: int | None,
    mosaic_x0: int | None,
    mosaic_origin_file: str | None,
    align_mosaic_origin_to_patch_size: int | None,
) -> tuple[int, int, int, int]:
    if mosaic_y0 is not None and mosaic_x0 is not None:
        return int(mosaic_y0), int(mosaic_x0), int(h_min), int(w_min)
    if mosaic_origin_file:
        p = Path(mosaic_origin_file)
        if p.is_file():
            with p.open("r") as f:
                meta = json.load(f)
            return (
                int(meta["y0"]),
                int(meta["x0"]),
                int(meta.get("h_min", h_min)),
                int(meta.get("w_min", w_min)),
            )
    if align_mosaic_origin_to_patch_size is not None and int(align_mosaic_origin_to_patch_size) > 0:
        return mosaic_center_origin(
            h_min,
            w_min,
            reference_patch_size=int(align_mosaic_origin_to_patch_size),
            stride=stride,
            patch_row=patch_row,
            patch_col=patch_col,
        )
    n_rows = max(1, (h_min - patch) // stride + 1)
    n_cols = max(1, (w_min - patch) // stride + 1)
    pr = n_rows // 2 if patch_row is None else max(0, min(int(patch_row), n_rows - 1))
    pc = n_cols // 2 if patch_col is None else max(0, min(int(patch_col), n_cols - 1))
    return pr * stride, pc * stride, int(h_min), int(w_min)


def _read_b432_hwc(path: Path) -> np.ndarray | None:
    """Load B4,B3,B2 as (H,W,C) float32 BOA reflectance ~0–1."""
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(path) as ds:
            if ds.count < 3:
                return None
            chw = ds.read((1, 2, 3)).astype(np.float32)
    except Exception:
        return None
    hwc = np.transpose(chw, (1, 2, 0))
    hwc = np.nan_to_num(hwc, nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(np.nanmax(hwc)) if hwc.size else 0.0
    if mx > 1.5:
        hwc = hwc / 10000.0
    return np.clip(hwc, 0.0, None).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Export S2 patch stack to satburst_synth format")
    p.add_argument("--tci_folder", type=str, required=True, help="Folder with S2 TCI images")
    p.add_argument("--output_dir", type=str, required=True, help="Output folder in satburst_synth format")
    p.add_argument("--patch_size", type=int, default=64, help="LR patch size")
    p.add_argument("--patch_row", type=int, default=None, help="Patch row index (default center)")
    p.add_argument("--patch_col", type=int, default=None, help="Patch col index (default center)")
    p.add_argument("--stride", type=int, default=None, help="Patch stride (default patch_size)")
    p.add_argument("--df", type=int, default=4, help="Upsampling factor for hr_ground_truth placeholder")
    p.add_argument(
        "--convert_raw_tif_to_rgb",
        action="store_true",
        help="If inputs are raw S2 tif reflectance, convert to display RGB before export.",
    )
    p.add_argument(
        "--raw_reflectance_scale",
        type=float,
        default=10000.0,
        help="Scale divisor for raw reflectance conversion (default: 10000).",
    )
    p.add_argument(
        "--raw_clip_percentile",
        type=float,
        default=99.5,
        help="Per-channel upper percentile for raw tif stretch (default: 99.5).",
    )
    p.add_argument(
        "--raw_gamma",
        type=float,
        default=0.7,
        help="Gamma for raw tif conversion (default: 0.7).",
    )
    p.add_argument(
        "--raw_b432_folder",
        type=str,
        default=None,
        help=(
            "Optional folder of 3-band GeoTIFFs (same filenames as tci_folder), e.g. "
            "…/geotiff_raw from download_s2_timeseries.py --write-raw-b432. "
            "Writes sample_XX_raw.npz (reflectance_b432 HWC float32) alongside LR PNGs."
        ),
    )
    p.add_argument(
        "--mosaic_y0",
        type=int,
        default=None,
        help="Fixed top-left row on the common mosaic (overrides grid center for this patch size).",
    )
    p.add_argument(
        "--mosaic_x0",
        type=int,
        default=None,
        help="Fixed top-left col on the common mosaic (use with --mosaic_y0).",
    )
    p.add_argument(
        "--mosaic_origin_file",
        type=str,
        default=None,
        help="JSON with y0/x0/h_min/w_min from a prior export (multi-size alignment).",
    )
    p.add_argument(
        "--align_mosaic_origin_to_patch_size",
        type=int,
        default=None,
        help=(
            "Use the center-patch origin computed for this reference LR size (e.g. 64) for "
            "all exports so s2_scene_center_v2_{64,128,256} share the same top-left footprint."
        ),
    )
    p.add_argument(
        "--write_mosaic_origin",
        type=str,
        default=None,
        help="Write mosaic_origin.json (y0, x0, h_min, w_min) to this path after resolving origin.",
    )
    args = p.parse_args()

    tci_folder = Path(args.tci_folder)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _list_images(tci_folder)
    if not paths:
        raise RuntimeError(f"No images found in {tci_folder}")

    raw_b432_folder: Path | None = Path(args.raw_b432_folder) if args.raw_b432_folder else None
    if raw_b432_folder is not None:
        if not HAS_RASTERIO:
            raise RuntimeError("--raw_b432_folder requires rasterio.")
        missing = [p.name for p in paths if not (raw_b432_folder / p.name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"raw_b432_folder {raw_b432_folder} missing {len(missing)} file(s) "
                f"(first few: {missing[:5]})"
            )

    patch = int(args.patch_size)
    stride = int(args.stride) if args.stride is not None else patch
    if patch <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")

    h_min, w_min = _get_common_hw(paths)
    y0, x0, h_min, w_min = _resolve_mosaic_origin(
        h_min,
        w_min,
        patch=patch,
        stride=stride,
        patch_row=args.patch_row,
        patch_col=args.patch_col,
        mosaic_y0=args.mosaic_y0,
        mosaic_x0=args.mosaic_x0,
        mosaic_origin_file=args.mosaic_origin_file,
        align_mosaic_origin_to_patch_size=args.align_mosaic_origin_to_patch_size,
    )
    y1 = y0 + patch
    x1 = x0 + patch
    if y1 > h_min or x1 > w_min:
        raise RuntimeError(
            f"Patch {patch}x{patch} at mosaic origin ({y0},{x0}) exceeds common crop {h_min}x{w_min}. "
            "Use a smaller patch_size or a different origin."
        )

    mosaic_meta = {
        "y0": int(y0),
        "x0": int(x0),
        "h_min": int(h_min),
        "w_min": int(w_min),
        "patch_size": int(patch),
        "reference_patch_size": int(args.align_mosaic_origin_to_patch_size or patch),
        "shared_eval_crop_lr_size": int(args.align_mosaic_origin_to_patch_size or patch),
    }
    if args.write_mosaic_origin:
        origin_path = Path(args.write_mosaic_origin)
        origin_path.parent.mkdir(parents=True, exist_ok=True)
        with origin_path.open("w") as f:
            json.dump(mosaic_meta, f, indent=2)
        print(f"Wrote mosaic origin metadata to {origin_path}")

    with open(out_dir / "mosaic_origin.json", "w") as f:
        json.dump(mosaic_meta, f, indent=2)

    print(f"Found {len(paths)} images")
    print(f"Common crop: {h_min}x{w_min}")
    print(f"Mosaic origin (y0,x0)=({y0},{x0}), exporting {patch}x{patch} -> [{y0}:{y1}, {x0}:{x1}]")

    transform_log = {}
    saved = 0
    first_patch_rgb = None

    for i, path in enumerate(paths):
        rgb01 = _read_rgb_float01(
            path,
            convert_raw_tif_to_rgb=bool(args.convert_raw_tif_to_rgb),
            raw_reflectance_scale=float(args.raw_reflectance_scale),
            raw_clip_percentile=float(args.raw_clip_percentile),
            raw_gamma=float(args.raw_gamma),
        )
        if rgb01 is None:
            continue
        rgb = (np.clip(rgb01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        rgb = _center_crop(rgb, h_min, w_min)
        p_rgb = rgb[y0:y1, x0:x1]
        if p_rgb.shape[0] != patch or p_rgb.shape[1] != patch:
            continue

        if first_patch_rgb is None:
            first_patch_rgb = p_rgb.copy()

        sample_name = f"sample_{saved:02d}"
        sample_file = f"{sample_name}.png"
        Image.fromarray(p_rgb, mode="RGB").save(out_dir / sample_file)

        entry = {
            "dx_pixels_hr": 0.0,
            "dy_pixels_hr": 0.0,
            "dx_pixels_lr": 0.0,
            "dy_pixels_lr": 0.0,
            "dx_percent": 0.0,
            "dy_percent": 0.0,
            "magnitude_pixels_hr": 0.0,
            "magnitude_pixels_lr": 0.0,
            "shape": [int(patch), int(patch), 3],
            "path": sample_file,
            "augmentation": "none",
            "source_file": path.name,
        }
        if raw_b432_folder is not None:
            raw_hwc = _read_b432_hwc(raw_b432_folder / path.name)
            if raw_hwc is None:
                raise RuntimeError(f"Could not read 3-band reflectance from {raw_b432_folder / path.name}")
            raw_hwc = _center_crop(raw_hwc, h_min, w_min)
            p_raw = raw_hwc[y0:y1, x0:x1]
            raw_name = f"{sample_name}_raw.npz"
            np.savez_compressed(out_dir / raw_name, reflectance_b432=p_raw.astype(np.float32))
            entry["raw_reflectance_npz"] = raw_name

        transform_log[sample_name] = entry
        saved += 1

    if saved == 0 or first_patch_rgb is None:
        raise RuntimeError("No patches were exported.")

    hr_h = int(patch * args.df)
    hr_w = int(patch * args.df)
    hr_placeholder = np.asarray(
        Image.fromarray(first_patch_rgb, mode="RGB").resize((hr_w, hr_h), resample=_RESAMPLE_BICUBIC),
        dtype=np.uint8,
    )
    Image.fromarray(hr_placeholder, mode="RGB").save(out_dir / "hr_ground_truth.png")

    with open(out_dir / "transform_log.json", "w") as f:
        json.dump(transform_log, f, indent=2)

    if raw_b432_folder is not None:
        meta = {
            "bands": ["B4", "B3", "B2"],
            "npz_array_key": "reflectance_b432",
            "layout": "HWC float32 BOA reflectance approximately 0 to 1",
            "post_sr_display_rgb": (
                "from s2_reflectance_utils import post_sr_reflectance_to_rgb_u8; "
                "rgb_hwc = post_sr_reflectance_to_rgb_u8(sr_output_chw)"
            ),
        }
        with open(out_dir / "reflectance_export_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Exported {saved} samples to {out_dir}")
    print(f"Wrote hr_ground_truth.png as bicubic(sample_00) at {hr_w}x{hr_h}")


if __name__ == "__main__":
    main()

