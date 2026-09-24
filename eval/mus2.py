"""MuS2-compatible discovery, preparation, and evaluation helpers.

The metric implementation intentionally mirrors the public MuS2 evaluator:
predictions are histogram-matched to WorldView-2, cPSNR chooses a small
integer translation while compensating a scalar bias, and cSSIM reuses that
translation. MuS2 masks use white/non-zero for excluded pixels.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity

MUS2_BAND_PAIRS = {"b2": 1, "b3": 2, "b4": 4, "b8": 6}
_IMAGE_SUFFIXES = {".jp2", ".tif", ".tiff", ".png"}
_DATE_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


def image_files(path: Path) -> list[Path]:
    """Return supported images directly below ``path`` in stable order."""
    if not path.is_dir():
        return []
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda item: item.name,
    )


def find_lr_files(scene_dir: Path, band: str) -> list[Path]:
    """Find MuS2 LR revisits in either the published or flattened layout."""
    nested = image_files(scene_dir / band / "lrs")
    return nested or image_files(scene_dir / band)


def find_hr_file(scene_dir: Path, band: str) -> Path | None:
    index = MUS2_BAND_PAIRS[band]
    hr_dir = scene_dir / "hr_resized"
    for suffix in (".tiff", ".tif", ".jp2", ".png"):
        candidate = hr_dir / f"mul_band_{index}{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(hr_dir.glob(f"mul_band_{index}.*")) if hr_dir.is_dir() else []
    return next((path for path in matches if path.suffix.lower() in _IMAGE_SUFFIXES), None)


def discover_scenes(root: Path, bands: Iterable[str] = MUS2_BAND_PAIRS) -> list[dict[str, Any]]:
    """Discover official MuS2 scene folders without assuming archive nesting."""
    root = root.resolve()
    scene_dirs: set[Path] = set()
    for hr_dir in root.rglob("hr_resized"):
        if hr_dir.is_dir():
            scene_dirs.add(hr_dir.parent)

    scenes: list[dict[str, Any]] = []
    for scene_dir in sorted(scene_dirs):
        available: dict[str, dict[str, Any]] = {}
        for band in bands:
            band = band.lower()
            if band not in MUS2_BAND_PAIRS:
                raise ValueError(f"unsupported MuS2 band {band!r}")
            lr_files = find_lr_files(scene_dir, band)
            hr_file = find_hr_file(scene_dir, band)
            if lr_files and hr_file is not None:
                available[band] = {
                    "lr_files": [str(path) for path in lr_files],
                    "hr_file": str(hr_file),
                }
        if available:
            scenes.append(
                {
                    "id": scene_dir.name,
                    "source_dir": str(scene_dir),
                    "bands": available,
                }
            )
    return scenes


def find_mask(mask_root: Path | None, scene_id: str, band: str, mode: str = "final") -> Path | None:
    """Locate a published MuS2 mask, tolerating archive wrapper directories."""
    if mask_root is None or mode == "none":
        return None
    folder_prefix = {
        "final": "final_masks",
        "relevance": "relevance_masks",
        "difference_newest": "difference_masks_newest",
        "perceptual": "perceptual_masks",
    }.get(mode)
    if folder_prefix is None:
        raise ValueError(f"unsupported MuS2 mask mode {mode!r}")
    roots = [
        path
        for path in mask_root.rglob(f"{folder_prefix}_{band}")
        if path.is_dir()
    ]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            path
            for path in root.iterdir()
            if path.is_file()
            and path.suffix.lower() in _IMAGE_SUFFIXES
            and scene_id in path.stem
        )
    return sorted(candidates)[0] if candidates else None


def acquisition_date(path: Path, fallback: str = "2000-01-01") -> str:
    """Extract an ISO date from common Sentinel product filenames."""
    match = _DATE_RE.search(path.name)
    if not match:
        return fallback
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def read_grayscale(path: Path) -> np.ndarray:
    """Read the first raster band while preserving its numeric range."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            array = src.read(1)
    except Exception:  # noqa: BLE001 - OpenCV supports non-geospatial test fixtures
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise ValueError(f"could not read image {path}")
        if array.ndim == 3:
            array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"expected one-band image at {path}, got shape {array.shape}")
    return np.asarray(array)


def _unit_float(array: np.ndarray) -> np.ndarray:
    result = np.nan_to_num(np.asarray(array, dtype=np.float64))
    if result.size and float(np.max(result)) > 1.0:
        result /= 255.0
    return result


def histogram_match_prediction(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match prediction values to the reference, as in the MuS2 evaluator."""
    return match_histograms(np.asarray(prediction), np.asarray(reference)).astype(np.float64)


def valid_mask_from_mus2(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """Convert MuS2's non-zero=excluded convention to a boolean valid mask."""
    if mask is None:
        return np.ones(shape, dtype=bool)
    resized = cv2.resize(
        np.asarray(mask, dtype=np.uint8),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    return ~resized.astype(bool)


def _bias_corrected_psnr(sr: np.ndarray, hr: np.ndarray, valid: np.ndarray) -> float:
    count = int(valid.sum())
    if count == 0:
        raise ValueError("MuS2 evaluation mask has no valid pixels")
    bias = float(((hr - sr) * valid).sum() / count)
    mse = float((np.square(hr - (sr + bias)) * valid).sum() / count)
    return float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))


def _bias_corrected_ssim(sr: np.ndarray, hr: np.ndarray, valid: np.ndarray) -> float:
    count = int(valid.sum())
    if count == 0:
        raise ValueError("MuS2 evaluation mask has no valid pixels")
    bias = float(((hr - sr) * valid).sum() / count)
    min_side = min(sr.shape)
    win_size = min(7, min_side if min_side % 2 else min_side - 1)
    if win_size < 3:
        raise ValueError("cSSIM requires aligned images at least 3x3")
    return float(
        structural_similarity(
            (sr + bias) * valid,
            hr * valid,
            data_range=1.0,
            win_size=win_size,
        )
    )


def align_official(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    max_shift: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], float]:
    """Select the translation that maximizes MuS2 cPSNR.

    This reproduces the public evaluator's ``range(2 * max_shift)`` search,
    including its asymmetric signed range ``[-max_shift, max_shift - 1]``.
    """
    if prediction.shape != reference.shape or reference.shape != valid.shape:
        raise ValueError(
            f"prediction, reference, and mask shapes must match; got "
            f"{prediction.shape}, {reference.shape}, {valid.shape}"
        )
    max_shift = int(max_shift)
    if max_shift < 0 or min(prediction.shape) <= 2 * max_shift:
        raise ValueError(f"invalid max_shift={max_shift} for image shape {prediction.shape}")
    if max_shift == 0:
        score = _bias_corrected_psnr(prediction, reference, valid)
        return prediction, reference, valid, (0, 0), score

    sr_crop = prediction[max_shift:-max_shift, max_shift:-max_shift]
    height, width = reference.shape
    best: tuple[np.ndarray, np.ndarray, tuple[int, int], float] | None = None
    for row in range(2 * max_shift):
        for col in range(2 * max_shift):
            hr_crop = reference[row : height - (2 * max_shift - row), col : width - (2 * max_shift - col)]
            mask_crop = valid[row : height - (2 * max_shift - row), col : width - (2 * max_shift - col)]
            score = _bias_corrected_psnr(sr_crop, hr_crop, mask_crop)
            if best is None or score > best[3]:
                best = (hr_crop, mask_crop, (row - max_shift, col - max_shift), score)
    assert best is not None
    return sr_crop, best[0], best[1], best[2], best[3]


def evaluate_arrays(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    excluded_mask: np.ndarray | None = None,
    max_shift: int = 3,
    lpips_model: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one single-band reconstruction with the MuS2 protocol."""
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError(f"prediction shape {prediction.shape} != reference shape {reference.shape}")
    matched = histogram_match_prediction(prediction, reference)
    if reference.size and float(np.max(reference)) > 1.0:
        # The public evaluator converts matched images to uint8 before metrics.
        matched = matched.astype(np.uint8)
        reference = reference.astype(np.uint8)
    sr = _unit_float(matched)
    hr = _unit_float(reference)
    valid = valid_mask_from_mus2(excluded_mask, hr.shape)
    sr_crop, hr_crop, valid_crop, shift, cpsnr = align_official(sr, hr, valid, max_shift=max_shift)
    cssim = _bias_corrected_ssim(sr_crop, hr_crop, valid_crop)
    result: dict[str, Any] = {
        "cPSNR": cpsnr,
        "cSSIM": cssim,
        "shift_yx": [int(shift[0]), int(shift[1])],
        "valid_fraction": float(valid_crop.mean()),
        "histogram_match": "prediction_to_worldview2",
        "mask_convention": "nonzero_excluded",
    }
    if lpips_model is not None:
        import torch

        # Official LPIPS is evaluated on the full, co-registered image rather
        # than reusing the cPSNR-selected integer shift.
        masked_sr = np.where(valid, sr, hr)
        sr_tensor = torch.from_numpy(masked_sr).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
        hr_tensor = torch.from_numpy(hr).float()[None, None].repeat(1, 3, 1, 1) * 2 - 1
        device = next(lpips_model.parameters()).device
        with torch.no_grad():
            result["LPIPS"] = float(lpips_model(sr_tensor.to(device), hr_tensor.to(device)).item())
    return result


def evaluate_files(
    prediction_path: Path,
    reference_path: Path,
    *,
    mask_path: Path | None = None,
    max_shift: int = 3,
    lpips_model: Any | None = None,
) -> dict[str, Any]:
    """File-based wrapper around :func:`evaluate_arrays`."""
    excluded = read_grayscale(mask_path) if mask_path is not None else None
    result = evaluate_arrays(
        read_grayscale(prediction_path),
        read_grayscale(reference_path),
        excluded_mask=excluded,
        max_shift=max_shift,
        lpips_model=lpips_model,
    )
    result.update(
        {
            "prediction": str(prediction_path),
            "reference": str(reference_path),
            "mask": str(mask_path) if mask_path is not None else None,
        }
    )
    return result


def bicubic_baseline(lr_paths: Iterable[Path], reference_shape: tuple[int, int]) -> np.ndarray:
    """Build MuS2's mean of bicubically upsampled LR revisits."""
    resized = [
        cv2.resize(
            read_grayscale(Path(path)).astype(np.float64),
            (reference_shape[1], reference_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        for path in lr_paths
    ]
    if not resized:
        raise ValueError("cannot build bicubic baseline without LR revisits")
    return np.mean(resized, axis=0)


def balanced_score(candidate: dict[str, float], bicubic: dict[str, float]) -> float:
    """Compute MuS2's balanced score B; lower than one beats bicubic."""
    terms = [
        bicubic["cPSNR"] / candidate["cPSNR"],
        bicubic["cSSIM"] / candidate["cSSIM"],
    ]
    if "LPIPS" in candidate and "LPIPS" in bicubic:
        terms.append(candidate["LPIPS"] / bicubic["LPIPS"])
    return float(np.mean(terms))
