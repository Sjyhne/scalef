"""Sentinel-2 revisit stack + NIB HR ortho dataset for ScaleF training."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data import (
    _build_input_coord_cache,
    _make_coord_grid,
    get_and_standardize_image,
    resolve_dataset_device,
)

DEFAULT_S2_DIR = Path("data/s2_revisits/bergen")
DEFAULT_NIB_ROOT = Path("data/nib_resampled")
DEFAULT_NIB_FOCUS_ROOT = Path("data/nib_focus_1m_worldcover/projects")
DEFAULT_NIB_NEW_ROOT = Path("data/nib_new_1m_worldcover/projects")
DEFAULT_SPATIAL_ALIGNMENT_PATH = Path("eval/spatial_alignment.json")
# Frozen identity frame: closest to the scene date among frames at or under this
# cell SCL cloud fraction. National 32VNM v2 bright squares tracked the
# *earliest* stack frame (often 10–14% cloud that still passed the 15% gate).
DEFAULT_MAX_BASE_CLOUD_FRAC = 0.02
S2_RGB_BANDS = (1, 2, 3)  # B04, B03, B02 in the RGBNIR GeoTIFF
S2_NIR_BAND = 4

# city id → NIB project folder (aois.json collection: focus + new exports)
FOCUS_PROJECT_BY_CITY = {
    "asker": "01_asker_akershus",
    "vennesla": "02_vennesla_agder",
    "trondheim": "03_trondheim_trondelag",
    "bergen": "04_bergen_vestland",
    "rana": "05_rana_nordland",
    "tromso": "06_tromso_troms",
    "amli": "07_amli_agder",
    "stavanger": "08_stavanger_rogaland",
    "algard": "01_eksport_7531643_1_tile01",
    "naerbo": "01_eksport_7531643_1_tile01",
    "flekkefjord": "02_eksport_7532043_1",
    "rafsbotn": "03_eksport_7532042_1_tile01",
    "nittedal": "04_eksport_7532041_1",
    "melhus": "05_eksport_7531640_1",
    "alta": "06_eksport_7528850_1",
    "karasjok": "08_eksport_7532042_1_tile03",
    "kautokeino": "09_eksport_7532042_1_tile04",
}

# Prefer focus package first, then the newer export package.
_NIB_PROJECT_ROOTS = (DEFAULT_NIB_FOCUS_ROOT, DEFAULT_NIB_NEW_ROOT)


def is_s2_dataset_request(args, name: str | None) -> bool:
    if getattr(args, "s2_dir", None) or getattr(args, "hr_path", None):
        return True
    text = str(name or "s2").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in {"s2", "s2_nib", "sentinel2", "sentinel-2"}:
        return True
    path = Path(text)
    if (path / "meta.json").is_file():
        return True
    return (Path("data/s2_revisits") / text / "meta.json").is_file()


def _format_gsd_m(gsd_m: float) -> str:
    if abs(gsd_m - round(gsd_m)) < 1e-6:
        return str(int(round(gsd_m)))
    return f"{gsd_m:g}"


def resolve_s2_dir(args, name: str | None = None) -> Path:
    explicit = getattr(args, "s2_dir", None)
    if explicit:
        path = Path(explicit)
        if not (path / "meta.json").is_file():
            raise FileNotFoundError(f"--s2-dir {path} has no meta.json")
        return path
    text = str(name or "s2").strip()
    candidate = Path(text)
    if (candidate / "meta.json").is_file():
        return candidate
    city_dir = Path("data/s2_revisits") / text
    if (city_dir / "meta.json").is_file():
        return city_dir
    if (DEFAULT_S2_DIR / "meta.json").is_file():
        return DEFAULT_S2_DIR
    raise FileNotFoundError(
        "No Sentinel-2 revisit directory found. Pass --s2-dir data/s2_revisits/bergen"
    )


def _city_id_from_s2_dir(s2_dir: Path) -> str:
    """Map a revisit folder back to the city id used for NIB focus lookup.

    Size-ladder dirs are ``{city}_lr{N}``. Affine-grid dirs are
    ``{city}_g{N}`` or ``{city}_g{N}_y{i}_x{j}``.
    """
    name = s2_dir.name
    if "_lr" in name:
        base, _, suffix = name.rpartition("_lr")
        if suffix.isdigit() and base:
            return base
    for city in sorted(FOCUS_PROJECT_BY_CITY, key=len, reverse=True):
        if name == city or name.startswith(f"{city}_"):
            return city
    return name


def resolve_focus_project_dir(city: str) -> Path | None:
    """Directory under a NIB package that holds the city's export + valid mask."""
    folder = FOCUS_PROJECT_BY_CITY.get(city)
    if not folder:
        return None
    for root in _NIB_PROJECT_ROOTS:
        proj = root / folder
        if proj.is_dir():
            return proj
    return None


def resolve_focus_hr_path(city: str) -> str | None:
    """NIB 1 m export via GDAL /vsizip (no extract).

    Searches ``nib_focus_1m_worldcover`` then ``nib_new_1m_worldcover``.
    Zip basename differs across packages; member is always ``Eksport-nib.tif``.
    Returned as ``str`` (not ``Path``) so ``/vsizip//abs/path.zip/...`` is preserved;
    ``pathlib.Path`` would collapse the double slash and break GDAL.
    """
    folder = FOCUS_PROJECT_BY_CITY.get(city)
    if not folder:
        return None
    zip_names = (
        "nib_export_with_worldcover_metadata.zip",
        "nib_export.zip",
    )
    for root in _NIB_PROJECT_ROOTS:
        for zip_name in zip_names:
            zpath = root / folder / zip_name
            if zpath.is_file():
                return f"/vsizip/{zpath.resolve()}/Eksport-nib.tif"
    return None


def resolve_hr_path(args, s2_dir: Path, hr_gsd_m: float) -> Path | str | None:
    """Resolve NIB/HR GT path, or None for production SR-only runs.

    Production Norway (and other SR-only runs) pass ``--allow_no_hr`` and never
    load aerial HR, even if a NIB package exists for the parent city id.
    Explicit ``--hr-path`` still wins (dev override).
    """
    explicit = getattr(args, "hr_path", None)
    if explicit:
        text = str(explicit)
        if text.startswith("/vsizip/"):
            return text
        path = Path(text)
        if not path.is_file():
            raise FileNotFoundError(f"--hr-path {path} does not exist")
        return path
    if bool(getattr(args, "allow_no_hr", False)):
        return None
    city = _city_id_from_s2_dir(s2_dir)
    # Prefer new focus exports when available; fall back to older nib_resampled.
    focus = resolve_focus_hr_path(city)
    if focus is not None:
        return focus
    nib_dir = DEFAULT_NIB_ROOT / city
    tag = _format_gsd_m(hr_gsd_m)
    matches = sorted(nib_dir.glob(f"*_{tag}m.tif"))
    if not matches:
        matches = sorted(nib_dir.glob(f"*_{tag}m.tiff"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No NIB HR GeoTIFF at {hr_gsd_m:g} m in {nib_dir} "
        f"(expected *_{tag}m.tif) and no focus package for {city!r}. "
        f"Pass --hr-path, or --allow_no_hr for production SR without GT."
    )


PROCESSING_BASELINES = Path(__file__).resolve().parent / "data" / "s2_revisits" / "processing_baselines.json"
_BASELINE_CACHE: dict[str, str | None] | None = None


def boa_add_offset(stac_id: str) -> float:
    """Reflectance offset to subtract for one L2A product (0.1 for processing baseline >= 04.00).

    Baselines from 04.00 onward store DN = 10000 * reflectance + 1000 (BOA_ADD_OFFSET = -1000).
    """
    global _BASELINE_CACHE
    if _BASELINE_CACHE is None:
        if not PROCESSING_BASELINES.is_file():
            raise FileNotFoundError(f"{PROCESSING_BASELINES} missing; run scripts/cache_processing_baselines.py")
        _BASELINE_CACHE = json.loads(PROCESSING_BASELINES.read_text())
    if stac_id not in _BASELINE_CACHE or _BASELINE_CACHE[stac_id] is None:
        raise KeyError(f"no processing baseline cached for {stac_id}; run scripts/cache_processing_baselines.py")
    major, minor = (int(p) for p in str(_BASELINE_CACHE[stac_id]).split("."))
    return 0.1 if (major, minor) >= (4, 0) else 0.0


def _l2a_to_reflectance(stack: np.ndarray) -> np.ndarray:
    arr = stack.astype(np.float32)
    peak = float(np.nanmax(arr)) if arr.size else 0.0
    if peak > 1.5:
        arr = arr / 10000.0
    return np.clip(np.nan_to_num(arr, nan=0.0), 0.0, 1.5)


def _window_from_meta(aoi: dict):
    from rasterio.windows import Window

    return Window(
        int(aoi["col_off"]),
        int(aoi["row_off"]),
        int(aoi["width"]),
        int(aoi["height"]),
    )


def _largest_ones_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest-area axis-aligned rectangle of True cells: (r0, r1, c0, c1) exclusive."""
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("valid mask must be a 2D array")
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)
    for i in range(h):
        row = mask[i]
        for j in range(w):
            heights[j] = heights[j] + 1 if row[j] else 0
        stack: list[int] = []
        for j in range(w + 1):
            cur = int(heights[j]) if j < w else 0
            while stack and int(heights[stack[-1]]) > cur:
                height = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                area = height * (j - left)
                r1 = i + 1
                r0 = r1 - height
                if area > best_area:
                    best_area = area
                    best = (r0, r1, left, j)
            stack.append(j)
    if best_area <= 0:
        raise ValueError("NIB HR does not overlap the Sentinel-2 AOI")
    return best


def _center_crop_hw(h: int, w: int, size_h: int, size_w: int) -> tuple[int, int, int, int]:
    size_h = min(int(size_h), h)
    size_w = min(int(size_w), w)
    r0 = max(0, (h - size_h) // 2)
    c0 = max(0, (w - size_w) // 2)
    return r0, r0 + size_h, c0, c0 + size_w


def _is_nib_focus_location(meta: dict, s2_dir: Path) -> bool:
    """True for NIB focus exports (fixed crop + masked eval)."""
    if meta.get("focus_project_folder"):
        return True
    return _city_id_from_s2_dir(s2_dir) in FOCUS_PROJECT_BY_CITY


def _build_hr_eval_mask(
    hr_cover: np.ndarray,
    s2_valid_lr: np.ndarray,
    hr_rgb: np.ndarray,
    df: int,
    *,
    erode_hr: int = 1,
) -> np.ndarray:
    """Boolean HR mask: NIB coverage ∩ valid S2 ∩ positive HR pixels."""
    s2_valid_hr = np.repeat(np.repeat(s2_valid_lr, df, axis=0), df, axis=1)
    hr_has_data = np.all(hr_rgb > 0, axis=-1)
    mask = hr_cover & s2_valid_hr & hr_has_data
    if erode_hr > 0 and np.any(mask):
        from scipy.ndimage import binary_erosion

        structure = np.ones((erode_hr * 2 + 1, erode_hr * 2 + 1), dtype=bool)
        mask = binary_erosion(mask, structure=structure)
    return mask


def _shift_from_alignment_entry(
    entry: dict | None, *, current_df: int | None
) -> tuple[float, float] | None:
    """Extract a scaled ``(dy, dx)`` from one alignment JSON entry."""
    if not entry:
        return None
    shift = entry.get("hr_shift_hr_px") or {}
    if "dy" not in shift or "dx" not in shift:
        return None
    dy, dx = float(shift["dy"]), float(shift["dx"])
    recorded_df = entry.get("df")
    if current_df is not None and recorded_df not in (None, 0):
        scale = float(current_df) / float(recorded_df)
        dy, dx = dy * scale, dx * scale
    return dy, dx


def load_hr_eval_shift(
    city: str,
    *,
    path: Path | str | None = None,
    current_df: int | None = None,
    s2_dir: Path | str | None = None,
    parent_tile_id: str | None = None,
) -> tuple[float, float] | None:
    """Load ``(dy, dx)`` HR-pixel shift to apply to NIB so it matches the S2 grid.

    Frozen shifts are in pixels of the alignment package's HR grid (usually
    ``df=4``, 2.5 m). When the current run uses another ``df``, scale by
    ``current_df / recorded_df`` so the physical offset is unchanged.

    Lookup order prefers the exact revisit folder, then an explicitly recorded
    LR512 parent patch (for nested children), then ``{city}_lr512``. This makes
    one independently estimated correction shared by every method and size
    derived from the same LR512 geographic parent.
    """
    resolved = resolve_hr_eval_alignment(
        city,
        path=path,
        current_df=current_df,
        s2_dir=s2_dir,
        parent_tile_id=parent_tile_id,
    )
    return tuple(resolved["shift_yx"]) if resolved is not None else None


def resolve_hr_eval_alignment(
    city: str,
    *,
    path: Path | str | None = None,
    current_df: int | None = None,
    s2_dir: Path | str | None = None,
    parent_tile_id: str | None = None,
) -> dict | None:
    """Resolve the shift and the exact manifest entry used for provenance."""
    import hashlib

    path = Path(path) if path is not None else DEFAULT_SPATIAL_ALIGNMENT_PATH
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    tiles = payload.get("tiles") or {}
    cities = payload.get("cities") or {}
    candidates: list[tuple[str, str, dict | None]] = []
    if s2_dir is not None:
        dirname = Path(s2_dir).name
        candidates.extend(
            (("tiles", dirname, tiles.get(dirname)), ("cities", dirname, cities.get(dirname)))
        )
        if parent_tile_id and parent_tile_id != dirname:
            candidates.extend(
                (
                    ("tiles", parent_tile_id, tiles.get(parent_tile_id)),
                    ("cities", parent_tile_id, cities.get(parent_tile_id)),
                )
            )
        parent_city = _city_id_from_s2_dir(Path(s2_dir))
        parent_lr512 = f"{parent_city}_lr512"
        if dirname != parent_lr512:
            candidates.extend(
                (
                    ("tiles", parent_lr512, tiles.get(parent_lr512)),
                    ("cities", parent_lr512, cities.get(parent_lr512)),
                )
            )
    candidates.extend((("tiles", city, tiles.get(city)), ("cities", city, cities.get(city))))
    for namespace, key, entry in candidates:
        shift = _shift_from_alignment_entry(entry, current_df=current_df)
        if shift is not None:
            return {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "entry_namespace": namespace,
                "entry_key": key,
                "shift_yx": [float(shift[0]), float(shift[1])],
                "recorded_df": entry.get("df"),
                "current_df": current_df,
                "alignment_scope": entry.get("alignment_scope"),
            }
    return None


def _apply_hr_spatial_shift(
    hr_rgb: np.ndarray,
    mask: np.ndarray,
    dy: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate NIB HR (+ mask) onto the S2 grid for evaluation."""
    from scipy.ndimage import shift as ndi_shift

    hr_out = ndi_shift(
        hr_rgb,
        shift=(float(dy), float(dx), 0.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    mask_out = ndi_shift(
        mask.astype(np.float32),
        shift=(float(dy), float(dx)),
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ) > 0.5
    hr_out = np.clip(hr_out.astype(np.float32), 0.0, 1.0)
    # Drop pixels that landed on empty after the shift.
    mask_out &= np.all(hr_out > 0, axis=-1)
    return hr_out, mask_out


_FRAME_DATE_IN_PATH = re.compile(r"_(\d{8})\.")
_NIB_DATE_IN_NAME = re.compile(r"_(\d{4}-\d{2}-\d{2})_")


def _parse_iso_date(value: str | date | datetime) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    if "T" in text:
        text = text.split("T", 1)[0]
    return date.fromisoformat(text[:10])


def _frame_acquisition_date(frame: dict) -> date:
    dt = frame.get("datetime")
    if dt:
        return _parse_iso_date(str(dt))
    match = _FRAME_DATE_IN_PATH.search(str(frame.get("path", "")))
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    raise ValueError(f"frame has no parseable datetime: {frame!r}")


def _nib_acquisition_date(hr_path: Path | str) -> date:
    path = Path(str(hr_path))
    match = _NIB_DATE_IN_NAME.search(path.name)
    if match:
        return _parse_iso_date(match.group(1))
    try:
        import rasterio

        with rasterio.open(hr_path) as src:
            for key in ("ACQUISITION_DATE", "acquisition_date"):
                value = (src.tags() or {}).get(key)
                if value:
                    return _parse_iso_date(str(value))
    except Exception:  # noqa: BLE001
        pass
    raise ValueError(f"could not parse NIB acquisition date from {hr_path}")


def _base_frame_index_for_nib(
    frames: list[dict],
    nib_date: date,
    *,
    cloud_fracs: list[float] | None = None,
    max_cloud_frac: float = DEFAULT_MAX_BASE_CLOUD_FRAC,
) -> int:
    """Frozen identity frame: closest to ``nib_date`` among nearly cloud-free revisits.

    Affine / radiometry freeze ``index 0``, so the caller should move this
    frame to the front of the stack. Without ``cloud_fracs``, this is
    date-only (legacy). If no frame is at or under ``max_cloud_frac``, pick
    the clearest, then closest date.
    """
    if not frames:
        raise ValueError("no frames")
    if cloud_fracs is not None and len(cloud_fracs) != len(frames):
        raise ValueError(
            f"cloud_fracs length {len(cloud_fracs)} != frames {len(frames)}"
        )

    def _key_date(idx: int) -> tuple:
        frame_date = _frame_acquisition_date(frames[idx])
        return (abs((frame_date - nib_date).days), frame_date.isoformat())

    if cloud_fracs is None:
        return min(range(len(frames)), key=_key_date)

    thr = float(max_cloud_frac)
    eligible = [
        i
        for i, cloud in enumerate(cloud_fracs)
        if np.isfinite(cloud) and float(cloud) <= thr
    ]
    if eligible:
        return min(eligible, key=_key_date)

    def _key_clearest(idx: int) -> tuple:
        cloud = float(cloud_fracs[idx])
        if not np.isfinite(cloud):
            cloud = 1.0
        return (cloud, *_key_date(idx))

    return min(range(len(frames)), key=_key_clearest)


def _index_to_front(index: int, n: int) -> list[int]:
    """Permutation that moves ``index`` to 0 and keeps other order."""
    idx = int(index)
    if idx < 0 or idx >= n:
        raise IndexError(f"index {idx} out of range for n={n}")
    return [idx] + [i for i in range(n) if i != idx]


def _harmonize_hr_histogram_match(hr_np: np.ndarray, base_lr: np.ndarray) -> np.ndarray:
    """SEN2NAIP cross-sensor harmonization: per-band histogram matching to S2.

    See Aybar et al., Scientific Data 2024 — match each NAIP/HR band to the
    reference S2 LR frame (closest in date to the NIB acquisition) before eval.
    """
    from skimage.exposure import match_histograms

    hr_np = np.asarray(hr_np, dtype=np.float32)
    base_lr = np.asarray(base_lr, dtype=np.float32)
    if hr_np.ndim != 3 or base_lr.ndim != 3 or hr_np.shape[-1] != base_lr.shape[-1]:
        raise ValueError(f"HR/LR shape mismatch: {hr_np.shape} vs {base_lr.shape}")

    out = np.zeros_like(hr_np)
    for c in range(int(hr_np.shape[-1])):
        src = hr_np[..., c]
        ref = base_lr[..., c]
        valid = src > 0
        if not np.any(valid) or ref.size < 2:
            out[..., c] = src
            continue
        matched = match_histograms(src, ref, channel_axis=None)
        out[..., c] = np.where(valid, np.clip(matched, 0.0, 1.0), 0.0)
    return out


def _fit_affine_1d(x: np.ndarray, y: np.ndarray, *, trim_percentiles: tuple[float, float] = (1.0, 99.0)) -> tuple[float, float]:
    """Fit y ~= a*x + b with simple trimmed least-squares.

    This is intentionally lightweight (no scikit-learn dependency): it robustly
    trims extreme values and then runs a 2-parameter linear least squares fit.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return 1.0, 0.0

    # If NIB (x) includes exact zeros as nodata, don't let them dominate.
    nonzero = x != 0
    if np.any(nonzero):
        x = x[nonzero]
        y = y[nonzero]
    if x.size < 2:
        return 1.0, 0.0

    lo_p, hi_p = trim_percentiles
    lo_x, hi_x = np.percentile(x, lo_p), np.percentile(x, hi_p)
    lo_y, hi_y = np.percentile(y, lo_p), np.percentile(y, hi_p)
    if lo_x == hi_x or lo_y == hi_y:
        # Degenerate case: fall back to mean matching.
        a = 1.0
        b = float(np.mean(y) - np.mean(x))
        return a, b

    m2 = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    x1 = x[m2]
    y1 = y[m2]
    if x1.size < 2:
        a = 1.0
        b = float(np.mean(y) - np.mean(x))
        return a, b

    a_b = np.linalg.lstsq(
        np.stack([x1, np.ones_like(x1)], axis=1),
        y1,
        rcond=None,
    )[0]
    a = float(a_b[0])
    b = float(a_b[1])
    return a, b


def _warp_rgb_to_grid(hr_path: Path | str, dst_crs, dst_transform, dst_h: int, dst_w: int):
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(hr_path) as src:
        count = min(3, int(src.count))
        rgb = np.zeros((count, dst_h, dst_w), dtype=np.float32)
        cover = np.zeros((dst_h, dst_w), dtype=np.uint8)
        source = src.read(out_dtype="float32")[:count]
        reproject(
            source=source,
            destination=rgb,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=0.0,
        )
        ones = np.ones((src.height, src.width), dtype=np.uint8)
        reproject(
            source=ones,
            destination=cover,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        src_crs = src.crs
        src_bounds = src.bounds
    peak = float(rgb.max()) if rgb.size else 0.0
    if peak > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0)), cover > 0, src_crs, src_bounds


def _window_fits(crop_win, width: int, height: int) -> bool:
    if crop_win is None:
        return False
    col = float(crop_win.col_off)
    row = float(crop_win.row_off)
    win_w = float(crop_win.width)
    win_h = float(crop_win.height)
    return (
        col >= 0.0
        and row >= 0.0
        and col + win_w <= float(width) + 1e-6
        and row + win_h <= float(height) + 1e-6
    )


def _read_frame_clear(
    s2_dir: Path,
    frame: dict,
    crop_win,
    *,
    height: int,
    width: int,
    dst_transform=None,
    dst_crs=None,
) -> np.ndarray:
    """Load a per-frame clear map (True = use in the loss). Missing mask → all clear.

    City OmniCloudMask files are already AOI-cropped and often a different
    size than the LR512 window into the full granule. Reproject those onto
    the LR grid. Full-granule SCL masks still use ``crop_win``.
    """
    from eval.s2_cloud_mask import clear_from_cloud_mask

    name = frame.get("cloud_mask")
    if not name:
        return np.ones((height, width), dtype=bool)
    path = s2_dir / str(name)
    if not path.is_file():
        return np.ones((height, width), dtype=bool)
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(path) as src:
        if int(src.width) == int(width) and int(src.height) == int(height):
            cloudy = src.read(1)
        elif _window_fits(crop_win, src.width, src.height):
            cloudy = src.read(1, window=crop_win)
        elif dst_transform is not None and src.crs is not None:
            cloudy = np.zeros((height, width), dtype=np.uint8)
            reproject(
                source=src.read(1),
                destination=cloudy,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs or src.crs,
                resampling=Resampling.nearest,
            )
        else:
            raise ValueError(
                f"{path}: cloud mask {src.width}×{src.height} cannot map onto "
                f"LR {(height, width)} (need a fitting crop window or dest transform)"
            )
    clear = clear_from_cloud_mask(cloudy)
    if clear.shape != (height, width):
        raise ValueError(
            f"{path}: cloud mask crop {clear.shape} != LR {(height, width)}"
        )
    return np.ascontiguousarray(clear, dtype=bool)


MIN_STATS_PIXELS = 256


def standardize_rgb_masked(
    rgb: torch.Tensor, stats_mask: np.ndarray | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Standardize ``[H,W,3]`` RGB with per-channel stats from ``stats_mask`` pixels.

    Clouds and nodata would otherwise skew the mean/std that set each frame's
    loss scale. Falls back to all pixels when too few are selected. Returns
    ``(standardized, mean[1,1,3], std[1,1,3], used_mask)``.
    """
    if stats_mask is None or int(np.count_nonzero(stats_mask)) < MIN_STATS_PIXELS:
        standardized, mean, std = get_and_standardize_image(rgb)
        return standardized, mean, std, False
    mask = torch.as_tensor(stats_mask, dtype=torch.bool, device=rgb.device)
    pixels = rgb[mask]
    mean = pixels.mean(dim=0).view(1, 1, -1)
    std = torch.clamp(pixels.std(dim=0), min=1e-8).view(1, 1, -1)
    return (rgb - mean) / std, mean, std, True


class S2NIBRevisitDataset(Dataset):
    """Windowed S2 RGB revisits with NIB HR warped onto the S2 AOI grid."""

    def __init__(self, args, name: str = "s2", training_device=None):
        import rasterio
        from rasterio.windows import transform as window_transform
        from rasterio.transform import Affine

        self.device = resolve_dataset_device(args, training_device=training_device)
        self.s2_dir = resolve_s2_dir(args, name)
        meta_path = self.s2_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text())
        frames = list(self.meta.get("frames") or [])
        if not frames:
            raise ValueError(f"{meta_path} has no frames")

        requested = int(getattr(args, "num_samples", 0) or 0)
        if requested > 0:
            frames = frames[:requested]
        if hasattr(args, "num_samples"):
            args.num_samples = len(frames)
        self.num_samples = len(frames)
        self.frames = frames

        aoi = dict(self.meta.get("aoi_window") or {})
        if not aoi:
            raise ValueError(f"{meta_path} missing aoi_window")
        west = int(getattr(args, "halo_lr_px_west", 0) or 0)
        north = int(getattr(args, "halo_lr_px_north", 0) or 0)
        if west or north:
            from eval.halo_consistency import expand_aoi_for_halo

            aoi = expand_aoi_for_halo(aoi, west_px=west, north_px=north)
            print(
                f"halo AOI expand west={west} north={north} → "
                f"col_off={aoi['col_off']} row_off={aoi['row_off']} "
                f"{aoi['width']}×{aoi['height']}",
                flush=True,
            )
        aoi_win = _window_from_meta(aoi)
        df = max(1, int(getattr(args, "df", 4) or 4))
        native_gsd = float(getattr(args, "s2_native_gsd_m", self.meta.get("resolution_m") or 10.0))
        hr_gsd = float(getattr(args, "hr_gsd_m", 0.0) or 0.0)
        if hr_gsd <= 0:
            hr_gsd = native_gsd / float(df)
        expected_df = int(round(native_gsd / hr_gsd))
        if expected_df != df:
            print(
                f"Warning: --df {df} does not match native {native_gsd:g} m / HR {hr_gsd:g} m "
                f"(={expected_df}×). Using --df for grid size."
            )
        self.df = df
        self.hr_gsd_m = hr_gsd
        self.native_gsd_m = native_gsd
        self.hr_path = resolve_hr_path(args, self.s2_dir, hr_gsd)
        self.has_hr_gt = self.hr_path is not None

        first_path = self.s2_dir / frames[0]["path"]
        with rasterio.open(first_path) as src:
            s2_crs = src.crs
            tile_transform = src.transform
            aoi_transform = window_transform(aoi_win, tile_transform)
            lr_h = int(aoi_win.height)
            lr_w = int(aoi_win.width)
            stack0 = src.read(window=aoi_win)

        hr_h = lr_h * df
        hr_w = lr_w * df
        hr_transform = Affine(hr_gsd, 0.0, aoi_transform.c, 0.0, -hr_gsd, aoi_transform.f)
        s2_valid = np.any(stack0 > 0, axis=0)
        nib_crs = None
        nib_bounds = None
        if self.has_hr_gt:
            hr_rgb, hr_cover, nib_crs, nib_bounds = _warp_rgb_to_grid(
                self.hr_path, s2_crs, hr_transform, hr_h, hr_w
            )
            cover_lr = hr_cover.reshape(lr_h, df, lr_w, df).all(axis=(1, 3))
            valid_lr = cover_lr & s2_valid
            self.use_fixed_nib_crop = _is_nib_focus_location(self.meta, self.s2_dir)
            if self.use_fixed_nib_crop:
                # Keep the full AOI rectangle; evaluate only where HR GT exists.
                r0, r1, c0, c1 = 0, lr_h, 0, lr_w
            else:
                r0, r1, c0, c1 = _largest_ones_rectangle(valid_lr)
                lr_size = int(getattr(args, "lr_size", 0) or 0)
                crop_h, crop_w = r1 - r0, c1 - c0
                if lr_size > 0:
                    side = min(lr_size, crop_h, crop_w)
                    cr0, cr1, cc0, cc1 = _center_crop_hw(crop_h, crop_w, side, side)
                    r0, r1, c0, c1 = r0 + cr0, r0 + cr1, c0 + cc0, c0 + cc1
        else:
            # Production: keep the full requested AOI; no NIB crop.
            hr_rgb = np.zeros((hr_h, hr_w, 3), dtype=np.float32)
            hr_cover = np.zeros((hr_h, hr_w), dtype=bool)
            self.use_fixed_nib_crop = False
            r0, r1, c0, c1 = 0, lr_h, 0, lr_w

        self.lr_row0 = int(aoi["row_off"]) + r0
        self.lr_col0 = int(aoi["col_off"]) + c0
        self.lr_height = r1 - r0
        self.lr_width = c1 - c0
        from rasterio.windows import Window

        crop_win = Window(self.lr_col0, self.lr_row0, self.lr_width, self.lr_height)
        with rasterio.open(first_path) as src:
            self.lr_transform = window_transform(crop_win, src.transform)
            self.crs = str(src.crs)
        # Same origin as the LR crop; pixel size is HR GSD.
        self.hr_transform = Affine(
            float(self.lr_transform.a) / float(df),
            0.0,
            float(self.lr_transform.c),
            0.0,
            float(self.lr_transform.e) / float(df),
            float(self.lr_transform.f),
        )

        hr_rgb = hr_rgb[r0 * df : r1 * df, c0 * df : c1 * df]
        if self.has_hr_gt:
            hr_cover_crop = hr_cover[r0 * df : r1 * df, c0 * df : c1 * df]
            s2_valid_crop = s2_valid[r0:r1, c0:c1]
            eval_mask = _build_hr_eval_mask(hr_cover_crop, s2_valid_crop, hr_rgb, df)
        else:
            eval_mask = np.zeros(hr_rgb.shape[:2], dtype=bool)
        self.hr_eval_mask = torch.as_tensor(eval_mask, dtype=torch.bool, device=self.device)
        self.hr_valid_fraction = float(eval_mask.mean()) if eval_mask.size else 0.0
        self.use_masked_eval = bool(
            self.has_hr_gt and self.use_fixed_nib_crop and self.hr_valid_fraction < 0.999
        )
        self.original_hr = torch.as_tensor(np.ascontiguousarray(hr_rgb), dtype=torch.float32, device=self.device)

        rgb_frames = []
        nir_frames = []
        means = []
        stds = []
        standardized = []
        clear_frames = []
        stats_mode = str(getattr(args, "lr_stats_pixels", "clear") or "clear").lower()
        if stats_mode not in {"clear", "all"}:
            raise ValueError(f"--lr_stats_pixels must be clear or all, got {stats_mode!r}")
        self.lr_stats_pixels = stats_mode
        self.lr_stats_masked = []
        boa_mode = str(getattr(args, "s2_boa_offset", "keep") or "keep")
        if boa_mode not in {"keep", "remove"}:
            raise ValueError(f"--s2_boa_offset must be keep or remove, got {boa_mode!r}")
        self.s2_boa_offset_mode = boa_mode
        self.s2_boa_offsets = []
        for frame in frames:
            path = self.s2_dir / frame["path"]
            with rasterio.open(path) as src:
                stack = src.read(window=crop_win)
            self.s2_boa_offsets.append(boa_add_offset(frame["stac_id"]) if boa_mode == "remove" else 0.0)
            refl = _l2a_to_reflectance(stack)
            rgb = np.transpose(refl[0:3], (1, 2, 0))
            nir = refl[3] if refl.shape[0] >= 4 else None
            clear = _read_frame_clear(
                self.s2_dir,
                frame,
                crop_win,
                height=self.lr_height,
                width=self.lr_width,
                dst_transform=self.lr_transform,
                dst_crs=self.crs,
            )
            rgb_t = torch.as_tensor(np.ascontiguousarray(rgb), dtype=torch.float32, device=self.device)
            stats_mask = None
            if stats_mode == "clear":
                stats_mask = clear & np.any(rgb > 0, axis=-1)
            rgb_std, mean, std, used_mask = standardize_rgb_masked(rgb_t, stats_mask)
            self.lr_stats_masked.append(bool(used_mask))
            rgb_frames.append(rgb_t)
            standardized.append(rgb_std)
            means.append(mean)
            stds.append(std)
            if nir is not None:
                nir_frames.append(torch.as_tensor(np.ascontiguousarray(nir), dtype=torch.float32, device=self.device))
            clear_frames.append(clear)

        self.lr_rgb = torch.stack(rgb_frames, dim=0)
        self.lr_standardized = torch.stack(standardized, dim=0)
        self.lr_mean = torch.stack(means, dim=0)
        self.lr_std = torch.stack(stds, dim=0)
        self.lr_nir = torch.stack(nir_frames, dim=0) if nir_frames else None
        self.lr_clear = torch.as_tensor(
            np.stack(clear_frames, axis=0), dtype=torch.bool, device=self.device
        )
        self.lr_clear_fractions = [float(c.mean()) for c in clear_frames]
        self.has_lr_cloud_mask = any(bool(fr.get("cloud_mask")) for fr in frames)

        center = self.meta.get("nib_acquisition_date") or self.meta.get("center_date")
        if self.has_hr_gt:
            try:
                nib_date = _nib_acquisition_date(self.hr_path)
            except ValueError:
                # Focus exports often lack acquisition date in the filename; use meta.
                if not center:
                    raise
                nib_date = _parse_iso_date(str(center))
        else:
            if center:
                nib_date = _parse_iso_date(str(center))
            else:
                nib_date = _frame_acquisition_date(frames[0])
        self.nib_acquisition_date = nib_date.isoformat()
        cloud_fracs = [1.0 - frac for frac in self.lr_clear_fractions]
        max_base_cloud = float(
            getattr(args, "max_base_cloud_frac", DEFAULT_MAX_BASE_CLOUD_FRAC)
        )
        force_raw = getattr(args, "force_base_date", None)
        self.force_base_date = None
        if force_raw:
            target = _parse_iso_date(str(force_raw))
            hits = [
                i
                for i in range(len(frames))
                if _frame_acquisition_date(frames[i]) == target
            ]
            if hits:
                def _force_key(idx: int) -> tuple:
                    cloud = float(cloud_fracs[idx])
                    if not np.isfinite(cloud):
                        cloud = 1.0
                    return (cloud, idx)

                base_src = min(hits, key=_force_key)
                self.force_base_date = target.isoformat()
            else:
                print(
                    f"WARN: --force_base_date {target.isoformat()} not in stack; "
                    "using the default identity rule"
                )
                base_src = _base_frame_index_for_nib(
                    frames,
                    nib_date,
                    cloud_fracs=cloud_fracs,
                    max_cloud_frac=max_base_cloud,
                )
        else:
            base_src = _base_frame_index_for_nib(
                frames,
                nib_date,
                cloud_fracs=cloud_fracs,
                max_cloud_frac=max_base_cloud,
            )
        if base_src != 0:
            order = _index_to_front(base_src, len(frames))
            frames = [frames[i] for i in order]
            self.frames = frames
            self.lr_rgb = self.lr_rgb[order]
            self.lr_standardized = self.lr_standardized[order]
            self.lr_mean = self.lr_mean[order]
            self.lr_std = self.lr_std[order]
            self.lr_clear = self.lr_clear[order]
            self.lr_clear_fractions = [self.lr_clear_fractions[i] for i in order]
            self.lr_stats_masked = [self.lr_stats_masked[i] for i in order]
            if self.lr_nir is not None:
                self.lr_nir = self.lr_nir[order]
            cloud_fracs = [cloud_fracs[i] for i in order]
            self.s2_boa_offsets = [self.s2_boa_offsets[i] for i in order]
        # Fitting sees DN/10000: per-frame standardization cancels a constant offset, and a
        # post-hoc subtraction keeps masks and dark (DN < 1000) pixels distinct from no-data zeros.
        self.eval_reflectance_offset = float(self.s2_boa_offsets[0]) if self.s2_boa_offsets else 0.0
        self.base_frame_index = 0
        self.base_frame_cloud_frac = float(cloud_fracs[0])
        self.max_base_cloud_frac = max_base_cloud
        base_frame = frames[0]
        base_frame_date = _frame_acquisition_date(base_frame).isoformat()
        self.base_frame_date = base_frame_date

        # Radiometric harmonization (NIB -> S2) for evaluation GT.
        #
        # SEN2NAIP cross-sensor protocol: per-band histogram matching of HR NIB to
        # the S2 revisit closest in date to the NIB acquisition. Training uses all
        # LR revisits; eval compares denormalized model output against harmonized HR.
        self.hr_harmonize_method = None
        if self.has_hr_gt and not bool(getattr(args, "no_hr_harmonize", False)):
            try:
                base_lr = self.lr_rgb[self.base_frame_index].detach().cpu().numpy()
                hr_np = self.original_hr.detach().cpu().numpy()  # [hr_h, hr_w, 3]
                hr_np = _harmonize_hr_histogram_match(hr_np, base_lr)
                self.original_hr = torch.as_tensor(
                    np.ascontiguousarray(hr_np),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.hr_harmonize_method = "histogram"
            except Exception as e:  # noqa: BLE001
                print(f"WARN: HR harmonization failed; using raw NIB HR. Error: {e}")

        # Spatial alignment (NIB -> S2 grid) for evaluation GT only.
        # Estimated offline via S2-bilinear vs NIB (eval/spatial_alignment.json).
        self.hr_spatial_shift = None
        self.hr_spatial_alignment = None
        if self.has_hr_gt and not bool(getattr(args, "no_hr_spatial_align", False)):
            city = _city_id_from_s2_dir(self.s2_dir)
            align_path = getattr(args, "spatial_alignment_path", None)
            patch_grid = self.meta.get("patch_grid") or {}
            alignment = resolve_hr_eval_alignment(
                city,
                path=align_path,
                current_df=int(df),
                s2_dir=self.s2_dir,
                parent_tile_id=patch_grid.get("parent_tile_id"),
            )
            shift = tuple(alignment["shift_yx"]) if alignment is not None else None
            if shift is not None:
                dy, dx = shift
                hr_np = self.original_hr.detach().cpu().numpy()
                mask_np = self.hr_eval_mask.detach().cpu().numpy()
                hr_np, mask_np = _apply_hr_spatial_shift(hr_np, mask_np, dy, dx)
                self.original_hr = torch.as_tensor(
                    np.ascontiguousarray(hr_np),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.hr_eval_mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                self.hr_valid_fraction = float(mask_np.mean())
                self.use_masked_eval = bool(
                    self.use_fixed_nib_crop and self.hr_valid_fraction < 0.999
                )
                self.hr_spatial_shift = {"dy": dy, "dx": dx}
                self.hr_spatial_alignment = alignment

        scale_factor = float(getattr(args, "scale_factor", df) or df)
        self.scale_factor = scale_factor
        self.input_coords = _build_input_coord_cache(
            self.lr_height,
            [scale_factor],
            device=self.device,
            lr_height=self.lr_height,
            lr_width=self.lr_width,
        )
        self.hr_coordinates = _make_coord_grid(
            self.original_hr.shape[0],
            self.original_hr.shape[1],
            device=self.device,
        )

        if self.has_hr_gt:
            print(
                f"S2/NIB dataset {self.s2_dir}: {self.num_samples} frames, "
                f"LR {self.lr_height}×{self.lr_width} @ {native_gsd:g} m, "
                f"HR {tuple(self.original_hr.shape[:2])} @ {hr_gsd:g} m "
                f"(df={df}), S2 {self.crs} ← NIB {nib_crs}, "
                f"HR file {Path(str(self.hr_path)).name}, "
                f"harmonize base frame {self.base_frame_index} "
                f"({base_frame['path']}, {base_frame_date}, "
                f"Δ{abs((_parse_iso_date(base_frame_date) - nib_date).days)}d from NIB {nib_date.isoformat()}), "
                f"HR eval mask {self.hr_valid_fraction * 100:.1f}% valid"
                + (" (masked eval)" if self.use_masked_eval else "")
                + (
                    f", HR spatial shift dy={self.hr_spatial_shift['dy']:+.2f} "
                    f"dx={self.hr_spatial_shift['dx']:+.2f} px"
                    if self.hr_spatial_shift
                    else ""
                )
            )
            if nib_crs is not None and str(s2_crs) != str(nib_crs):
                print(
                    f"  CRS caveat: NIB {nib_crs} warped onto S2 {s2_crs}; "
                    f"NIB bounds {tuple(nib_bounds)} vs S2 AOI window "
                    f"row/col {self.lr_row0},{self.lr_col0} size {self.lr_height}×{self.lr_width}."
                )
        else:
            print(
                f"S2 production dataset {self.s2_dir}: {self.num_samples} frames, "
                f"LR {self.lr_height}×{self.lr_width} @ {native_gsd:g} m, "
                f"SR grid {tuple(self.original_hr.shape[:2])} @ {hr_gsd:g} m "
                f"(df={df}), CRS {self.crs}, no HR GT (--allow_no_hr), "
                f"base frame 0 ({base_frame['path']}, {base_frame_date}, "
                f"cloud {self.base_frame_cloud_frac * 100:.1f}%, "
                f"Δ{abs((_parse_iso_date(base_frame_date) - nib_date).days)}d "
                f"from {nib_date.isoformat()})"
                + (
                    f", LR cloud mask mean clear "
                    f"{float(np.mean(self.lr_clear_fractions)) * 100:.1f}%"
                    if self.has_lr_cloud_mask
                    else ""
                )
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        sf = float(self.scale_factor)
        return {
            "input": self.input_coords[sf],
            "lr_target": self.lr_standardized[idx],
            "sample_id": torch.tensor(int(idx), dtype=torch.long, device=self.device),
        }

    def get_original_hr(self) -> torch.Tensor:
        return self.original_hr

    def get_hr_eval_mask(self) -> torch.Tensor:
        """HR-resolution boolean mask where harmonized GT is valid."""
        return self.hr_eval_mask

    def get_lr_clear(self) -> torch.Tensor:
        """Per-frame LR clear map ``[N,H,W]`` (True = not cloudy)."""
        return self.lr_clear

    def get_geo_meta(self) -> dict:
        """CRS + affine transforms for QGIS / GeoTIFF export."""
        return {
            "crs": getattr(self, "crs", None),
            "lr_transform": getattr(self, "lr_transform", None),
            "hr_transform": getattr(self, "hr_transform", None),
            "hr_gsd_m": float(getattr(self, "hr_gsd_m", 0.0) or 0.0),
            "native_gsd_m": float(getattr(self, "native_gsd_m", 0.0) or 0.0),
            "df": int(getattr(self, "df", 1) or 1),
        }

    def get_hr_coordinates(self) -> torch.Tensor:
        return self.hr_coordinates

    def get_lr_sample(self, idx: int) -> torch.Tensor:
        """Unstandardized RGB in CHW, reflectance ~[0, 1]."""
        return self.lr_rgb[idx].permute(2, 0, 1).contiguous()

    def get_lr_sample_hwc(self, idx: int) -> torch.Tensor:
        """Standardized RGB HWC."""
        return self.lr_standardized[idx]

    def get_lr_mean(self, idx: int) -> torch.Tensor:
        return self.lr_mean[idx]

    def get_lr_std(self, idx: int) -> torch.Tensor:
        return self.lr_std[idx]

    def get_lr_nir(self, idx: int) -> torch.Tensor | None:
        if self.lr_nir is None:
            return None
        return self.lr_nir[idx]
