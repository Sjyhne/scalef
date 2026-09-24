#!/usr/bin/env python3
"""Measure HR (NIB) ↔ S2 spatial offset via S2-bilinear vs harmonized HR.

SEN2NAIP harmonization is radiometric only. This script estimates the residual
sub-pixel shift between the harmonized NIB HR ground truth and the closest-date
S2 frame upsampled with bilinear interpolation:

  1. Phase correlation at HR (S2-bilinear vs NIB)
  2. Local exhaustive MSE refine at HR around that estimate

This matches the supervision baseline: exhaustive search of S2-bilinear vs
reference, without a full multi-LR-pixel HR grid search.

Example
-------
python scripts/audit_hr_lr_spatial_alignment.py --city asker --lr-size 512
python scripts/audit_hr_lr_spatial_alignment.py --all-focus --lr-size 512 --out-dir single_samples/spatial_audit
python scripts/audit_hr_lr_spatial_alignment.py --all-new --lr-size 0 \\
  --out-dir single_samples/spatial_audit_new \\
  --merge-into eval/spatial_alignment.json
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from torchmetrics.functional.image import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataset

FOCUS_CITIES = [
    "asker",
    "vennesla",
    "trondheim",
    "bergen",
    "rana",
    "tromso",
    "amli",
    "stavanger",
]

# New NIB exports in aois.json. B17 confirmatory tiles are {city}_lr512.
NEW_NIB_CITIES = [
    "algard",
    "naerbo",
    "flekkefjord",
    "rafsbotn",
    "nittedal",
    "melhus",
    "alta",
    "karasjok",
    "kautokeino",
]

B17_CITIES = tuple(FOCUS_CITIES + NEW_NIB_CITIES)


def _to_alignment_entry(rec: dict) -> dict:
    """Convert an audit record into eval/spatial_alignment.json city entry shape."""
    s2_dir = Path(rec["s2_dir"])
    entry = {
        "df": int(rec["df"]),
        "s2_dir_name": s2_dir.name,
        "lr_size": list(rec["lr_size"]),
        "hr_size": list(rec["hr_size"]),
        "base_frame_index": int(rec["base_frame_index"]),
        "base_frame_date": rec.get("base_frame_date"),
        "nib_acquisition_date": rec.get("nib_acquisition_date"),
        "valid_fraction": float(rec["valid_fraction"]),
        "hr_shift_hr_px": dict(rec["shift_apply_to_hr_for_eval"]),
        "s2_bilinear_shift_hr_px": dict(rec["shift_apply_to_s2_bilinear"]),
        "phase_init_hr_px": dict(rec["phase_init_hr_px"]),
        "before_align_s2_to_hr": dict(rec["before_align_s2_to_hr"]),
        "after_align_s2_to_hr": dict(rec["after_align_s2_to_hr"]),
        "delta": dict(rec["delta"]),
    }
    for key in (
        "tile_id",
        "parent_city",
        "alignment_scope",
        "status",
        "input_manifest_sha256",
        "s2_meta_sha256",
    ):
        if rec.get(key) is not None:
            entry[key] = rec[key]
    return entry


def merge_into_spatial_alignment(
    results: list[dict],
    *,
    alignment_path: Path,
    refine_radius_hr: float,
    refine_step_hr: float,
) -> None:
    """Upsert audit results into ``eval/spatial_alignment.json``."""
    from datetime import datetime, timezone

    if alignment_path.is_file():
        payload = json.loads(alignment_path.read_text())
    else:
        payload = {
            "version": 1,
            "method": "hr_phase_corr + local_hr_mse_refine",
            "notes": "",
            "cities": {},
        }
    cities = dict(payload.get("cities") or {})
    tiles = dict(payload.get("tiles") or {})
    for rec in results:
        entry = _to_alignment_entry(rec)
        tile_name = Path(rec["s2_dir"]).name
        old = cities.get(rec["city"])
        if old:
            old_name = old.get("s2_dir_name")
            if old_name and old_name != tile_name:
                tiles.setdefault(str(old_name), old)
        # Patch-grid audits must not replace the canonical city/LR512 fallback.
        if rec.get("alignment_scope") != "lr512_patch_exact":
            cities[rec["city"]] = entry
        tiles[tile_name] = entry
    payload["cities"] = dict(sorted(cities.items()))
    payload["tiles"] = dict(sorted(tiles.items()))
    payload["method"] = "hr_phase_corr + local_hr_mse_refine"
    payload["refine_radius_hr_px"] = float(refine_radius_hr)
    payload["refine_step_hr_px"] = float(refine_step_hr)
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    # Keep created_utc if present; refresh notes lightly.
    n_focus = sum(1 for c in FOCUS_CITIES if c in cities)
    n_new = sum(1 for c in NEW_NIB_CITIES if c in cities)
    payload["notes"] = (
        "Per-tile residual translation between S2-bilinear (base frame) and "
        "harmonized NIB HR. tiles[s2_dir.name] is the primary lookup; city keys "
        "remain as fallback. Apply hr_shift_hr_px to NIB HR for evaluation only; "
        "do not shift S2 LR revisits used for training. "
        f"Focus cities: {n_focus}; new NIB AOIs: {n_new}; tile entries: {len(tiles)}."
    )
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Merged {len(results)} cities → {alignment_path} (total {len(cities)})", flush=True)


def _dataset_args(s2_dir: Path) -> Namespace:
    return Namespace(
        dataset="s2",
        s2_dir=str(s2_dir),
        df=4,
        scale_factor=4,
        hr_gsd_m=0.0,
        s2_native_gsd_m=10.0,
        num_samples=0,
        lr_size=0,
        dataset_device="cpu",
        no_hr_harmonize=False,
        # Measure residual on raw (harmonized) NIB — do not apply a prior shift.
        no_hr_spatial_align=True,
    )


def _resolve_s2_dir(city: str, lr_size: int) -> Path:
    if lr_size > 0:
        variant = ROOT / "data" / "s2_revisits" / f"{city}_lr{lr_size}"
        if (variant / "meta.json").is_file():
            return variant
    base = ROOT / "data" / "s2_revisits" / city
    if not (base / "meta.json").is_file():
        raise FileNotFoundError(f"No S2 data for {city!r} under data/s2_revisits/")
    return base


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def _valid_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[:2], dtype=bool)
    for arr in arrays:
        mask &= np.all(arr > 0, axis=-1)
    return mask


def _masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(np.float64)
    n = max(1.0, m.sum())
    diff = (a - b)[..., :3] if a.ndim == 3 else (a - b)
    if diff.ndim == 3:
        err = (diff ** 2).mean(axis=-1)
    else:
        err = diff ** 2
    return float((err * m).sum() / n)


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask
    ga = _rgb_to_gray(a)[m]
    gb = _rgb_to_gray(b)[m]
    if ga.size < 16:
        return float("nan")
    ga = ga - ga.mean()
    gb = gb - gb.mean()
    denom = float(np.sqrt((ga ** 2).sum() * (gb ** 2).sum()))
    if denom <= 0:
        return float("nan")
    return float((ga * gb).sum() / denom)


def _shift_hwc(img: np.ndarray, dy: float, dx: float, order: int = 1) -> np.ndarray:
    out = ndi_shift(img, shift=(dy, dx, 0), order=order, mode="constant", cval=0.0, prefilter=False)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def _upsample_lr_bilinear(lr_hwc: np.ndarray, hr_h: int, hr_w: int) -> np.ndarray:
    return cv2.resize(lr_hwc, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)


def _metrics(ref: np.ndarray, mov: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compare ``mov`` to ``ref`` on ``mask`` (HR resolution, HWC float01)."""
    m = mask
    if not np.any(m):
        return {"psnr": float("nan"), "ssim": float("nan"), "ncc": float("nan"), "mse": float("nan")}

    ref_m = ref.copy()
    mov_m = mov.copy()
    ref_m[~m] = 0.0
    mov_m[~m] = 0.0

    ref_t = torch.from_numpy(ref_m.transpose(2, 0, 1)[None].astype(np.float32))
    mov_t = torch.from_numpy(mov_m.transpose(2, 0, 1)[None].astype(np.float32))
    return {
        "psnr": float(psnr_fn(mov_t, ref_t, data_range=1.0).item()),
        "ssim": float(ssim_fn(mov_t, ref_t, data_range=1.0).item()),
        "ncc": _masked_ncc(ref, mov, m),
        "mse": _masked_mse(ref, mov, m),
    }


def _phase_shift(ref_gray: np.ndarray, mov_gray: np.ndarray, mask: np.ndarray, upsample: int) -> tuple[float, float]:
    """Return (dy, dx) in HR pixels to apply to ``mov`` to align with ``ref``."""
    ref = ref_gray.copy()
    mov = mov_gray.copy()
    ref[~mask] = 0.0
    mov[~mask] = 0.0
    sh, _, _ = phase_cross_correlation(ref, mov, upsample_factor=int(upsample))
    # skimage returns shift to apply to mov: (row, col).
    return float(sh[0]), float(sh[1])


def _exhaustive_refine(
    ref: np.ndarray,
    mov: np.ndarray,
    mask: np.ndarray,
    dy0: float,
    dx0: float,
    *,
    radius_hr: float,
    step_hr: float,
) -> tuple[float, float, float]:
    """Search shifts at HR; return best (dy, dx, neg_mse).

    Scores on grayscale for speed; metrics after alignment stay on RGB.
    """
    ref_g = _rgb_to_gray(ref) if ref.ndim == 3 else ref
    mov_g = _rgb_to_gray(mov) if mov.ndim == 3 else mov
    best_dy, best_dx, best_score = dy0, dx0, float("-inf")
    steps = np.arange(-radius_hr, radius_hr + step_hr * 0.5, step_hr)
    for dy in steps:
        for dx in steps:
            cand_dy = dy0 + float(dy)
            cand_dx = dx0 + float(dx)
            shifted = ndi_shift(
                mov_g,
                shift=(cand_dy, cand_dx),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            score = -_masked_mse(ref_g, shifted, mask)
            if score > best_score:
                best_score = score
                best_dy, best_dx = cand_dy, cand_dx
    return best_dy, best_dx, best_score


def audit_city(
    city: str,
    *,
    lr_size: int = 512,
    refine_radius_hr: float = 2.0,
    refine_step_hr: float = 0.25,
    phase_upsample: int = 100,
    border_hr: int = 8,
    s2_dir: Path | None = None,
) -> dict:
    resolved = Path(s2_dir) if s2_dir is not None else _resolve_s2_dir(city, lr_size)
    return audit_s2_dir(
        resolved,
        city=city,
        refine_radius_hr=refine_radius_hr,
        refine_step_hr=refine_step_hr,
        phase_upsample=phase_upsample,
        border_hr=border_hr,
    )


def audit_s2_dir(
    s2_dir: Path,
    *,
    city: str | None = None,
    refine_radius_hr: float = 2.0,
    refine_step_hr: float = 0.25,
    phase_upsample: int = 100,
    border_hr: int = 8,
) -> dict:
    from s2_dataset import _city_id_from_s2_dir

    s2_dir = Path(s2_dir)
    city = city or _city_id_from_s2_dir(s2_dir)
    ds = get_dataset(_dataset_args(s2_dir))

    hr = ds.get_original_hr().detach().cpu().numpy()
    if hr.ndim == 3 and hr.shape[0] == 3:
        hr = np.transpose(hr, (1, 2, 0))
    hr = np.ascontiguousarray(hr, dtype=np.float32)

    base_idx = int(ds.base_frame_index)
    lr = ds.lr_rgb[base_idx].detach().cpu().numpy()
    lr = np.ascontiguousarray(lr, dtype=np.float32)

    df = int(ds.df)
    hr_h, hr_w = hr.shape[:2]
    s2_up = _upsample_lr_bilinear(lr, hr_h, hr_w)

    eval_mask = None
    if hasattr(ds, "get_hr_eval_mask"):
        eval_mask = ds.get_hr_eval_mask().detach().cpu().numpy()
    if eval_mask is None:
        eval_mask = _valid_mask(hr, s2_up)
    mask = eval_mask.copy()
    if border_hr > 0:
        mask[:border_hr, :] = False
        mask[-border_hr:, :] = False
        mask[:, :border_hr] = False
        mask[:, -border_hr:] = False

    if not np.any(mask):
        raise ValueError(
            f"{city}: empty HR eval mask (valid_fraction="
            f"{float(eval_mask.mean()) if eval_mask is not None else 0.0:.3f}); "
            f"check NIB/S2 overlap"
        )

    before = _metrics(hr, s2_up, mask)

    # HR search: phase correlation of S2-bilinear vs NIB, then local MSE refine.
    ref_gray = _rgb_to_gray(hr)
    mov_gray = _rgb_to_gray(s2_up)
    dy_phase, dx_phase = _phase_shift(ref_gray, mov_gray, mask, phase_upsample)
    dy, dx, _ = _exhaustive_refine(
        hr,
        s2_up,
        mask,
        dy_phase,
        dx_phase,
        radius_hr=refine_radius_hr,
        step_hr=refine_step_hr,
    )
    s2_aligned = _shift_hwc(s2_up, dy, dx)
    after = _metrics(hr, s2_aligned, mask)

    # HR shift that would align GT to the S2 grid (inverse of mov shift).
    hr_shift_dy, hr_shift_dx = -dy, -dx

    return {
        "city": city,
        "s2_dir": str(s2_dir),
        "hr_path": str(ds.hr_path),
        "method": "hr_phase_corr + local_hr_mse_refine",
        "df": df,
        "lr_size": [int(ds.lr_height), int(ds.lr_width)],
        "hr_size": [hr_h, hr_w],
        "base_frame_index": base_idx,
        "base_frame_date": getattr(ds, "base_frame_date", None),
        "nib_acquisition_date": getattr(ds, "nib_acquisition_date", None),
        "valid_fraction": float(mask.mean()),
        "shift_hr_px": {"dy": dy, "dx": dx, "dy_lr_equiv": dy / df, "dx_lr_equiv": dx / df},
        "shift_apply_to_s2_bilinear": {"dy": dy, "dx": dx},
        "shift_apply_to_hr_for_eval": {"dy": hr_shift_dy, "dx": hr_shift_dx},
        "phase_init_hr_px": {"dy": dy_phase, "dx": dx_phase},
        "refine_radius_hr_px": float(refine_radius_hr),
        "refine_step_hr_px": float(refine_step_hr),
        "before_align_s2_to_hr": before,
        "after_align_s2_to_hr": after,
        "delta": {
            "psnr": after["psnr"] - before["psnr"],
            "ssim": after["ssim"] - before["ssim"],
            "ncc": after["ncc"] - before["ncc"],
            "mse": before["mse"] - after["mse"],
        },
    }


def _print_row(rec: dict) -> None:
    sh = rec["shift_hr_px"]
    d = rec["delta"]
    b = rec["before_align_s2_to_hr"]
    a = rec["after_align_s2_to_hr"]
    print(
        f"{rec['city']:10s}  "
        f"shift=({sh['dy']:+.2f},{sh['dx']:+.2f}) HR px "
        f"[{sh['dy_lr_equiv']:+.2f},{sh['dx_lr_equiv']:+.2f} LR]  "
        f"PSNR {b['psnr']:.2f}→{a['psnr']:.2f} (Δ{d['psnr']:+.2f})  "
        f"SSIM {b['ssim']:.3f}→{a['ssim']:.3f}  "
        f"NCC {b['ncc']:.3f}→{a['ncc']:.3f}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", default=None, help="Single city id")
    ap.add_argument("--cities", nargs="+", default=None, help="Explicit city id list")
    ap.add_argument(
        "--s2-dir",
        type=Path,
        default=None,
        help="Audit this revisit folder instead of resolving {city}_lr{N}.",
    )
    ap.add_argument("--all-focus", action="store_true", help="Audit all 8 focus cities")
    ap.add_argument(
        "--all-new",
        action="store_true",
        help="Audit all 9 new NIB AOIs (prefer --lr-size 512 for B17)",
    )
    ap.add_argument(
        "--all-17",
        action="store_true",
        help="Audit all 17 confirmatory sites (8 focus + 9 new NIB)",
    )
    ap.add_argument(
        "--lr-size",
        type=int,
        default=512,
        help="Use {city}_lr{N} when present; 0 = parent revisit AOI (needed for new NIB sites)",
    )
    ap.add_argument(
        "--refine-radius-hr",
        type=float,
        default=2.0,
        help="Local exhaustive ± radius in HR pixels around phase-corr estimate",
    )
    ap.add_argument("--refine-step-hr", type=float, default=0.25, help="Local refine step in HR pixels")
    ap.add_argument("--phase-upsample", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "single_samples" / "spatial_audit")
    ap.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="Upsert results into this alignment JSON (e.g. eval/spatial_alignment.json).",
    )
    args = ap.parse_args()

    if args.s2_dir is not None and not (args.all_17 or args.all_focus or args.all_new or args.cities or args.city):
        cities = [None]
    elif args.all_17:
        cities = list(B17_CITIES)
    elif args.all_focus:
        cities = FOCUS_CITIES
    elif args.all_new:
        cities = NEW_NIB_CITIES
    elif args.cities:
        cities = list(args.cities)
    elif args.city:
        cities = [args.city]
    else:
        ap.error("Pass --city NAME, --cities ..., --s2-dir, --all-focus, --all-new, or --all-17")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for city in cities:
        try:
            rec = audit_city(
                city or "unknown",
                lr_size=args.lr_size,
                refine_radius_hr=args.refine_radius_hr,
                refine_step_hr=args.refine_step_hr,
                phase_upsample=args.phase_upsample,
                s2_dir=args.s2_dir,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"SKIP {city}: {exc}")
            continue
        results.append(rec)
        _print_row(rec)
        out_path = args.out_dir / f"{rec['city']}_spatial_alignment.json"
        out_path.write_text(json.dumps(rec, indent=2) + "\n")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {len(results)} city reports → {args.out_dir}")

    if args.merge_into is not None and results:
        merge_path = args.merge_into if args.merge_into.is_absolute() else ROOT / args.merge_into
        merge_into_spatial_alignment(
            results,
            alignment_path=merge_path,
            refine_radius_hr=args.refine_radius_hr,
            refine_step_hr=args.refine_step_hr,
        )


if __name__ == "__main__":
    main()
