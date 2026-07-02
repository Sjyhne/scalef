#!/usr/bin/env python3
"""Check which HR→LR operator best reproduces synth burst LR frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.s2_psf_forward import DSEN2_PSF_TRUNCATE_DEFAULT, degrade_hr_bchw
from utils import apply_shift_torch, bilinear_resize_torch


def load_rgb01(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float() / 255.0


def mse_psnr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    mse = float(F.mse_loss(a, b).item())
    psnr = float("inf") if mse <= 0 else -10.0 * np.log10(mse)
    return mse, psnr


def degrade_modes(hr_bchw: torch.Tensor, df: int, truncate: float) -> dict[str, torch.Tensor]:
    modes = {
        "bilinear": bilinear_resize_torch(
            hr_bchw, (hr_bchw.shape[2] // df, hr_bchw.shape[3] // df)
        ),
        "area": degrade_hr_bchw(hr_bchw, df, "area"),
        "s2_psf": degrade_hr_bchw(hr_bchw, df, "s2_psf", truncate=truncate),
        "s2_psf_m": degrade_hr_bchw(hr_bchw, df, "s2_psf_m", truncate=truncate),
    }
    return {k: v[0].permute(1, 2, 0).contiguous() for k, v in modes.items()}


def verify_scene(scene_dir: Path, sample_ids: list[int] | None = None) -> dict:
    meta_path = scene_dir / "synth_export_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    df = int(meta.get("df", 4))
    truncate = float(meta.get("s2_psf_truncate", DSEN2_PSF_TRUNCATE_DEFAULT))
    tagged = str(meta.get("degradation", "unknown"))

    hr = load_rgb01(scene_dir / "hr_ground_truth.png")
    hr_bchw = hr.permute(2, 0, 1).unsqueeze(0)

    with open(scene_dir / "transform_log.json") as f:
        tlog = json.load(f)

    if sample_ids is None:
        sample_ids = [0, 1, 2]

    rows = []
    for sid in sample_ids:
        key = f"sample_{sid:02d}"
        if key not in tlog:
            continue
        lr_path = scene_dir / tlog[key]["path"]
        lr_tgt = load_rgb01(lr_path)
        dx = float(tlog[key]["dx_pixels_hr"])
        dy = float(tlog[key]["dx_pixels_hr"])

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            hr_use = hr_bchw
        else:
            hr_use = apply_shift_torch(
                hr_bchw,
                dx=torch.tensor([dx], device=hr_bchw.device),
                dy=torch.tensor([dy], device=hr_bchw.device),
            )

        preds = degrade_modes(hr_use, df, truncate)
        best_mode = min(preds, key=lambda m: mse_psnr(preds[m], lr_tgt)[0])
        row = {
            "sample": key,
            "dx_hr": dx,
            "dy_hr": dy,
            "tagged_degradation": tagged,
            "best_mode": best_mode,
        }
        for mode, pred in preds.items():
            mse, psnr = mse_psnr(pred, lr_tgt)
            row[f"{mode}_mse"] = mse
            row[f"{mode}_psnr"] = psnr
        rows.append(row)
    return {"scene": str(scene_dir), "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roots",
        nargs="+",
        default=[
            "satsynthburst_data_s2psfm",
            "satburstsynth_data",
        ],
    )
    p.add_argument("--scene", default="UNHCR-YEMs035290_rgb")
    p.add_argument("--folder", default="scale_4_lr64_shift_1.0px_aug_none")
    p.add_argument("--samples", type=int, nargs="*", default=[0, 1, 8])
    args = p.parse_args()

    print("HR→LR reproduction check (lower MSE = better match to saved LR PNG)\n")
    for root in args.roots:
        scene_dir = Path(root) / args.scene / args.folder
        if not scene_dir.is_dir():
            print(f"[skip] missing {scene_dir}")
            continue
        result = verify_scene(scene_dir, args.samples)
        print(f"=== {result['scene']} ===")
        for row in result["rows"]:
            print(
                f"  {row['sample']} (dx={row['dx_hr']:.2f}, dy={row['dy_hr']:.2f}) "
                f"tag={row['tagged_degradation']} best={row['best_mode']}"
            )
            for mode in ("bilinear", "area", "s2_psf", "s2_psf_m"):
                print(
                    f"    {mode:10s}  MSE={row[f'{mode}_mse']:.6f}  "
                    f"PSNR={row[f'{mode}_psnr']:.2f} dB"
                )
        print()


if __name__ == "__main__":
    main()
