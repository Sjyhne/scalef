"""Dump per-frame INR affines and convert them to world metres."""

from __future__ import annotations

import json
from pathlib import Path


def affine_dump_dict(model, dataset) -> dict:
    """Serialise learned 2x3 affines plus the centre-point shift in metres."""
    geo = dataset.get_geo_meta() if hasattr(dataset, "get_geo_meta") else {}
    lr_t = geo.get("lr_transform")
    gsd = float(geo.get("native_gsd_m") or 10.0)
    height = int(getattr(dataset, "lr_height", 0) or 0)
    width = int(getattr(dataset, "lr_width", 0) or 0)
    if lr_t is not None:
        west, north = float(lr_t.c), float(lr_t.f)
        px, py = float(lr_t.a), float(lr_t.e)
    else:
        west = north = 0.0
        px, py = gsd, -gsd

    frames = []
    for i, param in enumerate(model.affine_params):
        A = param.detach().float().cpu().view(2, 3)
        a, b, tx = (float(A[0, 0]), float(A[0, 1]), float(A[0, 2]))
        d, e, ty = (float(A[1, 0]), float(A[1, 1]), float(A[1, 2]))
        u = v = 0.5
        up = a * u + b * v + tx
        vp = d * u + e * v + ty
        frames.append({
            "frame": i,
            "frozen": not bool(param.requires_grad),
            "matrix_2x3": [[a, b, tx], [d, e, ty]],
            "center_shift_east_m": (up - u) * width * px,
            "center_shift_north_m": (vp - v) * height * py,
        })

    return {
        "lr_height": height,
        "lr_width": width,
        "native_gsd_m": gsd,
        "west_m": west,
        "north_m": north,
        "pixel_east_m": px,
        "pixel_north_m": py,
        "row0": int(getattr(dataset, "lr_row0", 0) or 0),
        "col0": int(getattr(dataset, "lr_col0", 0) or 0),
        "frames": frames,
    }


def write_affine_dump(model, dataset, sample_dir: Path) -> Path:
    path = Path(sample_dir) / "affines.json"
    path.write_text(json.dumps(affine_dump_dict(model, dataset), indent=2) + "\n")
    return path
