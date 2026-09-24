import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.seam_pilot_3x3 import apply_gain, solve_gains_from_lr


def _write_rgb(path: Path, rgb: np.ndarray, *, x0: float, y0: float) -> None:
    h, w, _ = rgb.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 3,
        "dtype": "float32",
        "crs": "EPSG:32632",
        "transform": from_origin(x0, y0, 2.5, 2.5),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.transpose(rgb, (2, 0, 1)))


def test_solve_gains_recovers_known_offset(tmp_path: Path):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.05, 0.25, size=(64, 96, 3)).astype(np.float32)
    left = base[:, :64]
    right = np.clip(1.15 * base[:, 32:] + 0.02, 0, 1).astype(np.float32)
    p0 = tmp_path / "a.tif"
    p1 = tmp_path / "b.tif"
    _write_rgb(p0, left, x0=0.0, y0=160.0)
    _write_rgb(p1, right, x0=32 * 2.5, y0=160.0)
    solved = solve_gains_from_lr(
        [p0, p1], lf_k=5, margin=2, change_thr=0.2, lam_a=5.0, lam_b=5.0
    )
    g1 = solved["gains"][1]
    corrected = apply_gain(right, g1)
    # After correction, overlap with left should be closer than the raw 15% gain.
    raw = float(np.abs(left[:, 32:] - right[:, :32]).mean())
    after = float(np.abs(left[:, 32:] - corrected[:, :32]).mean())
    assert after < 0.5 * raw
    assert all(0.7 < a < 1.4 for a in g1["a"])
