"""Post-hoc sampling-geometry score for learned inter-revisit affines.

The score is a diagnostic of sampling phases. It is not an achieved spatial
resolution and not a probability of correct reconstruction. It is computed
after fitting and is not used to stop training or to select a checkpoint.

Coordinate convention
--------------------
The model builds pixel-centre coordinates in ``[0, 1)``::

    x = (col + 0.5) / width
    y = (row + 0.5) / height

``models.inr.apply_affine`` maps those coordinates with a learned 2×3 matrix,

    x' = a x + b y + tx
    y' = d x + e y + ty

before the field is encoded. Frame 0 is frozen to the identity, so the
displacement of frame ``t`` is the inter-revisit transform relative to the
base frame. One LR pixel is ``1/width`` in ``x`` and ``1/height`` in ``y``,
so the displacement in LR-pixel units is

    (dx, dy) = ((x' - x) * width, (y' - y) * height).

The fractional sampling offset is that displacement modulo one LR pixel.
Orthophoto registration shifts are not part of this calculation.

Score
-----
For a candidate factor ``r`` and one offset per usable revisit,

    P_{t,(k,ℓ)} = exp(2 π i (k δx_t + ℓ δy_t)),  k,ℓ = 0..r-1
    G_r = Pᴴ diag(w) P / Σ w
    S_r = 100 * λ_min(G_r)

``w_t`` is 1 when that revisit is usable at the location being scored.
Different pixel positions are separate locations; they are not extra rows of
``P``. Calculations use float64 and complex128. Eigenvalues of the full
``r² × r²`` Gram matrix are retained, so an underdetermined case keeps its
zero eigenvalues. Negative eigenvalues smaller in magnitude than ``1e-8`` are
treated as roundoff and set to zero. If no observation is usable the status
is ``no_data``.
"""

from __future__ import annotations

import numpy as np

EIGEN_CLIP = 1e-8
RANK_TOL = 1e-8


def sampling_offsets_lr(
    affines: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-affine displacements and fractional offsets at LR pixel centres.

    Parameters
    ----------
    affines:
        Learned matrices, shape ``(frames, 2, 3)``, in normalized coordinates.
    rows, cols:
        LR row and column indices, each shape ``(n_points,)``.

    Returns
    -------
    displacement, fractional:
        Both have shape ``(frames, n_points, 2)`` and order ``(x, y)``.
        ``displacement`` is in LR pixels. ``fractional`` is that displacement
        modulo 1.
    """
    matrices = np.asarray(affines, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 3):
        raise ValueError(f"affines must have shape (F, 2, 3), got {matrices.shape}")
    row = np.asarray(rows, dtype=np.float64).reshape(-1)
    col = np.asarray(cols, dtype=np.float64).reshape(-1)
    if row.shape != col.shape:
        raise ValueError("rows and cols must contain the same number of points")
    height = int(height)
    width = int(width)
    x = (col + 0.5) / width
    y = (row + 0.5) / height
    # (F, N)
    x_warp = matrices[:, 0, 0, None] * x + matrices[:, 0, 1, None] * y + matrices[:, 0, 2, None]
    y_warp = matrices[:, 1, 0, None] * x + matrices[:, 1, 1, None] * y + matrices[:, 1, 2, None]
    dx = (x_warp - x) * width
    dy = (y_warp - y) * height
    displacement = np.stack([dx, dy], axis=-1)
    fractional = np.mod(displacement, 1.0)
    return displacement, fractional


def _mode_axes(factor: int) -> tuple[np.ndarray, np.ndarray]:
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    ks, ls = np.meshgrid(np.arange(factor), np.arange(factor), indexing="ij")
    return ks.reshape(-1).astype(np.float64), ls.reshape(-1).astype(np.float64)


def phase_matrix(offsets: np.ndarray, factor: int) -> np.ndarray:
    """Complex phase matrix of shape ``(n_frames, factor²)``."""
    delta = np.asarray(offsets, dtype=np.float64)
    if delta.ndim != 2 or delta.shape[1] != 2:
        raise ValueError(f"offsets must have shape (T, 2), got {delta.shape}")
    ks, ls = _mode_axes(factor)
    phase = ks[None, :] * delta[:, 0:1] + ls[None, :] * delta[:, 1:2]
    return np.exp((2j * np.pi) * phase)


def sampling_geometry_score(offsets: np.ndarray, usable: np.ndarray, factor: int) -> dict:
    """Score one location. ``offsets`` is ``(T, 2)`` and ``usable`` is ``(T,)``."""
    delta = np.mod(np.asarray(offsets, dtype=np.float64), 1.0)
    weights = np.asarray(usable, dtype=bool).reshape(-1)
    if delta.shape != (weights.shape[0], 2):
        raise ValueError(
            f"offsets shape {delta.shape} does not match usable length {weights.shape[0]}"
        )
    n_usable = int(weights.sum())
    n_modes = int(factor) * int(factor)
    if n_usable == 0:
        return {
            "status": "no_data",
            "score": None,
            "rank": 0,
            "n_usable": 0,
            "n_modes": n_modes,
            "eigenvalues": None,
        }
    phases = phase_matrix(delta[weights], factor)
    gram = phases.conj().T @ phases
    gram = gram / n_usable
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues = np.linalg.eigvalsh(gram).real.astype(np.float64)
    if np.any(eigenvalues < -EIGEN_CLIP):
        raise FloatingPointError(
            f"Gram eigenvalue {float(eigenvalues.min())} is too negative to be roundoff"
        )
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    rank = int(np.count_nonzero(eigenvalues > RANK_TOL))
    return {
        "status": "ok",
        "score": float(100.0 * eigenvalues.min()),
        "rank": rank,
        "n_usable": n_usable,
        "n_modes": n_modes,
        "eigenvalues": eigenvalues,
    }


def circular_deviation(offsets: np.ndarray) -> np.ndarray:
    """Max toroidal distance from the circular mean, per coordinate.

    ``offsets`` has shape ``(n_points, 2)`` and is interpreted modulo 1.
    Returns shape ``(2,)``. An empty set returns zeros.
    """
    values = np.mod(np.asarray(offsets, dtype=np.float64), 1.0)
    if values.size == 0:
        return np.zeros(2, dtype=np.float64)
    deviations = []
    for axis in range(2):
        angles = np.exp((2j * np.pi) * values[:, axis])
        centre = np.angle(np.mean(angles)) / (2.0 * np.pi)
        centre = np.mod(centre, 1.0)
        delta = np.abs(values[:, axis] - centre)
        delta = np.minimum(delta, 1.0 - delta)
        deviations.append(float(np.max(delta)) if delta.size else 0.0)
    return np.asarray(deviations, dtype=np.float64)
