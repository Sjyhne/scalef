"""Fixed display mapping for RGB reflectance figures.

Paper panels use ``linear_stretch``: reflectance 0 maps to black and 0.25 to white,
with values outside that range clipped. The same stretch applies to every tile,
date, and method, so displayed brightness follows reflectance rather than scene
content. ``highlight_compress`` keeps the earlier Sentinel Hub
``HighlightCompressVisualizer(minValue=0, maxValue=0.4)`` mapping.
"""
from __future__ import annotations

import numpy as np

MIN_VALUE = 0.0
MAX_VALUE = 0.4
KNEE = 0.92
# Linear display range used for paper panels and production viewing.
STRETCH_MIN = 0.0
STRETCH_MAX = 0.25


def linear_stretch(
    rgb: np.ndarray, min_value: float = STRETCH_MIN, max_value: float = STRETCH_MAX
) -> np.ndarray:
    """Map reflectance ``min_value`` to black and ``max_value`` to white, clipping outside."""
    x = (np.asarray(rgb, dtype=np.float32) - min_value) / (max_value - min_value)
    return np.clip(x, 0.0, 1.0)


def highlight_compress(rgb: np.ndarray, min_value: float = MIN_VALUE, max_value: float = MAX_VALUE) -> np.ndarray:
    x = np.clip((np.asarray(rgb, dtype=np.float32) - min_value) / (max_value - min_value), 0.0, None)
    y = np.where(x <= KNEE, x, KNEE + (x - KNEE) * ((1.0 - KNEE) / (2.0 - KNEE)))
    return np.clip(y, 0.0, 1.0)
