"""Fixed display mapping for RGB reflectance figures.

Sentinel Hub ``HighlightCompressVisualizer(minValue=0, maxValue=0.4)``: linear up to
92% of the display range, then highlights are compressed so that reflectance 0.4 maps
to 0.926 and 0.8 saturates. The same mapping applies to every tile, date, and method,
so displayed brightness follows reflectance rather than scene content.
"""
from __future__ import annotations

import numpy as np

MIN_VALUE = 0.0
MAX_VALUE = 0.4
KNEE = 0.92


def highlight_compress(rgb: np.ndarray, min_value: float = MIN_VALUE, max_value: float = MAX_VALUE) -> np.ndarray:
    x = np.clip((np.asarray(rgb, dtype=np.float32) - min_value) / (max_value - min_value), 0.0, None)
    y = np.where(x <= KNEE, x, KNEE + (x - KNEE) * ((1.0 - KNEE) / (2.0 - KNEE)))
    return np.clip(y, 0.0, 1.0)
