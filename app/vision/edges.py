"""Edge-profile core shared by Width, Edge Presence, Pitch and 1-axis
Position Adjustment.

Positions are in pixel-edge coordinates: a sharp step between pixel 39 and
pixel 40 is at 40.0. Rising = dark -> light in the scan direction.
"""

from typing import List, Tuple

import cv2
import numpy as np


def profile(gray_patch: np.ndarray, direction: str = "x") -> np.ndarray:
    """Mean along the other axis: "x" averages the rows into a length-w array,
    "y" averages the columns into a length-h array."""
    if gray_patch.ndim == 3:
        gray_patch = cv2.cvtColor(gray_patch, cv2.COLOR_BGR2GRAY)
    return gray_patch.astype(np.float64).mean(axis=0 if direction == "x" else 1)


def gradient(p: np.ndarray, smooth: float = 1.0) -> np.ndarray:
    """Derivative of the profile after optional Gaussian smoothing (sigma in px)."""
    p = np.asarray(p, dtype=np.float64)
    if smooth and smooth > 0 and len(p) > 2:
        p = cv2.GaussianBlur(p.reshape(1, -1), (0, 0), float(smooth), borderType=cv2.BORDER_REPLICATE).ravel()
    return np.gradient(p) if len(p) > 1 else np.zeros_like(p)


def find_edges(p: np.ndarray, threshold: float = 20, polarity: str = "any", min_distance: float = 3,
               smooth: float = 1.0) -> List[Tuple[float, float]]:
    """Edges of a 1-D profile as (position, signed strength), sorted by position.

    Strength is the gradient there (positive rising, negative falling) and
    `threshold` applies to its magnitude. Positions are refined to sub-pixel
    with a parabola through the three gradient samples at the peak; of two
    edges closer than `min_distance` only the stronger is kept.
    """
    g = gradient(p, smooth)
    if len(g) < 3:
        return []
    signs = {"rising": (1,), "falling": (-1,)}.get(polarity, (1, -1))
    found: List[Tuple[float, float]] = []
    for s in signs:
        a = g * s
        for i in range(1, len(a) - 1):
            if a[i] < threshold or not (a[i] > a[i - 1] and a[i] >= a[i + 1]):
                continue
            denom = a[i - 1] - 2 * a[i] + a[i + 1]
            shift = 0.5 * (a[i - 1] - a[i + 1]) / denom if abs(denom) > 1e-12 else 0.0
            found.append((i + float(np.clip(shift, -0.5, 0.5)) + 0.5, float(g[i])))
    found.sort(key=lambda e: -abs(e[1]))
    kept: List[Tuple[float, float]] = []
    for e in found:
        if all(abs(e[0] - k[0]) >= min_distance for k in kept):
            kept.append(e)
    return sorted(kept)
