"""Perspective correction ("Top View"): straightens a 4-cornered area of an
angled camera image into a rectangle, as if seen from straight above.

A quad is 4 corner points as fractions of the raw frame ([[x, y], ...] in
0..1), so a job's top view survives a change of camera resolution, like
app/roi.py. Only the tilted plane is corrected, not lens curvature.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

Quad = List[List[float]]  # [[x, y]] x4: top-left, top-right, bottom-right, bottom-left

# Smaller areas are taken as a mis-placed corner, not a work area.
MIN_AREA_FRACTION = 0.02
MIN_OUTPUT_SIDE = 64


def order_corners(points: np.ndarray) -> np.ndarray:
    """4x2 points -> top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    ordered = np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])
    if len({tuple(p) for p in ordered}) < 4:
        # Strongly rotated quads can make x+y / y-x pick a point twice; fall
        # back to sorting by angle around the centre (clockwise from top-left).
        centre = pts.mean(axis=0)
        by_angle = pts[np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))]
        start = int(np.argmin(by_angle.sum(axis=1)))
        ordered = np.roll(by_angle, -start, axis=0)
    return ordered


def _is_convex(pts: np.ndarray) -> bool:
    """True if the ordered polygon turns the same way at every corner (so its
    edges don't cross)."""
    crosses = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(v > 0 for v in crosses) or all(v < 0 for v in crosses)


def normalize_quad(points: Optional[Sequence]) -> Optional[Quad]:
    """Validates a stored / edited quad: clamps to the frame and orders the
    corners. Returns None for no quad, malformed input, a crossed or
    non-convex shape, or an area under MIN_AREA_FRACTION of the frame."""
    if points is None:
        return None
    try:
        pts = np.clip(np.array(points, dtype=np.float64).reshape(4, 2), 0.0, 1.0)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(pts)):
        return None
    pts = order_corners(pts)
    if not _is_convex(pts):
        return None
    if cv2.contourArea(pts.astype(np.float32)) < MIN_AREA_FRACTION:
        return None
    return [[round(float(x), 4), round(float(y), 4)] for x, y in pts]


def quad_to_pixels(quad: Quad, width: int, height: int) -> np.ndarray:
    return np.array(quad, dtype=np.float32) * np.array([width, height], dtype=np.float32)


def output_size(quad_px: np.ndarray, max_side: int) -> Tuple[int, int]:
    """Rectangle size from the corners: width = longer of the top/bottom
    edges, height = longer of the left/right edges, scaled so the long side
    is at most max_side."""
    tl, tr, br, bl = quad_px
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    scale = min(1.0, max_side / max(width, height))
    return (max(MIN_OUTPUT_SIDE, int(round(width * scale))),
            max(MIN_OUTPUT_SIDE, int(round(height * scale))))


class PerspectiveWarp:
    """Warps frames of one size to the top view; the matrix is built once
    and reused until the quad or the frame size changes."""

    def __init__(self, quad: Quad) -> None:
        self.quad = quad
        self._frame_size: Optional[Tuple[int, int]] = None
        self._matrix: Optional[np.ndarray] = None
        self.size: Tuple[int, int] = (0, 0)

    def _prepare(self, width: int, height: int, max_side: Optional[int]) -> None:
        src = quad_to_pixels(self.quad, width, height)
        out_w, out_h = output_size(src, max_side or max(width, height))
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        self._matrix = cv2.getPerspectiveTransform(src, dst)
        self.size = (out_w, out_h)
        self._frame_size = (width, height)

    def apply(self, frame: np.ndarray, max_side: Optional[int] = None) -> np.ndarray:
        """max_side limits the output's long side (default: the frame's)."""
        height, width = frame.shape[:2]
        if self._frame_size != (width, height) or max_side is not None:
            self._prepare(width, height, max_side)
        return cv2.warpPerspective(frame, self._matrix, self.size, flags=cv2.INTER_LINEAR)
