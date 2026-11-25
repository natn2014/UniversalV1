"""Outline: % match of the part's outline with the taught master.

The master is stored as the gray ROI crop (so it is a viewable PNG); the edge
template and contour are derived from it and cached.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import Offset, Tool, overlay_item, register

SE3 = np.ones((3, 3), np.uint8)
# Hu-moment distances are tiny (circle vs square is about 0.03), so the
# distance is stretched before it becomes a percentage.
SHAPE_SHARPNESS = 25.0


def edge_map(gray: np.ndarray, canny_lo: float, canny_hi: float) -> np.ndarray:
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), canny_lo, canny_hi)


def largest_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    """Outer contour of the biggest blob after Otsu (the minority class is the part)."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) > mask.size / 2:
        mask = 255 - mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


@register("outline")
class OutlineTool(Tool):
    """`template` mode slides the dilated master edges over the ROI's edges
    (the ROI is `margin_px` larger than the template on each side, so a small
    shift is tolerated); `shape` mode compares the biggest contour's Hu
    moments, which ignores rotation and scale."""

    PARAMS = {"mode": "template", "canny_lo": 50, "canny_hi": 150, "margin_px": 8}
    DEFAULT_LOW = 70.0

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self._cache: Tuple = (None, None)

    def teach(self, frame: np.ndarray, offset: Optional[Offset] = None, **kw) -> None:
        patch, _ = self.crop(frame, offset)
        if patch.size:
            self.template = np.ascontiguousarray(self.gray(patch))

    def _margin(self, shape) -> int:
        m = int(self.margin_px)
        return m if min(shape[:2]) > 2 * m + 16 else 0

    def _derived(self):
        """(edge template, master contour) of the taught master, cached."""
        key = (id(self.template), self.canny_lo, self.canny_hi, self.margin_px)
        if self._cache[0] != key:
            m = self._margin(self.template.shape)
            edges = edge_map(self.template, self.canny_lo, self.canny_hi)
            inner = edges[m:edges.shape[0] - m, m:edges.shape[1] - m]
            self._cache = (key, (cv2.dilate(inner, SE3), largest_contour(self.template)))
        return self._cache[1]

    def process(self, patch, origin):
        if self.template is None:
            return self.fail("not taught")
        gray = self.gray(patch)
        if gray.shape != self.template.shape:
            return self.fail("ROI size changed: teach again")
        templ, master_contour = self._derived()
        if self.mode == "shape":
            return self._shape(gray, master_contour)
        return self._template(gray, templ)

    def _template(self, gray, templ):
        # Both sides dilated: otherwise the normalised score tops out near 0.58.
        edges = cv2.dilate(edge_map(gray, self.canny_lo, self.canny_hi), SE3)
        th, tw = templ.shape
        if cv2.countNonZero(templ) == 0 or cv2.countNonZero(edges) == 0:
            return self.result(0.0, text="no edges")
        res = cv2.matchTemplate(edges, templ, cv2.TM_CCORR_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(np.nan_to_num(res))
        value = float(np.clip(score, 0, 1)) * 100
        out = self.result(value, extra={"x": x, "y": y})
        out.overlay = [overlay_item("rect", [x, y, x + tw, y + th], out.ok, f"{value:.0f}%")]
        return out

    def _shape(self, gray, master_contour):
        contour = largest_contour(gray)
        if contour is None or master_contour is None:
            return self.result(0.0, text="no shape")
        dist = cv2.matchShapes(master_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
        value = 100.0 * math.exp(-dist * SHAPE_SHARPNESS)
        poly = cv2.approxPolyDP(contour, 2.0, True).reshape(-1, 2).tolist()
        out = self.result(value, extra={"distance": float(dist)})
        out.overlay = [overlay_item("poly", poly, out.ok, f"{value:.0f}%")]
        return out
