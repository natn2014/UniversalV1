"""Color tools: Color Area, Color Average (ΔE2000) and Color Prohibit.

Color Area and Color Prohibit share an HSV range (`HSV_PARAMS`); hue is
circular, so a range with h_lo > h_hi wraps through 0 (red).
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from ..roi import roi_to_pixels
from .base import Offset, Tool, overlay_item, register
from .preprocess import to_hsv

HSV_PARAMS = {"h_lo": 170, "h_hi": 10, "s_lo": 80, "s_hi": 255, "v_lo": 50, "v_hi": 255}
MIN_HALF_WIDTH = {"h": 8.0, "s": 40.0, "v": 40.0}
POINT_BOX = 9  # px side of the box sampled around a tapped point


def hsv_mask(hsv: np.ndarray, h_lo: int, h_hi: int, s_lo: int, s_hi: int, v_lo: int, v_hi: int,
             open_px: int = 0) -> np.ndarray:
    """255 where the pixel is inside the HSV range; optionally morphologically opened."""
    lo_sv, hi_sv = (s_lo, v_lo), (s_hi, v_hi)
    if h_lo <= h_hi:
        mask = cv2.inRange(hsv, (h_lo, *lo_sv), (h_hi, *hi_sv))
    else:
        mask = cv2.inRange(hsv, (h_lo, *lo_sv), (179, *hi_sv)) | cv2.inRange(hsv, (0, *lo_sv), (h_hi, *hi_sv))
    if open_px and open_px > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_px),) * 2))
    return mask


def learn_hsv_range(hsv_pixels: np.ndarray, k: float = 2.5) -> dict:
    """HSV range covering the sampled pixels (N x 3): circular mean +- spread
    for hue, mean +- k sigma for S and V, each at least the minimum width."""
    px = hsv_pixels.reshape(-1, 3).astype(np.float64)
    ang = px[:, 0] * (2 * math.pi / 180.0)
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    mean_h = (math.atan2(s, c) * 180.0 / (2 * math.pi)) % 180.0
    r = min(max(math.hypot(c, s), 1e-6), 1.0)
    sigma_h = math.sqrt(-2.0 * math.log(r)) * 180.0 / (2 * math.pi)
    half_h = max(k * sigma_h, MIN_HALF_WIDTH["h"])
    out = {}
    if half_h >= 90:
        out.update(h_lo=0, h_hi=179)
    else:
        out.update(h_lo=int(round(mean_h - half_h)) % 180, h_hi=int(round(mean_h + half_h)) % 180)
    for i, key in ((1, "s"), (2, "v")):
        mean, sd = px[:, i].mean(), px[:, i].std()
        half = max(k * sd, MIN_HALF_WIDTH[key])
        out[f"{key}_lo"] = int(max(0, round(mean - half)))
        out[f"{key}_hi"] = int(min(255, round(mean + half)))
    return out


class HsvTool(Tool):
    """Shared by the tools that threshold an HSV range and can learn it."""

    def hsv(self, patch: np.ndarray) -> np.ndarray:
        return to_hsv(self.prepared(patch))

    def mask(self, patch: np.ndarray) -> np.ndarray:
        return hsv_mask(self.hsv(patch), self.h_lo, self.h_hi, self.s_lo, self.s_hi,
                        self.v_lo, self.v_hi, getattr(self, "open_px", 0))

    def teach(self, frame: np.ndarray, offset: Optional[Offset] = None, point: Optional[Tuple[int, int]] = None,
              teach_roi=None, **kw) -> None:
        """Learns the range from a tapped point (9x9 px), a `teach_roi` (frame
        fractions) or, by default, the tool's own ROI."""
        h, w = frame.shape[:2]
        if point is not None:
            x, y = int(point[0]), int(point[1])
            r = POINT_BOX // 2
            sample = frame[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        elif teach_roi is not None:
            x1, y1, x2, y2 = roi_to_pixels(tuple(teach_roi), w, h)
            sample = frame[y1:y2, x1:x2]
        else:
            sample, _ = self.crop(frame, offset)
        if sample.size == 0:
            return
        self.set_params(learn_hsv_range(self.hsv(sample), self.k))


@register("color_area")
class ColorAreaTool(HsvTool):
    """Percentage of the ROI that matches the taught color."""

    PARAMS = {**HSV_PARAMS, "k": 2.5, "open_px": 0}

    def process(self, patch, origin):
        mask = self.mask(patch)
        value = cv2.countNonZero(mask) * 100.0 / mask.size
        overlay = []
        if value > 0:
            x, y, w, h = cv2.boundingRect(mask)
            overlay.append(overlay_item("mask_box", [x, y, x + w, y + h], True, f"{value:.1f}%"))
        res = self.result(value, overlay)
        for item in res.overlay:
            item["ok"] = res.ok
        return res
