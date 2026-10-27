"""Brightness tools: Area (mono), Brightness Average, Edge Pixels and the
shading-uniformity check. All work on the gray ROI after the `pre` chain."""

import cv2
import numpy as np

from .base import Tool, overlay_item, register


def binarize(gray: np.ndarray, method: str = "fixed", threshold: float = 128, polarity: str = "bright") -> np.ndarray:
    """255 where the pixel is brighter (or darker) than the threshold; Otsu picks it itself."""
    if method == "otsu":
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if polarity == "dark":
        return (gray < threshold).astype(np.uint8) * 255 if method == "fixed" else (gray <= threshold).astype(np.uint8) * 255
    return (gray > threshold).astype(np.uint8) * 255


@register("area")
class AreaTool(Tool):
    """Percentage of the ROI above (bright) or below (dark) a gray threshold."""

    PARAMS = {"method": "fixed", "threshold": 128, "polarity": "bright"}

    def process(self, patch, origin):
        mask = binarize(self.gray(patch), self.method, self.threshold, self.polarity)
        value = cv2.countNonZero(mask) * 100.0 / mask.size
        overlay = []
        if value > 0:
            x, y, w, h = cv2.boundingRect(mask)
            overlay.append(overlay_item("mask_box", [x, y, x + w, y + h], True, f"{value:.1f}%"))
        res = self.result(value, overlay)
        for item in res.overlay:
            item["ok"] = res.ok
        return res
