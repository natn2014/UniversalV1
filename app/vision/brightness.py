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


@register("brightness")
class BrightnessAverageTool(Tool):
    """Mean gray level of the ROI (or its standard deviation with `use_std`)."""

    PARAMS = {"use_std": False}

    def process(self, patch, origin):
        mean, std = cv2.meanStdDev(self.gray(patch))
        mean, std = float(mean[0][0]), float(std[0][0])
        return self.result(std if self.use_std else mean, extra={"mean": mean, "std": std})


@register("edge_pixels")
class EdgePixelsTool(Tool):
    """Number of Canny edge pixels in the ROI: scratches, burrs, texture."""

    PARAMS = {"canny_lo": 50, "canny_hi": 150}

    def process(self, patch, origin):
        gray = cv2.GaussianBlur(self.gray(patch), (3, 3), 0)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        count = cv2.countNonZero(edges)
        return self.result(count, extra={"density": count * 100.0 / edges.size})
