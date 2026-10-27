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


@register("shading")
class ShadingUniformityTool(Tool):
    """Spread (max - min) of the mean gray level over a grid of cells: uneven
    coating, discoloration, a weak lamp. Not an IV4 tool."""

    PARAMS = {"grid_x": 4, "grid_y": 4}
    DEFAULT_HIGH = 20.0

    def process(self, patch, origin):
        gray = self.gray(patch)
        gx, gy = int(self.grid_x), int(self.grid_y)
        cells = cv2.resize(gray, (gx, gy), interpolation=cv2.INTER_AREA).astype(np.float64)
        lo = np.unravel_index(int(np.argmin(cells)), cells.shape)
        hi = np.unravel_index(int(np.argmax(cells)), cells.shape)
        value = float(cells.max() - cells.min())
        h, w = gray.shape[:2]
        cw, ch = w / gx, h / gy

        def box(rc, label):
            r, c = rc
            return overlay_item("mask_box", [c * cw, r * ch, (c + 1) * cw, (r + 1) * ch], self.judge(value), label)

        overlay = [box(lo, f"min {cells[lo]:.0f}"), box(hi, f"max {cells[hi]:.0f}")]
        return self.result(value, overlay, extra={
            "cells": cells.round(2).tolist(), "argmin": [int(lo[0]), int(lo[1])], "argmax": [int(hi[0]), int(hi[1])]})
