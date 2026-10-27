"""Brightness tools: Area (mono), Brightness Average, Edge Pixels and the
shading-uniformity check. All work on the gray ROI after the `pre` chain."""

from typing import Optional, Tuple

import cv2
import numpy as np

from .base import Tool, overlay_item, region_item, register


def binarize(gray: np.ndarray, method: str = "fixed", threshold: float = 128, polarity: str = "bright") -> np.ndarray:
    """255 where the pixel is brighter (or darker) than the threshold; Otsu picks it itself."""
    if method == "otsu":
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if polarity == "dark":
        return (gray < threshold).astype(np.uint8) * 255 if method == "fixed" else (gray <= threshold).astype(np.uint8) * 255
    return (gray > threshold).astype(np.uint8) * 255


GRAY_STEP = 5  # one [+] / [-] step of the threshold
POINT_BOX = 9  # px side of the box sampled around a tapped point
EDGE_GAP = 15  # gray levels a pixel must differ from the tapped one to count as background


def pick_threshold(roi_gray: np.ndarray, tapped: float, polarity: Optional[str] = None) -> Tuple[int, str]:
    """Fixed threshold and polarity that count the tapped gray level: halfway
    between it and the background (the ROI pixels on the other side of it).
    Polarity is bright when the tap is in the upper half of the ROI's range."""
    p5, p95 = np.percentile(roi_gray, (5, 95))
    if polarity is None:
        polarity = "bright" if tapped >= (p5 + p95) / 2 else "dark"
    if polarity == "bright":
        other = roi_gray[roi_gray < tapped - EDGE_GAP]
        background = float(np.median(other)) if other.size else float(roi_gray.min())
    else:
        other = roi_gray[roi_gray > tapped + EDGE_GAP]
        background = float(np.median(other)) if other.size else float(roi_gray.max())
    return int(np.clip(round((tapped + background) / 2), 1, 254)), polarity


class GrayPick:
    """Brightness eyedropper for tools with `threshold` / `polarity` / `method`
    (Keyence "Brightness Extraction"): the first tap sets them, later taps
    only lower (bright) or raise (dark) the threshold so every tapped spot counts."""

    def gray_pick(self, frame, point, add=False, offset=None) -> None:
        x, y = int(point[0]), int(point[1])
        r = POINT_BOX // 2
        sample = frame[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        patch, _ = self.crop(frame, offset)
        if sample.size == 0 or patch.size == 0:
            return
        tapped = float(np.median(self.gray(sample)))
        keep = self.polarity if add else None
        threshold, polarity = pick_threshold(self.gray(patch), tapped, keep)
        if add:
            threshold = min(threshold, self.threshold) if polarity == "bright" else max(threshold, self.threshold)
        self.set_params({"method": "fixed", "threshold": int(threshold), "polarity": polarity})

    def gray_widen(self, steps: int) -> None:
        """Wider counts more pixels: a lower threshold for bright, higher for dark."""
        sign = -1 if self.polarity == "bright" else 1
        self.set_params({"method": "fixed",
                         "threshold": int(np.clip(self.threshold + sign * GRAY_STEP * steps, 0, 255))})


@register("area")
class AreaTool(GrayPick, Tool):
    """Percentage of the ROI above (bright) or below (dark) a gray threshold."""

    PARAMS = {"method": "fixed", "threshold": 128, "polarity": "bright"}
    PICK = "gray"

    def pick(self, frame, point, add=False, offset=None) -> None:
        self.gray_pick(frame, point, add, offset)

    def widen(self, steps: int) -> None:
        self.gray_widen(steps)

    def process(self, patch, origin):
        mask = binarize(self.gray(patch), self.method, self.threshold, self.polarity)
        value = cv2.countNonZero(mask) * 100.0 / mask.size
        overlay = []
        if value > 0:
            overlay.append(region_item(mask, True, f"{value:.1f}%"))
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
        overlay = [region_item(edges, color="lines")] if count else []
        return self.result(count, overlay, extra={"density": count * 100.0 / edges.size})


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
