"""Blob Count: number of blobs in a binarized ROI, filtered by size, shape."""

import math

import cv2
import numpy as np

from .brightness import binarize
from .base import overlay_item, register
from .color import HSV_PARAMS, HsvTool, open_mask


@register("blob_count")
class BlobCountTool(HsvTool):
    """`source` "gray" thresholds like Area, "color" uses the HSV range like
    Color Area. A blob counts when its area, aspect ratio (w/h) and
    circularity (4*pi*A/P^2) are inside the limits."""

    PARAMS = {
        "source": "gray", "method": "fixed", "threshold": 128, "polarity": "bright",
        **HSV_PARAMS, "k": 2.5,
        "open_px": 1, "area_min": 20, "area_max": 1e9, "aspect_min": 0.0, "aspect_max": 1e9, "circ_min": 0.0,
    }

    def blob_mask(self, patch: np.ndarray) -> np.ndarray:
        if self.source == "color":
            return self.mask(patch)
        return open_mask(binarize(self.gray(patch), self.method, self.threshold, self.polarity), self.open_px)

    @staticmethod
    def circularity(blob: np.ndarray) -> float:
        """4*pi*A/P^2 of a 0/255 mask holding one blob (1.0 = perfect circle)."""
        blob = cv2.copyMakeBorder(blob, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        # Smoothing the pixel staircase first: it would inflate the perimeter.
        perimeter = cv2.arcLength(cv2.approxPolyDP(c, 1.0, True), True)
        return min(1.0, 4 * math.pi * cv2.contourArea(c) / perimeter ** 2) if perimeter > 0 else 0.0

    def process(self, patch, origin):
        mask = self.blob_mask(patch)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs, overlay = [], []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if not self.area_min <= area <= self.area_max:
                continue
            if not self.aspect_min <= w / h <= self.aspect_max:
                continue
            circ = 1.0
            if self.circ_min > 0:
                circ = self.circularity((labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255)
                if circ < self.circ_min:
                    continue
            blobs.append({"x": x, "y": y, "w": w, "h": h, "area": area, "circularity": circ})
            overlay.append(overlay_item("rect", [x, y, x + w, y + h], True, str(len(blobs))))
        res = self.result(len(blobs), overlay, extra={"blobs": blobs})
        for item in res.overlay:
            item["ok"] = res.ok
        return res
