"""Measurement tools: Width, Edge Presence, Pitch and Diameter. Values are in
pixels, or mm when the job is calibrated (`Tool.mm_per_px`, set by the runner)."""

from typing import List, Optional, Tuple

import numpy as np

from .base import Tool, overlay_item, register
from .edges import find_edges, gradient, profile

Edge = Tuple[float, float]


def _matches(strength: float, polarity: str) -> bool:
    return polarity == "any" or (polarity == "rising") == (strength > 0)


class ScanTool(Tool):
    """Tools that scan a profile along `direction` ("x" or "y")."""

    def scan(self, patch: np.ndarray) -> np.ndarray:
        return profile(self.gray(patch), self.direction)

    def scale(self, px: float) -> float:
        return px * self.mm_per_px if self.mm_per_px else px

    @property
    def unit(self) -> str:
        return "mm" if self.mm_per_px else "px"

    def edge_line(self, pos: float, patch: np.ndarray, ok: bool, label: str = "") -> dict:
        h, w = patch.shape[:2]
        pts = [pos, 0, pos, h] if self.direction == "x" else [0, pos, w, pos]
        return overlay_item("line", pts, ok, label)


@register("width")
class WidthTool(ScanTool):
    """Distance between two edges along the scan direction."""

    PARAMS = {"direction": "x", "threshold": 20, "polarity_1": "any", "polarity_2": "any",
              "edge_pair": "outer", "smooth": 1.0}

    def pick_pair(self, edges: List[Edge]) -> Optional[Tuple[Edge, Edge]]:
        if self.edge_pair == "strongest":
            top = sorted(sorted(edges, key=lambda e: -abs(e[1]))[:2])
            pair = (top[0], top[1]) if len(top) == 2 else None
            if pair and _matches(pair[0][1], self.polarity_1) and _matches(pair[1][1], self.polarity_2):
                return pair
            return None
        if self.edge_pair == "first_rising_falling":
            for i, first in enumerate(edges):
                if first[1] > 0:
                    nxt = next((e for e in edges[i + 1:] if e[1] < 0), None)
                    return (first, nxt) if nxt else None
            return None
        first = next((e for e in edges if _matches(e[1], self.polarity_1)), None)
        last = next((e for e in reversed(edges) if _matches(e[1], self.polarity_2)), None)
        return (first, last) if first and last and last[0] > first[0] else None

    def process(self, patch, origin):
        edges = find_edges(self.scan(patch), self.threshold, "any", smooth=self.smooth)
        pair = self.pick_pair(edges)
        if pair is None:
            return self.fail("edge not found")
        (e1, _), (e2, _) = pair
        px = e2 - e1
        value = self.scale(px)
        h, w = patch.shape[:2]
        text = f"{value:.2f} {self.unit}"
        mid = h / 2 if self.direction == "x" else w / 2
        dim = [e1, mid, e2, mid] if self.direction == "x" else [mid, e1, mid, e2]
        res = self.result(value, text=text, extra={"edge1": e1, "edge2": e2, "px": px})
        res.overlay = [self.edge_line(e1, patch, res.ok), self.edge_line(e2, patch, res.ok),
                       overlay_item("line", dim, res.ok, text)]
        return res


@register("edge_presence")
class EdgePresenceTool(ScanTool):
    """Strength of the strongest edge of the wanted polarity; OK when it
    reaches `low` (default 20 gray levels per px)."""

    PARAMS = {"direction": "x", "polarity": "any", "smooth": 1.0}
    DEFAULT_LOW = 20.0

    def process(self, patch, origin):
        g = gradient(self.scan(patch), self.smooth)
        if len(g) == 0:
            return self.fail("empty ROI")
        signed = {"rising": g, "falling": -g}.get(self.polarity, np.abs(g))
        i = int(np.argmax(signed))
        value = max(float(signed[i]), 0.0)
        res = self.result(value, extra={"position": i + 0.5})
        res.overlay = [self.edge_line(i + 0.5, patch, res.ok, f"{value:.0f}")]
        return res
