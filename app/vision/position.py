"""Position Adjustment: finds the part inside a search area and reports how
far it moved from where it was taught, so every other tool's ROI can follow.

2-D mode is normalised template matching (coarse pass on a half-scale
pyramid, refined at full scale), optionally over a range of angles. The
1-axis "hi-speed" modes track the strongest edge of a row / column profile.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..roi import roi_to_pixels
from .base import Offset, Tool, ToolResult, overlay_item, register, rotate_points

REFINE_PX = 4  # full-scale search window around the coarse hit
MIN_PYRAMID_TEMPLATE = 16  # a half-scale template needs at least this many px per side


def _parabola(left: float, mid: float, right: float) -> float:
    """Sub-sample offset (-0.5..0.5) of the peak through three samples."""
    denom = left - 2 * mid + right
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def _best_match(search: np.ndarray, templ: np.ndarray) -> Tuple[float, int, int, np.ndarray]:
    res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
    res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
    _, score, _, (x, y) = cv2.minMaxLoc(res)
    return float(score), x, y, res


def _subpixel(res: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    fx = _parabola(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < res.shape[1] - 1 else 0.0
    fy = _parabola(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < res.shape[0] - 1 else 0.0
    return fx, fy


def _rotated(templ: np.ndarray, angle: float) -> np.ndarray:
    """The template turned by `angle` (counter-clockwise), corners filled with its mean."""
    if angle == 0:
        return templ
    h, w = templ.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(templ, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=float(templ.mean()))


def _strongest_edge(profile: np.ndarray, sign: int = 0) -> Optional[Tuple[float, float, int]]:
    """(position, strength, sign) of the steepest step in a 1-D profile.
    sign 0 = either direction, +1 rising only, -1 falling only."""
    p = cv2.GaussianBlur(profile.astype(np.float32).reshape(1, -1), (0, 0), 1.5).ravel()
    g = np.gradient(p)
    score = np.abs(g) if sign == 0 else g * sign
    i = int(np.argmax(score))
    if score[i] <= 0:
        return None
    pos = float(i)
    if 0 < i < len(g) - 1:
        pos += _parabola(abs(g[i - 1]), abs(g[i]), abs(g[i + 1]))
    return pos, float(score[i]), 1 if g[i] > 0 else -1


@register("position")
class PositionTool(Tool):
    """Its ROI is the search area; `template_roi` (frame fractions) is the
    part feature taught inside it."""

    PARAMS = {
        "mode": "2d", "template_roi": None, "angle_range": 0.0, "angle_step": 1.0,
        "min_score": 0.6, "pyramid": True,
        "taught_center": None, "ref_edge": None, "ref_strength": None, "ref_sign": 0,
    }

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self._last = Offset(found=False, score=0.0)

    def _search(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi_to_pixels(self.roi, w, h)
        return self.gray(frame[y1:y2, x1:x2]), (x1, y1)

    def teach(self, frame: np.ndarray, offset: Optional[Offset] = None, template_roi=None, **kw) -> None:
        if template_roi is not None:
            self.template_roi = [float(v) for v in template_roi]
        search, (sx, sy) = self._search(frame)
        if search.size == 0:
            return
        h, w = frame.shape[:2]
        if self.mode in ("x", "y"):
            prof = search.mean(axis=0 if self.mode == "x" else 1)
            edge = _strongest_edge(prof)
            if edge is not None:
                self.ref_edge = edge[0] + (sx if self.mode == "x" else sy)
                self.ref_strength, self.ref_sign = edge[1], edge[2]
            return
        if self.template_roi is None:
            sh, sw = search.shape[:2]
            tx1, ty1 = sx + sw // 4, sy + sh // 4
            box = (tx1, ty1, tx1 + sw // 2, ty1 + sh // 2)
            self.template_roi = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
        x1, y1, x2, y2 = roi_to_pixels(tuple(self.template_roi), w, h)
        # The template is cut from the same filtered image the search uses.
        full = self.gray(frame[sy:sy + search.shape[0], sx:sx + search.shape[1]])
        self.template = full[max(0, y1 - sy):y2 - sy, max(0, x1 - sx):x2 - sx].copy()
        self.taught_center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def offset(self) -> Offset:
        """The offset found by the last `run`."""
        return self._last

    def run(self, frame: np.ndarray, offset: Optional[Offset] = None) -> ToolResult:
        if frame is None or frame.size == 0:
            return self._publish(self.fail("empty frame"), Offset(found=False, score=0.0))
        search, origin = self._search(frame)
        if search.size == 0:
            return self._publish(self.fail("empty search area"), Offset(found=False, score=0.0))
        if self.mode in ("x", "y"):
            found = self._run_axis(search, origin)
        else:
            found = self._run_2d(search, origin)
        if isinstance(found, str):
            return self._publish(self.fail(found), Offset(found=False, score=0.0))
        return self._publish(*found)

    def _publish(self, result: ToolResult, off: Offset) -> ToolResult:
        self._last = off
        result.extra.update(dx=off.dx, dy=off.dy, angle=off.angle, score=off.score, found=off.found)
        return result

    def _run_2d(self, search: np.ndarray, origin: Tuple[int, int]):
        templ = self.template
        if templ is None or self.taught_center is None:
            return "not taught"
        th, tw = templ.shape[:2]
        if search.shape[0] < th or search.shape[1] < tw:
            return "search area smaller than template"
        if float(search.std()) < 1.0:
            return "no part"
        angles = [0.0]
        if self.angle_range > 0:
            step = max(float(self.angle_step), 0.1)
            n = int(self.angle_range // step)
            angles = [i * step for i in range(-n, n + 1)]
        use_pyr = self.pyramid and min(th, tw) >= 2 * MIN_PYRAMID_TEMPLATE
        small = cv2.resize(search, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) if use_pyr else search
        best = (-2.0, 0, 0, 0.0)  # score, x, y (in `small` pixels), angle
        for angle in angles:
            t = _rotated(templ, angle)
            if use_pyr:
                t = cv2.resize(t, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if small.shape[0] < t.shape[0] or small.shape[1] < t.shape[1]:
                continue
            score, x, y, _ = _best_match(small, t)
            if score > best[0]:
                best = (score, x, y, angle)
        if best[0] < -1.5:
            return "search area smaller than template"
        _, x, y, angle = best
        if use_pyr:
            x, y = x * 2, y * 2
        t = _rotated(templ, angle)
        x0, y0 = max(0, x - REFINE_PX), max(0, y - REFINE_PX)
        x1 = min(search.shape[1], x + tw + REFINE_PX)
        y1 = min(search.shape[0], y + th + REFINE_PX)
        score, rx, ry, res = _best_match(search[y0:y1, x0:x1], t)
        fx, fy = _subpixel(res, rx, ry)
        cx = origin[0] + x0 + rx + fx + tw / 2.0
        cy = origin[1] + y0 + ry + fy + th / 2.0
        off = Offset(cx - self.taught_center[0], cy - self.taught_center[1], angle, score, score >= self.min_score)
        corners = [(cx - tw / 2, cy - th / 2), (cx + tw / 2, cy - th / 2), (cx + tw / 2, cy + th / 2), (cx - tw / 2, cy + th / 2)]
        poly = [list(p) for p in rotate_points(corners, (cx, cy), angle)]
        overlay = self._search_overlay(search, origin, off.found) + [
            overlay_item("poly", poly, off.found, f"{score * 100:.0f}%"),
            overlay_item("line", [cx - 8, cy, cx + 8, cy], off.found),
            overlay_item("line", [cx, cy - 8, cx, cy + 8], off.found),
        ]
        return self.result(score * 100, overlay, ok=off.found), off

    def _run_axis(self, search: np.ndarray, origin: Tuple[int, int]):
        if self.ref_edge is None or not self.ref_strength:
            return "not taught"
        axis_x = self.mode == "x"
        prof = search.mean(axis=0 if axis_x else 1)
        edge = _strongest_edge(prof, int(self.ref_sign))
        if edge is None:
            return "edge not found"
        pos = edge[0] + (origin[0] if axis_x else origin[1])
        score = min(1.0, edge[1] / self.ref_strength)
        shift = pos - self.ref_edge
        off = Offset(shift if axis_x else 0.0, 0.0 if axis_x else shift, 0.0, score, score >= self.min_score)
        sh, sw = search.shape[:2]
        if axis_x:
            line = [pos, origin[1], pos, origin[1] + sh]
        else:
            line = [origin[0], pos, origin[0] + sw, pos]
        overlay = self._search_overlay(search, origin, off.found) + [overlay_item("line", line, off.found, f"{shift:+.1f}px")]
        return self.result(score * 100, overlay, ok=off.found), off

    def _search_overlay(self, search: np.ndarray, origin: Tuple[int, int], ok: bool) -> List[dict]:
        h, w = search.shape[:2]
        return [overlay_item("rect", [origin[0], origin[1], origin[0] + w, origin[1] + h], ok, self.name)]


def offset_from_result(result: Optional[ToolResult]) -> Offset:
    """The Offset a position result carries (no part when there is none)."""
    if result is None:
        return Offset()
    e = result.extra
    return Offset(e.get("dx", 0.0), e.get("dy", 0.0), e.get("angle", 0.0), e.get("score", 0.0), bool(e.get("found", False)))
