"""Pattern Search (Keyence ShapeTrax style): finds every copy of a taught
pattern inside the ROI, at any angle, and counts them.

Matching runs on edge-strength maps (smoothed Sobel magnitude), not on gray
levels, so the outline of the pattern is what counts and a change of lighting
or background shade matters little. A coarse pass over every angle on a
downscaled image collects candidates; each is then refined at full scale
around its angle and position. Found copies are drawn as the pattern's own
edges, turned and placed where they were found.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import Offset, Tool, overlay_item, region_item, register

MIN_COARSE_TEMPLATE = 12  # px (square root of the area) the downscaled template keeps at least
COARSE_SLACK = 0.85  # coarse candidates may score this much lower than min_score
COARSE_BLUR = 1.0  # extra sigma on the coarse maps, so a coarser angle step still hits
COARSE_TOL_PX = 2.0  # how far a corner may be off at the coarse scale between two angles
MAX_COARSE_STEP = 12.0  # degrees
MAX_REFINE_ANGLES = 9
BORDER_BAND_PX = 3
EDGE_BLUR = 1.5  # sigma: tolerance of about a pixel for edges that moved a little
# OpenCV releases the GIL while matching, so the angles run side by side.
_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="pattern-search")


def edge_strength(gray: np.ndarray, edge_min: float) -> np.ndarray:
    """Smoothed gradient magnitude (float32), with weak texture below
    `edge_min` set to zero."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    mag[mag < edge_min] = 0
    return cv2.GaussianBlur(mag, (0, 0), EDGE_BLUR)


def padded_square(img: np.ndarray) -> np.ndarray:
    """`img` centred on a zero square as wide as its diagonal, so turning it
    never cuts a corner off (zero = no edge, which matches nothing)."""
    h, w = img.shape[:2]
    d = int(math.ceil(math.hypot(w, h))) | 1
    out = np.zeros((d, d), img.dtype)
    y0, x0 = (d - h) // 2, (d - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def turned(img: np.ndarray, angle: float) -> np.ndarray:
    """Counter-clockwise turn about the centre (same sense as Position)."""
    if angle == 0:
        return img
    d = img.shape[0]
    m = cv2.getRotationMatrix2D(((d - 1) / 2.0, (d - 1) / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (d, d), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _match(search: np.ndarray, templ: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalised correlation over the turned pattern's footprint only: the
    padding around it must not be compared with whatever is next to a copy.
    (TM_CCORR_NORMED would be cheaper, but on edge maps that are mostly zero it
    scores near 1 in empty background.)"""
    res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED, mask=mask)
    res[~np.isfinite(res)] = 0.0  # a flat window divides by zero under a mask
    return np.clip(res, -1.0, 1.0, out=res)


def _footprint(shape: Tuple[int, int], angle: float, scale: float = 1.0) -> np.ndarray:
    """0/1 float mask of the template rectangle, padded and turned. A band
    along its border is left out: the edge filter sees the crop's border
    there, where the live image carries on, so those pixels never match."""
    h, w = shape
    band = BORDER_BAND_PX if min(h, w) > 4 * BORDER_BAND_PX else 1
    inner = np.zeros(shape, np.float32)
    inner[band:h - band, band:w - band] = 1
    m = turned(padded_square(inner), angle)
    if scale < 1:
        m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return (m > 0.5).astype(np.float32)


@register("pattern_search")
class PatternSearchTool(Tool):
    """Its ROI is the search area; `template_roi` (frame fractions) is the
    pattern taught inside it. The value is the number of copies found."""

    PARAMS = {
        "template_roi": None, "angle_range": 180.0, "angle_step": 5.0,
        "min_score": 0.88, "max_count": 10, "edge_min": 20.0, "overlap": 0.5,
    }
    DEFAULT_LOW = 1.0
    SLOW = True  # a search over every angle is heavy: runs in the slow-tool thread, like OCR

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self._cache: Tuple = (None, None)

    # --- teaching -------------------------------------------------------

    def teach(self, frame: np.ndarray, offset: Optional[Offset] = None, template_roi=None, **kw) -> None:
        """Cuts the pattern from the filtered ROI: the dragged `template_roi`,
        or else the middle half of the ROI."""
        if template_roi is not None:
            self.template_roi = [float(v) for v in template_roi]
        patch, (sx, sy) = self.crop(frame, offset)
        if patch.size == 0:
            return
        gray = self.gray(patch)
        h, w = frame.shape[:2]
        if self.template_roi is None:
            ph, pw = gray.shape[:2]
            box = (sx + pw // 4, sy + ph // 4, sx + pw // 4 + pw // 2, sy + ph // 4 + ph // 2)
            self.template_roi = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
        # Exact pixels: roi_to_pixels would grow a small mark to its 32 px minimum.
        fx1, fy1, fx2, fy2 = self.template_roi
        x1, y1, x2, y2 = int(round(fx1 * w)), int(round(fy1 * h)), int(round(fx2 * w)), int(round(fy2 * h))
        x1, y1 = max(x1, sx), max(y1, sy)
        x2, y2 = min(x2, sx + gray.shape[1]), min(y2, sy + gray.shape[0])
        if x2 - x1 < 4 or y2 - y1 < 4:
            return
        self.template = np.ascontiguousarray(gray[y1 - sy:y2 - sy, x1 - sx:x2 - sx])

    # --- derived templates (cached) -------------------------------------

    def _coarse_step(self, scale: float) -> float:
        """Angle step of the coarse pass: as coarse as the pattern's size
        allows (a small pattern turns little at its corners), never finer
        than `angle_step`. The full-scale pass then finds the exact angle."""
        radius = 0.5 * math.hypot(*self.template.shape[:2]) * scale
        tolerated = math.degrees(math.atan(COARSE_TOL_PX / max(radius, 1.0)))
        return max(float(self.angle_step), min(MAX_COARSE_STEP, tolerated))

    def _angles(self, step: float) -> List[float]:
        rng = float(np.clip(self.angle_range, 0, 180))
        step = max(step, 0.5)
        if rng <= 0:
            return [0.0]
        n = int(rng // step)
        angles = [i * step for i in range(-n, n + 1)]
        if rng >= 180:  # -180 and +180 are the same turn
            angles = [a for a in angles if a > -180]
        return angles

    def _scale(self) -> float:
        side = math.sqrt(self.template.shape[0] * self.template.shape[1])  # long text shrinks too
        scale = 1.0
        while side * scale / 2 >= MIN_COARSE_TEMPLATE and scale > 0.125:
            scale /= 2
        return scale

    def _derived(self) -> Dict:
        key = (id(self.template), self.template.shape, self.edge_min, self.angle_range, self.angle_step)
        if self._cache[0] != key:
            feat = padded_square(edge_strength(self.template, self.edge_min))
            scale = self._scale()
            step = self._coarse_step(scale)
            coarse = {}
            for angle in self._angles(step):
                t = turned(feat, angle)
                if scale < 1:
                    t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                coarse[angle] = (cv2.GaussianBlur(t, (0, 0), COARSE_BLUR),
                                 _footprint(self.template.shape[:2], angle, scale))
            # What is drawn at each copy: the pattern's own (thin) Canny edges.
            outline = padded_square(cv2.Canny(cv2.GaussianBlur(self.template, (3, 3), 0), 80, 200))
            self._cache = (key, {"feat": feat, "scale": scale, "step": step, "coarse": coarse, "outline": outline})
        return self._cache[1]

    # --- searching ------------------------------------------------------

    def process(self, patch, origin):
        if self.template is None:
            return self.fail("not taught")
        d = self._derived()
        feat = d["feat"]
        size = feat.shape[0]
        search = edge_strength(self.gray(patch), self.edge_min)
        if search.shape[0] < size or search.shape[1] < size:
            return self.fail("search area smaller than the turned pattern")
        if float(feat.max()) <= 0:
            return self.fail("pattern has no edges: teach again")
        if float(search.max()) <= 0:
            return self.result(0.0, extra={"matches": []})
        candidates = self._coarse(search, d)
        matches = self._refine(search, feat, candidates, d)
        overlay = []
        for i, m in enumerate(matches, 1):
            x, y = int(round(m["x"] - size / 2)), int(round(m["y"] - size / 2))
            overlay.append({**region_item(turned(d["outline"], m["angle"]), color="lines"), "pts": [x, y]})
            reach = min(self.template.shape[:2]) / 2  # just above the mark itself, not its padded square
            overlay.append(overlay_item("text", [m["x"] - reach, m["y"] - reach - 4], True,
                                        f"{i}: {m['score'] * 100:.0f}% {m['angle']:+.0f}°"))
        res = self.result(len(matches), overlay, extra={"matches": matches})
        for item in res.overlay:
            item["ok"] = res.ok
        return res

    def _coarse(self, search: np.ndarray, d: Dict) -> List[Tuple[float, float, float, float]]:
        """(score, cx, cy, angle) in full-scale search pixels, best first,
        one per place (the best angle there)."""
        scale = d["scale"]
        small = search if scale >= 1 else cv2.resize(search, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), COARSE_BLUR)
        jobs = [(angle, t, m) for angle, (t, m) in d["coarse"].items()
                if small.shape[0] >= t.shape[0] and small.shape[1] >= t.shape[1]]
        results = _POOL.map(lambda job: (job[0], _match(small, job[1], job[2])), jobs)
        best_score, best_angle = None, None
        for angle, res in results:
            if best_score is None:
                best_score, best_angle = res, np.full(res.shape, angle, np.float32)
            else:
                h = min(best_score.shape[0], res.shape[0])
                w = min(best_score.shape[1], res.shape[1])
                better = res[:h, :w] > best_score[:h, :w]
                best_score[:h, :w][better] = res[:h, :w][better]
                best_angle[:h, :w][better] = angle
        if best_score is None:
            return []
        t_side = next(iter(d["coarse"].values()))[0].shape[0]
        radius = max(1, int(t_side * self.overlap / 2))
        floor = self.min_score * COARSE_SLACK
        scores = best_score.copy()
        out = []
        limit = max(1, int(self.max_count)) * 3
        while len(out) < limit:
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score < floor:
                break
            cx, cy = (x + t_side / 2) / scale, (y + t_side / 2) / scale
            out.append((float(score), cx, cy, float(best_angle[y, x])))
            cv2.circle(scores, (x, y), radius, -1.0, -1)
        return out

    def _refine(self, search, feat, candidates, d) -> List[Dict]:
        """Full-scale score around each candidate's place and angle; keeps the
        ones above `min_score` that do not overlap a better one."""
        size = feat.shape[0]
        scale = d["scale"]
        slack = int(math.ceil(2 / scale)) + 2
        half = d["step"] / 2 if self.angle_range > 0 else 0.0
        fine = max(float(self.angle_step), 2 * half / (MAX_REFINE_ANGLES - 1), 0.5)
        offsets = [0.0] if half == 0 else [-half + i * fine for i in range(int(2 * half / fine + 1e-6) + 1)]
        found: List[Dict] = []
        for _, cx, cy, angle in candidates:
            x0 = int(max(0, round(cx - size / 2) - slack))
            y0 = int(max(0, round(cy - size / 2) - slack))
            x1 = int(min(search.shape[1], round(cx + size / 2) + slack + 1))
            y1 = int(min(search.shape[0], round(cy + size / 2) + slack + 1))
            window = search[y0:y1, x0:x1]
            if window.shape[0] < size or window.shape[1] < size:
                continue
            best = None
            for a in (angle + o for o in offsets):
                res = _match(window, turned(feat, a), _footprint(self.template.shape[:2], a))
                _, score, _, (x, y) = cv2.minMaxLoc(res)
                if best is None or score > best[0]:
                    best = (float(score), x0 + x + size / 2.0, y0 + y + size / 2.0, a)
            score, mx, my, a = best
            if score < self.min_score:
                continue
            if any(math.hypot(mx - f["x"], my - f["y"]) < min(self.template.shape[:2]) * self.overlap
                   for f in found):
                continue
            found.append({"x": mx, "y": my, "angle": (a + 180) % 360 - 180, "score": score})
        found.sort(key=lambda f: -f["score"])
        return found[:max(1, int(self.max_count))]
