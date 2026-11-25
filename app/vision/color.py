"""Color tools: Color Area, Color Average (ΔE2000) and Color Prohibit.

Color Area and Color Prohibit share an HSV range (`HSV_PARAMS`); hue is
circular, so a range with h_lo > h_hi wraps through 0 (red).
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from ..roi import roi_to_pixels
from .base import Offset, Tool, overlay_item, region_item, register
from .preprocess import to_hsv


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


HSV_PARAMS = {"h_lo": 170, "h_hi": 10, "s_lo": 80, "s_hi": 255, "v_lo": 50, "v_hi": 255}
MIN_HALF_WIDTH = {"h": 8.0, "s": 40.0, "v": 40.0}
POINT_BOX = 9  # px side of the box sampled around a tapped point
WIDEN_STEP = {"h": 3, "s": 10, "v": 10}  # one [+] / [-] step, each side
FULL_HUE = {"h_lo": 0, "h_hi": 179}


def point_sample(frame: np.ndarray, point: Tuple[int, int]) -> np.ndarray:
    """The POINT_BOX square around a tapped point (clipped to the frame)."""
    x, y = int(point[0]), int(point[1])
    r = POINT_BOX // 2
    return frame[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]


def _hue_arc(lo: int, hi: int) -> Tuple[int, int]:
    """(start, length) of a hue range on the 0..179 circle."""
    return lo, (hi - lo) % 180


def _is_full_hue(r: dict) -> bool:
    return _hue_arc(r["h_lo"], r["h_hi"])[1] >= 179


def merge_hsv_range(a: dict, b: dict) -> dict:
    """Smallest HSV range covering both: the shorter hue arc that holds both
    arcs, and the union of the S and V spans."""
    out = {}
    if _is_full_hue(a) or _is_full_hue(b):
        out.update(FULL_HUE)
    else:
        arcs = [_hue_arc(a["h_lo"], a["h_hi"]), _hue_arc(b["h_lo"], b["h_hi"])]
        best = min(((start, max((s - start) % 180 + n for s, n in arcs)) for start, _ in arcs),
                   key=lambda c: c[1])
        out.update(FULL_HUE if best[1] >= 179 else {"h_lo": best[0], "h_hi": (best[0] + best[1]) % 180})
    for key in ("s", "v"):
        out[f"{key}_lo"] = min(a[f"{key}_lo"], b[f"{key}_lo"])
        out[f"{key}_hi"] = max(a[f"{key}_hi"], b[f"{key}_hi"])
    return out


def widen_hsv_range(r: dict, steps: int) -> dict:
    """The range grown (or shrunk, steps < 0) by WIDEN_STEP per side, never
    narrower than the minimum widths. A full hue circle stays full."""
    out = dict(r)
    if not _is_full_hue(r):
        start, length = _hue_arc(r["h_lo"], r["h_hi"])
        grow = WIDEN_STEP["h"] * steps
        new_len = max(length + 2 * grow, int(MIN_HALF_WIDTH["h"]))
        if new_len >= 179:
            out.update(FULL_HUE)
        else:
            new_start = start - (new_len - length) // 2
            out.update(h_lo=new_start % 180, h_hi=(new_start + new_len) % 180)
    for key in ("s", "v"):
        lo, hi = r[f"{key}_lo"], r[f"{key}_hi"]
        grow = WIDEN_STEP[key] * steps
        new_lo, new_hi = max(0, lo - grow), min(255, hi + grow)
        min_width = int(MIN_HALF_WIDTH[key] / 2)
        if new_hi - new_lo < min_width:  # narrowed past the minimum: keep the middle
            mid = (lo + hi) // 2
            new_lo, new_hi = max(0, mid - min_width // 2), min(255, mid + min_width // 2)
        out[f"{key}_lo"], out[f"{key}_hi"] = int(new_lo), int(new_hi)
    return out


def open_mask(mask: np.ndarray, open_px: int) -> np.ndarray:
    """Morphological open that also removes specks on the ROI edge (an
    erosion with a 0 border; OpenCV's default +inf border keeps them)."""
    if not open_px or open_px <= 1:
        return mask
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_px),) * 2)
    return cv2.dilate(cv2.erode(mask, se, borderType=cv2.BORDER_CONSTANT, borderValue=0), se)


def hsv_mask(hsv: np.ndarray, h_lo: int, h_hi: int, s_lo: int, s_hi: int, v_lo: int, v_hi: int,
             open_px: int = 0) -> np.ndarray:
    """255 where the pixel is inside the HSV range; optionally morphologically opened."""
    lo_sv, hi_sv = (s_lo, v_lo), (s_hi, v_hi)
    if h_lo <= h_hi:
        mask = cv2.inRange(hsv, (h_lo, *lo_sv), (h_hi, *hi_sv))
    else:
        mask = cv2.inRange(hsv, (h_lo, *lo_sv), (179, *hi_sv)) | cv2.inRange(hsv, (0, *lo_sv), (h_hi, *hi_sv))
    return open_mask(mask, open_px)


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

    PICK = "color"

    def hsv_range(self) -> dict:
        return {key: getattr(self, key) for key in HSV_PARAMS}

    def pick(self, frame, point, add=False, offset=None) -> None:
        sample = point_sample(frame, point)
        if sample.size == 0:
            return
        learned = learn_hsv_range(self.hsv(sample), self.k)
        self.set_params(merge_hsv_range(self.hsv_range(), learned) if add else learned)

    def widen(self, steps: int) -> None:
        self.set_params(widen_hsv_range(self.hsv_range(), steps))

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
            sample = point_sample(frame, point)
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
            overlay.append(region_item(mask, True, f"{value:.1f}%"))
        res = self.result(value, overlay)
        for item in res.overlay:
            item["ok"] = res.ok
        return res


def delta_e_2000(lab1, lab2) -> float:
    """CIEDE2000 colour difference between two L*a*b* triples (Sharma 2005)."""
    L1, a1, b1 = (float(v) for v in lab1)
    L2, a2, b2 = (float(v) for v in lab2)
    c_bar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2.0
    g = 0.5 * (1 - math.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if c1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if c2p else 0.0
    dL, dC = L2 - L1, c2p - c1p
    if c1p * c2p == 0:
        dh = 0.0
    else:
        dh = h2p - h1p
        dh -= 360 * round(dh / 360)
    dH = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh) / 2)
    Lp, Cp = (L1 + L2) / 2.0, (c1p + c2p) / 2.0
    if c1p * c2p == 0:
        hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp = (h1p + h2p) / 2.0
    else:
        hp = (h1p + h2p + (360 if h1p + h2p < 360 else -360)) / 2.0
    t = (1 - 0.17 * math.cos(math.radians(hp - 30)) + 0.24 * math.cos(math.radians(2 * hp))
         + 0.32 * math.cos(math.radians(3 * hp + 6)) - 0.20 * math.cos(math.radians(4 * hp - 63)))
    d_theta = 30 * math.exp(-(((hp - 275) / 25.0) ** 2))
    rc = 2 * math.sqrt(Cp ** 7 / (Cp ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (Lp - 50) ** 2 / math.sqrt(20 + (Lp - 50) ** 2)
    sc, sh = 1 + 0.045 * Cp, 1 + 0.015 * Cp * t
    rt = -math.sin(math.radians(2 * d_theta)) * rc
    return math.sqrt((dL / sl) ** 2 + (dC / sc) ** 2 + (dH / sh) ** 2 + rt * (dC / sc) * (dH / sh))


def mean_lab(bgr: np.ndarray) -> list:
    """Mean colour of a BGR patch as real L*a*b* (OpenCV 8-bit Lab is L*255/100, a+128, b+128)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.mean(lab)[:3]
    return [l * 100.0 / 255.0, a - 128.0, b - 128.0]


@register("color_average")
class ColorAverageTool(Tool):
    """ΔE2000 between the ROI's mean color and the taught master color."""

    PARAMS = {"ref_lab": None}
    DEFAULT_HIGH = 5.0

    def teach(self, frame: np.ndarray, offset: Optional[Offset] = None, **kw) -> None:
        patch, _ = self.crop(frame, offset)
        if patch.size:
            self.ref_lab = mean_lab(to_bgr(self.prepared(patch)))

    def process(self, patch, origin):
        if self.ref_lab is None:
            return self.fail("not taught")
        lab = mean_lab(to_bgr(self.prepared(patch)))
        de = delta_e_2000(self.ref_lab, lab)
        res = self.result(de, extra={"lab": lab})
        res.text = "L*a*b* " + " ".join(f"{v:.1f}" for v in lab)
        return res


@register("color_prohibit")
class ColorProhibitTool(HsvTool):
    """NG when a blob of the forbidden color at least `min_blob_px` large
    appears anywhere in the ROI. The value is the largest blob's area (px)."""

    PARAMS = {**HSV_PARAMS, "k": 2.5, "open_px": 2, "min_blob_px": 30}

    def __init__(self, *args, **kw) -> None:
        if len(args) < 4 and "high" not in kw:
            kw["high"] = kw.get("min_blob_px", self.PARAMS["min_blob_px"]) - 1
        super().__init__(*args, **kw)

    def process(self, patch, origin):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(self.mask(patch), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        value = float(areas.max()) if len(areas) else 0.0
        found = np.flatnonzero(areas >= self.min_blob_px)
        overlay = []
        if len(found):
            # The forbidden pixels in red; the boxes keep a tiny spot easy to find.
            keep = np.zeros(n, np.uint8)
            keep[found + 1] = 255
            overlay.append(region_item(keep[labels], False, color="red"))
        for i in found:
            x, y, w, h, area = stats[i + 1]
            overlay.append(overlay_item("rect", [int(x), int(y), int(x + w), int(y + h)], False, f"{int(area)}px"))
        return self.result(value, overlay, extra={"blobs": int(len(found))})
