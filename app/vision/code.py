"""Code tool: reads a QR / 1-D code in the ROI with pyzbar (a cv2 QR detector
is the fallback) and checks it against an expected text or pattern. IV4 has
no such tool; UniversalV1 already reads codes for the part-number scan."""

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .. import optional_deps
from ..barcode import decode_scanned_text
from .base import Tool, overlay_item, register

Decoded = Tuple[str, str, List[List[float]]]  # text, symbology, polygon


def _zbar(gray: np.ndarray, symbology: str) -> Optional[Decoded]:
    if optional_deps.zbar_decode is None:
        return None
    for r in optional_deps.zbar_decode(gray):
        kind = str(r.type)
        if symbology != "any" and symbology.lower().replace("-", "") not in kind.lower().replace("-", ""):
            continue
        text = r.data.decode("utf-8", "replace")
        return text, kind, [[float(p.x), float(p.y)] for p in r.polygon]
    return None


def _cv_qr(gray: np.ndarray) -> Optional[Decoded]:
    try:
        text, points, _ = cv2.QRCodeDetector().detectAndDecode(gray)
    except cv2.error:
        return None
    if not text:
        return None
    poly = [[float(x), float(y)] for x, y in points.reshape(-1, 2)] if points is not None else []
    return text, "QRCODE", poly


def read_code(gray: np.ndarray, symbology: str = "any") -> Optional[Decoded]:
    """Tries pyzbar, then (only when the symbology is set to QR) OpenCV's
    detector, then both again at 2x scale. With "any", pyzbar alone: it reads
    QR too, and the OpenCV detector costs about 16 ms on every miss."""
    wants_qr = symbology.lower() in ("qr", "qrcode")
    for scale in (1, 2):
        img = gray if scale == 1 else cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        found = _zbar(img, symbology) or (_cv_qr(img) if wants_qr else None)
        if found:
            text, kind, poly = found
            return text, kind, [[x / scale, y / scale] for x, y in poly]
    return None


@register("code")
class CodeTool(Tool):
    """Value 1 when a code was read (0 otherwise); `text` is the decoded string."""

    PARAMS = {"symbology": "any", "expected": "", "pattern": "", "use_job_part_no": False}
    DEFAULT_LOW = 1.0

    def process(self, patch, origin):
        found = read_code(np.ascontiguousarray(self.gray(patch)), self.symbology)
        if found is None:
            return self.result(0, text="", extra={"reason": "no code"})
        raw, kind, poly = found
        text = decode_scanned_text(raw)
        expected = self.part_no if self.use_job_part_no else self.expected
        reason = ""
        if self.use_job_part_no and not expected:
            reason = "no job part number"
        elif expected and text != expected:
            reason = "text differs from expected"
        elif self.pattern:
            try:
                if re.fullmatch(self.pattern, text) is None:
                    reason = "text does not match pattern"
            except re.error:
                reason = "bad pattern"
        res = self.result(1, text=text, extra={"reason": reason, "symbology": kind, "raw": raw}, ok=not reason)
        if poly:
            res.overlay.append(overlay_item("poly", poly, res.ok, text))
        else:
            res.overlay.append(overlay_item("text", [2, 14], res.ok, text))
        return res
