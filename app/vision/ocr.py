"""OCR read: prepares the ROI (shading correction, scale, binarize, deskew)
and reads it with Tesseract. The engine sits behind `OcrEngine` so another
one can replace it.

TODO: add a PaddleOCR engine (better on curved / low-contrast print); it only
has to implement `OcrEngine.read`.
"""

import math
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .base import Tool, overlay_item, register
from .preprocess import to_gray

TARGET_CHAR_PX = 32
MAX_SCALE = 4.0
MAX_DESKEW_DEG = 15.0
DEFAULT_OCR_PRE = [{"op": "flat_field", "kernel": 51}, {"op": "clahe", "clip": 2.0, "tiles": 8}]


class OcrEngine(Protocol):
    def read(self, binary: np.ndarray, psm: int = 7, whitelist: str = "") -> Tuple[str, float]:
        """Text and mean confidence (0..100) of a dark-on-light image."""


class TesseractEngine:
    def __init__(self, module) -> None:
        self._tess = module

    def read(self, binary: np.ndarray, psm: int = 7, whitelist: str = "") -> Tuple[str, float]:
        config = f"--psm {int(psm)}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        data = self._tess.image_to_data(binary, config=config, output_type=self._tess.Output.DICT)
        lines, confs = {}, []
        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if not word.strip() or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word.strip())
            confs.append(conf)
        text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
        return text, float(np.mean(confs)) if confs else 0.0


def get_engine() -> Optional[OcrEngine]:
    from ..optional_deps import TESSERACT
    return TesseractEngine(TESSERACT) if TESSERACT is not None else None


def _binarize_dark_on_light(gray: np.ndarray) -> np.ndarray:
    """Otsu, inverted when needed so the text (the minority) ends up dark."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(binary) < binary.size / 2:
        binary = 255 - binary
    return binary


def estimate_char_height(binary: np.ndarray) -> float:
    """Median height of the dark blobs (characters) of a dark-on-light image."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(255 - binary, connectivity=8)
    heights = [s[cv2.CC_STAT_HEIGHT] for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 6]
    return float(np.median(heights)) if heights else 0.0


def estimate_skew(binary: np.ndarray) -> float:
    """Baseline angle in degrees (image y down, positive = text slopes down to
    the right) from the principal axis of the dark pixels; 0 when unclear."""
    ys, xs = np.nonzero(binary == 0)
    if len(xs) < 20:
        return 0.0
    cov = np.cov(np.vstack([xs, ys]).astype(np.float64))
    vals, vecs = np.linalg.eigh(cov)
    if vals[1] < 2.0 * vals[0]:
        return 0.0  # not elongated: no reliable direction
    vx, vy = vecs[:, 1]
    angle = math.degrees(math.atan2(vy, vx))
    angle = (angle + 90) % 180 - 90  # fold into -90..90
    return angle if abs(angle) < MAX_DESKEW_DEG else 0.0


def deskew(binary: np.ndarray) -> np.ndarray:
    angle = estimate_skew(binary)
    if abs(angle) < 0.2:
        return binary
    pad = 10
    padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    h, w = padded.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(padded, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return _binarize_dark_on_light(out)


def prepare_for_ocr(gray: np.ndarray, char_height_px: float = 0) -> np.ndarray:
    """Scale up small print, binarize dark-on-light, straighten a tilted line.
    `gray` has already been through the tool's `pre` chain."""
    binary = _binarize_dark_on_light(gray)
    height = char_height_px or estimate_char_height(binary)
    if 0 < height < TARGET_CHAR_PX:
        scale = min(TARGET_CHAR_PX / height, MAX_SCALE)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        binary = _binarize_dark_on_light(gray)
    return deskew(binary)


@register("ocr")
class OcrTool(Tool):
    """Reads the text in the ROI; the value is the mean confidence (0..100)."""

    PARAMS = {"psm": 7, "whitelist": "", "char_height_px": 0}
    DEFAULT_PRE = DEFAULT_OCR_PRE
    DEFAULT_LOW = 60.0
    engine: Optional[OcrEngine] = None  # set to override (tests, another engine)

    def prepare(self, patch: np.ndarray) -> np.ndarray:
        return prepare_for_ocr(self.gray(patch), self.char_height_px)

    def read(self, patch: np.ndarray) -> Tuple[str, float]:
        engine = self.engine or get_engine()
        if engine is None:
            raise RuntimeError("OCR engine not installed")
        return engine.read(self.prepare(patch), self.psm, self.whitelist)

    def process(self, patch, origin):
        try:
            text, conf = self.read(patch)
        except RuntimeError as exc:
            return self.fail(str(exc))
        res = self.result(conf, text=text, extra={"conf": conf})
        res.overlay = [overlay_item("text", [2, 14], res.ok, text)]
        return res
