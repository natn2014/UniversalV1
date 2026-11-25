"""Camera image adjustments (Settings > Camera): mirror / flip for how the
camera is mounted, and colour enhancement. Applied by the video worker right
after each frame is read, so the AI, the tools, captures and the Top View
editor all see the same image. Steps at their neutral value are skipped."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageAdjust:
    mirror: bool = False  # left-right
    flip: bool = False  # upside down
    brightness: int = 0  # -100..100, added to every channel
    contrast: int = 100  # %, 50..200
    saturation: int = 100  # %, 0..200
    gamma: float = 1.0  # 0.5..2.0; above 1 lifts the shadows
    clahe: bool = False  # local contrast (CLAHE on the lightness)

    @property
    def is_neutral(self) -> bool:
        return self == ImageAdjust()

    def to_settings(self) -> dict:
        return {f"camera_{k}": v for k, v in asdict(self).items()}

    @classmethod
    def from_settings(cls, settings) -> "ImageAdjust":
        return cls(**{k: settings[f"camera_{k}"] for k in asdict(cls())})

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """BGR frame in, adjusted BGR frame out (the input is not modified)."""
        if self.mirror and self.flip:
            frame = cv2.flip(frame, -1)
        elif self.mirror:
            frame = cv2.flip(frame, 1)
        elif self.flip:
            frame = cv2.flip(frame, 0)
        if self.contrast != 100 or self.brightness:
            frame = cv2.convertScaleAbs(frame, alpha=self.contrast / 100.0, beta=float(self.brightness))
        if self.saturation != 100:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=self.saturation / 100.0)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if self.gamma != 1.0:
            frame = cv2.LUT(frame, _gamma_lut(self.gamma))
        if self.clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[..., 0] = _CLAHE.apply(lab[..., 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_LUTS: dict = {}


def _gamma_lut(gamma: float) -> np.ndarray:
    key = round(gamma, 2)
    if key not in _LUTS:
        _LUTS[key] = np.array([255 * (i / 255) ** (1 / key) for i in range(256)]).clip(0, 255).astype(np.uint8)
    return _LUTS[key]
