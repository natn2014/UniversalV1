"""mm calibration: one job-level `mm_per_px`, taught from a feature of known size."""

from typing import Optional

import numpy as np

from ..roi import Roi
from .measure import WidthTool


def calibrate_from_width(frame: np.ndarray, roi: Roi, known_mm: float, direction: str = "x") -> Optional[float]:
    """mm per pixel from a width of `known_mm` inside `roi`; None if no
    width could be measured or the size is not positive."""
    if known_mm is None or known_mm <= 0:
        return None
    res = WidthTool(roi=roi, direction=direction, follow_position=False).run(frame)
    px = res.extra.get("px") if res.ok else None
    return known_mm / px if px else None
