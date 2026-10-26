"""Detection area (ROI): the part of the camera frame the model looks at.

Stored as fractions of the frame (x1, y1, x2, y2 in 0..1), so a job's ROI
survives a change of camera resolution. None = the whole frame.
"""

from typing import List, Optional, Sequence, Tuple

Roi = Tuple[float, float, float, float]
PixelBox = Tuple[int, int, int, int]

# Smaller areas are taken as a mis-tap, not a detection area.
MIN_ROI_FRACTION = 0.03
# The crop handed to the model is never smaller than this, in pixels.
MIN_ROI_PX = 32


def normalize_roi(roi: Optional[Sequence[float]]) -> Optional[Roi]:
    """Validates a stored / drawn ROI: clamps to the frame, orders the
    corners, and returns None for no ROI, a too-small area or the full frame."""
    if roi is None:
        return None
    try:
        x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in roi)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x2 - x1 < MIN_ROI_FRACTION or y2 - y1 < MIN_ROI_FRACTION:
        return None
    if (x1, y1, x2, y2) == (0.0, 0.0, 1.0, 1.0):
        return None
    return (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))


def roi_to_pixels(roi: Roi, width: int, height: int) -> PixelBox:
    """ROI -> pixel box inside a width x height frame, at least MIN_ROI_PX
    on each side (or the whole frame if it's smaller than that)."""
    x1, y1 = int(roi[0] * width), int(roi[1] * height)
    x2, y2 = int(round(roi[2] * width)), int(round(roi[3] * height))
    min_w, min_h = min(MIN_ROI_PX, width), min(MIN_ROI_PX, height)
    if x2 - x1 < min_w:
        x1 = max(0, min(x1, width - min_w))
        x2 = x1 + min_w
    if y2 - y1 < min_h:
        y1 = max(0, min(y1, height - min_h))
        y2 = y1 + min_h
    return x1, y1, x2, y2


def offset_detections(detections: List[dict], dx: int, dy: int) -> None:
    """Moves crop-relative boxes back into frame coordinates, in place."""
    for det in detections:
        det["x1"] += dx
        det["x2"] += dx
        det["y1"] += dy
        det["y2"] += dy


def detections_in_crop(detections: List[dict], box: PixelBox) -> List[dict]:
    """Copies of the detections relative to `box`, clipped to it; boxes
    entirely outside are dropped."""
    bx1, by1, bx2, by2 = box
    out = []
    for det in detections:
        x1, y1 = max(det["x1"], bx1) - bx1, max(det["y1"], by1) - by1
        x2, y2 = min(det["x2"], bx2) - bx1, min(det["y2"], by2) - by1
        if x2 > x1 and y2 > y1:
            out.append({**det, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out
