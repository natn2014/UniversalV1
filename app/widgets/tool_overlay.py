"""Drawing the plain-dict overlays that vision tools return (see
app/vision/base.py) and formatting a result's value for the tool table."""

from typing import List

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from ..theme import NG_RED, OK_GREEN
from ..vision.base import ToolResult

# Verdict colors are dark for chips; on the camera image they need a lift.
OVERLAY_OK = QColor(OK_GREEN).lighter(170)
OVERLAY_NG = QColor(NG_RED).lighter(130)
MM_KINDS = {"width", "diameter"}
# Counted pixels (Area, Color Area): see-through yellow with a darker edge.
TEXT_SHADOW = QColor(10, 10, 10)
# Premultiplied BGRA (the byte order of QImage.Format_ARGB32_Premultiplied)
# for pixel classes 0 = clear, 1 = counted, 2 = edge of the counted area.
# Fills are at alpha 90, so the image shows through.
REGION_PALETTES = {
    "yellow": [(0, 0, 0, 0), (0, 75, 90, 90), (0, 120, 176, 255)],
    "red": [(0, 0, 0, 0), (25, 25, 85, 90), (30, 30, 190, 255)],
    # Edge maps are one pixel wide, so every pixel is "edge": solid bright yellow.
    "lines": [(0, 0, 0, 0), (0, 214, 255, 255), (0, 214, 255, 255)],
}
REGION_LUTS = {name: [np.array([c[i] for c in bgra] + [0] * 253, np.uint8) for i in range(4)]
               for name, bgra in REGION_PALETTES.items()}
EDGE_KERNEL = np.ones((5, 5), np.uint8)  # 2 px edge


def draw_overlay(painter: QPainter, items: List[dict]) -> None:
    """Draws overlay items in frame pixels; green when ok, red when not.
    Labels are placed like the YOLO box labels (above the box corner)."""
    pen = QPen()
    pen.setWidth(2)
    for item in items:
        color = OVERLAY_OK if item.get("ok", True) else OVERLAY_NG
        pen.setColor(color)
        pen.setStyle(Qt.DashLine if item["type"] == "mask_box" else Qt.SolidLine)
        painter.setPen(pen)
        pts, label = item["pts"], item.get("label", "")
        kind = item["type"]
        anchor = None
        if kind in ("rect", "mask_box") and len(pts) == 4:
            x1, y1, x2, y2 = (int(round(v)) for v in pts)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            anchor = (x1, max(0, y1 - 6))
        elif kind == "line" and len(pts) == 4:
            painter.drawLine(*(int(round(v)) for v in pts))
            anchor = (int(round((pts[0] + pts[2]) / 2)), int(round((pts[1] + pts[3]) / 2)) - 4)
        elif kind == "circle" and len(pts) == 3:
            cx, cy, r = pts
            painter.drawEllipse(QPointF(cx, cy), r, r)
            anchor = (int(cx - r), max(0, int(cy - r) - 6))
        elif kind == "poly" and len(pts) >= 2:
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
            anchor = (int(min(p[0] for p in pts)), max(0, int(min(p[1] for p in pts)) - 6))
        elif kind == "region" and len(pts) == 2 and item.get("mask") is not None:
            draw_region(painter, int(round(pts[0])), int(round(pts[1])), item["mask"], item.get("color", "yellow"))
            anchor = (int(round(pts[0])) + 4, int(round(pts[1])) + 16)
        elif kind == "text" and len(pts) == 2:
            anchor = (int(pts[0]), int(pts[1]))
        if label and anchor is not None:
            if kind == "text":  # often sits on a filled region: dark shadow keeps it readable
                painter.setPen(TEXT_SHADOW)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    painter.drawText(anchor[0] + dx, anchor[1] + dy, label)
                painter.setPen(pen)
            painter.drawText(anchor[0], anchor[1], label)


def region_image(mask: np.ndarray, color: str = "yellow") -> QImage:
    """See-through fill over the counted pixels, with a darker 2 px edge;
    clear elsewhere (holes too)."""
    _, on = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
    inner = cv2.erode(on, EDGE_KERNEL, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    classes = cv2.subtract(cv2.add(on, on), inner)  # 0 clear, 1 inside, 2 edge
    luts = REGION_LUTS.get(color, REGION_LUTS["yellow"])
    bgra = cv2.merge([cv2.LUT(classes, lut) for lut in luts])
    h, w = mask.shape[:2]
    return QImage(bgra.data, w, h, 4 * w, QImage.Format_ARGB32_Premultiplied).copy()


def draw_region(painter: QPainter, x: int, y: int, mask: np.ndarray, color: str = "yellow") -> None:
    if mask.size:
        painter.drawImage(x, y, region_image(mask, color))


def tool_value_text(result: ToolResult, calibrated: bool) -> str:
    """Value with 2 decimals, plus the mm unit for measurements once the job
    is calibrated. Code / OCR tools show their text when they have one, and
    a slow tool's result shows how old it is ("…" while still waiting)."""
    if result.extra.get("pending"):
        return "…"
    if result.kind in ("ocr", "code") and result.text:
        text = result.text.splitlines()[0][:24]
    else:
        text = f"{result.value:.2f}"
        if calibrated and result.kind in MM_KINDS:
            text += " mm"
    if "age_ms" in result.extra:
        text += f" · {result.extra['age_ms'] / 1000.0:.1f} s"
    return text
