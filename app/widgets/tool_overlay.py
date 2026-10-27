"""Drawing the plain-dict overlays that vision tools return (see
app/vision/base.py) and formatting a result's value for the tool table."""

from typing import List

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF

from ..theme import NG_RED, OK_GREEN
from ..vision.base import ToolResult

# Verdict colors are dark for chips; on the camera image they need a lift.
OVERLAY_OK = QColor(OK_GREEN).lighter(170)
OVERLAY_NG = QColor(NG_RED).lighter(130)
MM_KINDS = {"width", "diameter"}
# Counted pixels (Area, Color Area): see-through yellow with a darker edge.
LABEL_CHIP = QColor(0, 0, 0, 185)  # dark chip behind every label: readable on any image
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


def overlay_scale(painter: QPainter) -> float:
    """Overlays are drawn in frame pixels and the frame is then shrunk to fit
    the view, so lines and text grow with the frame: 1.0 for a 640 px frame."""
    return max(1.0, painter.device().width() / 640.0)


def use_label_font(painter: QPainter) -> None:
    """Bold label font, about 1.6 % of the frame width (min 13 px): about
    20 px on screen for a 1280 px frame on the 1920 x 1080 panel."""
    font = QFont(painter.font())
    font.setPixelSize(max(13, round(10 * overlay_scale(painter))))
    font.setBold(True)
    painter.setFont(font)


def draw_label(painter: QPainter, x: float, y: float, text: str, color) -> None:
    """`text` with its baseline at (x, y) on a dark chip, kept inside the
    image; colored like the box it belongs to."""
    if not text:
        return
    fm = painter.fontMetrics()
    pad = max(2, fm.height() // 8)
    w, h = fm.horizontalAdvance(text) + 2 * pad, fm.height() + pad
    dev_w, dev_h = painter.device().width(), painter.device().height()
    left = int(min(max(0, x - pad), max(0, dev_w - w)))
    top = int(min(max(0, y - fm.ascent() - pad // 2), max(0, dev_h - h)))
    painter.save()
    painter.fillRect(left, top, w, h, LABEL_CHIP)
    painter.setPen(QColor(color))
    painter.drawText(left + pad, top + pad // 2 + fm.ascent(), text)
    painter.restore()


def draw_overlay(painter: QPainter, items: List[dict]) -> None:
    """Draws overlay items in frame pixels; green when ok, red when not.
    Labels are placed like the YOLO box labels (above the box corner)."""
    use_label_font(painter)
    pen = QPen()
    pen.setWidth(max(2, round(2 * overlay_scale(painter))))
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
            anchor = (int(round(pts[0])) + 4, int(round(pts[1])) + painter.fontMetrics().ascent() + 6)
        elif kind == "text" and len(pts) == 2:
            anchor = (int(pts[0]), int(pts[1]))
        if label and anchor is not None:
            draw_label(painter, anchor[0], anchor[1], label, color)


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
