"""Drawing the plain-dict overlays that vision tools return (see
app/vision/base.py) and formatting a result's value for the tool table."""

from typing import List

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF

from ..theme import NG_RED, OK_GREEN
from ..vision.base import ToolResult

# Verdict colors are dark for chips; on the camera image they need a lift.
OVERLAY_OK = QColor(OK_GREEN).lighter(170)
OVERLAY_NG = QColor(NG_RED).lighter(130)
MM_KINDS = {"width", "diameter"}


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
        elif kind == "text" and len(pts) == 2:
            anchor = (int(pts[0]), int(pts[1]))
        if label and anchor is not None:
            painter.drawText(anchor[0], anchor[1], label)


def tool_value_text(result: ToolResult, calibrated: bool) -> str:
    """Value with 2 decimals, plus the mm unit for measurements once the job
    is calibrated. Code / OCR tools show their text when they have one."""
    if result.kind in ("ocr", "code") and result.text:
        return result.text.splitlines()[0][:24]
    text = f"{result.value:.2f}"
    if calibrated and result.kind in MM_KINDS:
        text += " mm"
    return text
