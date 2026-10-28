"""Drawing the plain-dict overlays that vision tools return (see
app/vision/base.py) and formatting a result's value for the tool table."""

from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
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
# On screen, lines and labels have a fixed size whatever the camera resolution.
SCREEN_FONT_PX = 15
LABEL_GAP = 3  # between a label and the shape it names


class ScreenPainter(QPainter):
    """VideoView's painter. Between begin_frame() and end_frame() it is
    scaled so overlays are given in frame pixels like when they are burned
    into an image, but the helpers below draw lines with a fixed screen width
    and labels unscaled: sharp text, the same size at any frame resolution."""

    frame_bounds: Optional[QRectF] = None

    def begin_frame(self, image_rect: QRectF, scale: float) -> None:
        self.save()
        self.translate(image_rect.topLeft())
        self.scale(scale, scale)
        self.setClipRect(QRectF(0, 0, image_rect.width() / scale, image_rect.height() / scale))
        self.frame_bounds = QRectF(image_rect)

    def end_frame(self) -> None:
        self.frame_bounds = None
        self.restore()


def on_screen(painter: QPainter) -> bool:
    return getattr(painter, "frame_bounds", None) is not None


def overlay_pen(painter: QPainter, color, width: float = 2.0, style=Qt.SolidLine) -> QPen:
    """`width` screen pixels on the view; burned into a frame, it grows with
    the frame like the image does when shrunk to fit."""
    pen = QPen(QColor(color))
    pen.setStyle(style)
    if on_screen(painter):
        pen.setCosmetic(True)
        pen.setWidthF(width)
    else:
        pen.setWidth(max(round(width), round(width * overlay_scale(painter))))
    return pen


def overlay_scale(painter: QPainter) -> float:
    """For overlays burned into a frame (NG log images), which is later shown
    shrunk: lines and text grow with the frame, 1.0 for a 640 px frame."""
    return max(1.0, painter.device().width() / 640.0)


def use_label_font(painter: QPainter) -> None:
    """Bold label font: SCREEN_FONT_PX on the view; burned into a frame,
    about 1.6 % of the frame width (min 13 px)."""
    font = QFont(painter.font())
    font.setPixelSize(SCREEN_FONT_PX if on_screen(painter) else max(13, round(10 * overlay_scale(painter))))
    font.setBold(True)
    painter.setFont(font)


def draw_label(painter: QPainter, x: float, y: float, text: str, color, place: str = "above") -> None:
    """`text` on a dark chip, colored like the shape it names, at the frame
    point (x, y): "above" it, "inside" (below and right of it, for a box's
    top-left corner) or "center"-ed on it. Kept inside the image; too long a
    text is cut with "…"."""
    if not text:
        return
    painter.save()
    use_label_font(painter)
    if on_screen(painter):
        point = painter.transform().map(QPointF(x, y))
        bounds = painter.frame_bounds
        gap = LABEL_GAP
        painter.resetTransform()  # text in screen pixels, not scaled with the frame
    else:
        point = QPointF(x, y)
        bounds = QRectF(0, 0, painter.device().width(), painter.device().height())
        gap = LABEL_GAP * overlay_scale(painter)
    fm = painter.fontMetrics()
    pad = max(2, fm.height() // 8)
    text = fm.elidedText(text, Qt.ElideRight, int(bounds.width()) - 2 * pad)
    w, h = fm.horizontalAdvance(text) + 2 * pad, fm.height() + pad
    if place == "inside":
        left, top = point.x() + gap, point.y() + gap
    elif place == "center":
        left, top = point.x() - w / 2, point.y() - h / 2
    else:
        left, top = point.x() - pad, point.y() - gap - h
    left = min(max(bounds.left(), left), max(bounds.left(), bounds.right() - w))
    top = min(max(bounds.top(), top), max(bounds.top(), bounds.bottom() - h))
    painter.fillRect(QRectF(left, top, w, h), LABEL_CHIP)
    painter.setPen(QColor(color))
    painter.drawText(QPointF(left + pad, top + pad / 2 + fm.ascent()), text)
    painter.restore()


def draw_overlay(painter: QPainter, items: List[dict]) -> None:
    """Draws overlay items given in frame pixels; green when ok, red when not.
    Labels sit just above their shape, like the YOLO box labels."""
    for item in items:
        color = OVERLAY_OK if item.get("ok", True) else OVERLAY_NG
        style = Qt.DashLine if item["type"] == "mask_box" else Qt.SolidLine
        painter.setPen(overlay_pen(painter, color, 2, style))
        painter.setBrush(Qt.NoBrush)
        pts, label = item["pts"], item.get("label", "")
        kind = item["type"]
        anchor, place = None, item.get("place", "above")
        if kind in ("rect", "mask_box") and len(pts) == 4:
            x1, y1, x2, y2 = pts
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            anchor = (x1, y1)
        elif kind == "line" and len(pts) == 4:
            painter.drawLine(QPointF(pts[0], pts[1]), QPointF(pts[2], pts[3]))
            anchor = ((pts[0] + pts[2]) / 2, (pts[1] + pts[3]) / 2)
        elif kind == "circle" and len(pts) == 3:
            cx, cy, r = pts
            painter.drawEllipse(QPointF(cx, cy), r, r)
            anchor = (cx - r, cy - r)
        elif kind == "poly" and len(pts) >= 2:
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
            anchor = (min(p[0] for p in pts), min(p[1] for p in pts))
        elif kind == "region" and len(pts) == 2 and item.get("mask") is not None:
            draw_region(painter, int(round(pts[0])), int(round(pts[1])), item["mask"], item.get("color", "yellow"))
            anchor, place = (pts[0], pts[1]), "inside"
        elif kind == "text" and len(pts) == 2:
            anchor = (pts[0], pts[1])
        if label and anchor is not None:
            draw_label(painter, anchor[0], anchor[1], label, color, place)


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
