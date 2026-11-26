import math
from typing import List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


# Keypoint dots by index: the same order as the training preview's colours.
KPT_COLORS = ["#ff5050", "#3cc8ff", "#50dc50", "#dc50dc", "#ffc83c"]


class AnnotatedImageWidget(QWidget):
    """Displays an image scaled to fit, with clickable detection boxes drawn on top.
    In draw mode, dragging draws a new box instead (emitted in image pixels).

    Boxes may carry pose keypoints, `"kpts": [(x, y, v)]` in image pixels
    (v 0 = not placed, 1 = hidden, 2 = visible). They are drawn as numbered
    dots with an arrow from point 1 to point 0 (tail -> head). In point mode
    a tap emits `point_placed`; dragging a dot of the selected box emits
    `point_moved` when released. Which point comes next is the caller's job."""

    box_clicked = Signal(int)
    box_drawn = Signal(int, int, int, int)  # x1, y1, x2, y2
    point_placed = Signal(float, float)  # image pixels
    point_moved = Signal(int, float, float)  # keypoint index, image pixels

    MIN_BOX_PX = 4
    POINT_HIT_PX = 16  # screen pixels: finger-sized

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._boxes: List[dict] = []
        self._selected_index: int = -1
        self._draw_rect = QRectF()
        self._draw_mode = False
        self._drag_start: Optional[QPointF] = None  # image coordinates
        self._drag_end: Optional[QPointF] = None
        self._point_mode = False
        self._dragged_point: Optional[int] = None
        self.setMinimumSize(320, 240)

    def set_point_mode(self, enabled: bool) -> None:
        self._point_mode = enabled
        self._dragged_point = None
        if enabled:
            self._draw_mode = False
            self._drag_start = self._drag_end = None
        self.setCursor(Qt.PointingHandCursor if enabled else
                       (Qt.CrossCursor if self._draw_mode else Qt.ArrowCursor))
        self.update()

    def point_mode(self) -> bool:
        return self._point_mode

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self._drag_start = self._drag_end = None
        if enabled:
            self._point_mode = False
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def set_image(self, pixmap: QPixmap, boxes: List[dict]) -> None:
        self._pixmap = pixmap
        self._boxes = boxes
        self._selected_index = -1
        self.update()

    def set_selected_index(self, index: int) -> None:
        self._selected_index = index
        self.update()

    def pixmap(self) -> Optional[QPixmap]:
        return self._pixmap

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b0d10"))

        if self._pixmap is None or self._pixmap.isNull():
            painter.end()
            return

        pix_w, pix_h = self._pixmap.width(), self._pixmap.height()
        widget_w, widget_h = self.width(), self.height()
        scale = min(widget_w / pix_w, widget_h / pix_h)
        draw_w, draw_h = pix_w * scale, pix_h * scale
        offset_x = (widget_w - draw_w) / 2
        offset_y = (widget_h - draw_h) / 2
        self._draw_rect = QRectF(offset_x, offset_y, draw_w, draw_h)

        painter.drawPixmap(self._draw_rect, self._pixmap, QRectF(self._pixmap.rect()))

        label_font = QFont()
        label_font.setPointSize(10)
        label_font.setBold(True)
        painter.setFont(label_font)

        for i, box in enumerate(self._boxes):
            selected = i == self._selected_index
            color = QColor("#ff5252") if selected else QColor("#7fd0ff")
            pen = QPen(color)
            pen.setWidth(3 if selected else 2)
            painter.setPen(pen)

            x1 = offset_x + box["x1"] * scale
            y1 = offset_y + box["y1"] * scale
            x2 = offset_x + box["x2"] * scale
            y2 = offset_y + box["y2"] * scale
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            painter.drawText(int(x1), max(0, int(y1) - 6), box.get("class_name", ""))
            self._paint_keypoints(painter, box.get("kpts") or [], offset_x, offset_y, scale, selected)

        if self._drag_start is not None and self._drag_end is not None:
            pen = QPen(QColor("#ffd54f"))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(
                offset_x + self._drag_start.x() * scale, offset_y + self._drag_start.y() * scale,
                (self._drag_end.x() - self._drag_start.x()) * scale,
                (self._drag_end.y() - self._drag_start.y()) * scale).normalized())

        painter.end()

    def _paint_keypoints(self, painter: QPainter, kpts, offset_x: float, offset_y: float,
                         scale: float, selected: bool) -> None:
        pts = [QPointF(offset_x + x * scale, offset_y + y * scale) if v else None for x, y, v in kpts]
        if len(pts) >= 2 and pts[0] is not None and pts[1] is not None:
            _arrow(painter, pts[1], pts[0], QColor("#ffd54f"), 3 if selected else 2)
        radius = 7 if selected else 5
        for i, ((_, _, v), p) in enumerate(zip(kpts, pts)):
            if p is None:
                continue
            color = QColor(KPT_COLORS[i % len(KPT_COLORS)])
            painter.setPen(QPen(QColor("#0b0d10") if v == 2 else color, 2))
            painter.setBrush(color if v == 2 else Qt.NoBrush)  # hidden: hollow
            painter.drawEllipse(p, radius, radius)
            painter.setPen(QPen(color))
            painter.drawText(int(p.x() + radius + 2), int(p.y() - radius), str(i))
        painter.setBrush(Qt.NoBrush)

    def _point_at(self, pos: QPointF) -> Optional[int]:
        """Index of the selected box's keypoint under a widget position."""
        if not (0 <= self._selected_index < len(self._boxes)) or self._pixmap is None:
            return None
        scale = self._draw_rect.width() / max(1, self._pixmap.width())
        best, best_d = None, float(self.POINT_HIT_PX)
        for i, (x, y, v) in enumerate(self._boxes[self._selected_index].get("kpts") or []):
            if not v:
                continue
            sx, sy = self._draw_rect.x() + x * scale, self._draw_rect.y() + y * scale
            d = math.hypot(pos.x() - sx, pos.y() - sy)
            if d <= best_d:
                best, best_d = i, d
        return best

    def _to_image(self, pos: QPointF, clamp: bool = False) -> Optional[QPointF]:
        """Widget position -> image pixels; None outside the image unless clamped."""
        if not clamp and not self._draw_rect.contains(pos):
            return None
        scale = self._draw_rect.width() / self._pixmap.width()
        x = (pos.x() - self._draw_rect.x()) / scale
        y = (pos.y() - self._draw_rect.y()) / scale
        if clamp:
            x = min(max(x, 0.0), self._pixmap.width() - 1)
            y = min(max(y, 0.0), self._pixmap.height() - 1)
        return QPointF(x, y)

    def mousePressEvent(self, event) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        grabbed = None if self._draw_mode else self._point_at(event.position())
        if grabbed is not None:
            self._dragged_point = grabbed
            return
        point = self._to_image(event.position())
        if point is None:
            return
        if self._point_mode:
            self.point_placed.emit(point.x(), point.y())
            return
        if self._draw_mode:
            self._drag_start = self._drag_end = point
            return
        img_x, img_y = point.x(), point.y()

        best_index = -1
        best_area = None
        for i, box in enumerate(self._boxes):
            if box["x1"] <= img_x <= box["x2"] and box["y1"] <= img_y <= box["y2"]:
                area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                if best_area is None or area < best_area:
                    best_area = area
                    best_index = i

        if best_index >= 0:
            self._selected_index = best_index
            self.update()
            self.box_clicked.emit(best_index)

    def mouseMoveEvent(self, event) -> None:
        if self._dragged_point is not None:
            point = self._to_image(event.position(), clamp=True)
            kpts = self._boxes[self._selected_index]["kpts"]
            _, _, v = kpts[self._dragged_point]
            kpts[self._dragged_point] = (point.x(), point.y(), v)
            self.update()
            return
        if self._drag_start is None:
            return
        self._drag_end = self._to_image(event.position(), clamp=True)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._dragged_point is not None:
            index, self._dragged_point = self._dragged_point, None
            x, y, _ = self._boxes[self._selected_index]["kpts"][index]
            self.point_moved.emit(index, x, y)
            return
        if self._drag_start is None:
            return
        end = self._to_image(event.position(), clamp=True)
        start = self._drag_start
        self._drag_start = self._drag_end = None
        self.update()
        x1, x2 = sorted((int(start.x()), int(end.x())))
        y1, y2 = sorted((int(start.y()), int(end.y())))
        if x2 - x1 >= self.MIN_BOX_PX and y2 - y1 >= self.MIN_BOX_PX:
            self.box_drawn.emit(x1, y1, x2, y2)


def _arrow(painter: QPainter, start: QPointF, end: QPointF, color: QColor, width: int) -> None:
    """A line from start to end with a head at end."""
    painter.setPen(QPen(color, width))
    painter.drawLine(start, end)
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    size = 8 + 2 * width
    for side in (-0.45, 0.45):
        a = angle + math.pi + side
        painter.drawLine(end, QPointF(end.x() + size * math.cos(a), end.y() + size * math.sin(a)))
