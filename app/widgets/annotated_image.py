from typing import List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class AnnotatedImageWidget(QWidget):
    """Displays an image scaled to fit, with clickable detection boxes drawn on top.
    In draw mode, dragging draws a new box instead (emitted in image pixels)."""

    box_clicked = Signal(int)
    box_drawn = Signal(int, int, int, int)  # x1, y1, x2, y2

    MIN_BOX_PX = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._boxes: List[dict] = []
        self._selected_index: int = -1
        self._draw_rect = QRectF()
        self._draw_mode = False
        self._drag_start: Optional[QPointF] = None  # image coordinates
        self._drag_end: Optional[QPointF] = None
        self.setMinimumSize(320, 240)

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self._drag_start = self._drag_end = None
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
        point = self._to_image(event.position())
        if point is None:
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
        if self._drag_start is None:
            return
        self._drag_end = self._to_image(event.position(), clamp=True)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
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
