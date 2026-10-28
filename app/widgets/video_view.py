from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..roi import normalize_roi
from .tool_overlay import ScreenPainter


class VideoView(QWidget):
    """Live frame scaled to fit. Boxes and labels are painted over it here,
    at screen size (see set_frame), not burned into the frame. In ROI mode,
    dragging on the image draws a new detection area, emitted as fractions of
    the frame (x1, y1, x2, y2)."""

    roi_drawn = Signal(tuple)
    point_picked = Signal(tuple)  # (x, y) as fractions of the frame, in point mode

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._overlay: Optional[Callable] = None
        self._image_rect = QRectF()
        self._roi_mode = False
        self._drag_start: Optional[QPointF] = None  # fractions of the frame
        self._drag_end: Optional[QPointF] = None
        self._overlay_text: Optional[str] = None
        self._roi_hint = "Drag on the image to set the detection area"
        self._point_mode = False

    def setPixmap(self, pixmap: QPixmap) -> None:
        self.set_frame(pixmap)

    def set_frame(self, pixmap: QPixmap, overlay: Optional[Callable] = None) -> None:
        """`overlay(painter)` draws in frame pixels, after the frame is drawn;
        use the tool_overlay helpers so lines and text keep screen size."""
        self._pixmap = pixmap
        self._overlay = overlay
        self.update()

    def pixmap(self) -> Optional[QPixmap]:
        return self._pixmap

    def set_overlay_text(self, text: Optional[str]) -> None:
        """Large warning drawn over the image (e.g. camera lost); None hides it."""
        self._overlay_text = text
        self.update()

    def set_roi_mode(self, enabled: bool, hint: Optional[str] = None) -> None:
        self._roi_mode = enabled
        self._roi_hint = hint or "Drag on the image to set the detection area"
        self._drag_start = self._drag_end = None
        self.setCursor(Qt.CrossCursor if enabled or self._point_mode else Qt.ArrowCursor)
        self.update()

    def set_point_mode(self, enabled: bool) -> None:
        """A tap on the image is emitted as `point_picked` instead of drawing a box."""
        self._point_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled or self._roi_mode else Qt.ArrowCursor)

    def paintEvent(self, event) -> None:
        painter = ScreenPainter(self)
        painter.fillRect(self.rect(), QColor("#0b0d10"))
        if self._pixmap is None or self._pixmap.isNull():
            self._draw_overlay(painter)
            painter.end()
            return
        scale = min(self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        w, h = self._pixmap.width() * scale, self._pixmap.height() * scale
        self._image_rect = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(self._image_rect, self._pixmap, QRectF(self._pixmap.rect()))
        if self._overlay is not None:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.begin_frame(self._image_rect, scale)
            try:
                self._overlay(painter)
            finally:
                painter.end_frame()

        if self._roi_mode:
            painter.setPen(QColor("#ffd54f"))
            painter.drawText(self._image_rect.adjusted(8, 6, -8, -6), Qt.AlignTop | Qt.AlignLeft,
                             self._roi_hint)
        if self._drag_start is not None and self._drag_end is not None:
            r = self._image_rect
            pen = QPen(QColor("#ffd54f"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRectF(
                QPointF(r.x() + self._drag_start.x() * r.width(), r.y() + self._drag_start.y() * r.height()),
                QPointF(r.x() + self._drag_end.x() * r.width(), r.y() + self._drag_end.y() * r.height()),
            ).normalized())
        self._draw_overlay(painter)
        painter.end()

    def _draw_overlay(self, painter: QPainter) -> None:
        if not self._overlay_text:
            return
        painter.fillRect(self.rect(), QColor(0, 0, 0, 170))
        font = painter.font()
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#ff5a4a"))
        painter.drawText(self.rect().adjusted(20, 0, -20, 0), Qt.AlignCenter | Qt.TextWordWrap, self._overlay_text)

    def _to_fraction(self, pos: QPointF) -> QPointF:
        r = self._image_rect
        return QPointF(min(max((pos.x() - r.x()) / r.width(), 0.0), 1.0),
                       min(max((pos.y() - r.y()) / r.height(), 0.0), 1.0))

    def mousePressEvent(self, event) -> None:
        if self._point_mode and not self._image_rect.isEmpty() and self._image_rect.contains(event.position()):
            p = self._to_fraction(event.position())
            self.point_picked.emit((p.x(), p.y()))
            return
        if not self._roi_mode or self._image_rect.isEmpty():
            return
        if self._image_rect.contains(event.position()):
            self._drag_start = self._drag_end = self._to_fraction(event.position())

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            return
        self._drag_end = self._to_fraction(event.position())
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is None:
            return
        start, end = self._drag_start, self._to_fraction(event.position())
        self._drag_start = self._drag_end = None
        self.update()
        roi = normalize_roi((start.x(), start.y(), end.x(), end.y()))
        if roi is not None:
            self.roi_drawn.emit(roi)
