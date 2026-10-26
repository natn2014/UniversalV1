import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QEventPoint, QPainter, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..theme import make_icon


class ZoomImageView(QWidget):
    """Image view with pinch-zoom for the touch panel: two fingers zoom around
    their midpoint (and pan as they move), one finger drags, double-tap fits.
    Mouse: wheel zooms at the cursor, drag pans, double-click fits."""

    FIT_RATIO = 0.85   # opens at ~85% of the available area
    MAX_ZOOM = 8.0     # relative to the fitted size
    MIN_ZOOM = 0.5
    TAP_SLOP = 12      # px a press may move and still count as a tap

    zoom_changed = Signal(float)   # zoom relative to the fitted size
    backdrop_tapped = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.setMinimumSize(1, 1)
        self._pixmap = QPixmap()
        self._fit_scale = 1.0
        self._scale = 1.0
        self._offset = QPointF(0, 0)   # image centre relative to widget centre
        self._fitted = True
        self._pinch_start = None       # (dist, scale, image point under midpoint)
        self._drag_last: Optional[QPointF] = None
        self._press_pos: Optional[QPointF] = None
        self._moved = False

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.fit()

    def fit(self) -> None:
        self._fitted = True
        self._offset = QPointF(0, 0)
        if not self._pixmap.isNull() and self.width() > 0 and self.height() > 0:
            self._fit_scale = self.FIT_RATIO * min(
                self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        self._scale = self._fit_scale
        self._changed()

    def zoom_by(self, factor: float, anchor: Optional[QPointF] = None) -> None:
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        self._zoom_to(self._scale * factor, self._to_image(anchor), anchor)

    def _changed(self) -> None:
        self.zoom_changed.emit(self._scale / self._fit_scale if self._fit_scale else 1.0)
        self.update()

    def _center(self) -> QPointF:
        return QPointF(self.width() / 2, self.height() / 2) + self._offset

    def _to_image(self, pos: QPointF) -> QPointF:
        return (pos - self._center()) / self._scale

    def _zoom_to(self, scale: float, image_point: QPointF, screen_point: QPointF) -> None:
        """Set scale while keeping image_point under screen_point."""
        lo, hi = self._fit_scale * self.MIN_ZOOM, self._fit_scale * self.MAX_ZOOM
        self._scale = max(lo, min(hi, scale))
        widget_center = QPointF(self.width() / 2, self.height() / 2)
        self._offset = screen_point - widget_center - image_point * self._scale
        self._fitted = False
        self._changed()

    def _image_rect(self) -> QRectF:
        w = self._pixmap.width() * self._scale
        h = self._pixmap.height() * self._scale
        c = self._center()
        return QRectF(c.x() - w / 2, c.y() - h / 2, w, h)

    def paintEvent(self, event) -> None:
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(self._image_rect(), self._pixmap, QRectF(self._pixmap.rect()))
        painter.end()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fitted:
            self.fit()

    # --- touch -----------------------------------------------------------
    def event(self, event) -> bool:
        if event.type() in (QEvent.TouchBegin, QEvent.TouchUpdate,
                            QEvent.TouchEnd, QEvent.TouchCancel):
            self._handle_touch(event)
            event.accept()
            return True
        return super().event(event)

    def _handle_touch(self, event) -> None:
        points = [p for p in event.points() if p.state() != QEventPoint.Released]
        if event.type() == QEvent.TouchBegin:
            self._press_pos = event.points()[0].position()
            self._moved = False

        if len(points) >= 2:
            a, b = points[0].position(), points[1].position()
            mid = (a + b) / 2
            dist = max(1.0, ((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2) ** 0.5)
            if self._pinch_start is None:
                self._pinch_start = (dist, self._scale, self._to_image(mid))
            start_dist, start_scale, image_point = self._pinch_start
            self._zoom_to(start_scale * dist / start_dist, image_point, mid)
            self._moved = True
            self._drag_last = None
        elif len(points) == 1:
            pos = points[0].position()
            if self._pinch_start is not None:
                # Lifted one finger of a pinch: continue as a drag from here.
                self._pinch_start = None
                self._drag_last = pos
            elif self._drag_last is not None:
                self._pan(pos - self._drag_last)
                self._drag_last = pos
            else:
                self._drag_last = pos
            if self._press_pos is not None and (pos - self._press_pos).manhattanLength() > self.TAP_SLOP:
                self._moved = True

        if event.type() in (QEvent.TouchEnd, QEvent.TouchCancel):
            pos = event.points()[0].position()
            if event.type() == QEvent.TouchEnd and not self._moved:
                self._on_tap(pos)
            self._pinch_start = None
            self._drag_last = None
            self._press_pos = None

    def _pan(self, delta: QPointF) -> None:
        self._offset += delta
        self._fitted = False
        self.update()

    def _on_tap(self, pos: QPointF) -> None:
        now = time.monotonic()
        if now - getattr(self, "_last_tap", 0.0) < 0.35:
            self._last_tap = 0.0
            self.fit()
            return
        self._last_tap = now
        if not self._image_rect().contains(pos):
            self.backdrop_tapped.emit()

    # --- mouse -----------------------------------------------------------
    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120
        if steps:
            self.zoom_by(1.2 ** steps, event.position())

    def mousePressEvent(self, event) -> None:
        self._press_pos = event.position()
        self._drag_last = event.position()
        self._moved = False

    def mouseMoveEvent(self, event) -> None:
        if self._drag_last is None:
            return
        if (event.position() - self._press_pos).manhattanLength() > self.TAP_SLOP:
            self._moved = True
        if self._moved:
            self._pan(event.position() - self._drag_last)
        self._drag_last = event.position()

    def mouseReleaseEvent(self, event) -> None:
        if not self._moved and not self._image_rect().contains(event.position()):
            self.backdrop_tapped.emit()
        self._drag_last = None
        self._press_pos = None

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()


class ImageOverlay(QWidget):
    """Full-window popup showing one NG image, with a close button top right.
    Closes on the button, Esc, or a tap outside the image."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("imageOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.title_label = QLabel()
        self.title_label.setObjectName("overlayTitle")
        self.zoom_label = QLabel()
        self.zoom_label.setObjectName("overlayZoom")

        self.zoom_out_button = QPushButton("−")
        self.zoom_in_button = QPushButton("+")
        self.fit_button = QPushButton("Fit")
        for button in (self.zoom_out_button, self.zoom_in_button, self.fit_button):
            button.setFixedHeight(44)
            button.setMinimumWidth(56)

        self.close_button = QPushButton()
        self.close_button.setObjectName("overlayClose")
        self.close_button.setIcon(make_icon("close"))
        self.close_button.setIconSize(QSize(24, 24))
        self.close_button.setFixedSize(44, 44)
        self.close_button.clicked.connect(self.hide)

        self.view = ZoomImageView()
        self.view.zoom_changed.connect(
            lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.view.backdrop_tapped.connect(self.hide)
        self.zoom_out_button.clicked.connect(lambda: self.view.zoom_by(1 / 1.25))
        self.zoom_in_button.clicked.connect(lambda: self.view.zoom_by(1.25))
        self.fit_button.clicked.connect(self.view.fit)

        top_row = QHBoxLayout()
        top_row.addWidget(self.title_label)
        top_row.addStretch(1)
        top_row.addWidget(self.zoom_label)
        top_row.addWidget(self.zoom_out_button)
        top_row.addWidget(self.zoom_in_button)
        top_row.addWidget(self.fit_button)
        top_row.addSpacing(16)
        top_row.addWidget(self.close_button)

        hint = QLabel("Pinch with 2 fingers to zoom · drag to move · double-tap to fit")
        hint.setObjectName("overlayHint")
        hint.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 8)
        layout.addLayout(top_row)
        layout.addWidget(self.view, 1)
        layout.addWidget(hint)
        self.hide()

    def show_image(self, image_path: Path, title: str) -> None:
        self.title_label.setText(title)
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self.setFocus()
        # After show(), so the view has its real size when fitting.
        self.view.set_pixmap(QPixmap(str(image_path)))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.hide()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.view.zoom_by(1.25)
        elif event.key() == Qt.Key_Minus:
            self.view.zoom_by(1 / 1.25)
        else:
            super().keyPressEvent(event)
