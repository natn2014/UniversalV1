from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..perspective import PerspectiveWarp, normalize_quad, order_corners, output_size, quad_to_pixels
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon
from ..widgets.layout import bottom_bar, top_bar

DEFAULT_INSET = 0.1
PREVIEW_MAX_SIDE = 360


def bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy())


class QuadEditor(QWidget):
    """Camera snapshot with 4 draggable corner handles. Dragging moves a
    handle; tapping elsewhere moves the nearest handle to that spot."""

    quad_changed = Signal()

    HANDLE_RADIUS = 18  # touch-sized

    def __init__(self, pixmap: QPixmap, points: List[List[float]], parent=None) -> None:
        super().__init__(parent)
        self._pixmap = pixmap
        self.points = [list(p) for p in points]  # fractions of the image
        self._image_rect = QRectF()
        self._drag_index = -1
        self.valid = True
        self.setMinimumSize(480, 360)

    def set_points(self, points: List[List[float]]) -> None:
        self.points = [list(p) for p in points]
        self.update()
        self.quad_changed.emit()

    def _to_widget(self, p) -> QPointF:
        r = self._image_rect
        return QPointF(r.x() + p[0] * r.width(), r.y() + p[1] * r.height())

    def _to_fraction(self, pos: QPointF) -> List[float]:
        r = self._image_rect
        return [min(max((pos.x() - r.x()) / r.width(), 0.0), 1.0),
                min(max((pos.y() - r.y()) / r.height(), 0.0), 1.0)]

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b0d10"))
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / pw, self.height() / ph)
        w, h = pw * scale, ph * scale
        self._image_rect = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        painter.drawPixmap(self._image_rect, self._pixmap, QRectF(self._pixmap.rect()))

        # Outline in corner order (TL, TR, BR, BL) whatever order the handles
        # were dragged into; red while the shape can't be used.
        color = QColor("#ffd54f") if self.valid else QColor("#ff5252")
        ordered = order_corners(np.array(self.points))
        polygon = QPolygonF([self._to_widget(p) for p in ordered])
        painter.setPen(QPen(color, 2))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 40))
        painter.drawPolygon(polygon)

        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)
        for i, p in enumerate(self.points):
            centre = self._to_widget(p)
            r = self.HANDLE_RADIUS
            painter.setPen(QPen(QColor("#1a1a1a"), 2))
            painter.setBrush(QBrush(QColor("#ffb300") if i == self._drag_index else color))
            painter.drawEllipse(centre, r, r)
            painter.setPen(QPen(QColor("#1a1a1a"), 1))
            if i == self._drag_index:
                # Crosshair instead of the number while dragging, so the
                # exact corner point stays visible.
                painter.drawLine(QPointF(centre.x() - r + 4, centre.y()), QPointF(centre.x() + r - 4, centre.y()))
                painter.drawLine(QPointF(centre.x(), centre.y() - r + 4), QPointF(centre.x(), centre.y() + r - 4))
            else:
                painter.drawText(QRectF(centre.x() - r, centre.y() - r, 2 * r, 2 * r), Qt.AlignCenter, str(i + 1))
        painter.end()

    def _nearest(self, pos: QPointF) -> int:
        dists = [(self._to_widget(p) - pos).manhattanLength() for p in self.points]
        return int(np.argmin(dists))

    def mousePressEvent(self, event) -> None:
        if self._image_rect.isEmpty():
            return
        pos = event.position()
        self._drag_index = self._nearest(pos)
        if (self._to_widget(self.points[self._drag_index]) - pos).manhattanLength() > 2 * self.HANDLE_RADIUS:
            self.points[self._drag_index] = self._to_fraction(pos)  # tap-to-place
            self.quad_changed.emit()
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_index < 0:
            return
        self.points[self._drag_index] = self._to_fraction(event.position())
        self.update()
        self.quad_changed.emit()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_index = -1
        self.update()


class PerspectiveDialog(QDialog):
    """Top View setup: place the 4 corners of the work area on a camera
    snapshot; Apply straightens that area into a rectangle.

    After exec(): result() == Accepted and `quad` holds the corners, or
    None when the user chose Remove Top View."""

    def __init__(self, frame: np.ndarray, quad: Optional[list] = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Top View")
        self.setModal(True)
        self.resize(1010, 740)
        self._frame = frame
        self.quad: Optional[list] = None

        start = quad or self._default_points()
        self.editor = QuadEditor(bgr_to_pixmap(frame), start)
        self.editor.quad_changed.connect(self._on_quad_changed)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(220)
        self.preview_label.setStyleSheet("background: #0b0d10;")
        self.size_label = QLabel()
        self.size_label.setObjectName("panelText")
        self.size_label.setWordWrap(True)

        self.remove_button = QPushButton("Remove Top View")
        self.remove_button.setIcon(make_icon("trash", "#e0e0e0"))
        self.remove_button.clicked.connect(self._on_remove)
        self.reset_button = QPushButton("Reset Corners")
        self.reset_button.setIcon(make_icon("refresh", "#e0e0e0"))
        self.reset_button.clicked.connect(lambda: self.editor.set_points(self._default_points()))
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setIcon(make_icon("close", "#e0e0e0"))
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setIcon(make_icon("play"))
        self.apply_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.apply_button.clicked.connect(self._on_apply)
        for button in (self.remove_button, self.reset_button, self.cancel_button):
            button.setFixedHeight(44)
        self.apply_button.setFixedSize(170, 44)
        self.remove_button.setEnabled(quad is not None)

        self._build_layout()
        self._on_quad_changed()

    @staticmethod
    def _default_points() -> List[List[float]]:
        a, b = DEFAULT_INSET, 1 - DEFAULT_INSET
        return [[a, a], [b, a], [b, b], [a, b]]

    def _build_layout(self) -> None:
        top, top_layout = top_bar((10, 6, 10, 6))
        title = QLabel("Top View")
        title.setObjectName("monitorInfo")
        hint = QLabel("Drag the 4 handles onto the corners of the work area")
        hint.setObjectName("hint")
        top_layout.addWidget(title)
        top_layout.addStretch(1)
        top_layout.addWidget(hint)

        editor_frame = QFrame()
        editor_frame.setObjectName("videoFrame")
        editor_layout = QVBoxLayout(editor_frame)
        editor_layout.setContentsMargins(2, 2, 2, 2)
        editor_layout.addWidget(self.editor)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_panel.setFixedWidth(300)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        header = QLabel("Result Preview")
        header.setObjectName("panelHeader")
        side_layout.addWidget(header)
        side_layout.addWidget(self.preview_label)
        info = QWidget()
        info.setObjectName("counterBox")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.addWidget(self.size_label)
        side_layout.addWidget(info)
        side_layout.addStretch(1)

        content = QHBoxLayout()
        content.addWidget(editor_frame, 1)
        content.addWidget(side_panel)

        bottom, bottom_layout = bottom_bar()
        bottom_layout.addWidget(self.remove_button)
        bottom_layout.addWidget(self.reset_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.cancel_button)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(content, 1)
        layout.addWidget(bottom)

    @Slot()
    def _on_quad_changed(self) -> None:
        quad = normalize_quad(self.editor.points)
        self.editor.valid = quad is not None
        self.editor.update()
        self.apply_button.setEnabled(quad is not None)
        if quad is None:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Corners don't form a usable 4-sided area")
            self.size_label.setText("Move the handles so the outline doesn't cross itself.")
            return
        h, w = self._frame.shape[:2]
        out_w, out_h = output_size(quad_to_pixels(quad, w, h), max(w, h))
        preview = PerspectiveWarp(quad).apply(self._frame, max_side=PREVIEW_MAX_SIDE)
        pixmap = bgr_to_pixmap(preview).scaled(
            self.preview_label.width() - 8, max(self.preview_label.height(), 220) - 8,
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)
        self.size_label.setText(f"Output: {out_w} × {out_h} px\nThe Monitor and the model will use this view.")

    @Slot()
    def _on_apply(self) -> None:
        self.quad = normalize_quad(self.editor.points)
        if self.quad is not None:
            self.accept()

    @Slot()
    def _on_remove(self) -> None:
        self.quad = None
        self.accept()
