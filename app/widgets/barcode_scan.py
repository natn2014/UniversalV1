from typing import Optional

import cv2
from PySide6.QtCore import QRectF, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..barcode import decode_scanned_text
from ..camera import CAMERA_BACKEND
from ..optional_deps import zbar_decode
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon


class BarcodeWorker(QThread):
    frame_ready = Signal(QImage)
    barcode_found = Signal(str)
    status = Signal(str)

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._running = False
        self._last_emitted: Optional[str] = None

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = cv2.VideoCapture(self._camera_index, CAMERA_BACKEND)
        if not cap.isOpened():
            self.status.emit("Camera open failed.")
            return

        self._running = True
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.status.emit("Frame grab failed.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            # .copy() for the same reason VideoWorker does it: rgb is a view into
            # frame's buffer, which cv2 overwrites on the next cap.read().
            image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(image)

            if zbar_decode is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                try:
                    results = zbar_decode(gray)
                except Exception:
                    results = []
                if results:
                    raw_text = results[0].data.decode("utf-8", errors="ignore").strip()
                    text = decode_scanned_text(raw_text)
                    if text and text != self._last_emitted:
                        self._last_emitted = text
                        self.barcode_found.emit(text)

        cap.release()
        self.status.emit("Scanner stopped.")


class BarcodeScanWidget(QWidget):
    """Displays a live pixmap scaled to fit, with a red horizontal guide line."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self.setMinimumSize(480, 320)

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b0d10"))

        if self._pixmap is not None and not self._pixmap.isNull():
            pix_w, pix_h = self._pixmap.width(), self._pixmap.height()
            widget_w, widget_h = self.width(), self.height()
            scale = min(widget_w / pix_w, widget_h / pix_h)
            draw_w, draw_h = pix_w * scale, pix_h * scale
            offset_x = (widget_w - draw_w) / 2
            offset_y = (widget_h - draw_h) / 2
            draw_rect = QRectF(offset_x, offset_y, draw_w, draw_h)
            painter.drawPixmap(draw_rect, self._pixmap, QRectF(self._pixmap.rect()))

            y = offset_y + draw_h / 2
            outline_pen = QPen(QColor(0, 0, 0, 160))
            outline_pen.setWidth(5)
            painter.setPen(outline_pen)
            painter.drawLine(int(offset_x), int(y), int(offset_x + draw_w), int(y))
            line_pen = QPen(QColor("#ff1744"))
            line_pen.setWidth(2)
            painter.setPen(line_pen)
            painter.drawLine(int(offset_x), int(y), int(offset_x + draw_w), int(y))
        else:
            y = self.height() / 2
            pen = QPen(QColor("#ff1744"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(0, int(y), self.width(), int(y))

        painter.end()


class BarcodeScanDialog(QDialog):
    load_requested = Signal(str)
    save_requested = Signal(str)

    def __init__(self, camera_index: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scan Part Number")
        self.setModal(True)
        self.resize(760, 620)

        self.preview_widget = BarcodeScanWidget()

        self.detected_field = QLineEdit()
        self.detected_field.setObjectName("readout")
        self.detected_field.setReadOnly(True)
        self.detected_field.setPlaceholderText("Waiting for barcode...")

        self.manual_field = QLineEdit()
        self.manual_field.setPlaceholderText("Or type part number manually")
        self.manual_load_button = QPushButton("Load")
        self.manual_load_button.setIcon(make_icon("folder", "#ffffff"))
        self.manual_load_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.manual_load_button.clicked.connect(self._on_manual_load)
        self.manual_save_button = QPushButton("Save")
        self.manual_save_button.setIcon(make_icon("save", "#e0e0e0"))
        self.manual_save_button.clicked.connect(self._on_manual_save)
        for button in (self.manual_load_button, self.manual_save_button):
            button.setFixedHeight(40)
            button.setMinimumWidth(96)

        self.close_button = QPushButton("Close")
        self.close_button.setIcon(make_icon("close", "#e0e0e0"))
        self.close_button.setFixedHeight(44)
        self.close_button.setMinimumWidth(140)
        self.close_button.clicked.connect(self.reject)

        # Same frame as the Monitor tab: title strip, framed camera view,
        # readout + manual entry, action bar with Close bottom right.
        top_bar = QFrame()
        top_bar.setObjectName("monitorTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 6, 10, 6)
        title = QLabel("Scan Part Number")
        title.setObjectName("monitorInfo")
        hint = QLabel("Align the barcode with the red line")
        hint.setObjectName("hint")
        top_layout.addWidget(title)
        top_layout.addStretch(1)
        top_layout.addWidget(hint)

        preview_frame = QFrame()
        preview_frame.setObjectName("videoFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(2, 2, 2, 2)
        preview_layout.addWidget(self.preview_widget)

        entry_card = QFrame()
        entry_card.setObjectName("card")
        entry_layout = QVBoxLayout(entry_card)
        entry_layout.setContentsMargins(12, 10, 12, 12)
        entry_layout.setSpacing(8)
        detected_row = QHBoxLayout()
        detected_row.addWidget(QLabel("Detected:"))
        detected_row.addWidget(self.detected_field, 1)
        entry_layout.addLayout(detected_row)
        manual_row = QHBoxLayout()
        manual_row.addWidget(self.manual_field, 1)
        manual_row.addWidget(self.manual_load_button)
        manual_row.addWidget(self.manual_save_button)
        entry_layout.addLayout(manual_row)

        bottom_bar = QFrame()
        bottom_bar.setObjectName("monitorBottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 4, 6, 4)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.close_button)

        layout = QVBoxLayout()
        layout.addWidget(top_bar)
        layout.addWidget(preview_frame, 1)
        layout.addWidget(entry_card)
        layout.addWidget(bottom_bar)
        self.setLayout(layout)

        self._worker = BarcodeWorker(camera_index)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.barcode_found.connect(self._on_barcode_found)
        self._worker.status.connect(self._on_status)
        self._worker.start()

    @Slot(QImage)
    def _on_frame(self, image: QImage) -> None:
        self.preview_widget.set_image(QPixmap.fromImage(image))

    @Slot(str)
    def _on_barcode_found(self, text: str) -> None:
        self.detected_field.setText(text)
        self.manual_field.setText(text)
        self.load_requested.emit(text)

    @Slot(str)
    def _on_status(self, message: str) -> None:
        self.detected_field.setPlaceholderText(message)

    @Slot()
    def _on_manual_load(self) -> None:
        text = self.manual_field.text().strip()
        if text:
            self.load_requested.emit(text)

    @Slot()
    def _on_manual_save(self) -> None:
        text = self.manual_field.text().strip()
        if text:
            self.save_requested.emit(text)

    def _stop_worker(self) -> None:
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)

    def reject(self) -> None:
        self._stop_worker()
        super().reject()

    def closeEvent(self, event) -> None:
        self._stop_worker()
        super().closeEvent(event)
