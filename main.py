import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml
from PySide6.QtCore import Qt, QRectF, QThread, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from patlite_lr6 import TowerLight, LED_ON, BUZZER_OFF, BUZZER_ON, BUZZER_PATTERN_NAMES
except Exception:
    TowerLight = None
    BUZZER_OFF = 0x0
    BUZZER_ON = 0x1
    BUZZER_PATTERN_NAMES = [
        ("Off", 0x0), ("Continuous", 0x1), ("Pattern 1", 0x2),
        ("Pattern 2", 0x3), ("Pattern 3", 0x4), ("Pattern 4", 0x5),
    ]

try:
    import torch
except Exception:
    torch = None

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None


DARK_INDUSTRIAL_QSS = """
QWidget {
    background-color: #1e2228;
    color: #e0e0e0;
    font-size: 13pt;
}
QTabWidget::pane {
    border: 1px solid #3a4048;
    background-color: #262b33;
}
QTabBar::tab {
    background-color: #2c313a;
    color: #9aa5b1;
    padding: 10px 24px;
    border: 1px solid #3a4048;
    border-bottom: none;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #0d47a1;
    color: #ffffff;
}
QTabBar::tab:hover:!selected {
    background-color: #3a4048;
}
QGroupBox {
    border: 1px solid #3a4048;
    border-radius: 4px;
    margin-top: 14px;
    padding: 10px;
    font-weight: 600;
    color: #7fd0ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    background-color: #2c313a;
    border: 1px solid #4a5560;
    border-radius: 3px;
    padding: 6px 14px;
}
QPushButton:hover {
    background-color: #3a4048;
    border-color: #5a6570;
}
QPushButton:pressed {
    background-color: #14171c;
}
QLabel#judgementBanner {
    font-size: 15pt;
    font-weight: 800;
    border: 2px solid #10131a;
    border-radius: 18px;
    padding: 4px 18px;
}
"""

ALLOWED_FPS = [24, 30, 60]

# CAP_DSHOW is Windows-only; use V4L2 on Linux (Jetson) and let OpenCV
# auto-pick elsewhere.
CAMERA_BACKEND = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
CAPTURES_IMAGES_DIR = CAPTURES_DIR / "images"
CAPTURES_LABELS_DIR = CAPTURES_DIR / "labels"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PART_CONFIGS_DIR = Path(__file__).resolve().parent / "part_configs"
LAST_PART_FILE = PART_CONFIGS_DIR / ".last_part"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_part_no(part_no: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", part_no.strip())
    return cleaned.strip("_") or "unknown_part"


# AIM Code 39 Extended decoding, ported verbatim from natn2014/Hioki_data_logger's
# main.py (AIM_MAP + decode_barcode + _decode_model_text) for consistency with the
# barcode content produced by the same label-printing/scanning conventions. zbar
# does NOT decode these escape sequences itself (verified empirically: a raw scan
# of a "+A" Code39 payload comes back as the literal two characters "+A", not the
# Full-ASCII-mapped "a") — this post-processing step is required.
AIM_MAP = {
    '/A': ' ',  '/B': '!',  '/C': '"',  '/D': ',',
    '/E': '%',  '/F': '&',  '/G': "'",  '/H': '(',
    '/I': ')',  '/J': '*',  '/K': '+',  '/L': '/',
    '/M': ':',  '/N': ';',  '/O': '<',  '/P': '=',
    '/Q': '>',  '/R': '?',  '/S': '@',  '/T': '[',
    '/U': '\\', '/V': ']',  '/W': '^',  '/X': '_',
    '/Y': '`',  '/Z': '{',
}


def decode_barcode(raw: str) -> str:
    """Decode AIM Code 39 Extended barcode string to a plain model number.

    Strips the leading check-digit character, converts /X escape pairs to their
    real characters, and stops at /D (field separator).
    """
    if not raw:
        return ''
    s = raw[1:]  # strip check-digit / scanner prefix
    result = ''
    i = 0
    while i < len(s):
        if s[i] == '/' and i + 1 < len(s):
            code = s[i:i + 2].upper()
            if code == '/D':
                break
            if code in AIM_MAP:
                result += AIM_MAP[code]
                i += 2
                continue
        result += s[i]
        i += 1
    return result.strip()


def decode_scanned_text(raw: str) -> str:
    """Unified decode for barcode-scanned text (not manual entry).

    Priority:
    1. Dollar-delimited  — PREFIX$[id]MODEL$SUFFIX  (label-printer format)
    2. AIM Code 39 Extended — /X escape sequences   (scanner format)
    3. Plain text — returned as-is after strip
    """
    raw = raw.strip()
    if not raw:
        return ''
    if '$' in raw:
        first = raw.find('$')
        after = raw[first + 1:]
        second = after.find('$')
        return (after[:second] if second != -1 else after).strip()
    # AIM Code 39: presence of /[A-Z] escape pair signals encoded barcode
    if any(raw[i] == '/' and i + 1 < len(raw) and raw[i + 1].isupper()
           for i in range(len(raw))):
        decoded = decode_barcode(raw)
        if decoded:
            return decoded
    return raw


def find_cameras(max_index: int = 10) -> List[int]:
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, CAMERA_BACKEND)
        if cap.isOpened():
            available.append(idx)
        cap.release()
    return available


def read_yolo_label_file(label_path: Path) -> List[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, w, h = parts
        boxes.append(
            {
                "cls_id": int(cls_id),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
            }
        )
    return boxes


def write_yolo_label_file(label_path: Path, boxes: List[dict]) -> None:
    lines = [
        f"{box['cls_id']} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}"
        for box in boxes
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def yolo_box_to_pixel(box: dict, img_w: int, img_h: int) -> tuple:
    cx, cy, w, h = box["cx"] * img_w, box["cy"] * img_h, box["w"] * img_w, box["h"] * img_h
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return x1, y1, x2, y2


def pixel_box_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> dict:
    return {
        "cx": ((x1 + x2) / 2) / img_w,
        "cy": ((y1 + y2) / 2) / img_h,
        "w": (x2 - x1) / img_w,
        "h": (y2 - y1) / img_h,
    }


def nearest_allowed_fps(value: float) -> int:
    if value <= 1:
        return 30
    return min(ALLOWED_FPS, key=lambda x: abs(x - value))


class VideoWorker(QThread):
    frame_ready = Signal(QImage, list)
    status = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._camera_index: Optional[int] = None
        self._model_path: Optional[Path] = None
        self._model = None
        self._target_fps: int = 30

    def set_camera_index(self, index: Optional[int]) -> None:
        self._camera_index = index

    def set_model_path(self, path: Optional[Path]) -> None:
        self._model_path = path
        self._model = None

    def stop(self) -> None:
        self._running = False

    def _load_model(self) -> None:
        if self._model_path is None:
            return
        if YOLO is None:
            self.status.emit("Ultralytics not available. Install requirements.")
            return
        try:
            self._model = YOLO(str(self._model_path))
            self.status.emit(f"Model loaded: {self._model_path.name}")
        except Exception as exc:
            self.status.emit(f"Model load failed: {exc}")
            self._model = None

    def _extract_detections(self, results) -> List[dict]:
        detections: List[dict] = []
        if results is None:
            return detections
        boxes = results.boxes
        names = results.names
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            class_name = str(names.get(cls_id, cls_id))
            label = f"{class_name} {conf:.2f}"
            x1, y1, x2, y2 = xyxy.tolist()
            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                    "class_name": class_name,
                    "conf": conf,
                    "cls_id": cls_id,
                }
            )
        return detections

    def run(self) -> None:
        if self._camera_index is None:
            self.status.emit("Select a camera.")
            return

        cap = cv2.VideoCapture(self._camera_index, CAMERA_BACKEND)
        if not cap.isOpened():
            self.status.emit("Camera open failed.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        self._target_fps = nearest_allowed_fps(fps)
        self.status.emit(f"Camera FPS: {fps:.2f} -> Target FPS: {self._target_fps}")

        if self._model_path is not None and self._model is None:
            self._load_model()

        self._running = True
        frame_period = 1.0 / self._target_fps

        while self._running:
            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                self.status.emit("Frame grab failed.")
                break

            detections = []
            if self._model is not None:
                try:
                    results = self._model(frame, verbose=False)[0]
                except Exception as exc:
                    self.status.emit(f"Inference error: {exc}")
                    results = None
                detections = self._extract_detections(results)

            elapsed = time.perf_counter() - start_time

            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            else:
                frames_to_skip = int(elapsed / frame_period) - 1
                for _ in range(max(0, frames_to_skip)):
                    cap.grab()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            # .copy() forces a deep copy so the QImage doesn't reference a
            # numpy buffer that gets overwritten/freed on the next iteration.
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_ready.emit(image, detections)

        cap.release()
        self.status.emit("Stopped.")


class AnnotatedImageWidget(QWidget):
    """Displays an image scaled to fit, with clickable detection boxes drawn on top."""

    box_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._boxes: List[dict] = []
        self._selected_index: int = -1
        self._draw_rect = QRectF()
        self.setMinimumSize(320, 240)

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
        painter.fillRect(self.rect(), QColor("#1e2228"))

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

        painter.end()

    def mousePressEvent(self, event) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        pos = event.position()
        if not self._draw_rect.contains(pos):
            return

        scale = self._draw_rect.width() / self._pixmap.width()
        img_x = (pos.x() - self._draw_rect.x()) / scale
        img_y = (pos.y() - self._draw_rect.y()) / scale

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


class TrainingWorker(QThread):
    status_update = Signal(str)
    training_complete = Signal(str)
    training_error = Signal(str)

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            if YOLO is None:
                self.training_error.emit("Ultralytics not available.")
                return

            model = YOLO(self.config["model_path"])

            def _on_epoch_end(trainer):
                try:
                    self.status_update.emit(f"Epoch {trainer.epoch + 1}/{trainer.epochs}")
                except Exception:
                    pass

            model.add_callback("on_train_epoch_end", _on_epoch_end)

            self.status_update.emit("Training started...")
            results = model.train(
                data=self.config["data_yaml"],
                epochs=self.config["epochs"],
                imgsz=self.config["imgsz"],
                batch=self.config["batch"],
                device=self.config["device"],
                project=self.config["project"],
                name=self.config["name"],
                patience=self.config["patience"],
                optimizer=self.config["optimizer"],
                lr0=self.config["lr0"],
                verbose=True,
                plots=False,
                val=True,
                amp=True,
                workers=0,
            )
            best_path = Path(results.save_dir) / "weights" / "best.pt"
            self.training_complete.emit(str(best_path))
        except Exception as exc:
            self.training_error.emit(f"Training failed: {exc}")


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
        painter.fillRect(self.rect(), QColor("#1e2228"))

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
        self.resize(640, 520)

        self.preview_widget = BarcodeScanWidget()

        self.detected_field = QLineEdit()
        self.detected_field.setReadOnly(True)
        self.detected_field.setPlaceholderText("Waiting for barcode...")

        self.manual_field = QLineEdit()
        self.manual_field.setPlaceholderText("Or type part number manually")
        self.manual_load_button = QPushButton("Load")
        self.manual_load_button.clicked.connect(self._on_manual_load)
        self.manual_save_button = QPushButton("Save")
        self.manual_save_button.clicked.connect(self._on_manual_save)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.preview_widget, 1)
        layout.addWidget(QLabel("Detected:"))
        layout.addWidget(self.detected_field)
        manual_row = QHBoxLayout()
        manual_row.addWidget(self.manual_field, 1)
        manual_row.addWidget(self.manual_load_button)
        manual_row.addWidget(self.manual_save_button)
        layout.addLayout(manual_row)
        layout.addWidget(self.close_button)
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


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Object Detection Concept")
        self._worker = VideoWorker()
        self._worker.frame_ready.connect(self.on_frame)
        self._worker.status.connect(self.on_status)
        self._class_counts = {}
        self._class_colors = {}
        self._current_pixmap = None
        self._selected_classes = set()
        self._confidence_threshold = 0.0
        self._max_detections = 0  # 0 = unlimited
        self._model_classes = []  # Store all classes from model
        self._tower = None
        self._tower_connected = False
        self._target_class_1: Optional[str] = None
        self._target_class_2: Optional[str] = None
        self._target_gate: str = "OR"
        self._last_tower_state = None
        self._buzzer_enabled: bool = False
        self._buzzer_pattern: int = BUZZER_ON
        self._current_frame_image: Optional[QImage] = None
        self._current_frame_detections: list = []
        self._train_pairs: list = []
        self._train_current_index: int = -1
        self._train_boxes: list = []
        self._train_selected_box_index: int = -1
        self._training_worker: Optional[TrainingWorker] = None
        self._current_part_no: Optional[str] = None
        self._barcode_dialog: Optional[BarcodeScanDialog] = None

        self.model_label = QLabel("No model selected")
        self.load_model_button = QPushButton("Load .pt Model")
        self.load_model_button.clicked.connect(self.load_model)

        self.camera_combo = QComboBox()
        self.scan_button = QPushButton("Scan Cameras")
        self.scan_button.clicked.connect(self.scan_cameras)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_stream)
        self.capture_button = QPushButton("Capture Frame")
        self.capture_button.clicked.connect(self.capture_frame)

        self.status_label = QLabel("Ready")
        self.tower_status_label = QLabel("Tower Light: Not Connected")

        self.buzzer_enabled_checkbox = QCheckBox("Enable Buzzer on NG")
        self.buzzer_enabled_checkbox.setChecked(False)
        self.buzzer_enabled_checkbox.toggled.connect(self.on_buzzer_enabled_changed)

        self.buzzer_pattern_combo = QComboBox()
        for name, value in BUZZER_PATTERN_NAMES:
            if value == BUZZER_OFF:
                continue
            self.buzzer_pattern_combo.addItem(name, value)
        self.buzzer_pattern_combo.currentIndexChanged.connect(self.on_buzzer_pattern_changed)

        self.buzzer_test_button = QPushButton("Test")
        self.buzzer_test_button.clicked.connect(self.on_buzzer_test_clicked)

        self.part_config_label = QLabel("No part scanned")
        self.scan_barcode_button = QPushButton("Scan Barcode...")
        self.scan_barcode_button.clicked.connect(self._open_barcode_scan_dialog)

        self.judgement_banner = QLabel()
        self.judgement_banner.setObjectName("judgementBanner")
        self.judgement_banner.setAlignment(Qt.AlignCenter)
        self.judgement_banner.setFixedSize(130, 36)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Class", "Count"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumWidth(220)

        # Filter panel
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(0)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        
        self.confidence_label = QLabel("Confidence: 0%")

        self.max_detections_label = QLabel("Max Detections (0 = unlimited):")
        self.max_detections_spinbox = QSpinBox()
        self.max_detections_spinbox.setMinimum(0)
        self.max_detections_spinbox.setMaximum(100)
        self.max_detections_spinbox.setValue(0)
        self.max_detections_spinbox.valueChanged.connect(self.on_max_detections_changed)

        self.target_class_label = QLabel("Target Classes:")
        self.target_class_combo_1 = QComboBox()
        self.target_class_combo_1.addItem("-- None --", None)
        self.target_class_combo_1.currentIndexChanged.connect(self.on_target_class_1_changed)

        self.target_gate_combo = QComboBox()
        self.target_gate_combo.addItem("OR")
        self.target_gate_combo.addItem("AND")
        self.target_gate_combo.currentTextChanged.connect(self.on_target_gate_changed)

        self.target_class_combo_2 = QComboBox()
        self.target_class_combo_2.addItem("-- None --", None)
        self.target_class_combo_2.currentIndexChanged.connect(self.on_target_class_2_changed)

        # Scrollable class filters
        self.class_filters_scroll = QScrollArea()
        self.class_filters_scroll.setWidgetResizable(True)
        self.class_filters_container = QWidget()
        self.class_filters_layout = QVBoxLayout()
        self.class_filters_container.setLayout(self.class_filters_layout)
        self.class_filters_scroll.setWidget(self.class_filters_container)
        self.class_filters_scroll.setMinimumHeight(200)

        # Train tab widgets
        self.train_model_label = QLabel("No model selected")
        self.train_select_folder_button = QPushButton("Select Image Folder")
        self.train_select_folder_button.clicked.connect(self._select_train_folder)
        self.train_folder_label = QLabel("(none selected)")

        self.train_list_widget = QListWidget()
        self.train_list_widget.currentRowChanged.connect(self._load_train_index)

        self.train_prev_button = QPushButton("Previous")
        self.train_prev_button.clicked.connect(self._train_prev)
        self.train_next_button = QPushButton("Next")
        self.train_next_button.clicked.connect(self._train_next)
        self.train_counter_label = QLabel("Image 0 / 0")

        self.train_image_widget = AnnotatedImageWidget()
        self.train_image_widget.box_clicked.connect(self._on_train_box_clicked)

        self.train_selected_label = QLabel("Selected box: none")
        self.train_class_combo = QComboBox()
        self.train_confirm_button = QPushButton("Confirm")
        self.train_confirm_button.clicked.connect(self._on_train_confirm)

        self.train_start_button = QPushButton("Start Training")
        self.train_start_button.clicked.connect(self._start_training)
        self.train_status_label = QLabel("Ready")

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_monitor_tab(), "Monitor")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs.addTab(self._build_job_change_tab(), "Job Change")
        self.tabs.addTab(self._build_train_tab(), "Train")
        self._refresh_job_buttons()

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.tabs, 1)
        root_layout.addWidget(self.status_label)
        self.setLayout(root_layout)

        self.scan_cameras()
        self._connect_tower_light()
        self._update_judgement_banner("YELLOW")
        self._auto_load_last_config()

    def _build_monitor_tab(self) -> QWidget:
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.start_button)
        top_bar.addWidget(self.stop_button)
        top_bar.addWidget(self.capture_button)
        top_bar.addStretch(1)
        top_bar.addWidget(self.judgement_banner)

        content_row = QHBoxLayout()
        content_row.addWidget(self.video_label, 1)
        content_row.addWidget(self.table)

        layout = QVBoxLayout()
        layout.addLayout(top_bar)
        layout.addLayout(content_row, 1)

        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _build_settings_tab(self) -> QWidget:
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.load_model_button)
        model_layout.addWidget(self.model_label, 1)
        model_group.setLayout(model_layout)

        camera_group = QGroupBox("Camera")
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.scan_button)
        camera_layout.addStretch(1)
        camera_group.setLayout(camera_layout)

        filters_group = QGroupBox("Detection Filters")
        filters_layout = QVBoxLayout()
        filters_layout.addWidget(QLabel("Confidence Threshold:"))
        filters_layout.addWidget(self.confidence_slider)
        filters_layout.addWidget(self.confidence_label)
        filters_layout.addSpacing(10)
        filters_layout.addWidget(self.max_detections_label)
        filters_layout.addWidget(self.max_detections_spinbox)
        filters_layout.addSpacing(10)
        filters_layout.addWidget(self.target_class_label)
        target_row = QHBoxLayout()
        target_row.addWidget(self.target_class_combo_1)
        target_row.addWidget(self.target_gate_combo)
        target_row.addWidget(self.target_class_combo_2)
        filters_layout.addLayout(target_row)
        filters_layout.addSpacing(10)
        filters_layout.addWidget(QLabel("Classes to Show:"))
        filters_layout.addWidget(self.class_filters_scroll, 1)
        filters_group.setLayout(filters_layout)

        tower_group = QGroupBox("Tower Light")
        tower_status_row = QHBoxLayout()
        tower_status_row.addWidget(self.tower_status_label)
        tower_status_row.addStretch(1)

        buzzer_row = QHBoxLayout()
        buzzer_row.addWidget(self.buzzer_enabled_checkbox)
        buzzer_row.addWidget(QLabel("Style:"))
        buzzer_row.addWidget(self.buzzer_pattern_combo)
        buzzer_row.addWidget(self.buzzer_test_button)
        buzzer_row.addStretch(1)

        tower_layout = QVBoxLayout()
        tower_layout.addLayout(tower_status_row)
        tower_layout.addLayout(buzzer_row)
        tower_group.setLayout(tower_layout)

        layout = QVBoxLayout()
        layout.addWidget(model_group)
        layout.addWidget(camera_group)
        layout.addWidget(filters_group, 1)
        layout.addWidget(tower_group)

        content = QWidget()
        content.setLayout(layout)

        # Wrapped in a scroll area so the tab degrades gracefully on smaller
        # displays (e.g. 1024x768 industrial panels) instead of forcing the
        # window to grow past the screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _build_job_change_tab(self) -> QWidget:
        part_group = QGroupBox("Part Config")
        part_layout = QHBoxLayout()
        part_layout.addWidget(QLabel("Current Part:"))
        part_layout.addWidget(self.part_config_label, 1)
        part_layout.addWidget(self.scan_barcode_button)
        part_group.setLayout(part_layout)

        jobs_group = QGroupBox("Saved Jobs")
        self.job_buttons_layout = QVBoxLayout()
        self.job_buttons_container = QWidget()
        self.job_buttons_container.setLayout(self.job_buttons_layout)

        jobs_scroll = QScrollArea()
        jobs_scroll.setWidgetResizable(True)
        jobs_scroll.setWidget(self.job_buttons_container)

        jobs_layout = QVBoxLayout()
        jobs_layout.addWidget(jobs_scroll)
        jobs_group.setLayout(jobs_layout)

        layout = QVBoxLayout()
        layout.addWidget(part_group)
        layout.addWidget(jobs_group, 1)

        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _refresh_job_buttons(self) -> None:
        while self.job_buttons_layout.count() > 0:
            item = self.job_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not PART_CONFIGS_DIR.exists():
            self.job_buttons_layout.addWidget(QLabel("No saved jobs yet."))
            return

        config_paths = sorted(PART_CONFIGS_DIR.glob("*.json"))
        if not config_paths:
            self.job_buttons_layout.addWidget(QLabel("No saved jobs yet."))
            return

        for config_path in config_paths:
            try:
                config = json.loads(config_path.read_text())
            except Exception:
                continue
            part_no = config.get("part_no") or config_path.stem
            button = QPushButton(part_no)
            button.clicked.connect(lambda checked, p=part_no: self._on_job_button_clicked(p))
            self.job_buttons_layout.addWidget(button)

        self.job_buttons_layout.addStretch(1)

    @Slot(str)
    def _on_job_button_clicked(self, part_no: str) -> None:
        config = self._load_part_config(part_no)
        if config is None:
            self.status_label.setText(f"Config for '{part_no}' no longer exists.")
            self._refresh_job_buttons()
            return
        model_ok = self._apply_config(config)
        self._set_current_part(part_no)
        if model_ok:
            self.status_label.setText(f"Applied config for '{part_no}'.")

    def _build_train_tab(self) -> QWidget:
        model_group = QGroupBox("Active Model")
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.train_model_label, 1)
        model_group.setLayout(model_layout)

        folder_group = QGroupBox("Dataset Folder")
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.train_select_folder_button)
        folder_layout.addWidget(self.train_folder_label, 1)
        folder_group.setLayout(folder_layout)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.train_prev_button)
        nav_row.addWidget(self.train_counter_label)
        nav_row.addWidget(self.train_next_button)

        image_column = QVBoxLayout()
        image_column.addWidget(self.train_image_widget, 1)
        image_column.addLayout(nav_row)

        correction_group = QGroupBox("Box Correction")
        correction_layout = QVBoxLayout()
        correction_layout.addWidget(self.train_selected_label)
        correction_layout.addWidget(self.train_class_combo)
        correction_layout.addWidget(self.train_confirm_button)
        correction_layout.addStretch(1)
        correction_group.setLayout(correction_layout)
        correction_group.setMaximumWidth(220)

        self.train_list_widget.setMaximumWidth(200)

        middle_row = QHBoxLayout()
        middle_row.addLayout(image_column, 1)
        middle_row.addWidget(self.train_list_widget)
        middle_row.addWidget(correction_group)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.train_start_button)
        bottom_row.addWidget(self.train_status_label, 1)

        layout = QVBoxLayout()
        layout.addWidget(model_group)
        layout.addWidget(folder_group)
        layout.addLayout(middle_row, 1)
        layout.addLayout(bottom_row)

        content = QWidget()
        content.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    @Slot()
    def scan_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = find_cameras()
        if not cameras:
            self.camera_combo.addItem("No camera", None)
            return
        for idx in cameras:
            self.camera_combo.addItem(f"Camera {idx}", idx)

    @Slot()
    def load_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO .pt Model",
            "",
            "PyTorch Model (*.pt)",
        )
        if not file_path:
            return
        self._load_model_from_path(Path(file_path))

    def _load_model_from_path(self, model_path: Path) -> None:
        self.model_label.setText(model_path.name)
        self._worker.set_model_path(model_path)
        self._sync_train_model_label()
        self.status_label.setText("Loading model...")

        # Load model to extract class names
        if YOLO is not None:
            try:
                model = YOLO(str(model_path))
                self._model_classes = list(model.names.values())
                self._populate_class_filters()
                self.status_label.setText(f"Model loaded: {len(self._model_classes)} classes")
            except Exception as exc:
                self.status_label.setText(f"Failed to load model: {exc}")
                self._model_classes = []
        else:
            self.status_label.setText("Ultralytics not available.")
            self._model_classes = []

    def _populate_class_filters(self) -> None:
        """Create checkboxes for all model classes."""
        # Clear existing checkboxes
        while self.class_filters_layout.count() > 0:
            item = self.class_filters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._selected_classes.clear()
        
        # Create checkbox for each class
        for class_name in sorted(self._model_classes):
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, cn=class_name: self._toggle_class_filter(cn, checked)
            )
            self.class_filters_layout.addWidget(checkbox)
            self._selected_classes.add(class_name)

        # Repopulate target-class combos from the same source list
        for combo in (self.target_class_combo_1, self.target_class_combo_2):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("-- None --", None)
            for class_name in sorted(self._model_classes):
                combo.addItem(class_name, class_name)
            combo.blockSignals(False)
        self._target_class_1 = None
        self._target_class_2 = None

        # Repopulate the Train tab's correction combo from the same source list
        self.train_class_combo.blockSignals(True)
        self.train_class_combo.clear()
        for class_name in sorted(self._model_classes):
            self.train_class_combo.addItem(class_name)
        self.train_class_combo.blockSignals(False)

    def _sync_train_model_label(self) -> None:
        if self._worker._model_path is not None:
            self.train_model_label.setText(self._worker._model_path.name)
        else:
            self.train_model_label.setText("No model selected")

    def _set_current_part(self, part_no: str) -> None:
        self._current_part_no = part_no
        self.part_config_label.setText(part_no)
        PART_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        LAST_PART_FILE.write_text(part_no)

    def _collect_current_config(self) -> dict:
        return {
            "part_no": self._current_part_no,
            "model_path": str(self._worker._model_path) if self._worker._model_path else None,
            "confidence_threshold": self._confidence_threshold,
            "max_detections": self._max_detections,
            "target_class_1": self._target_class_1,
            "target_class_2": self._target_class_2,
            "target_gate": self._target_gate,
            "selected_classes": sorted(self._selected_classes),
        }

    def _save_part_config(self, part_no: str, config: dict) -> None:
        PART_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        path = PART_CONFIGS_DIR / f"{sanitize_part_no(part_no)}.json"
        path.write_text(json.dumps(config, indent=2))
        self._refresh_job_buttons()

    def _load_part_config(self, part_no: str) -> Optional[dict]:
        path = PART_CONFIGS_DIR / f"{sanitize_part_no(part_no)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _apply_config(self, config: dict) -> bool:
        """Applies a saved config to the UI. Returns False if the config's model
        file was missing on disk (filters still get applied to whatever model is
        currently loaded), True otherwise."""
        model_ok = True
        model_path_str = config.get("model_path")
        if model_path_str:
            model_path = Path(model_path_str)
            current = self._worker._model_path
            if model_path.exists() and current != model_path:
                self._load_model_from_path(model_path)
            elif not model_path.exists():
                model_ok = False
                self.status_label.setText(
                    f"Config model not found on disk: {model_path.name} — filters applied to current model."
                )

        self.confidence_slider.setValue(int(round(config.get("confidence_threshold", 0.0) * 100)))
        self.max_detections_spinbox.setValue(int(config.get("max_detections", 0)))

        idx1 = self.target_class_combo_1.findData(config.get("target_class_1"))
        self.target_class_combo_1.setCurrentIndex(idx1 if idx1 >= 0 else 0)
        idx2 = self.target_class_combo_2.findData(config.get("target_class_2"))
        self.target_class_combo_2.setCurrentIndex(idx2 if idx2 >= 0 else 0)
        self.target_gate_combo.setCurrentText(config.get("target_gate", "OR"))

        selected = set(config.get("selected_classes", []))
        for i in range(self.class_filters_layout.count()):
            widget = self.class_filters_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(widget.text() in selected)

        return model_ok

    def _auto_load_last_config(self) -> None:
        if not LAST_PART_FILE.exists():
            return
        part_no = LAST_PART_FILE.read_text().strip()
        if not part_no:
            return
        config = self._load_part_config(part_no)
        if config is None:
            return
        model_ok = self._apply_config(config)
        self._set_current_part(part_no)
        if model_ok:
            self.status_label.setText(f"Auto-loaded last config: '{part_no}'.")

    @Slot()
    def _open_barcode_scan_dialog(self) -> None:
        if self._worker.isRunning():
            self.status_label.setText("Stop the Monitor stream before scanning a barcode (camera is in use).")
            return
        index = self.camera_combo.currentData()
        if index is None:
            self.status_label.setText("Select a valid camera.")
            return

        self._barcode_dialog = BarcodeScanDialog(int(index), parent=self)
        self._barcode_dialog.load_requested.connect(self._on_part_load_requested)
        self._barcode_dialog.save_requested.connect(self._on_part_save_requested)
        self._barcode_dialog.finished.connect(self._on_barcode_dialog_finished)
        self._barcode_dialog.show()

    def _on_barcode_dialog_finished(self) -> None:
        self._barcode_dialog = None

    @Slot(str)
    def _on_part_load_requested(self, part_no: str) -> None:
        config = self._load_part_config(part_no)
        if config is None:
            answer = QMessageBox.question(
                self,
                "No Config Found",
                f"No config found for '{part_no}'. Save current settings as a new config for this part?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer == QMessageBox.Yes:
                self._set_current_part(part_no)
                self._save_part_config(part_no, self._collect_current_config())
                self.status_label.setText(f"Saved new config for '{part_no}'.")
            return

        answer = QMessageBox.question(
            self,
            "Config Found",
            f"Found saved config for '{part_no}'. Apply?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            model_ok = self._apply_config(config)
            self._set_current_part(part_no)
            if model_ok:
                self.status_label.setText(f"Applied config for '{part_no}'.")

    @Slot(str)
    def _on_part_save_requested(self, part_no: str) -> None:
        self._set_current_part(part_no)
        self._save_part_config(part_no, self._collect_current_config())
        self.status_label.setText(f"Saved config for '{part_no}'.")

    @Slot(int)
    def on_target_class_1_changed(self, index: int) -> None:
        self._target_class_1 = self.target_class_combo_1.currentData()

    @Slot(int)
    def on_target_class_2_changed(self, index: int) -> None:
        self._target_class_2 = self.target_class_combo_2.currentData()

    @Slot(str)
    def on_target_gate_changed(self, text: str) -> None:
        self._target_gate = text

    def _connect_tower_light(self) -> None:
        if TowerLight is None:
            self.tower_status_label.setText("Tower Light: Driver Unavailable")
            return
        try:
            self._tower = TowerLight(backend="auto")
            self._tower.open()
            self._tower_connected = True
            self.tower_status_label.setText(f"Tower Light: Connected ({self._tower.backend_name})")
        except Exception:
            self._tower = None
            self._tower_connected = False
            self.tower_status_label.setText("Tower Light: Not Connected")

    def _compute_judgement_state(self, filtered_detections: list) -> str:
        present_classes = {d.get("class_name") for d in filtered_detections}
        targets = [c for c in (self._target_class_1, self._target_class_2) if c]
        if targets:
            if self._target_gate == "AND":
                is_match = all(t in present_classes for t in targets)
            else:
                is_match = any(t in present_classes for t in targets)
            if is_match:
                return "GREEN"
        if filtered_detections:
            return "RED"
        return "YELLOW"

    def _update_judgement_banner(self, state: str) -> None:
        text, bg, fg = {
            "GREEN": ("OK", "#2e7d32", "#ffffff"),
            "RED": ("NG", "#c0392b", "#ffffff"),
            "YELLOW": ("STANDBY", "#f4d03f", "#1a1a1a"),
        }[state]
        self.judgement_banner.setText(text)
        self.judgement_banner.setStyleSheet(
            f"QLabel#judgementBanner {{ background-color: {bg}; color: {fg}; }}"
        )

    def _compute_buzzer_pattern(self, state: Optional[str]) -> int:
        if self._buzzer_enabled and state == "RED":
            return self._buzzer_pattern
        return BUZZER_OFF

    @Slot(bool)
    def on_buzzer_enabled_changed(self, checked: bool) -> None:
        self._buzzer_enabled = checked
        self._sync_buzzer_now()

    @Slot(int)
    def on_buzzer_pattern_changed(self, index: int) -> None:
        self._buzzer_pattern = self.buzzer_pattern_combo.currentData()
        self._sync_buzzer_now()

    @Slot()
    def on_buzzer_test_clicked(self) -> None:
        if not self._tower_connected or self._tower is None:
            self.status_label.setText("Tower Light: Not Connected — cannot test buzzer.")
            return
        try:
            self._tower.set_buzzer(self._buzzer_pattern, limit=1)
        except Exception:
            self._tower_connected = False
            self.tower_status_label.setText("Tower Light: Disconnected")

    def _sync_buzzer_now(self) -> None:
        """Immediately re-applies buzzer on/off, independent of whether the
        judgement state itself just changed — so toggling the checkbox or
        switching styles takes effect right away, not on the next transition.

        If NG is already active, applies the style continuously (real alarm).
        Otherwise, enabling or changing style while idle plays a short preview
        beep (limit=1, auto-stops via the tower's own repeat-count) so the
        operator can hear the selected style without needing a live NG."""
        if not self._tower_connected or self._tower is None:
            return
        try:
            if self._last_tower_state == "RED" and self._buzzer_enabled:
                self._tower.set_buzzer(self._buzzer_pattern)
            elif self._buzzer_enabled:
                self._tower.set_buzzer(self._buzzer_pattern, limit=1)
            else:
                self._tower.set_buzzer(BUZZER_OFF)
        except Exception:
            self._tower_connected = False
            self.tower_status_label.setText("Tower Light: Disconnected")

    def _update_tower_light(self, state: str) -> None:
        if not self._tower_connected or self._tower is None:
            return

        if state == self._last_tower_state:
            return

        try:
            if state == "GREEN":
                self._tower.set_tower(green=LED_ON)
            elif state == "RED":
                self._tower.set_tower(red=LED_ON)
            else:
                self._tower.set_tower(yellow=LED_ON)
            self._tower.set_buzzer(self._compute_buzzer_pattern(state))
            self._last_tower_state = state
        except Exception:
            self._tower_connected = False
            self.tower_status_label.setText("Tower Light: Disconnected")

    def _shutdown_tower_light(self) -> None:
        if self._tower is not None:
            try:
                self._tower.reset()
            except Exception:
                pass
            try:
                self._tower.close()
            except Exception:
                pass
        self._tower = None
        self._tower_connected = False
        self._last_tower_state = None
        self.tower_status_label.setText("Tower Light: Not Connected")

    @Slot()
    def start_stream(self) -> None:
        if self._worker.isRunning():
            self.status_label.setText("Already running.")
            return
        if self._training_worker is not None and self._training_worker.isRunning():
            self.status_label.setText("Training in progress. Wait for it to finish.")
            return
        index = self.camera_combo.currentData()
        if index is None:
            self.status_label.setText("Select a valid camera.")
            return
        self._worker.set_camera_index(int(index))
        self._worker.start()
        if not self._tower_connected:
            self._connect_tower_light()

    @Slot()
    def stop_stream(self) -> None:
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        self._shutdown_tower_light()

    @Slot()
    def capture_frame(self) -> None:
        if self._current_frame_image is None:
            self.status_label.setText("No frame to capture.")
            return

        CAPTURES_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        CAPTURES_LABELS_DIR.mkdir(parents=True, exist_ok=True)

        model_stem = (
            Path(self._worker._model_path).stem if self._worker._model_path else "capture"
        )
        stem = f"{model_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        image_path = CAPTURES_IMAGES_DIR / f"{stem}.jpg"
        label_path = CAPTURES_LABELS_DIR / f"{stem}.txt"

        pixmap = QPixmap.fromImage(self._current_frame_image)
        if not pixmap.save(str(image_path), "JPG"):
            self.status_label.setText("Frame save failed.")
            return

        img_w = self._current_frame_image.width()
        img_h = self._current_frame_image.height()
        boxes = []
        for det in self._current_frame_detections:
            box = pixel_box_to_yolo(det["x1"], det["y1"], det["x2"], det["y2"], img_w, img_h)
            box["cls_id"] = det["cls_id"]
            boxes.append(box)
        write_yolo_label_file(label_path, boxes)

        self.status_label.setText(f"Captured: {image_path.name} ({len(boxes)} boxes)")

    @Slot(int)
    def on_confidence_changed(self, value: int) -> None:
        self._confidence_threshold = value / 100.0
        self.confidence_label.setText(f"Confidence: {value}%")

    @Slot(int)
    def on_max_detections_changed(self, value: int) -> None:
        self._max_detections = value

    def _toggle_class_filter(self, class_name: str, checked: bool) -> None:
        if checked:
            self._selected_classes.add(class_name)
        else:
            self._selected_classes.discard(class_name)

    @Slot(QImage, list)
    def on_frame(self, image: QImage, detections: list) -> None:
        pixmap = QPixmap.fromImage(image)
        
        # Filter detections based on selected classes and confidence threshold
        filtered_detections = []
        for det in detections:
            class_name = det.get("class_name", "unknown")
            conf = det.get("conf", 0.0)

            # Filter by selected classes and confidence
            if class_name in self._selected_classes and conf >= self._confidence_threshold:
                filtered_detections.append(det)

        # Highest-confidence first; cap to the configured maximum (0 = unlimited)
        filtered_detections.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
        if self._max_detections > 0:
            filtered_detections = filtered_detections[: self._max_detections]

        detections = filtered_detections  # Use filtered for drawing and table
        self._current_frame_image = image
        self._current_frame_detections = detections
        state = self._compute_judgement_state(detections)
        self._update_judgement_banner(state)
        self._update_tower_light(state)

        if detections:
            # Assign colors to each detection based on class name
            for det in detections:
                class_name = det.get("class_name", "unknown")
                if class_name not in self._class_colors:
                    hue = (len(self._class_colors) * 37) % 360
                    self._class_colors[class_name] = QColor.fromHsv(hue, 200, 255)
                det["color"] = self._class_colors[class_name]
            
            painter = QPainter(pixmap)
            pen = QPen()
            pen.setWidth(2)
            for det in detections:
                color = det.get("color", Qt.green)
                pen.setColor(color)
                painter.setPen(pen)
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.drawText(x1, max(0, y1 - 6), det["label"])
            painter.end()
        self._current_pixmap = pixmap
        self.video_label.setPixmap(pixmap)
        self._update_class_table(detections)

    def _update_class_table(self, detections: list) -> None:
        self._class_counts.clear()
        for det in detections:
            class_name = det.get("class_name", "unknown")
            self._class_counts[class_name] = self._class_counts.get(class_name, 0) + 1

        self.table.setRowCount(len(self._class_counts))
        for row, (class_name, count) in enumerate(sorted(self._class_counts.items())):
            color = self._class_colors.get(class_name, Qt.white)

            class_item = QTableWidgetItem(class_name)
            count_item = QTableWidgetItem(str(count))
            brush = QBrush(color)
            class_item.setForeground(brush)
            count_item.setForeground(brush)

            self.table.setItem(row, 0, class_item)
            self.table.setItem(row, 1, count_item)

    @Slot(str)
    def on_status(self, message: str) -> None:
        self.status_label.setText(message)

    @Slot()
    def _select_train_folder(self) -> None:
        default_dir = str(CAPTURES_IMAGES_DIR) if CAPTURES_IMAGES_DIR.exists() else ""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", default_dir)
        if folder:
            self._scan_train_folder(Path(folder))

    def _scan_train_folder(self, images_dir: Path) -> None:
        labels_dir = images_dir.parent / "labels"
        self.train_folder_label.setText(str(images_dir))

        pairs = []
        skipped = 0
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))
            else:
                skipped += 1

        self._train_pairs = pairs
        self.train_list_widget.blockSignals(True)
        self.train_list_widget.clear()
        for image_path, _ in pairs:
            self.train_list_widget.addItem(image_path.name)
        self.train_list_widget.blockSignals(False)

        if pairs:
            self.train_status_label.setText(
                f"Loaded {len(pairs)} pairs ({skipped} skipped: no label)"
            )
            self._load_train_index(0)
        else:
            self._train_current_index = -1
            self.train_image_widget.set_image(QPixmap(), [])
            self.train_counter_label.setText("Image 0 / 0")
            self.train_status_label.setText(
                f"No valid image/label pairs found ({skipped} skipped: no label)"
            )

    def _load_train_index(self, index: int) -> None:
        if index < 0 or index >= len(self._train_pairs):
            return
        self._train_current_index = index
        image_path, label_path = self._train_pairs[index]

        pixmap = QPixmap(str(image_path))
        img_w, img_h = pixmap.width(), pixmap.height()

        boxes = []
        for box in read_yolo_label_file(label_path):
            x1, y1, x2, y2 = yolo_box_to_pixel(box, img_w, img_h)
            cls_id = box["cls_id"]
            class_name = (
                self._model_classes[cls_id]
                if 0 <= cls_id < len(self._model_classes)
                else str(cls_id)
            )
            boxes.append(
                {
                    "cls_id": cls_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_name": class_name,
                }
            )

        self._train_boxes = boxes
        self._train_selected_box_index = -1
        self.train_image_widget.set_image(pixmap, boxes)
        self.train_counter_label.setText(f"Image {index + 1} / {len(self._train_pairs)}")
        self.train_selected_label.setText("Selected box: none")

        self.train_list_widget.blockSignals(True)
        self.train_list_widget.setCurrentRow(index)
        self.train_list_widget.blockSignals(False)

    @Slot()
    def _train_prev(self) -> None:
        if self._train_current_index > 0:
            self._load_train_index(self._train_current_index - 1)

    @Slot()
    def _train_next(self) -> None:
        if self._train_current_index < len(self._train_pairs) - 1:
            self._load_train_index(self._train_current_index + 1)

    @Slot(int)
    def _on_train_box_clicked(self, index: int) -> None:
        if index < 0 or index >= len(self._train_boxes):
            return
        self._train_selected_box_index = index
        box = self._train_boxes[index]
        self.train_selected_label.setText(f"Selected box: {box['class_name']}")
        combo_index = self.train_class_combo.findText(box["class_name"])
        if combo_index >= 0:
            self.train_class_combo.setCurrentIndex(combo_index)

    @Slot()
    def _on_train_confirm(self) -> None:
        if self._train_selected_box_index < 0 or self._train_current_index < 0:
            return
        new_class_name = self.train_class_combo.currentText()
        if new_class_name not in self._model_classes:
            return
        new_cls_id = self._model_classes.index(new_class_name)

        box = self._train_boxes[self._train_selected_box_index]
        box["cls_id"] = new_cls_id
        box["class_name"] = new_class_name
        self.train_image_widget.set_image(self.train_image_widget.pixmap(), self._train_boxes)
        self.train_image_widget.set_selected_index(self._train_selected_box_index)

        _, label_path = self._train_pairs[self._train_current_index]
        pixmap = self.train_image_widget.pixmap()
        img_w, img_h = pixmap.width(), pixmap.height()
        out_boxes = [
            {
                **pixel_box_to_yolo(b["x1"], b["y1"], b["x2"], b["y2"], img_w, img_h),
                "cls_id": b["cls_id"],
            }
            for b in self._train_boxes
        ]
        write_yolo_label_file(label_path, out_boxes)
        self.train_status_label.setText(f"Updated label: {label_path.name}")

    @Slot()
    def _start_training(self) -> None:
        if self._worker.isRunning():
            self.train_status_label.setText("Stop the Monitor stream before starting training.")
            return
        if self._training_worker is not None and self._training_worker.isRunning():
            self.train_status_label.setText("Training already in progress.")
            return
        if self._worker._model_path is None:
            self.train_status_label.setText("Load a model first.")
            return
        if not self._train_pairs:
            self.train_status_label.setText("Select a folder with valid image/label pairs first.")
            return

        images_dir = self._train_pairs[0][0].parent
        dataset_root = images_dir.parent
        data_yaml_path = dataset_root / "data.yaml"
        with data_yaml_path.open("w") as f:
            yaml.safe_dump(
                {
                    "path": str(dataset_root.resolve()),
                    "train": "images",
                    "val": "images",
                    "nc": len(self._model_classes),
                    "names": list(self._model_classes),
                },
                f,
                sort_keys=False,
            )

        # Release the Monitor tab's loaded model before training — otherwise it
        # stays resident in memory (the stream only stops running, it never
        # clears self._worker._model), and a second full model plus training's
        # own memory needs can exceed this device's RAM. See: OOM kills observed
        # in journalctl during training with both loaded at once.
        self._worker._model = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = "0" if (torch is not None and torch.cuda.is_available()) else "cpu"
        config = {
            "model_path": str(self._worker._model_path),
            "data_yaml": str(data_yaml_path),
            "epochs": 50,
            "imgsz": 640,
            "batch": 8,
            "patience": 20,
            "optimizer": "auto",
            "lr0": 0.01,
            "device": device,
            "project": str(dataset_root / "runs"),
            "name": "train",
        }

        self._training_worker = TrainingWorker(config)
        self._training_worker.status_update.connect(self._on_training_status)
        self._training_worker.training_complete.connect(self._on_training_complete)
        self._training_worker.training_error.connect(self._on_training_error)
        self._training_worker.finished.connect(self._on_training_thread_finished)
        self.train_start_button.setEnabled(False)
        self._training_worker.start()

    @Slot(str)
    def _on_training_status(self, message: str) -> None:
        self.train_status_label.setText(message)

    @Slot(str)
    def _on_training_complete(self, best_path: str) -> None:
        self.train_status_label.setText(f"Training complete. Best model: {best_path}")

    @Slot(str)
    def _on_training_error(self, message: str) -> None:
        self.train_status_label.setText(message)

    def _on_training_thread_finished(self) -> None:
        self.train_start_button.setEnabled(True)
        self._training_worker = None

    def closeEvent(self, event) -> None:
        self.stop_stream()
        if self._barcode_dialog is not None:
            self._barcode_dialog.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_INDUSTRIAL_QSS)
    window = MainWindow()
    window.resize(1050, 720)
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()