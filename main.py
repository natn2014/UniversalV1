import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QEventPoint,
    QFont,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
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


# Keyence-style skin: near-black frame, glossy gray gradient buttons, blue
# gloss for the selected tab, dark header strips and a light-gray side panel.
GLOSS_GRAY = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6b737e, stop:0.48 #4a525c,"
    " stop:0.52 #3c434c, stop:1 #2e343b)"
)
GLOSS_GRAY_HOVER = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7d8692, stop:0.48 #5a636e,"
    " stop:0.52 #4b535d, stop:1 #3a414a)"
)
GLOSS_GRAY_PRESSED = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2e343b, stop:0.5 #3c434c,"
    " stop:1 #4a525c)"
)
GLOSS_ORANGE = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffcf73, stop:0.48 #f0a330,"
    " stop:0.52 #e08e16, stop:1 #b86e0a)"
)
GLOSS_BLUE = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a9be6, stop:0.48 #2b6fc9,"
    " stop:0.52 #1f5fb8, stop:1 #0d47a1)"
)
DARK_STRIP = (
    "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2f343b, stop:1 #16191d)"
)

# QSS can only draw sub-control arrows from image files, so the spinbox
# arrows are written once as tiny SVGs next to the app.
UI_ASSETS_DIR = Path(__file__).resolve().parent / ".ui_assets"


def _write_arrow_svgs() -> tuple:
    UI_ASSETS_DIR.mkdir(exist_ok=True)
    paths = []
    for name, points in (("arrow_up", "2,9 7,3 12,9"), ("arrow_down", "2,4 7,10 12,4")):
        path = UI_ASSETS_DIR / f"{name}.svg"
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="13" viewBox="0 0 14 13">'
            f'<polygon points="{points}" fill="#e8eaed"/></svg>'
        )
        if not path.exists() or path.read_text() != svg:
            path.write_text(svg)
        paths.append(path.as_posix())
    return tuple(paths)


ARROW_UP_SVG, ARROW_DOWN_SVG = _write_arrow_svgs()

DARK_INDUSTRIAL_QSS = f"""
QWidget {{
    background-color: #15181c;
    color: #e0e0e0;
    font-size: 13pt;
}}
QLabel, QCheckBox, QRadioButton {{
    background: transparent;
}}
QTabWidget::pane {{
    border: 1px solid #0b0d10;
    background-color: #1f2329;
}}
QTabBar::tab {{
    background: {GLOSS_GRAY};
    color: #d0d5db;
    padding: 10px 24px;
    border: 1px solid #0b0d10;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background: {GLOSS_BLUE};
    color: #ffffff;
}}
QTabBar::tab:hover:!selected {{
    background: {GLOSS_GRAY_HOVER};
}}
QGroupBox {{
    border: 1px solid #0b0d10;
    border-radius: 4px;
    margin-top: 14px;
    padding: 10px;
    font-weight: 600;
    color: #7fd0ff;
    background-color: #1f2329;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}}
QPushButton {{
    background: {GLOSS_GRAY};
    color: #f0f2f4;
    border: 1px solid #0e1013;
    border-radius: 3px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background: {GLOSS_GRAY_HOVER};
}}
QPushButton:pressed {{
    background: {GLOSS_GRAY_PRESSED};
}}
QPushButton:checked {{
    background: {GLOSS_ORANGE};
    color: #1a1a1a;
}}
QPushButton:disabled {{
    background: #2a2f36;
    color: #7a838d;
}}
QFrame#monitorTopBar, QFrame#monitorBottomBar {{
    background: {DARK_STRIP};
    border: 1px solid #0b0d10;
    border-radius: 3px;
}}
QFrame#monitorTopBar QLabel, QFrame#monitorBottomBar QLabel {{
    background: transparent;
}}
QFrame#videoFrame {{
    background-color: #0b0d10;
    border: 2px solid #e0a526;
}}
QFrame#videoFrame QLabel {{
    background-color: #0b0d10;
}}
QFrame#sidePanel {{
    background-color: #d5d9de;
    border: 1px solid #0b0d10;
}}
QLineEdit, QComboBox, QSpinBox {{
    background-color: #0f1114;
    color: #e8eaed;
    border: 1px solid #3a4048;
    border-radius: 3px;
    padding: 4px 8px;
    min-height: 28px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border-color: #e0a526;
}}
QSpinBox {{
    padding-right: 40px;
}}
QSpinBox::up-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    width: 36px;
    background: {GLOSS_GRAY};
    border: 1px solid #0e1013;
}}
QSpinBox::up-button {{
    subcontrol-position: top right;
    border-top-right-radius: 3px;
}}
QSpinBox::down-button {{
    subcontrol-position: bottom right;
    border-bottom-right-radius: 3px;
}}
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {{
    background: {GLOSS_GRAY_PRESSED};
}}
QSpinBox::up-arrow {{
    image: url({ARROW_UP_SVG});
    width: 14px;
    height: 13px;
}}
QSpinBox::down-arrow {{
    image: url({ARROW_DOWN_SVG});
    width: 14px;
    height: 13px;
}}
QComboBox QAbstractItemView {{
    background-color: #1f2329;
    color: #e8eaed;
    selection-background-color: #1f5fb8;
    border: 1px solid #0b0d10;
}}
QLineEdit#readout {{
    background-color: #0b0d10;
    color: #ffcf73;
    font-size: 18pt;
    font-weight: 800;
    border: 1px solid #e0a526;
    min-height: 40px;
}}
QSlider::groove:horizontal {{
    height: 8px;
    background: #0f1114;
    border: 1px solid #3a4048;
    border-radius: 4px;
}}
QSlider::sub-page:horizontal {{
    background: {GLOSS_BLUE};
    border: 1px solid #0b0d10;
    border-radius: 4px;
}}
QSlider::handle:horizontal {{
    background: {GLOSS_GRAY};
    border: 1px solid #0e1013;
    width: 22px;
    margin: -9px 0;
    border-radius: 4px;
}}
QScrollBar:vertical {{
    background: #15181c;
    width: 14px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #4a525c;
    min-height: 32px;
    border-radius: 5px;
    margin: 2px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
QFrame#card {{
    background-color: #1f2329;
    border: 1px solid #0b0d10;
    border-radius: 4px;
}}
QLabel#cardHeader {{
    background: {DARK_STRIP};
    color: #7fd0ff;
    font-weight: 700;
    padding: 6px 12px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QLabel#hint {{
    color: #9aa5b1;
    font-size: 11pt;
}}
QLabel#currentPart {{
    color: #ffcf73;
    font-size: 18pt;
    font-weight: 800;
}}
QPushButton#jobButton {{
    font-size: 15pt;
    font-weight: 700;
    min-height: 72px;
}}
QLabel#panelText {{
    color: #1a1d22;
}}
QFrame#sidePanel QListWidget {{
    background-color: #d5d9de;
    alternate-background-color: #c8cdd3;
    color: #1a1d22;
    border: none;
}}
QFrame#sidePanel QListWidget::item {{
    padding: 4px 6px;
}}
QFrame#sidePanel QListWidget::item:selected {{
    background: {GLOSS_BLUE};
    color: #ffffff;
}}
QLabel#panelHeader {{
    background: {DARK_STRIP};
    color: #ffffff;
    font-weight: 700;
    padding: 6px 10px;
}}
QFrame#sidePanel QTableWidget {{
    background-color: #d5d9de;
    alternate-background-color: #c8cdd3;
    color: #1a1d22;
    gridline-color: #aab1b9;
    border: none;
}}
QFrame#sidePanel QHeaderView::section {{
    background: {GLOSS_GRAY};
    color: #ffffff;
    border: 1px solid #0e1013;
    padding: 4px;
    font-weight: 600;
}}
QWidget#counterBox, QWidget#clearBox {{
    background: transparent;
}}
QFrame#counterTile {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #262b32, stop:1 #0f1114);
    border: 1px solid #0b0d10;
    border-radius: 6px;
}}
QLabel#counterTitle {{
    color: #ffffff;
    font-size: 13pt;
    font-weight: 800;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 3px 0;
}}
QLabel#counterValue {{
    font-size: 40pt;
    font-weight: 800;
    padding: 0 4px 4px 4px;
}}
QPushButton#counterReset {{
    font-size: 11pt;
    padding: 4px 10px;
}}
QFrame#ngPartGroup {{
    background-color: #1f2329;
    border: 1px solid #0b0d10;
    border-radius: 4px;
}}
QLabel#ngPartHeader {{
    background: {DARK_STRIP};
    color: #ffffff;
    font-weight: 700;
    padding: 6px 10px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QToolButton#ngThumb {{
    background-color: #0b0d10;
    color: #c7cdd4;
    border: 2px solid #b83225;
    border-radius: 4px;
    padding: 3px;
    font-size: 10pt;
}}
QToolButton#ngThumb:hover {{
    border-color: #ff5a4a;
    background-color: #1a1d22;
}}
QWidget#imageOverlay {{
    background-color: rgba(0, 0, 0, 215);
}}
QLabel#overlayTitle {{
    color: #ffffff;
    font-size: 15pt;
    font-weight: 700;
}}
QLabel#overlayZoom {{
    color: #ffcf73;
    font-weight: 700;
    min-width: 70px;
}}
QLabel#overlayHint {{
    color: #9aa5b1;
    font-size: 11pt;
}}
QPushButton#overlayClose {{
    background: {GLOSS_GRAY};
    border: 1px solid #0e1013;
    border-radius: 22px;
    padding: 0;
}}
QPushButton#overlayClose:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e0685a, stop:1 #8e2419);
}}
QLabel#monitorInfo {{
    color: #c7cdd4;
    font-weight: 600;
}}
QLabel#judgementBanner {{
    font-size: 15pt;
    font-weight: 800;
    border: 2px solid #0b0d10;
    border-radius: 18px;
    padding: 4px 18px;
}}
"""

ALLOWED_FPS = [24, 30, 60]

# CAP_DSHOW is Windows-only; use V4L2 on Linux (Jetson) and let OpenCV
# auto-pick elsewhere.
CAMERA_BACKEND = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY

CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
CAPTURES_IMAGES_DIR = CAPTURES_DIR / "images"
CAPTURES_LABELS_DIR = CAPTURES_DIR / "labels"
# NG log: ng_logs/<part>/<timestamp>.jpg (annotated frame) + <timestamp>_thumb.jpg
NG_LOGS_DIR = Path(__file__).resolve().parent / "ng_logs"
NG_THUMB_SUFFIX = "_thumb.jpg"
NG_THUMB_SIZE = QSize(160, 120)  # 5 across fit the 1024-wide kiosk screen
NG_THUMBS_PER_PART = 5
NO_PART_NAME = "No Part"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PART_CONFIGS_DIR = Path(__file__).resolve().parent / "part_configs"
LAST_PART_FILE = PART_CONFIGS_DIR / ".last_part"
APP_SETTINGS_FILE = Path(__file__).resolve().parent / "app_settings.json"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_part_no(part_no: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", part_no.strip())
    return cleaned.strip("_") or "unknown_part"


# Buzzer defaults to on so NG buzzes out of the box — the app auto-starts on
# boot, so an unpersisted default-off would silently disable the alarm.
DEFAULT_APP_SETTINGS = {
    "buzzer_enabled": True,
    "buzzer_pattern": BUZZER_ON,
    "ok_beep_enabled": True,
    "ok_beep_ms": 50,
    # Minimum time a STANDBY/OK/NG judgement is held before it may change.
    "state_hold_ms": 1000,
    # OK/NG cycle counters, persisted so a kiosk restart doesn't lose them.
    "ok_count": 0,
    "ng_count": 0,
}


def make_icon(kind: str, color: str = "#ffffff") -> QIcon:
    """Small vector icons drawn in code, so no image files or emoji fonts are
    needed on the Jetson. Drawn on a 20x20 grid at 2x for crisp edges."""
    scale = 2
    pixmap = QPixmap(20 * scale, 20 * scale)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(scale, scale)
    fill = QColor(color)
    stroke = QPen(fill, 1.8)
    stroke.setCapStyle(Qt.RoundCap)

    if kind == "play":
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawPolygon(QPolygonF([QPointF(6, 3.5), QPointF(16.5, 10), QPointF(6, 16.5)]))
    elif kind == "stop":
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawRoundedRect(QRectF(4.5, 4.5, 11, 11), 1.5, 1.5)
    elif kind == "camera":
        painter.setPen(stroke)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(2.5, 6, 15, 10), 2, 2)
        painter.drawRect(QRectF(7, 3.5, 6, 2.5))
        painter.drawEllipse(QPointF(10, 11), 3, 3)
    elif kind == "close":
        pen = QPen(fill, 2.4)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(5, 5), QPointF(15, 15))
        painter.drawLine(QPointF(15, 5), QPointF(5, 15))
    elif kind in ("speaker_on", "speaker_off"):
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawPolygon(QPolygonF([
            QPointF(2.5, 7.5), QPointF(6, 7.5), QPointF(10.5, 3.5),
            QPointF(10.5, 16.5), QPointF(6, 12.5), QPointF(2.5, 12.5),
        ]))
        painter.setPen(stroke)
        painter.setBrush(Qt.NoBrush)
        if kind == "speaker_on":
            painter.drawArc(QRectF(9, 6.5, 6, 7), -60 * 16, 120 * 16)
            painter.drawArc(QRectF(8.5, 3.5, 9.5, 13), -60 * 16, 120 * 16)
        else:
            painter.drawLine(QPointF(13, 7.5), QPointF(18, 12.5))
            painter.drawLine(QPointF(18, 7.5), QPointF(13, 12.5))

    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return QIcon(pixmap)


# Run button looks: colour signals what a click will *do*, never status
# (status has its own chip next to the Start/Stop button). Blue gloss for
# Start mirrors the "Run" button of the Keyence reference UI.
RUN_BUTTON_STYLES = {
    "start": ("#5a9be6", "#1f5fb8", "#0d47a1"),
    "stop": ("#e0685a", "#b83225", "#8e2419"),
}


def gloss_gradient(top: str, mid: str, bottom: str) -> str:
    return (
        f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {top}, stop:0.48 {mid},"
        f" stop:0.52 {QColor(mid).darker(112).name()}, stop:1 {bottom})"
    )


def action_button_qss(top: str, mid: str, bottom: str) -> str:
    hover = gloss_gradient(QColor(top).lighter(112).name(), QColor(mid).lighter(112).name(),
                           QColor(bottom).lighter(112).name())
    return (
        f"QPushButton {{ background: {gloss_gradient(top, mid, bottom)}; color: #ffffff;"
        f" border: 1px solid #0b0d10; border-radius: 4px; font-weight: 700;"
        f" font-size: 14pt; padding: 6px 16px; }}"
        f"QPushButton:hover {{ background: {hover}; }}"
        f"QPushButton:pressed {{ background: {gloss_gradient(bottom, mid, top)}; }}"
        f"QPushButton:disabled {{ background: #2a2f36; color: #7a838d; }}"
    )


def load_app_settings() -> dict:
    settings = dict(DEFAULT_APP_SETTINGS)
    try:
        data = json.loads(APP_SETTINGS_FILE.read_text())
        for key, default in DEFAULT_APP_SETTINGS.items():
            if key in data:
                settings[key] = type(default)(data[key])
    except (OSError, ValueError, TypeError):
        pass
    return settings


def save_app_settings(settings: dict) -> None:
    data = {}
    try:
        data = json.loads(APP_SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        pass
    data.update(settings)
    APP_SETTINGS_FILE.write_text(json.dumps(data, indent=2))


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

    def start(self) -> None:
        # Set before the thread begins, not inside run(): a stop() arriving
        # during camera open or model load must not be overwritten.
        self._running = True
        super().start()

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

        if self._running and self._model_path is not None and self._model is None:
            self._load_model()

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
        self.manual_load_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.manual_load_button.clicked.connect(self._on_manual_load)
        self.manual_save_button = QPushButton("Save")
        self.manual_save_button.clicked.connect(self._on_manual_save)
        for button in (self.manual_load_button, self.manual_save_button):
            button.setFixedHeight(40)
            button.setMinimumWidth(96)

        self.close_button = QPushButton("Close")
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
        app_settings = load_app_settings()
        self._buzzer_enabled: bool = app_settings["buzzer_enabled"]
        self._buzzer_pattern: int = app_settings["buzzer_pattern"]
        self._ok_beep_enabled: bool = app_settings["ok_beep_enabled"]
        self._ok_beep_ms: int = app_settings["ok_beep_ms"]
        self._ok_beep_token: int = 0
        self._state_hold_ms: int = app_settings["state_hold_ms"]
        self._judged_state: Optional[str] = None
        self._judged_since: float = 0.0
        self._ok_count: int = app_settings["ok_count"]
        self._pending_ng_save: bool = False
        self._ng_count: int = app_settings["ng_count"]
        self._current_frame_image: Optional[QImage] = None
        self._current_frame_detections: list = []
        self._train_pairs: list = []
        self._train_current_index: int = -1
        self._train_boxes: list = []
        self._train_selected_box_index: int = -1
        self._training_worker: Optional[TrainingWorker] = None
        self._current_part_no: Optional[str] = None
        self._barcode_dialog: Optional[BarcodeScanDialog] = None
        self._stop_requested: bool = False

        self.model_label = QLabel("No model selected")
        self.load_model_button = QPushButton("Load .pt Model")
        self.load_model_button.clicked.connect(self.load_model)

        self.camera_combo = QComboBox()
        self.scan_button = QPushButton("Scan Cameras")
        self.scan_button.clicked.connect(self.scan_cameras)

        toolbar_height = 40
        icon_size = QSize(20, 20)

        self.run_button = QPushButton()
        self.run_button.setIconSize(icon_size)
        self.run_button.setFixedSize(170, 44)
        self.run_button.clicked.connect(self.toggle_stream)
        self.run_status_label = QLabel()
        self.run_status_label.setMinimumWidth(120)
        # Follow the worker's real state, so the button also flips back if the
        # stream ends on its own (camera open failure, lost frames).
        self._worker.started.connect(self._refresh_run_button)
        self._worker.finished.connect(self._refresh_run_button)
        self._refresh_run_button()

        self.capture_button = QPushButton("Capture")
        self.capture_button.setIcon(make_icon("camera", "#e0e0e0"))
        self.capture_button.setIconSize(icon_size)
        self.capture_button.setFixedHeight(toolbar_height)
        self.capture_button.clicked.connect(self.capture_frame)

        self.status_label = QLabel("Ready")
        self.tower_status_label = QLabel("Tower Light: Not Connected")

        # Saved values are applied before signals are connected, so restoring
        # them at startup doesn't fire a preview beep or rewrite the file.
        self.buzzer_enabled_checkbox = QCheckBox("Enable Buzzer")
        self.buzzer_enabled_checkbox.setChecked(self._buzzer_enabled)
        self.buzzer_enabled_checkbox.toggled.connect(self.on_buzzer_enabled_changed)

        self.buzzer_pattern_combo = self._make_buzzer_style_combo(self._buzzer_pattern)
        self._buzzer_pattern = self.buzzer_pattern_combo.currentData()
        self.buzzer_pattern_combo.currentIndexChanged.connect(self.on_buzzer_pattern_changed)

        self.ok_beep_checkbox = QCheckBox("Beep on OK")
        self.ok_beep_checkbox.setChecked(self._ok_beep_enabled)
        self.ok_beep_checkbox.toggled.connect(self.on_ok_beep_enabled_changed)

        self.ok_beep_duration_spinbox = QSpinBox()
        self.ok_beep_duration_spinbox.setRange(20, 1000)
        self.ok_beep_duration_spinbox.setSingleStep(10)
        self.ok_beep_duration_spinbox.setSuffix(" ms")
        self.ok_beep_duration_spinbox.setValue(self._ok_beep_ms)
        self._ok_beep_ms = self.ok_beep_duration_spinbox.value()
        self.ok_beep_duration_spinbox.valueChanged.connect(self.on_ok_beep_ms_changed)

        self.ok_beep_test_button = QPushButton("Test")

        self.state_hold_spinbox = QSpinBox()
        self.state_hold_spinbox.setRange(0, 5000)
        self.state_hold_spinbox.setSingleStep(100)
        self.state_hold_spinbox.setSuffix(" ms")
        self.state_hold_spinbox.setValue(self._state_hold_ms)
        self._state_hold_ms = self.state_hold_spinbox.value()
        self.state_hold_spinbox.valueChanged.connect(self.on_state_hold_changed)
        self.ok_beep_test_button.clicked.connect(self.on_ok_beep_test_clicked)

        # Quick-access on/off toggle on the Monitor tab, mirroring the checkbox.
        self.buzzer_toggle_button = QPushButton()
        self.buzzer_toggle_button.setIconSize(icon_size)
        self.buzzer_toggle_button.setFixedSize(140, toolbar_height)
        self.buzzer_toggle_button.setCheckable(True)
        self.buzzer_toggle_button.setChecked(self._buzzer_enabled)
        self.buzzer_toggle_button.toggled.connect(self.buzzer_enabled_checkbox.setChecked)
        self._refresh_buzzer_toggle_button()

        self.buzzer_test_button = QPushButton("Test")
        self.buzzer_test_button.clicked.connect(self.on_buzzer_test_clicked)

        self.part_config_label = QLabel("No part scanned")
        self.scan_barcode_button = QPushButton("Scan Barcode...")
        self.scan_barcode_button.setFixedHeight(44)
        self.scan_barcode_button.setMinimumWidth(200)
        self.scan_barcode_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.scan_barcode_button.clicked.connect(self._open_barcode_scan_dialog)

        self.judgement_banner = QLabel()
        self.judgement_banner.setObjectName("judgementBanner")
        self.judgement_banner.setAlignment(Qt.AlignCenter)
        self.judgement_banner.setFixedSize(130, 36)

        self.monitor_info_label = QLabel()
        self.monitor_info_label.setObjectName("monitorInfo")
        self._refresh_monitor_info()

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
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

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
        self.train_select_folder_button = QPushButton("Select Folder...")
        self.train_select_folder_button.clicked.connect(self._select_train_folder)
        self.train_folder_label = QLabel("(none selected)")

        self.train_list_widget = QListWidget()
        self.train_list_widget.currentRowChanged.connect(self._load_train_index)
        # Capture names are long; elide the left so the date and time stay visible.
        self.train_list_widget.setTextElideMode(Qt.ElideLeft)
        self.train_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.train_prev_button = QPushButton("◀ Prev")
        self.train_prev_button.clicked.connect(self._train_prev)
        self.train_next_button = QPushButton("Next ▶")
        self.train_next_button.clicked.connect(self._train_next)
        self.train_counter_label = QLabel("Image 0 / 0")

        self.train_image_widget = AnnotatedImageWidget()
        self.train_image_widget.box_clicked.connect(self._on_train_box_clicked)

        self.train_selected_label = QLabel("Selected box: none")
        self.train_class_combo = QComboBox()
        self.train_confirm_button = QPushButton("Confirm")
        self.train_confirm_button.clicked.connect(self._on_train_confirm)

        self.train_start_button = QPushButton("Start Training")
        self.train_start_button.setIcon(make_icon("play"))
        self.train_start_button.setIconSize(QSize(20, 20))
        self.train_start_button.setFixedHeight(44)
        self.train_start_button.setMinimumWidth(190)
        self.train_start_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.train_start_button.clicked.connect(self._start_training)
        self.train_status_label = QLabel("Ready")
        # Long training messages must not widen the bottom bar.
        self.train_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.train_status_label.setStyleSheet("color: #c7cdd4;")

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_monitor_tab(), "Monitor")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs.addTab(self._build_job_change_tab(), "Job Change")
        self.tabs.addTab(self._build_train_tab(), "Train")
        self.log_tab_index = self.tabs.addTab(self._build_log_tab(), "Log")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._refresh_job_buttons()
        self._ng_log_dirty = True
        self.image_overlay = ImageOverlay(self)

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.tabs, 1)
        root_layout.addWidget(self.status_label)
        self.setLayout(root_layout)

        self.scan_cameras()
        self._connect_tower_light()
        self._update_judgement_banner("YELLOW")
        self._auto_load_last_config()

    def _build_monitor_tab(self) -> QWidget:
        # Layout follows the Keyence reference: info strip + judgement pill on
        # top, framed image with a light side panel, action bar at the bottom
        # with Run on the far right.
        top_bar = QFrame()
        top_bar.setObjectName("monitorTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 4, 6, 4)
        top_layout.addWidget(self.monitor_info_label)
        top_layout.addStretch(1)
        top_layout.addWidget(self.judgement_banner)

        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(2, 2, 2, 2)
        video_layout.addWidget(self.video_label)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        panel_header = QLabel("Detections")
        panel_header.setObjectName("panelHeader")
        side_layout.addWidget(panel_header)
        side_layout.addWidget(self.table, 1)
        side_layout.addWidget(self._build_cycle_counters())

        content_row = QHBoxLayout()
        content_row.addWidget(video_frame, 1)
        content_row.addWidget(side_panel)

        bottom_bar = QFrame()
        bottom_bar.setObjectName("monitorBottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 4, 6, 4)
        bottom_layout.addWidget(self.capture_button)
        bottom_layout.addWidget(self.buzzer_toggle_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.run_status_label)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.run_button)

        layout = QVBoxLayout()
        layout.addWidget(top_bar)
        layout.addLayout(content_row, 1)
        layout.addWidget(bottom_bar)

        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _make_counter_tile(self, title: str, top: str, bottom: str, digits: str):
        tile = QFrame()
        tile.setObjectName("counterTile")
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(0, 0, 0, 0)
        tile_layout.setSpacing(0)
        title_label = QLabel(title)
        title_label.setObjectName("counterTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            f"QLabel#counterTitle {{ background: {gloss_gradient(top, bottom, bottom)}; }}"
        )
        value_label = QLabel("0")
        value_label.setObjectName("counterValue")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet(f"QLabel#counterValue {{ color: {digits}; }}")
        tile_layout.addWidget(title_label)
        tile_layout.addWidget(value_label, 1)
        return tile, value_label

    def _build_cycle_counters(self) -> QWidget:
        ok_tile, self.ok_count_label = self._make_counter_tile(
            "OK", "#5cbf60", "#2e7d32", "#4cd964")
        ng_tile, self.ng_count_label = self._make_counter_tile(
            "NG", "#e0685a", "#b83225", "#ff5a4a")

        self.counter_reset_button = QPushButton("Reset Count")
        self.counter_reset_button.setObjectName("counterReset")
        self.counter_reset_button.clicked.connect(self.on_reset_counters_clicked)

        tiles_row = QHBoxLayout()
        tiles_row.setSpacing(6)
        tiles_row.addWidget(ok_tile, 1)
        tiles_row.addWidget(ng_tile, 1)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        reset_row.addWidget(self.counter_reset_button)

        box = QWidget()
        box.setObjectName("counterBox")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(tiles_row)
        layout.addLayout(reset_row)
        self._refresh_cycle_counters()
        return box

    def _refresh_cycle_counters(self) -> None:
        self.ok_count_label.setText(str(self._ok_count))
        self.ng_count_label.setText(str(self._ng_count))

    def _count_cycle(self, state: str) -> None:
        if state == "GREEN":
            self._ok_count += 1
        elif state == "RED":
            self._ng_count += 1
            # The frame is annotated later in on_frame; save it there.
            self._pending_ng_save = True
        else:
            return
        self._refresh_cycle_counters()
        self._save_app_settings()

    @Slot()
    def on_reset_counters_clicked(self) -> None:
        answer = QMessageBox.question(
            self, "Reset Count",
            f"Reset cycle counters?\n\nOK: {self._ok_count}    NG: {self._ng_count}",
        )
        if answer != QMessageBox.Yes:
            return
        self._ok_count = 0
        self._ng_count = 0
        self._refresh_cycle_counters()
        self._save_app_settings()

    def _build_log_tab(self) -> QWidget:
        self.ng_log_container = QWidget()
        self.ng_log_layout = QVBoxLayout(self.ng_log_container)
        self.ng_log_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.ng_log_container)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ng_log_summary_label = QLabel()
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._refresh_ng_log)
        open_row = QHBoxLayout()
        open_row.addWidget(self.ng_log_summary_label)
        open_row.addStretch(1)
        open_row.addWidget(refresh_button)

        layout = QVBoxLayout()
        layout.addLayout(open_row)
        layout.addWidget(scroll, 1)
        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def _save_ng_image(self, pixmap: QPixmap) -> None:
        part = self._current_part_no or NO_PART_NAME
        part_dir = NG_LOGS_DIR / sanitize_part_no(part)
        try:
            part_dir.mkdir(parents=True, exist_ok=True)
            (part_dir / ".part_name").write_text(part)
            stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
            pixmap.save(str(part_dir / f"{stamp}.jpg"), "JPG", 90)
            pixmap.scaled(NG_THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation).save(
                str(part_dir / f"{stamp}{NG_THUMB_SUFFIX}"), "JPG", 85)
        except OSError as exc:
            self.status_label.setText(f"NG image save failed: {exc}")
            return
        # Rebuild only if the operator is looking at the Log tab right now;
        # otherwise defer to the next time it's opened (keeps Monitor fast).
        self._ng_log_dirty = True
        if self.tabs.currentIndex() == self.log_tab_index:
            self._refresh_ng_log()

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        if index == self.log_tab_index and self._ng_log_dirty:
            self._refresh_ng_log()

    def _collect_ng_log(self) -> list:
        """[(part_name, [thumb paths newest first], total count)], parts with
        the most recent NG first."""
        groups = []
        if not NG_LOGS_DIR.is_dir():
            return groups
        for part_dir in NG_LOGS_DIR.iterdir():
            if not part_dir.is_dir():
                continue
            # Timestamped names sort chronologically.
            thumbs = sorted(part_dir.glob(f"*{NG_THUMB_SUFFIX}"), reverse=True)
            if not thumbs:
                continue
            name_file = part_dir / ".part_name"
            name = name_file.read_text().strip() if name_file.exists() else part_dir.name
            groups.append((name, thumbs, len(thumbs)))
        groups.sort(key=lambda g: g[1][0].name, reverse=True)
        return groups

    @staticmethod
    def _ng_stamp_text(thumb_path: Path) -> str:
        stamp = thumb_path.name[: -len(NG_THUMB_SUFFIX)]
        try:
            t = time.strptime(stamp[:15], "%Y%m%d_%H%M%S")
            return time.strftime("%Y-%m-%d %H:%M:%S", t)
        except ValueError:
            return stamp

    @Slot()
    def _refresh_ng_log(self) -> None:
        self._ng_log_dirty = False
        while self.ng_log_layout.count():
            item = self.ng_log_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()

        groups = self._collect_ng_log()
        total = sum(g[2] for g in groups)
        self.ng_log_summary_label.setText(
            f"NG images: {total}   ·   Parts: {len(groups)}   (latest part on top)")

        if not groups:
            empty = QLabel("No NG recorded yet.")
            empty.setStyleSheet("color: #9aa5b1; padding: 20px;")
            self.ng_log_layout.addWidget(empty)

        for part_name, thumbs, count in groups:
            group = QFrame()
            group.setObjectName("ngPartGroup")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 8)
            header = QLabel(
                f"Part: {part_name}   ·   NG {count}   ·   Last: {self._ng_stamp_text(thumbs[0])}")
            header.setObjectName("ngPartHeader")
            group_layout.addWidget(header)

            row = QHBoxLayout()
            row.setContentsMargins(8, 4, 8, 0)
            row.setSpacing(8)
            for thumb in thumbs[:NG_THUMBS_PER_PART]:
                row.addWidget(self._make_ng_thumb_button(part_name, thumb))
            row.addStretch(1)
            group_layout.addLayout(row)
            self.ng_log_layout.addWidget(group)
        self.ng_log_layout.addStretch(1)

    def _make_ng_thumb_button(self, part_name: str, thumb: Path) -> QToolButton:
        full_image = thumb.with_name(thumb.name[: -len(NG_THUMB_SUFFIX)] + ".jpg")
        stamp_text = self._ng_stamp_text(thumb)
        button = QToolButton()
        button.setObjectName("ngThumb")
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setIcon(QIcon(str(thumb)))
        button.setIconSize(NG_THUMB_SIZE)
        button.setText(stamp_text)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(
            lambda: self.image_overlay.show_image(full_image, f"NG  ·  {part_name}  ·  {stamp_text}"))
        return button

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "image_overlay") and self.image_overlay.isVisible():
            self.image_overlay.setGeometry(self.rect())

    @staticmethod
    def _make_card(title: str):
        """Section card with a dark header strip (Keyence-style section bar)."""
        card = QFrame()
        card.setObjectName("card")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QLabel(title)
        header.setObjectName("cardHeader")
        outer.addWidget(header)
        body = QVBoxLayout()
        body.setContentsMargins(12, 10, 12, 12)
        body.setSpacing(8)
        outer.addLayout(body, 1)
        return card, body

    @staticmethod
    def _row(*items) -> QHBoxLayout:
        """HBox from widgets; None adds a stretch."""
        row = QHBoxLayout()
        for item in items:
            if item is None:
                row.addStretch(1)
            else:
                row.addWidget(item)
        return row

    def _build_settings_tab(self) -> QWidget:
        model_card, model_body = self._make_card("Model")
        model_body.addLayout(self._row(self.load_model_button, self.model_label))
        self.model_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        camera_card, camera_body = self._make_card("Camera")
        camera_body.addLayout(self._row(QLabel("Camera:"), self.camera_combo, self.scan_button, None))

        tower_card, tower_body = self._make_card("Tower Light & Buzzer")
        tower_body.addWidget(self.tower_status_label)
        tower_body.addWidget(self.buzzer_enabled_checkbox)
        tower_body.addLayout(self._row(
            QLabel("NG alarm:"), self.buzzer_pattern_combo, None, self.buzzer_test_button))
        tower_body.addLayout(self._row(
            self.ok_beep_checkbox, None, self.ok_beep_duration_spinbox, self.ok_beep_test_button))
        tower_body.addLayout(self._row(QLabel("State hold time:"), None, self.state_hold_spinbox))
        hold_hint = QLabel("Minimum time between STANDBY / OK / NG changes (0 = off)")
        hold_hint.setObjectName("hint")
        hold_hint.setWordWrap(True)
        tower_body.addWidget(hold_hint)

        filters_card, filters_body = self._make_card("Detection Filters")
        self.confidence_label.setStyleSheet("color: #ffcf73; font-weight: 700;")
        filters_body.addLayout(self._row(QLabel("Confidence threshold"), None, self.confidence_label))
        filters_body.addWidget(self.confidence_slider)
        filters_body.addSpacing(6)
        self.max_detections_spinbox.setFixedWidth(120)
        filters_body.addLayout(self._row(self.max_detections_label, None, self.max_detections_spinbox))
        filters_body.addSpacing(6)
        filters_body.addWidget(self.target_class_label)
        filters_body.addLayout(self._row(
            self.target_class_combo_1, self.target_gate_combo, self.target_class_combo_2))
        filters_body.addSpacing(6)
        filters_body.addWidget(QLabel("Classes to show:"))
        filters_body.addWidget(self.class_filters_scroll, 1)

        left_column = QVBoxLayout()
        left_column.addWidget(model_card)
        left_column.addWidget(camera_card)
        left_column.addWidget(tower_card)
        left_column.addStretch(1)

        columns = QHBoxLayout()
        columns.addLayout(left_column, 1)
        columns.addWidget(filters_card, 1)

        content = QWidget()
        content.setLayout(columns)

        # Wrapped in a scroll area so the tab degrades gracefully if a panel
        # is smaller than the 1024x768 kiosk screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_job_change_tab(self) -> QWidget:
        top_bar = QFrame()
        top_bar.setObjectName("monitorTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 4, 10, 4)
        caption = QLabel("Current Part:")
        caption.setObjectName("monitorInfo")
        self.part_config_label.setObjectName("currentPart")
        top_layout.addWidget(caption)
        top_layout.addWidget(self.part_config_label)
        top_layout.addStretch(1)

        jobs_card, jobs_body = self._make_card("Saved Jobs")
        self.job_buttons_layout = QGridLayout()
        self.job_buttons_layout.setSpacing(10)
        grid_widget = QWidget()
        grid_widget.setObjectName("clearBox")
        grid_widget.setLayout(self.job_buttons_layout)
        self.job_buttons_container = QWidget()
        self.job_buttons_container.setObjectName("clearBox")
        container_layout = QVBoxLayout(self.job_buttons_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(grid_widget)
        container_layout.addStretch(1)

        jobs_scroll = QScrollArea()
        jobs_scroll.setWidgetResizable(True)
        jobs_scroll.setFrameShape(QFrame.NoFrame)
        jobs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        jobs_scroll.setWidget(self.job_buttons_container)
        jobs_scroll.viewport().setObjectName("clearBox")
        jobs_body.addWidget(jobs_scroll, 1)

        bottom_bar = QFrame()
        bottom_bar.setObjectName("monitorBottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(10, 4, 6, 4)
        hint = QLabel("Tap a job to switch  ·  Scan a barcode to load or create a job")
        hint.setObjectName("hint")
        bottom_layout.addWidget(hint)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.scan_barcode_button)

        layout = QVBoxLayout()
        layout.addWidget(top_bar)
        layout.addWidget(jobs_card, 1)
        layout.addWidget(bottom_bar)

        tab = QWidget()
        tab.setLayout(layout)
        return tab

    JOB_GRID_COLUMNS = 4

    def _refresh_job_buttons(self) -> None:
        while self.job_buttons_layout.count() > 0:
            item = self.job_buttons_layout.takeAt(0)
            if item.widget():
                # Hide now: deleteLater only runs once control returns to the
                # event loop, and until then the old tile would still paint.
                item.widget().hide()
                item.widget().deleteLater()

        config_paths = sorted(PART_CONFIGS_DIR.glob("*.json")) if PART_CONFIGS_DIR.exists() else []
        if not config_paths:
            empty = QLabel("No saved jobs yet. Scan a barcode to create one.")
            empty.setObjectName("hint")
            self.job_buttons_layout.addWidget(empty, 0, 0)
            return

        position = 0
        for config_path in config_paths:
            try:
                config = json.loads(config_path.read_text())
            except Exception:
                continue
            part_no = config.get("part_no") or config_path.stem
            model_path = config.get("model_path")
            model = Path(model_path).stem if model_path else "no model"
            # Tile shows the part and its model; the active job is highlighted.
            button = QPushButton(f"{part_no}\n{model}")
            button.setObjectName("jobButton")
            button.setCheckable(True)
            button.setChecked(part_no == self._current_part_no)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(lambda checked, p=part_no: self._on_job_button_clicked(p))
            row, col = divmod(position, self.JOB_GRID_COLUMNS)
            self.job_buttons_layout.addWidget(button, row, col)
            position += 1
        for col in range(self.JOB_GRID_COLUMNS):
            self.job_buttons_layout.setColumnStretch(col, 1)

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
        # Same frame as the Monitor tab: info strip on top, framed image with a
        # light side panel, action bar with the main action bottom right.
        top_bar = QFrame()
        top_bar.setObjectName("monitorTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 4, 6, 4)
        model_caption = QLabel("Active Model:")
        model_caption.setObjectName("monitorInfo")
        top_layout.addWidget(model_caption)
        top_layout.addWidget(self.train_model_label)
        top_layout.addSpacing(24)
        dataset_caption = QLabel("Dataset:")
        dataset_caption.setObjectName("monitorInfo")
        top_layout.addWidget(dataset_caption)
        top_layout.addWidget(self.train_folder_label, 1)
        top_layout.addWidget(self.train_select_folder_button)

        image_frame = QFrame()
        image_frame.setObjectName("videoFrame")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(2, 2, 2, 2)
        image_layout.addWidget(self.train_image_widget)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_panel.setFixedWidth(260)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)

        images_header = QLabel("Images")
        images_header.setObjectName("panelHeader")
        side_layout.addWidget(images_header)
        side_layout.addWidget(self.train_list_widget, 1)

        correction_header = QLabel("Box Correction")
        correction_header.setObjectName("panelHeader")
        side_layout.addWidget(correction_header)
        correction_box = QWidget()
        correction_box.setObjectName("counterBox")
        correction_layout = QVBoxLayout(correction_box)
        correction_layout.setContentsMargins(8, 8, 8, 8)
        correction_layout.setSpacing(6)
        self.train_selected_label.setObjectName("panelText")
        self.train_selected_label.setWordWrap(True)
        class_caption = QLabel("Change class to:")
        class_caption.setObjectName("panelText")
        correction_layout.addWidget(self.train_selected_label)
        correction_layout.addWidget(class_caption)
        correction_layout.addWidget(self.train_class_combo)
        correction_layout.addWidget(self.train_confirm_button)
        side_layout.addWidget(correction_box)

        content_row = QHBoxLayout()
        content_row.addWidget(image_frame, 1)
        content_row.addWidget(side_panel)

        bottom_bar = QFrame()
        bottom_bar.setObjectName("monitorBottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 4, 6, 4)
        bottom_layout.addWidget(self.train_prev_button)
        bottom_layout.addWidget(self.train_counter_label)
        bottom_layout.addWidget(self.train_next_button)
        bottom_layout.addSpacing(12)
        bottom_layout.addWidget(self.train_status_label, 1)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.train_start_button)

        layout = QVBoxLayout()
        layout.addWidget(top_bar)
        layout.addLayout(content_row, 1)
        layout.addWidget(bottom_bar)

        tab = QWidget()
        tab.setLayout(layout)
        return tab

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
        self._refresh_monitor_info()
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
        self._refresh_monitor_info()
        if hasattr(self, "job_buttons_layout"):
            self._refresh_job_buttons()

    def _refresh_monitor_info(self) -> None:
        part = self._current_part_no or "—"
        model_path = self._worker._model_path
        model = Path(model_path).stem if model_path else "—"
        self.monitor_info_label.setText(f"Part: {part}   ·   Model: {model}")

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
        gloss = gloss_gradient(QColor(bg).lighter(135).name(), bg, QColor(bg).darker(125).name())
        self.judgement_banner.setStyleSheet(
            f"QLabel#judgementBanner {{ background: {gloss}; color: {fg}; }}"
        )

    def _latch_judgement(self, raw_state: str) -> str:
        """Holds each STANDBY/OK/NG judgement for at least _state_hold_ms, so a
        flickering detection can't make the banner and tower light chatter.
        No queueing: after the hold, the current raw state is taken as-is."""
        now = time.monotonic()
        if self._judged_state is None or (
            raw_state != self._judged_state
            and (now - self._judged_since) * 1000 >= self._state_hold_ms
        ):
            # One cycle = each time the held judgement becomes OK or NG.
            self._judged_state = raw_state
            self._judged_since = now
            self._count_cycle(raw_state)
        return self._judged_state

    def _reset_judgement(self) -> None:
        self._judged_state = None
        self._update_judgement_banner("YELLOW")

    def _compute_buzzer_pattern(self, state: Optional[str]) -> int:
        if self._buzzer_enabled and state == "RED":
            return self._buzzer_pattern
        return BUZZER_OFF

    def _make_buzzer_style_combo(self, selected_pattern: int) -> QComboBox:
        combo = QComboBox()
        for name, value in BUZZER_PATTERN_NAMES:
            if value == BUZZER_OFF:
                continue
            combo.addItem(name, value)
        index = combo.findData(selected_pattern)
        if index >= 0:
            combo.setCurrentIndex(index)
        return combo

    def _save_app_settings(self) -> None:
        save_app_settings({
            "buzzer_enabled": self._buzzer_enabled,
            "buzzer_pattern": self._buzzer_pattern,
            "ok_beep_enabled": self._ok_beep_enabled,
            "ok_beep_ms": self._ok_beep_ms,
            "state_hold_ms": self._state_hold_ms,
            "ok_count": self._ok_count,
            "ng_count": self._ng_count,
        })

    @Slot(int)
    def on_state_hold_changed(self, value: int) -> None:
        self._state_hold_ms = value
        self._save_app_settings()

    @Slot(bool)
    def on_buzzer_enabled_changed(self, checked: bool) -> None:
        self._buzzer_enabled = checked
        self.buzzer_toggle_button.setChecked(checked)
        self._refresh_buzzer_toggle_button()
        self._save_app_settings()
        self._sync_buzzer_now()

    @Slot(int)
    def on_buzzer_pattern_changed(self, index: int) -> None:
        self._buzzer_pattern = self.buzzer_pattern_combo.currentData()
        self._save_app_settings()
        self._sync_buzzer_now()

    @Slot(bool)
    def on_ok_beep_enabled_changed(self, checked: bool) -> None:
        self._ok_beep_enabled = checked
        self._save_app_settings()
        if checked and self._buzzer_enabled:
            self._play_ok_beep()

    @Slot(int)
    def on_ok_beep_ms_changed(self, value: int) -> None:
        # No preview here: spinbox arrows fire on every step, so use Test.
        self._ok_beep_ms = value
        self._save_app_settings()

    @Slot()
    def on_ok_beep_test_clicked(self) -> None:
        if not self._tower_connected or self._tower is None:
            self.status_label.setText("Tower Light: Not Connected — cannot test buzzer.")
            return
        self._play_ok_beep()

    def _play_ok_beep(self) -> None:
        """Short confirmation beep: the built-in patterns are all too long, so
        switch the tone on and turn it off again after _ok_beep_ms."""
        if not self._tower_connected or self._tower is None:
            return
        self._ok_beep_token += 1
        token = self._ok_beep_token
        try:
            self._tower.set_buzzer(BUZZER_ON)
        except Exception:
            self._tower_connected = False
            self.tower_status_label.setText("Tower Light: Disconnected")
            return
        QTimer.singleShot(self._ok_beep_ms, lambda: self._end_ok_beep(token))

    def _end_ok_beep(self, token: int) -> None:
        # A newer beep owns the buzzer now; let its own timer end it.
        if token != self._ok_beep_token:
            return
        if not self._tower_connected or self._tower is None:
            return
        try:
            # Restore whatever the current state wants — silent for OK, but the
            # NG alarm if the state turned NG during the beep.
            self._tower.set_buzzer(self._compute_buzzer_pattern(self._last_tower_state))
        except Exception:
            self._tower_connected = False
            self.tower_status_label.setText("Tower Light: Disconnected")

    def _play_one_shot(self, pattern: int) -> None:
        """Plays one cycle of a pattern; the tower stops it by itself."""
        if not self._tower_connected or self._tower is None:
            self.status_label.setText("Tower Light: Not Connected — cannot test buzzer.")
            return
        try:
            self._tower.set_buzzer(pattern, limit=1)
        except Exception:
            self._tower_connected = False
            self.tower_status_label.setText("Tower Light: Disconnected")

    def _refresh_buzzer_toggle_button(self) -> None:
        # Neutral button; the speaker icon carries the on/muted state, so green
        # stays reserved for the Start action.
        if self._buzzer_enabled:
            self.buzzer_toggle_button.setText("Buzzer On")
            self.buzzer_toggle_button.setIcon(make_icon("speaker_on", "#e0e0e0"))
            self.buzzer_toggle_button.setStyleSheet("")
        else:
            self.buzzer_toggle_button.setText("Buzzer Off")
            self.buzzer_toggle_button.setIcon(make_icon("speaker_off", "#e57373"))
            self.buzzer_toggle_button.setStyleSheet("color: #9aa5b1;")

    @Slot()
    def on_buzzer_test_clicked(self) -> None:
        self._play_one_shot(self._buzzer_pattern)

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
            # Buzzer first: set_tower() re-sends the driver's remembered buzzer
            # value, so it must already be correct or a stale one chirps briefly.
            self._tower.set_buzzer(self._compute_buzzer_pattern(state))
            if state == "GREEN":
                self._tower.set_tower(green=LED_ON)
            elif state == "RED":
                self._tower.set_tower(red=LED_ON)
            else:
                self._tower.set_tower(yellow=LED_ON)
            self._last_tower_state = state
            # Beep after set_tower(), which re-sends the driver's remembered
            # (off) buzzer value and would otherwise cancel the tone.
            if state == "GREEN" and self._buzzer_enabled and self._ok_beep_enabled:
                self._play_ok_beep()
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
    def toggle_stream(self) -> None:
        if self._worker.isRunning():
            self.stop_stream()
        else:
            self.start_stream()

    def _refresh_run_button(self) -> None:
        running = self._worker.isRunning()
        if not running:
            self._stop_requested = False
        stopping = running and self._stop_requested

        if stopping:
            text, icon, style = "Stopping…", "stop", RUN_BUTTON_STYLES["stop"]
            status, dot = "Stopping…", "#f4d03f"
        elif running:
            text, icon, style = "Stop", "stop", RUN_BUTTON_STYLES["stop"]
            status, dot = "Running", "#2ecc71"
        else:
            text, icon, style = "Start", "play", RUN_BUTTON_STYLES["start"]
            status, dot = "Stopped", "#9aa5b1"

        self.run_button.setEnabled(not stopping)
        self.run_button.setText(text)
        self.run_button.setIcon(make_icon(icon))
        self.run_button.setStyleSheet(action_button_qss(*style))
        self.run_status_label.setText(
            f'<span style="color:{dot}; font-size:16pt;">●</span>'
            f'&nbsp;<span style="color:{dot}; font-weight:600;">{status}</span>'
        )

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
        self._reset_judgement()
        self._worker.start()
        self._refresh_run_button()
        if not self._tower_connected:
            self._connect_tower_light()

    @Slot()
    def stop_stream(self) -> None:
        # Non-blocking: the worker can take seconds to notice (first inference
        # warms up the GPU), so the button shows "Stopping..." until the
        # worker's finished signal flips it back to Start.
        if self._worker.isRunning():
            self._stop_requested = True
            self._worker.stop()
        self._shutdown_tower_light()
        self._reset_judgement()
        self._refresh_run_button()

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
        state = self._latch_judgement(self._compute_judgement_state(detections))
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
        if self._pending_ng_save:
            self._pending_ng_save = False
            self._save_ng_image(pixmap)

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
        # Last two path parts fit the top strip; full path in the tooltip.
        self.train_folder_label.setText(str(Path(*images_dir.parts[-2:])))
        self.train_folder_label.setToolTip(str(images_dir))

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
        # Must fully join: destroying a QThread that's still running aborts
        # the whole process ("QThread: Destroyed while thread is still running").
        self._worker.wait()
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