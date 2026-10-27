from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

from .paths import ROOT_DIR


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

# Verdict colors, identical on the banner, the tool overlays and the result
# tables. Reserved: don't reuse green / red for decoration.
OK_GREEN = "#2e7d32"
NG_RED = "#c0392b"
STANDBY_YELLOW = "#f4d03f"

# QSS can only draw sub-control arrows from image files, so the spinbox
# arrows are written once as tiny SVGs next to the app.
UI_ASSETS_DIR = ROOT_DIR / ".ui_assets"


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
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: #0f1114;
    color: #e8eaed;
    border: 1px solid #3a4048;
    border-radius: 3px;
    padding: 4px 8px;
    min-height: 28px;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: #e0a526;
}}
QSpinBox, QDoubleSpinBox {{
    padding-right: 40px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 36px;
    background: {GLOSS_GRAY};
    border: 1px solid #0e1013;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
    border-top-right-radius: 3px;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
    border-bottom-right-radius: 3px;
}}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
    background: {GLOSS_GRAY_PRESSED};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({ARROW_UP_SVG});
    width: 14px;
    height: 13px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
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
QPushButton#toolTile {{
    text-align: left;
    padding: 0;
}}
QLabel#toolTileTitle {{
    font-size: 14pt;
    font-weight: 700;
    color: #f0f2f4;
}}
QLabel#toolTileUse {{
    font-size: 10pt;
    color: #c4cad1;
}}
QLabel#dangerTitle {{
    color: #ff7a6b;
    font-size: 15pt;
    font-weight: 800;
}}
QLabel#dangerText {{
    color: #ffb4a8;
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
QLabel#masterCaption {{
    color: #1a1d22;
    font-size: 9pt;
}}
QPushButton#masterEmpty {{
    color: #4a525c;
    background: transparent;
    border: 1px dashed #8a939d;
    border-radius: 3px;
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
    elif kind == "autocapture":
        # Camera with a small "A" badge: captures on its own.
        painter.setPen(stroke)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(1.5, 6, 12, 9), 2, 2)
        painter.drawEllipse(QPointF(7.5, 10.5), 2.5, 2.5)
        painter.drawLine(QPointF(15, 17), QPointF(16.75, 11))
        painter.drawLine(QPointF(16.75, 11), QPointF(18.5, 17))
        painter.drawLine(QPointF(15.6, 15), QPointF(17.9, 15))
    elif kind == "master":
        # Picture frame with a star: the reference image.
        painter.setPen(stroke)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(2.5, 3.5, 15, 13), 1.5, 1.5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        painter.drawPolygon(QPolygonF([
            QPointF(10, 6), QPointF(11.2, 8.8), QPointF(14.2, 9), QPointF(11.9, 10.9),
            QPointF(12.6, 13.9), QPointF(10, 12.3), QPointF(7.4, 13.9), QPointF(8.1, 10.9),
            QPointF(5.8, 9), QPointF(8.8, 8.8),
        ]))
    elif kind == "close":
        pen = QPen(fill, 2.4)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(5, 5), QPointF(15, 15))
        painter.drawLine(QPointF(15, 5), QPointF(5, 15))
    elif kind == "roi":
        # Four corner brackets around a dashed frame.
        painter.setPen(stroke)
        painter.setBrush(Qt.NoBrush)
        for x, y, dx, dy in ((3, 3, 1, 1), (17, 3, -1, 1), (3, 17, 1, -1), (17, 17, -1, -1)):
            painter.drawLine(QPointF(x, y), QPointF(x + 4 * dx, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + 4 * dy))
        dash = QPen(fill, 1.2, Qt.DashLine)
        painter.setPen(dash)
        painter.drawRect(QRectF(7, 7, 6, 6))
    elif kind == "perspective":
        # Tilted trapezoid becoming a square: the top-view correction.
        painter.setPen(stroke)
        painter.setBrush(Qt.NoBrush)
        painter.drawPolygon(QPolygonF([QPointF(4, 6), QPointF(9, 6), QPointF(10.5, 15), QPointF(1.5, 15)]))
        painter.drawRect(QRectF(12, 6, 6.5, 9))
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
