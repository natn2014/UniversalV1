"""Industrial HMI theme for PySide6 (Keyence-style vision-system look).

Drop this file into a project and:

    from industrial_theme import INDUSTRIAL_QSS, apply_theme, ...
    app = QApplication(sys.argv)
    apply_theme(app)

Extracted from a production wire-harness inspection kiosk (Jetson, 1024x768
touch panel). Run `python industrial_theme.py` for a demo window of every
building block; `--screenshot out.png` renders it offscreen.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QEventPoint, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------- palette ---
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
UI_ASSETS_DIR = Path.home() / ".cache" / "industrial_hmi_ui"


def _write_arrow_svgs() -> tuple:
    UI_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
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

INDUSTRIAL_QSS = f"""
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


# ------------------------------------------------ icons and gloss helpers ---
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


# ------------------------------------------------------------- builders ---
def apply_theme(app: QApplication) -> None:
    """App-wide stylesheet. Per-widget setStyleSheet() calls still win."""
    app.setStyleSheet(INDUSTRIAL_QSS)


def make_strip(position: str = "top") -> tuple:
    """Dark gradient bar (top info strip or bottom action bar).
    Returns (frame, QHBoxLayout)."""
    frame = QFrame()
    frame.setObjectName("monitorTopBar" if position == "top" else "monitorBottomBar")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(10, 4, 6, 4)
    return frame, layout


def make_info_label(text: str = "") -> QLabel:
    """Muted bold caption for strips, e.g. 'Part: 42931 · Model: v2'."""
    label = QLabel(text)
    label.setObjectName("monitorInfo")
    return label


def make_hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    return label


def make_framed(widget: QWidget) -> QFrame:
    """Black well with the amber outline used around camera/image views."""
    frame = QFrame()
    frame.setObjectName("videoFrame")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(widget)
    return frame


def make_panel_header(title: str) -> QLabel:
    header = QLabel(title)
    header.setObjectName("panelHeader")
    return header


def make_side_panel(width: Optional[int] = None) -> tuple:
    """Light-gray side panel (tables/lists read dark-on-light, like the
    reference's settings pane). Add make_panel_header() + content to it.
    Returns (frame, QVBoxLayout)."""
    frame = QFrame()
    frame.setObjectName("sidePanel")
    if width:
        frame.setFixedWidth(width)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    return frame, layout


def make_card(title: str) -> tuple:
    """Dark section card with a header strip (replaces QGroupBox).
    Returns (frame, body QVBoxLayout)."""
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


def make_row(*items) -> QHBoxLayout:
    """HBox from widgets; None inserts a stretch."""
    row = QHBoxLayout()
    for item in items:
        if item is None:
            row.addStretch(1)
        else:
            row.addWidget(item)
    return row


def make_clear_box() -> QWidget:
    """Transparent container, for layouts nested inside light panels."""
    box = QWidget()
    box.setObjectName("clearBox")
    return box


def make_primary_button(text: str, kind: str = "start", icon: Optional[str] = None,
                        height: int = 44, min_width: int = 170) -> QPushButton:
    """The one main action of a screen (bottom-right). kind: 'start' = blue
    gloss, 'stop' = red gloss (also used for destructive actions)."""
    button = QPushButton(text)
    if icon:
        button.setIcon(make_icon(icon))
        button.setIconSize(QSize(20, 20))
    button.setFixedHeight(height)
    button.setMinimumWidth(min_width)
    button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES[kind]))
    return button


def set_primary_button_kind(button: QPushButton, kind: str) -> None:
    button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES[kind]))


JUDGEMENT_COLORS = {
    # state: (text, background, foreground)
    "OK": ("OK", "#2e7d32", "#ffffff"),
    "NG": ("NG", "#c0392b", "#ffffff"),
    "STANDBY": ("STANDBY", "#f4d03f", "#1a1a1a"),
}


def make_judgement_pill(width: int = 130, height: int = 36) -> QLabel:
    pill = QLabel()
    pill.setObjectName("judgementBanner")
    pill.setAlignment(Qt.AlignCenter)
    pill.setFixedSize(width, height)
    set_judgement(pill, "STANDBY")
    return pill


def set_judgement(pill: QLabel, state: str, colors: dict = JUDGEMENT_COLORS) -> None:
    """Glossy status pill; state is a key of `colors`."""
    text, bg, fg = colors[state]
    pill.setText(text)
    gloss = gloss_gradient(QColor(bg).lighter(135).name(), bg, QColor(bg).darker(125).name())
    pill.setStyleSheet(f"QLabel#judgementBanner {{ background: {gloss}; color: {fg}; }}")


def make_status_chip() -> QLabel:
    """'● Running' style state text; pair with set_status_chip()."""
    chip = QLabel()
    chip.setMinimumWidth(120)
    return chip


def set_status_chip(chip: QLabel, text: str, color: str) -> None:
    chip.setText(
        f'<span style="color:{color}; font-size:16pt;">●</span>'
        f'&nbsp;<span style="color:{color}; font-weight:600;">{text}</span>'
    )


def make_counter_tile(title: str, top: str, bottom: str, digits: str) -> tuple:
    """Big eye-catching counter (e.g. OK / NG cycles). Returns (tile, value QLabel).
    OK: make_counter_tile("OK", "#5cbf60", "#2e7d32", "#4cd964")
    NG: make_counter_tile("NG", "#e0685a", "#b83225", "#ff5a4a")"""
    tile = QFrame()
    tile.setObjectName("counterTile")
    layout = QVBoxLayout(tile)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    title_label = QLabel(title)
    title_label.setObjectName("counterTitle")
    title_label.setAlignment(Qt.AlignCenter)
    title_label.setStyleSheet(
        f"QLabel#counterTitle {{ background: {gloss_gradient(top, bottom, bottom)}; }}")
    value = QLabel("0")
    value.setObjectName("counterValue")
    value.setAlignment(Qt.AlignCenter)
    value.setStyleSheet(f"QLabel#counterValue {{ color: {digits}; }}")
    layout.addWidget(title_label)
    layout.addWidget(value, 1)
    return tile, value


def make_tile_button(text: str, checked: bool = False) -> QPushButton:
    """Large touch tile for choosing one item (jobs, recipes, modes).
    Checkable: the checked tile shows the orange 'selected tool' gloss."""
    button = QPushButton(text)
    button.setObjectName("jobButton")
    button.setCheckable(True)
    button.setChecked(checked)
    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return button


def style_panel_table(table) -> None:
    """Read-only table for a light side panel: no row numbers, columns fill."""
    from PySide6.QtWidgets import QHeaderView, QTableWidget
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionMode(QTableWidget.NoSelection)
    table.setAlternatingRowColors(True)


def style_panel_list(list_widget) -> None:
    """List for a light side panel. Elide on the left: file names usually end
    in the part that matters (dates, numbers)."""
    list_widget.setTextElideMode(Qt.ElideLeft)
    list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


def clear_layout(layout) -> None:
    """Remove every widget from a layout. hide() first: deleteLater only runs
    when control returns to the event loop, until then old widgets still paint."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.deleteLater()


class ConfirmByTypingDialog(QDialog):
    """Destructive-action confirmation: the red button enables only once the
    exact name is typed (case-sensitive), so a stray tap can't delete data."""

    def __init__(self, title: str, name: str, consequence: str, action_text: str = "Remove",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(560, 320)
        self._name = name

        top, top_layout = make_strip("top")
        heading = QLabel(title)
        heading.setObjectName("dangerTitle")
        top_layout.addWidget(heading)
        top_layout.addStretch(1)

        card = QFrame()
        card.setObjectName("card")
        body = QVBoxLayout(card)
        body.setContentsMargins(12, 10, 12, 12)
        body.setSpacing(8)
        warning = QLabel(consequence)
        warning.setObjectName("dangerText")
        warning.setWordWrap(True)
        prompt = QLabel(f"To confirm, type  <b>{name}</b>  below:")
        self.field = QLineEdit()
        self.field.setPlaceholderText("Type it here")
        body.addWidget(warning)
        body.addWidget(prompt)
        body.addWidget(self.field)
        body.addStretch(1)

        cancel = QPushButton("Cancel")
        cancel.setFixedHeight(44)
        cancel.setMinimumWidth(140)
        cancel.clicked.connect(self.reject)
        self.action_button = make_primary_button(action_text, "stop", min_width=140)
        self.action_button.clicked.connect(self.accept)
        self.action_button.setEnabled(False)
        bottom, bottom_layout = make_strip("bottom")
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(cancel)
        bottom_layout.addWidget(self.action_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addWidget(card, 1)
        layout.addWidget(bottom)

        self.field.textChanged.connect(
            lambda text: self.action_button.setEnabled(text.strip() == self._name))
        self.field.setFocus()


# ------------------------------------------- image popup with pinch zoom ---
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


# ----------------------------------------------------------------- demo ---
def _demo_window() -> QWidget:
    from PySide6.QtWidgets import QSpinBox, QSlider, QTabWidget, QTableWidget, QCheckBox

    root = QWidget()
    root.setWindowTitle("Industrial HMI theme demo")
    tabs = QTabWidget()

    # Monitor-style screen: info strip / framed view + light side panel / action bar
    monitor = QWidget()
    top, top_layout = make_strip("top")
    top_layout.addWidget(make_info_label("Part: 42931   ·   Model: demo"))
    top_layout.addStretch(1)
    pill = make_judgement_pill()
    set_judgement(pill, "OK")
    top_layout.addWidget(pill)

    view = QLabel("camera / image view")
    view.setAlignment(Qt.AlignCenter)
    view.setMinimumSize(480, 320)
    side, side_layout = make_side_panel(260)
    side_layout.addWidget(make_panel_header("Detections"))
    table = QTableWidget(2, 2)
    table.setHorizontalHeaderLabels(["Class", "Count"])
    style_panel_table(table)
    side_layout.addWidget(table, 1)
    counters = make_clear_box()
    counters_layout = QHBoxLayout(counters)
    ok_tile, ok_value = make_counter_tile("OK", "#5cbf60", "#2e7d32", "#4cd964")
    ng_tile, ng_value = make_counter_tile("NG", "#e0685a", "#b83225", "#ff5a4a")
    ok_value.setText("128")
    ng_value.setText("7")
    counters_layout.addWidget(ok_tile)
    counters_layout.addWidget(ng_tile)
    side_layout.addWidget(counters)
    content = QHBoxLayout()
    content.addWidget(make_framed(view), 1)
    content.addWidget(side)

    bottom, bottom_layout = make_strip("bottom")
    capture = QPushButton("Capture")
    capture.setIcon(make_icon("camera", "#e0e0e0"))
    capture.setFixedHeight(40)
    toggle = QPushButton("Buzzer On")
    toggle.setCheckable(True)
    toggle.setChecked(True)
    toggle.setFixedHeight(40)
    chip = make_status_chip()
    set_status_chip(chip, "Running", "#2ecc71")
    run = make_primary_button("Stop", "stop", icon="stop")
    bottom_layout.addWidget(capture)
    bottom_layout.addWidget(toggle)
    bottom_layout.addStretch(1)
    bottom_layout.addWidget(chip)
    bottom_layout.addWidget(run)

    monitor_layout = QVBoxLayout(monitor)
    monitor_layout.addWidget(top)
    monitor_layout.addLayout(content, 1)
    monitor_layout.addWidget(bottom)
    tabs.addTab(monitor, "Monitor")

    # Settings-style screen: two columns of cards
    settings = QWidget()
    left_card, left_body = make_card("Tower Light & Buzzer")
    left_body.addWidget(QCheckBox("Enable Buzzer"))
    spin = QSpinBox()
    spin.setRange(0, 5000)
    spin.setValue(1000)
    spin.setSuffix(" ms")
    left_body.addLayout(make_row(QLabel("State hold time:"), None, spin))
    left_body.addWidget(make_hint("Minimum time between STANDBY / OK / NG changes"))
    right_card, right_body = make_card("Detection Filters")
    slider = QSlider(Qt.Horizontal)
    slider.setValue(60)
    right_body.addWidget(slider)
    combo = QComboBox()
    combo.addItems(["Lead", "NoTape"])
    right_body.addWidget(combo)
    right_body.addStretch(1)
    columns = QHBoxLayout(settings)
    left_column = QVBoxLayout()
    left_column.addWidget(left_card)
    left_column.addStretch(1)
    columns.addLayout(left_column, 1)
    columns.addWidget(right_card, 1)
    tabs.addTab(settings, "Settings")

    layout = QVBoxLayout(root)
    layout.addWidget(tabs)
    return root


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app)
    window = _demo_window()
    window.resize(1024, 768)
    if "--screenshot" in sys.argv:
        out = sys.argv[sys.argv.index("--screenshot") + 1]
        window.show()
        app.processEvents()
        app.processEvents()
        window.grab().save(out)
        print(f"saved {out}")
    else:
        window.show()
        sys.exit(app.exec())
