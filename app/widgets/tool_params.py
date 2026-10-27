"""Add-tool tiles and the parameter form of the Tools tab."""

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QScroller,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..paths import ROOT_DIR
from ..theme import RUN_BUTTON_STYLES, action_button_qss
from ..vision.base import Tool
from .layout import bottom_bar, make_card, top_bar

TOUCH_HEIGHT = 48

KIND_LABELS = {
    "position": "Position\nAdjust",
    "color_area": "Color\nArea", "color_average": "Color\nAverage", "color_prohibit": "Color\nProhibit",
    "area": "Area\n(mono)", "brightness": "Brightness\nAverage", "edge_pixels": "Edge\nPixels", "shading": "Shading\nUniformity",
    "width": "Width", "edge_presence": "Edge\nPresence", "pitch": "Pitch", "diameter": "Diameter",
    "ocr": "OCR\nText", "code": "Code\nQR / 1D",
    "outline": "Outline", "pattern_search": "Pattern\nSearch", "blob_count": "Blob\nCount",
}
CATEGORIES = [
    ("Position", ["position"]),
    ("Color", ["color_area", "color_average", "color_prohibit"]),
    ("Brightness", ["area", "brightness", "edge_pixels", "shading"]),
    ("Measure", ["width", "edge_presence", "pitch", "diameter"]),
    ("Text & Code", ["ocr", "code"]),
    ("Pattern & Count", ["outline", "pattern_search", "blob_count"]),
]
# The case each tool suits, shown under its tile with an example picture
# (drawn by tools/make_tool_thumbs.py).
KIND_USES = {
    "position": "Follow a part that moves or turns",
    "color_area": "Right wire color is there",
    "color_average": "Cap / LED / label is the taught color",
    "color_prohibit": "No stain or foreign color",
    "area": "Terminal or part present (bright / dark)",
    "brightness": "LED on, lamp lit",
    "edge_pixels": "Print or label present",
    "shading": "Dent, scratch or uneven surface",
    "width": "Strip length, wire or gap size",
    "edge_presence": "Terminal fully inserted",
    "pitch": "Pin spacing, bent pins",
    "diameter": "Hole or ring size",
    "ocr": "Read lot / date / part text",
    "code": "Read QR or barcode label",
    "outline": "Right part shape",
    "pattern_search": "Find & count a mark at any angle",
    "blob_count": "Count terminals, find a missing one",
}
THUMBS_DIR = ROOT_DIR / "app" / "assets" / "tool_thumbs"
THUMB_WIDTH, THUMB_HEIGHT = 192, 120


def kind_title(kind: str) -> str:
    return KIND_LABELS.get(kind, kind).replace("\n", " ")


def kind_thumbnail(kind: str) -> QPixmap:
    """Example picture of the case the tool suits; null if it is missing."""
    pixmap = QPixmap(str(THUMBS_DIR / f"{kind}.png"))
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(THUMB_WIDTH, THUMB_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ToolTile(QPushButton):
    """Touch tile: example picture, tool name and what it is for."""

    def __init__(self, kind: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("toolTile")
        self.setAccessibleName(kind_title(kind))
        self.setToolTip(KIND_USES.get(kind, ""))
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignCenter)
        self.picture.setFixedSize(THUMB_WIDTH, THUMB_HEIGHT)
        pixmap = kind_thumbnail(kind)
        if pixmap.isNull():
            self.picture.hide()
        else:
            self.picture.setPixmap(pixmap)
        title = QLabel(kind_title(kind))
        title.setObjectName("toolTileTitle")
        use = QLabel(KIND_USES.get(kind, ""))
        use.setObjectName("toolTileUse")
        use.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        picture_row = QHBoxLayout()
        picture_row.addStretch(1)
        picture_row.addWidget(self.picture)
        picture_row.addStretch(1)
        layout.addLayout(picture_row)
        layout.addWidget(title)
        layout.addWidget(use)
        layout.addStretch(1)
        for label in (self.picture, title, use):
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setMinimumHeight(THUMB_HEIGHT + 90)
        self.setMinimumWidth(THUMB_WIDTH + 18)


class AddToolDialog(QDialog):
    """A grid of touch tiles, one per tool kind, grouped by category. Each
    tile shows an example of the case the tool suits."""

    COLUMNS = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Tool")
        self.setModal(True)
        self.resize(4 * (THUMB_WIDTH + 34) + 60, 820)
        self.selected_kind: Optional[str] = None
        self.tiles: Dict[str, QPushButton] = {}

        top, top_layout = top_bar()
        title = QLabel("Add Tool: pick a kind, then draw its area on the image")
        title.setObjectName("monitorInfo")
        top_layout.addWidget(title)
        top_layout.addStretch(1)

        body = QWidget()
        body.setObjectName("clearBox")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        for category, kinds in CATEGORIES:
            card, card_body = make_card(category)
            grid = QGridLayout()
            grid.setSpacing(8)
            for i, kind in enumerate(kinds):
                tile = ToolTile(kind)
                tile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                tile.clicked.connect(lambda _=False, k=kind: self._pick(k))
                self.tiles[kind] = tile
                grid.addWidget(tile, i // self.COLUMNS, i % self.COLUMNS)
            for col in range(self.COLUMNS):
                grid.setColumnStretch(col, 1)
            card_body.addLayout(grid)
            body_layout.addWidget(card)
        body_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        scroll.viewport().setObjectName("clearBox")
        QScroller.grabGesture(scroll.viewport(), QScroller.LeftMouseButtonGesture)  # drag to scroll on touch

        cancel = QPushButton("Cancel")
        cancel.setFixedHeight(44)
        cancel.setMinimumWidth(140)
        cancel.clicked.connect(self.reject)
        bottom, bottom_layout = bottom_bar()
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addWidget(scroll, 1)
        layout.addWidget(bottom)

    def _pick(self, kind: str) -> None:
        self.selected_kind = kind
        self.accept()


class DeleteToolDialog(QDialog):
    """Touch confirm for removing one tool: its picture and name, and two big
    buttons far apart. No typing, so it is quick with a finger, and Delete is
    never the default button, so a stray tap or Enter cannot remove it."""

    BUTTON_HEIGHT = 64

    def __init__(self, tool: Tool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Delete Tool")
        self.setModal(True)
        self.resize(560, 380)

        top, top_layout = top_bar()
        title = QLabel("Delete this tool?")
        title.setObjectName("dangerTitle")
        top_layout.addWidget(title)
        top_layout.addStretch(1)

        picture = QLabel()
        picture.setFixedSize(THUMB_WIDTH, THUMB_HEIGHT)
        pixmap = kind_thumbnail(tool.kind)
        if pixmap.isNull():
            picture.hide()
        else:
            picture.setPixmap(pixmap)
        name = QLabel(tool.name)
        name.setObjectName("toolTileTitle")
        name.setWordWrap(True)
        kind = QLabel(kind_title(tool.kind))
        kind.setObjectName("toolTileUse")
        note = QLabel("Its area and taught data are removed from the job.")
        note.setObjectName("dangerText")
        note.setWordWrap(True)
        text = QVBoxLayout()
        text.addWidget(name)
        text.addWidget(kind)
        text.addStretch(1)
        text.addWidget(note)
        card = QFrame()
        card.setObjectName("card")
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(14)
        card_layout.addWidget(picture, 0, Qt.AlignTop)
        card_layout.addLayout(text, 1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.delete_button = QPushButton("Delete")
        self.delete_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["stop"]))
        self.delete_button.setAutoDefault(False)
        self.delete_button.clicked.connect(self.accept)
        bottom, bottom_layout = bottom_bar()
        for button in (self.cancel_button, self.delete_button):
            button.setFixedHeight(self.BUTTON_HEIGHT)
            button.setMinimumWidth(180)
        bottom_layout.addWidget(self.cancel_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.delete_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addWidget(card, 1)
        layout.addWidget(bottom)
        self.cancel_button.setFocus()


# Values that are taught or internal, not for hand editing.
HIDDEN_PARAMS = {"pre", "template_roi", "taught_center", "ref_edge", "ref_strength", "ref_sign", "ref_lab"}
_EDGE = ["any", "rising", "falling"]
CHOICES = {
    ("position", "mode"): ["2d", "x", "y"],
    ("outline", "mode"): ["template", "shape"],
    ("area", "polarity"): ["bright", "dark"],
    ("blob_count", "polarity"): ["bright", "dark"],
    ("pitch", "polarity"): _EDGE,
    "direction": ["x", "y"],
    "polarity": _EDGE, "polarity_1": _EDGE, "polarity_2": _EDGE,
    "edge_pair": ["outer", "strongest", "first_rising_falling"],
    "method": ["fixed", "otsu"],
    "source": ["gray", "color"],
    "measure": ["count", "mean", "min", "max"],
    "symbology": ["any", "qrcode", "code39", "code128", "ean13"],
}


def _title(key: str) -> str:
    return key.replace("_", " ").capitalize()


class ToolParamsForm(QWidget):
    """A form built from `tool.params()`: bool -> check box, int / float ->
    spin box, str with known choices -> combo box, other str -> line edit.
    Emits `changed` after a value was written into the tool."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tool: Optional[Tool] = None
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self.widgets: Dict[str, QWidget] = {}

    def set_tool(self, tool: Optional[Tool]) -> None:
        self._tool = None
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self.widgets = {}
        if tool is None:
            return
        for key, value in tool.params().items():
            widget = self._make_widget(tool, key, value)
            if widget is None:
                continue
            widget.setMinimumHeight(TOUCH_HEIGHT)
            self.widgets[key] = widget
            self._layout.addRow(_title(key), widget)
        self._tool = tool

    def _make_widget(self, tool: Tool, key: str, value) -> Optional[QWidget]:
        if key in HIDDEN_PARAMS:
            return None
        if isinstance(value, bool):
            box = QCheckBox("On")
            box.setChecked(value)
            box.toggled.connect(lambda v, k=key: self._write(k, bool(v)))
            return box
        choices = CHOICES.get((tool.kind, key)) or CHOICES.get(key)
        if choices and isinstance(value, str):
            combo = QComboBox()
            combo.addItems(choices if value in choices else [value] + choices)
            combo.setCurrentText(value)
            combo.currentTextChanged.connect(lambda v, k=key: self._write(k, v))
            return combo
        if isinstance(value, int):
            spin = QSpinBox()
            spin.setRange(-10 ** 9, 10 ** 9)
            spin.setValue(value)
            spin.valueChanged.connect(lambda v, k=key: self._write(k, int(v)))
            return spin
        if isinstance(value, float):
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(0.1 if abs(value) < 10 else 1.0)
            spin.setValue(value)
            spin.valueChanged.connect(lambda v, k=key: self._write(k, float(v)))
            return spin
        if isinstance(value, str):
            edit = QLineEdit(value)
            edit.textChanged.connect(lambda v, k=key: self._write(k, v))
            return edit
        return None

    def _write(self, key: str, value) -> None:
        if self._tool is not None:
            self._tool.set_params({key: value})
            self.changed.emit()
