"""Add-tool tiles and the parameter form of the Tools tab."""

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..vision.base import Tool
from .layout import bottom_bar, make_card, top_bar

TOUCH_HEIGHT = 48

KIND_LABELS = {
    "position": "Position\nAdjust",
    "color_area": "Color\nArea", "color_average": "Color\nAverage", "color_prohibit": "Color\nProhibit",
    "area": "Area\n(mono)", "brightness": "Brightness\nAverage", "edge_pixels": "Edge\nPixels", "shading": "Shading\nUniformity",
    "width": "Width", "edge_presence": "Edge\nPresence", "pitch": "Pitch", "diameter": "Diameter",
    "ocr": "OCR\nText", "code": "Code\nQR / 1D",
    "outline": "Outline", "blob_count": "Blob\nCount",
}
CATEGORIES = [
    ("Position", ["position"]),
    ("Color", ["color_area", "color_average", "color_prohibit"]),
    ("Brightness", ["area", "brightness", "edge_pixels", "shading"]),
    ("Measure", ["width", "edge_presence", "pitch", "diameter"]),
    ("Text & Code", ["ocr", "code"]),
    ("Pattern & Count", ["outline", "blob_count"]),
]
HSV_KINDS = {"color_area", "color_prohibit", "blob_count"}


def kind_title(kind: str) -> str:
    return KIND_LABELS.get(kind, kind).replace("\n", " ")


class AddToolDialog(QDialog):
    """A grid of touch tiles, one per tool kind, grouped by category."""

    COLUMNS = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Tool")
        self.setModal(True)
        self.resize(760, 560)
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
                tile = QPushButton(KIND_LABELS[kind])
                tile.setObjectName("jobButton")
                tile.setMinimumHeight(72)
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
