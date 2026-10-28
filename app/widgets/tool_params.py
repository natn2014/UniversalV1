"""Tool Catalog (add a tool), delete confirm and the parameter form of the Tools tab."""

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
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
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..paths import ROOT_DIR
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon
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
# What each tool does and when to pick it: the Application Navigator text of
# the Tool Catalog (as on Keyence's CV-X).
KIND_HELP = {
    "position": "Finds a taught feature of the part and measures how far it moved or turned. "
                "Other tools follow it, so their areas stay on the part.",
    "color_area": "Counts the pixels inside the taught color range, as a % of the area. "
                  "Use Pick Color and tap the color on the image.",
    "color_average": "Compares the average color of the area with the taught master (ΔE). "
                     "Small ΔE = same color.",
    "color_prohibit": "NG as soon as a spot of the forbidden color, bigger than the minimum size, "
                      "appears anywhere in the area.",
    "area": "Counts bright (or dark) pixels as a % of the area. "
            "Use Pick Gray and tap the part to set the threshold.",
    "brightness": "Average gray level of the area: is a lamp or LED on, is the part there.",
    "edge_pixels": "Counts edge pixels in the area: print, labels and texture have many, "
                   "a blank surface has few.",
    "shading": "Splits the area into cells and compares their brightness. "
               "A dent, scratch or stain makes one cell differ.",
    "width": "Finds two edges along a line and measures the distance between them, "
             "in px or mm once calibrated.",
    "edge_presence": "Is there an edge along the line: a terminal pushed fully home, "
                     "a part edge in place.",
    "pitch": "Finds every edge along a line and measures the spacing: pin pitch, bent or missing pins.",
    "diameter": "Fits a circle to the edges in the area and measures its diameter.",
    "ocr": "Reads printed text (lot, date, part number) and checks it against the expected text.",
    "code": "Reads QR codes and 1D barcodes and checks the content.",
    "outline": "Compares the part's outline with the taught master, at a fixed place and angle.",
    "pattern_search": "Finds every copy of a taught pattern in the area, at any angle, and counts them.",
    "blob_count": "Counts separate bright, dark or colored blobs that pass the size and shape filters.",
}
# The question under the category tiles, like the CV-X Tool Catalog asks it.
CATEGORY_QUESTIONS = {
    "Position": "Does the part move or turn between shots?",
    "Color": "What do you want to check about the color?",
    "Brightness": "What should the brightness tell you?",
    "Measure": "What do you want to measure?",
    "Text & Code": "What should be read?",
    "Pattern & Count": "Find a shape, or count things?",
}
CATALOG_TIP = "When the part's position varies, add Position Adjust first: the other tools then follow it."
THUMBS_DIR = ROOT_DIR / "app" / "assets" / "tool_thumbs"
THUMB_WIDTH, THUMB_HEIGHT = 192, 120
TILE_THUMB = (128, 80)  # tool tiles in the catalog
ICON_THUMB = (72, 45)  # category tiles
NAV_THUMB = (256, 160)  # Application Navigator


def kind_title(kind: str) -> str:
    return KIND_LABELS.get(kind, kind).replace("\n", " ")


def kind_thumbnail(kind: str, size=(THUMB_WIDTH, THUMB_HEIGHT)) -> QPixmap:
    """Example picture of the case the tool suits; null if it is missing."""
    pixmap = QPixmap(str(THUMBS_DIR / f"{kind}.png"))
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)


class CatalogTile(QPushButton):
    """Checkable catalog tile: a picture with a bold name under it. Light
    gloss, orange when selected (CV-X Tool Catalog)."""

    double_clicked = Signal()

    def __init__(self, picture_kind: str, text: str, size, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("catalogTile")
        self.setCheckable(True)
        self.setAccessibleName(text)
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignCenter)
        self.picture.setFixedSize(*size)
        pixmap = kind_thumbnail(picture_kind, size)
        if pixmap.isNull():
            self.picture.hide()
        else:
            self.picture.setPixmap(pixmap)
        self.caption = QLabel(text)
        self.caption.setObjectName("catalogTileText")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self.picture, 0, Qt.AlignHCenter)
        layout.addWidget(self.caption)
        for label in (self.picture, self.caption):
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setMinimumSize(size[0] + 16, size[1] + 44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def mouseDoubleClickEvent(self, event) -> None:
        super().mouseDoubleClickEvent(event)
        self.double_clicked.emit()


class ToolTile(CatalogTile):
    """A tool in the catalog: its example picture and name."""

    def __init__(self, kind: str, parent=None) -> None:
        super().__init__(kind, kind_title(kind), TILE_THUMB, parent)
        self.kind = kind
        self.setToolTip(KIND_USES.get(kind, ""))


class AddToolDialog(QDialog):
    """Tool Catalog, after Keyence's CV-X: pick a category, then a tool; the
    Application Navigator on the right says what the selected tool does and
    shows an example. Add (or a double tap on the tool) confirms. Sized to fit
    the window it opens over, with nothing to scroll."""

    COLUMNS = 4
    last_category = CATEGORIES[0][0]  # the catalog opens where it was left

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tool Catalog")
        self.setModal(True)
        self.selected_kind: Optional[str] = None
        self.tiles: Dict[str, ToolTile] = {}
        self.category_tiles: Dict[str, CatalogTile] = {}

        top, top_layout = top_bar()
        title = QLabel("Tool Catalog")
        title.setObjectName("catalogTitle")
        top_layout.addWidget(title)
        top_layout.addStretch(1)

        # Left: categories, the question, the tools of the category.
        catalog = QFrame()
        catalog.setObjectName("catalogPanel")
        left = QVBoxLayout(catalog)
        left.setContentsMargins(10, 8, 10, 10)
        left.setSpacing(8)
        heading = QLabel("Category")
        heading.setObjectName("catalogHeading")
        left.addWidget(heading)
        cat_grid = QGridLayout()
        cat_grid.setSpacing(6)
        self._category_group = QButtonGroup(self)
        for i, (category, kinds) in enumerate(CATEGORIES):
            tile = CatalogTile(kinds[0], category, ICON_THUMB)
            tile.clicked.connect(lambda _=False, c=category: self.show_category(c))
            self._category_group.addButton(tile)
            self.category_tiles[category] = tile
            cat_grid.addWidget(tile, i // 3, i % 3)
        left.addLayout(cat_grid)
        self.question = QLabel()
        self.question.setObjectName("catalogQuestion")
        left.addWidget(self.question)
        self._tool_group = QButtonGroup(self)
        self._pages = QStackedWidget()
        self._pages.setObjectName("clearBox")  # the light catalog panel shows through
        self._page_of: Dict[str, int] = {}
        for category, kinds in CATEGORIES:
            page = QWidget()
            page.setObjectName("clearBox")
            grid = QGridLayout(page)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(8)
            for i, kind in enumerate(kinds):
                tile = ToolTile(kind)
                tile.clicked.connect(lambda _=False, k=kind: self.select(k))
                tile.double_clicked.connect(lambda k=kind: self._add(k))
                self._tool_group.addButton(tile)
                self.tiles[kind] = tile
                grid.addWidget(tile, i // self.COLUMNS, i % self.COLUMNS)
            for col in range(self.COLUMNS):
                grid.setColumnStretch(col, 1)
            grid.setRowStretch(grid.rowCount(), 1)
            self._page_of[category] = self._pages.addWidget(page)
        left.addWidget(self._pages, 1)

        # Right: the Application Navigator.
        nav = QFrame()
        nav.setObjectName("navigator")
        nav.setFixedWidth(NAV_THUMB[0] + 28)
        right = QVBoxLayout(nav)
        right.setContentsMargins(14, 12, 14, 14)
        right.setSpacing(8)
        self.nav_title = QLabel()
        self.nav_title.setObjectName("navTitle")
        self.nav_title.setWordWrap(True)
        self.nav_text = QLabel()
        self.nav_text.setObjectName("navText")
        self.nav_text.setWordWrap(True)
        self.nav_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        app_heading = QLabel("Application")
        app_heading.setObjectName("navHeading")
        self.nav_use = QLabel()
        self.nav_use.setObjectName("navText")
        self.nav_use.setWordWrap(True)
        self.nav_picture = QLabel()
        self.nav_picture.setObjectName("navPicture")
        self.nav_picture.setFixedSize(NAV_THUMB[0] + 4, NAV_THUMB[1] + 4)
        self.nav_picture.setAlignment(Qt.AlignCenter)
        right.addWidget(self.nav_title)
        right.addWidget(self.nav_text, 1)
        right.addWidget(app_heading)
        right.addWidget(self.nav_use)
        right.addWidget(self.nav_picture, 0, Qt.AlignHCenter)

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(catalog, 1)
        body.addWidget(nav)

        info = QLabel("i")
        info.setObjectName("infoIcon")
        info.setFixedSize(28, 28)
        info.setAlignment(Qt.AlignCenter)
        tip = QLabel(CATALOG_TIP)
        tip.setObjectName("catalogTip")
        tip.setWordWrap(True)
        tip.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.add_button = QPushButton("Add")
        self.add_button.setIcon(make_icon("plus", "#ffffff"))
        self.add_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.add_button.clicked.connect(lambda: self._add(self.selected_kind))
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setIcon(make_icon("close", "#e0e0e0"))
        self.cancel_button.clicked.connect(self.reject)
        for button in (self.add_button, self.cancel_button):
            button.setFixedHeight(48)
            button.setMinimumWidth(150)
        bottom, bottom_layout = bottom_bar()
        bottom_layout.addWidget(info)
        bottom_layout.addWidget(tip, 1)
        bottom_layout.addWidget(self.cancel_button)
        bottom_layout.addWidget(self.add_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(body, 1)
        layout.addWidget(bottom)

        self._fit_to(parent)
        self.show_category(AddToolDialog.last_category)

    def _fit_to(self, parent) -> None:
        """As big as the catalog likes, never bigger than the window under it."""
        avail = parent.window().size() if parent is not None else None
        width, height = 1180, 720
        if avail is not None and avail.width() > 0:
            width = min(width, avail.width() - 24)
            height = min(height, avail.height() - 24)
        self.resize(max(width, 640), max(height, 480))

    def show_category(self, category: str) -> None:
        if category not in self._page_of:
            category = CATEGORIES[0][0]
        AddToolDialog.last_category = category
        self.category_tiles[category].setChecked(True)
        self.question.setText("▼  " + CATEGORY_QUESTIONS.get(category, "Pick a tool"))
        self._pages.setCurrentIndex(self._page_of[category])
        kinds = dict(CATEGORIES)[category]
        self.select(self.selected_kind if self.selected_kind in kinds else kinds[0])

    def select(self, kind: str) -> None:
        """Highlights the tool and shows it in the Application Navigator."""
        self.selected_kind = kind
        self.tiles[kind].setChecked(True)
        self.nav_title.setText(kind_title(kind))
        self.nav_text.setText(KIND_HELP.get(kind, ""))
        self.nav_use.setText(KIND_USES.get(kind, ""))
        self.nav_picture.setPixmap(kind_thumbnail(kind, NAV_THUMB))
        self.add_button.setText(f"Add {kind_title(kind)}")

    def _add(self, kind: Optional[str]) -> None:
        if kind is None:
            return
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
        self.delete_button.setIcon(make_icon("trash", "#ffffff"))
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
