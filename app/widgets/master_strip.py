from typing import List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..autocapture import Master

COLUMNS = 3
THUMB = QSize(68, 51)
# Border per state: no master near / nearest but outside the threshold / match.
BORDER = {"idle": "#9aa5b1", "near": "#f0a020", "match": "#2ecc71"}


class MasterStrip(QWidget):
    """The job's masters as small thumbnails in the Monitor side panel; the
    one nearest the live view is outlined (green = within the threshold).
    Tapping any thumbnail opens the Masters dialog."""

    edit_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("clearBox")
        self._tiles: List[tuple] = []  # (button, caption)
        self._highlight: Optional[tuple] = None
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(4)
        self._empty = QPushButton("No masters — tap to set")
        self._empty.setObjectName("masterEmpty")
        self._empty.setFixedHeight(36)
        self._empty.clicked.connect(self.edit_requested)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 0)
        layout.addWidget(self._empty)
        layout.addLayout(self._grid)

    def set_masters(self, masters: List[Master]) -> None:
        for button, caption in self._tiles:
            for widget in (button, caption):
                self._grid.removeWidget(widget)
                widget.hide()
                widget.deleteLater()
        self._tiles = []
        self._highlight = None
        for index, master in enumerate(masters):
            button = QPushButton()
            button.setFixedSize(THUMB.width() + 6, THUMB.height() + 6)
            button.setIconSize(THUMB)
            pixmap = QPixmap(str(master.path))
            if not pixmap.isNull():
                button.setIcon(QIcon(pixmap.scaled(THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            button.setToolTip(f"Master {index + 1} — tap to edit masters")
            button.clicked.connect(self.edit_requested)
            caption = QLabel(f"Master {index + 1}")
            caption.setObjectName("masterCaption")
            caption.setAlignment(Qt.AlignHCenter)
            caption.setFixedWidth(button.width())
            row, col = divmod(index, COLUMNS)
            self._grid.addWidget(button, row * 2, col, Qt.AlignHCenter)
            self._grid.addWidget(caption, row * 2 + 1, col, Qt.AlignHCenter)
            self._tiles.append((button, caption))
            self._style(button, "idle")
        self._empty.setVisible(not masters)

    def set_match(self, index: Optional[int], matched: bool) -> None:
        """Outlines master `index` (None = no live frame / no masters)."""
        highlight = (index, matched) if index is not None else None
        if highlight == self._highlight:
            return  # most frames: nothing changed, skip restyling
        self._highlight = highlight
        for i, (button, _) in enumerate(self._tiles):
            state = "idle" if i != index else ("match" if matched else "near")
            self._style(button, state)

    @staticmethod
    def _style(button: QPushButton, state: str) -> None:
        width = 1 if state == "idle" else 3
        button.setStyleSheet(
            f"QPushButton {{ background: #0b0d10; border: {width}px solid {BORDER[state]};"
            " border-radius: 3px; padding: 0px; }")
