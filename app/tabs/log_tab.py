from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QToolButton, QVBoxLayout, QWidget

from ..ng_log import collect_ng_log, full_image_for_thumb, ng_stamp_text
from ..paths import NG_THUMB_SIZE, NG_THUMBS_PER_PART
from ..theme import make_icon


class LogTab(QWidget):
    """Latest NG images grouped by part; tap a thumbnail to open it full size."""

    image_requested = Signal(Path, str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dirty = True

        self.ng_log_container = QWidget()
        self.ng_log_layout = QVBoxLayout(self.ng_log_container)
        self.ng_log_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.ng_log_container)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ng_log_summary_label = QLabel()
        refresh_button = QPushButton("Refresh")
        refresh_button.setIcon(make_icon("refresh", "#e0e0e0"))
        refresh_button.clicked.connect(self.refresh)
        open_row = QHBoxLayout()
        open_row.addWidget(self.ng_log_summary_label)
        open_row.addStretch(1)
        open_row.addWidget(refresh_button)

        layout = QVBoxLayout(self)
        layout.addLayout(open_row)
        layout.addWidget(scroll, 1)

    def mark_dirty(self) -> None:
        # Rebuild only if the operator is looking at the Log tab right now;
        # otherwise defer to the next time it's opened (keeps Monitor fast).
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    @Slot()
    def refresh(self) -> None:
        self._dirty = False
        while self.ng_log_layout.count():
            item = self.ng_log_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()

        groups = collect_ng_log()
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
                f"Part: {part_name}   ·   NG {count}   ·   Last: {ng_stamp_text(thumbs[0])}")
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
        full_image = full_image_for_thumb(thumb)
        stamp_text = ng_stamp_text(thumb)
        button = QToolButton()
        button.setObjectName("ngThumb")
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setIcon(QIcon(str(thumb)))
        button.setIconSize(NG_THUMB_SIZE)
        button.setText(stamp_text)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(
            lambda: self.image_requested.emit(full_image, f"NG  ·  {part_name}  ·  {stamp_text}"))
        return button
