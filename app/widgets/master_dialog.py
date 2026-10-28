from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..autocapture import MAX_MASTERS, Master, hamming, image_hash, qimage_to_array, save_master
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon

THUMB_W, THUMB_H = 220, 150
GRID_COLUMNS = 3
# Live distance readout refresh, for tuning the match threshold.
DISTANCE_REFRESH_MS = 300


class _MasterCard(QFrame):
    """One master: thumbnail, replace / remove, live distance."""

    def __init__(self, master: Optional[Master], new_image: Optional[QImage]) -> None:
        super().__init__()
        self.setObjectName("card")
        self.master = master  # saved master, kept unless its image is replaced
        self.new_image = new_image  # grabbed in this dialog, not yet saved
        self.hash: Optional[int] = None

        self.title = QLabel()
        self.thumb = QLabel()
        self.thumb.setObjectName("videoFrame")
        self.thumb.setFixedSize(THUMB_W, THUMB_H)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.grab_button = QPushButton("Set from live")
        self.grab_button.setIcon(make_icon("camera", "#e0e0e0"))
        self.remove_button = QPushButton("Remove")
        self.remove_button.setIcon(make_icon("trash", "#ffffff"))
        self.remove_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["stop"]))
        for button in (self.grab_button, self.remove_button):
            button.setFixedHeight(40)
        self.distance_label = QLabel()
        self.distance_label.setObjectName("hint")

        buttons = QHBoxLayout()
        buttons.addWidget(self.grab_button, 1)
        buttons.addWidget(self.remove_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.title)
        layout.addWidget(self.thumb, 0, Qt.AlignHCenter)
        layout.addLayout(buttons)
        layout.addWidget(self.distance_label)

        if new_image is not None:
            self.set_image(new_image)
        elif master is not None:
            self.hash = master.hash
            self._show(QPixmap(str(master.path)))

    def set_image(self, image: QImage) -> None:
        self.new_image = image
        self.hash = image_hash(qimage_to_array(image))
        self._show(QPixmap.fromImage(image))

    def _show(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            self.thumb.setText("(image missing)")
            return
        self.thumb.setPixmap(pixmap.scaled(THUMB_W - 4, THUMB_H - 4, Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))


class MasterDialog(QDialog):
    """Adds, replaces and removes the job's auto-capture master images: a
    live frame like any of them is saved for labelling. Nothing is written
    until Save; Cancel drops every change, removals included.

    ensure_live(then) starts the camera if needed and calls `then` once
    frames flow, so the user never has to start it first."""

    def __init__(self, masters: List[Master], part_no: str,
                 grab_frame: Callable[[], Optional[QImage]],
                 ensure_live: Callable[[Callable[[], None]], bool], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto Capture Masters")
        self.setModal(True)
        self.resize(900, 640)
        self._part_no = part_no
        self._grab_frame = grab_frame
        self._ensure_live = ensure_live
        self._closed = False  # queued camera actions are dropped after close
        self._cards: List[_MasterCard] = []
        self.entries: list = []

        top_bar = QFrame()
        top_bar.setObjectName("monitorTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 6, 10, 6)
        title = QLabel(f"Masters — Job {part_no}")
        title.setObjectName("monitorInfo")
        self.count_label = QLabel()
        self.count_label.setObjectName("hint")
        top_layout.addWidget(title)
        top_layout.addStretch(1)
        top_layout.addWidget(self.count_label)

        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        # Always first: what Add Master / Set from live will take.
        self.live_card = QFrame()
        self.live_card.setObjectName("card")
        live_layout = QVBoxLayout(self.live_card)
        live_layout.setContentsMargins(8, 6, 8, 8)
        live_title = QLabel("Live view")
        live_title.setObjectName("monitorInfo")
        self.live_thumb = QLabel("Starting camera...")
        self.live_thumb.setObjectName("videoFrame")
        self.live_thumb.setFixedSize(THUMB_W, THUMB_H)
        self.live_thumb.setAlignment(Qt.AlignCenter)
        live_hint = QLabel("Show the part, then press Add Master.")
        live_hint.setObjectName("hint")
        live_hint.setWordWrap(True)
        live_layout.addWidget(live_title)
        live_layout.addWidget(self.live_thumb, 0, Qt.AlignHCenter)
        live_layout.addWidget(live_hint)
        live_layout.addStretch(1)
        self.empty_label = QLabel("No masters yet — show the part to the camera and press Add Master.")
        self.empty_label.setObjectName("hint")
        grid_box = QWidget()
        grid_box.setObjectName("clearBox")
        box_layout = QVBoxLayout(grid_box)
        box_layout.addWidget(self.empty_label)
        box_layout.addLayout(self.grid)
        box_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(grid_box)
        scroll.viewport().setObjectName("clearBox")

        hint = QLabel("Frames that match any master are saved to the job's auto capture folder, "
                      "for labelling in the Train tab. Use several masters for different views or "
                      "variants. Distance = difference from the live view, lower is closer.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)

        self.add_button = QPushButton("Add Master")
        self.add_button.setIcon(make_icon("plus", "#e0e0e0"))
        self.add_button.clicked.connect(self._add_from_live)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setIcon(make_icon("close", "#e0e0e0"))
        self.cancel_button.clicked.connect(self.reject)
        self.save_button = QPushButton("Save")
        self.save_button.setIcon(make_icon("save", "#ffffff"))
        self.save_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.save_button.clicked.connect(self._save)
        for button in (self.add_button, self.cancel_button, self.save_button):
            button.setFixedHeight(44)
            button.setMinimumWidth(140)
        self.error_label = QLabel()
        self.error_label.setObjectName("dangerText")
        self.info_label = QLabel()
        self.info_label.setObjectName("hint")

        bottom_bar = QFrame()
        bottom_bar.setObjectName("monitorBottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 4, 6, 4)
        bottom_layout.addWidget(self.add_button)
        bottom_layout.addWidget(self.info_label)
        bottom_layout.addWidget(self.error_label, 1)
        bottom_layout.addWidget(self.cancel_button)
        bottom_layout.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top_bar)
        layout.addWidget(scroll, 1)
        layout.addWidget(hint)
        layout.addWidget(bottom_bar)

        for master in masters[:MAX_MASTERS]:
            self._add_card(master, None)
        self._relayout()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_distances)
        self._timer.start(DISTANCE_REFRESH_MS)
        self._refresh_distances()
        if not self._ensure_live(lambda: None):  # start the live view right away
            self.live_thumb.setText("Camera not available\n(see the status bar)")

    # --- Cards ------------------------------------------------------------

    def _add_card(self, master: Optional[Master], image: Optional[QImage]) -> _MasterCard:
        card = _MasterCard(master, image)
        card.grab_button.clicked.connect(lambda _=False, c=card: self._replace_from_live(c))
        card.remove_button.clicked.connect(lambda _=False, c=card: self._confirm_remove(c))
        self._cards.append(card)
        return card

    def _relayout(self) -> None:
        for card in self._cards:
            self.grid.removeWidget(card)
        self.grid.addWidget(self.live_card, 0, 0)
        for index, card in enumerate(self._cards):
            card.title.setText(f"Master {index + 1}")
            row, col = divmod(index + 1, GRID_COLUMNS)
            self.grid.addWidget(card, row, col)
        self.empty_label.setVisible(not self._cards)
        self.add_button.setEnabled(len(self._cards) < MAX_MASTERS)
        self.count_label.setText(f"{len(self._cards)} / {MAX_MASTERS}")

    def _when_live(self, action: Callable[[QImage], None]) -> None:
        """Runs action(live frame) now, or as soon as the camera delivers."""
        self.error_label.setText("")
        image = self._grab_frame()
        if image is not None:
            action(image)
            return
        self.info_label.setText("Starting camera...")

        def run() -> None:
            self.info_label.setText("")
            frame = self._grab_frame()
            if frame is not None and not self._closed:
                action(frame)

        if not self._ensure_live(run):
            self.info_label.setText("")
            self.error_label.setText("Camera could not start — see the status bar.")

    def _add_from_live(self) -> None:
        if len(self._cards) < MAX_MASTERS:
            self._when_live(self._add_image)

    def _add_image(self, image: QImage) -> None:
        if len(self._cards) >= MAX_MASTERS:
            return
        self._add_card(None, image)
        self._relayout()
        self._refresh_distances()

    def _replace_from_live(self, card: _MasterCard) -> None:
        def replace(image: QImage) -> None:
            if card in self._cards:
                card.set_image(image)
                self._refresh_distances()
        self._when_live(replace)

    def _confirm_remove(self, card: _MasterCard) -> None:
        # No is the default, so a double tap or Enter can't remove by accident.
        index = self._cards.index(card)
        answer = QMessageBox.question(
            self, "Remove Master",
            f"Remove master {index + 1}?\n\n"
            "Images already captured are kept. Nothing is saved until you press Save.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self._cards.remove(card)
        self.grid.removeWidget(card)
        card.hide()
        card.deleteLater()
        self._relayout()

    def _refresh_distances(self) -> None:
        image = self._grab_frame()
        if image is not None:
            self.live_thumb.setPixmap(QPixmap.fromImage(image).scaled(
                THUMB_W - 4, THUMB_H - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        live_hash = image_hash(qimage_to_array(image)) if image is not None else None
        for card in self._cards:
            if card.hash is None:
                card.distance_label.setText("")
            elif live_hash is None:
                card.distance_label.setText("Distance now: — (no live frame)")
            else:
                card.distance_label.setText(f"Distance now: {hamming(live_hash, card.hash)}")

    def done(self, result: int) -> None:
        self._closed = True
        self._timer.stop()
        super().done(result)

    # --- Save -------------------------------------------------------------

    def _save(self) -> None:
        entries = []
        for card in self._cards:
            if card.new_image is not None:
                try:
                    image = save_master(self._part_no, card.new_image)
                except OSError as exc:
                    self.error_label.setText(f"Save failed: {exc}")
                    return
                entries.append({"image": image})
            elif card.master is not None:
                entries.append({"image": card.master.image})
        self.entries = entries
        self.accept()
