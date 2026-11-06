from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..augmentation import PRESET_HINTS, PRESET_LABELS, build_hyp
from ..theme import RUN_BUTTON_STYLES, action_button_qss
from .layout import bottom_bar, make_card, row, top_bar

PREVIEW_COUNT = 6
TILE_SIZE = 180


class PreviewWorker(QThread):
    """Builds augmented samples off the UI thread (Ultralytics takes a few
    seconds to load the first time)."""

    done = Signal(list)  # [QImage]
    failed = Signal(str)

    def __init__(self, data_yaml: Path, hyp: dict) -> None:
        super().__init__()
        self._data_yaml = data_yaml
        self._hyp = hyp

    def run(self) -> None:
        try:
            from ..augment_preview import preview_samples
            images = []
            for bgr in preview_samples(self._data_yaml, self._hyp, count=PREVIEW_COUNT):
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                images.append(QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy())
            self.done.emit(images)
        except Exception as exc:
            self.failed.emit(f"Preview failed: {exc}")


class TrainingOptionsDialog(QDialog):
    """Augmentation preset, flips and epochs, with a preview of training
    samples exactly as Ultralytics will produce them."""

    def __init__(self, preset: str, mirror: bool, flip_ud: bool, epochs: int,
                 data_yaml: Optional[Path], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training Options")
        self.setModal(True)
        self.resize(1000, 700)
        self._data_yaml = data_yaml
        self._worker: Optional[PreviewWorker] = None
        self._pending = False

        self.preset_combo = QComboBox()
        for key, label in PRESET_LABELS.items():
            self.preset_combo.addItem(label, key)
        self.preset_combo.setCurrentIndex(max(0, self.preset_combo.findData(preset)))
        self.preset_hint = QLabel()
        self.preset_hint.setObjectName("hint")
        self.preset_hint.setWordWrap(True)
        self.mirror_check = QCheckBox("Mirror left-right")
        self.mirror_check.setChecked(mirror)
        self.flip_ud_check = QCheckBox("Flip upside-down")
        self.flip_ud_check.setChecked(flip_ud)
        self.flip_hint = QLabel("Leave off for printed text and parts whose left/right or up/down matters.")
        self.flip_hint.setObjectName("hint")
        self.flip_hint.setWordWrap(True)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setSingleStep(10)
        self.epochs_spin.setValue(epochs)
        self.epochs_spin.setFixedWidth(130)

        self.preview_button = QPushButton("Refresh Preview")
        self.preview_button.setFixedHeight(40)
        self.preview_button.clicked.connect(self.refresh_preview)
        self.preview_status = QLabel()
        self.preview_status.setObjectName("panelText")
        self.preview_status.setWordWrap(True)
        self.tiles = []
        for _ in range(PREVIEW_COUNT):
            tile = QLabel()
            tile.setAlignment(Qt.AlignCenter)
            tile.setFixedSize(TILE_SIZE, TILE_SIZE)
            tile.setStyleSheet("background: #0b0d10;")
            self.tiles.append(tile)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedHeight(44)
        self.cancel_button.clicked.connect(self.reject)
        self.save_button = QPushButton("Save")
        self.save_button.setFixedSize(170, 44)
        self.save_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.save_button.clicked.connect(self.accept)

        for signal in (self.preset_combo.currentIndexChanged, self.mirror_check.toggled,
                       self.flip_ud_check.toggled):
            signal.connect(self._on_options_changed)

        self._build_layout()
        self._update_hint()
        if data_yaml is None:
            self.preview_button.setEnabled(False)
            self.preview_status.setText("Open a dataset with labelled images in the Train tab to see a preview.")
        else:
            self.refresh_preview()

    def _build_layout(self) -> None:
        top, top_layout = top_bar((10, 6, 10, 6))
        title = QLabel("Training Options")
        title.setObjectName("monitorInfo")
        hint = QLabel("Augmentation makes varied copies of each image while training")
        hint.setObjectName("hint")
        top_layout.addWidget(title)
        top_layout.addStretch(1)
        top_layout.addWidget(hint)

        card, body = make_card("Augmentation")
        body.addLayout(row(QLabel("Preset:"), self.preset_combo, None))
        body.addWidget(self.preset_hint)
        body.addSpacing(8)
        body.addWidget(self.mirror_check)
        body.addWidget(self.flip_ud_check)
        body.addWidget(self.flip_hint)
        body.addSpacing(8)
        body.addLayout(row(QLabel("Epochs:"), None, self.epochs_spin))
        epochs_hint = QLabel("More epochs = longer training; stops early when it stops improving.")
        epochs_hint.setObjectName("hint")
        epochs_hint.setWordWrap(True)
        body.addWidget(epochs_hint)
        body.addStretch(1)
        card.setFixedWidth(330)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        header = QLabel("Preview — what the trainer sees")
        header.setObjectName("panelHeader")
        side_layout.addWidget(header)
        grid_box = QWidget()
        grid = QGridLayout(grid_box)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)
        for i, tile in enumerate(self.tiles):
            grid.addWidget(tile, i // 3, i % 3)
        side_layout.addWidget(grid_box, 0, Qt.AlignHCenter)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(8, 0, 8, 8)
        status_row.addWidget(self.preview_status, 1)
        status_row.addWidget(self.preview_button)
        side_layout.addLayout(status_row)
        side_layout.addStretch(1)

        content = QHBoxLayout()
        content.addWidget(card)
        content.addWidget(side_panel, 1)

        bottom, bottom_layout = bottom_bar()
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.cancel_button)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(content, 1)
        layout.addWidget(bottom)

    # --- Values -----------------------------------------------------------

    @property
    def preset(self) -> str:
        return self.preset_combo.currentData()

    @property
    def mirror(self) -> bool:
        return self.mirror_check.isChecked()

    @property
    def flip_ud(self) -> bool:
        return self.flip_ud_check.isChecked()

    @property
    def epochs(self) -> int:
        return self.epochs_spin.value()

    def _update_hint(self) -> None:
        self.preset_hint.setText(PRESET_HINTS[self.preset])

    @Slot()
    def _on_options_changed(self, *_) -> None:
        self._update_hint()
        if self._data_yaml is not None:
            self.refresh_preview()

    # --- Preview ----------------------------------------------------------

    @Slot()
    def refresh_preview(self) -> None:
        if self._data_yaml is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._pending = True  # re-run with the newest options when this one ends
            return
        self.preview_status.setText("Making preview...")
        self.preview_button.setEnabled(False)
        self._worker = PreviewWorker(self._data_yaml, build_hyp(self.preset, self.mirror, self.flip_ud))
        self._worker.done.connect(self._on_preview_done)
        self._worker.failed.connect(self._on_preview_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    @Slot(list)
    def _on_preview_done(self, images: list) -> None:
        for tile, image in zip(self.tiles, images):
            tile.setPixmap(QPixmap.fromImage(image).scaled(
                TILE_SIZE, TILE_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview_status.setText("Random samples. Refresh to see others.")

    @Slot(str)
    def _on_preview_failed(self, message: str) -> None:
        self.preview_status.setText(message)

    @Slot()
    def _on_worker_finished(self) -> None:
        self.preview_button.setEnabled(True)
        if self._pending:
            self._pending = False
            self.refresh_preview()

    def done(self, result: int) -> None:
        # Don't destroy the dialog under a running preview thread.
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        super().done(result)
