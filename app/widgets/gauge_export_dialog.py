"""Export Gauge Data: copy a date range of a job's daily CSV files to a USB drive."""

from datetime import date
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..gauge_log import days_size, export_days, usb_drives
from ..i18n import tr
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon
from .layout import bottom_bar, make_card, row, top_bar

DRIVE_SCAN_MS = 2000
ROW_HEIGHT = 48
COMBO_WIDTH = 300


class _ExportWorker(QThread):
    """Copying and flushing to a USB stick can take many seconds."""

    done = Signal(str, int)
    failed = Signal(str)

    def __init__(self, part: Optional[str], days: List[date], drive: Path) -> None:
        super().__init__()
        self._args = (part, days, drive)

    def run(self) -> None:
        try:
            self.done.emit(str(export_days(*self._args)), len(self._args[1]))
        except OSError as exc:
            self.failed.emit(str(exc))


class GaugeExportDialog(QDialog):
    def __init__(self, part: Optional[str], days: List[date], parent: Optional[QWidget] = None) -> None:
        """`days`: the days this job has data for, newest first."""
        super().__init__(parent)
        self._part = part
        self._days = days
        self._drives: List[Path] = []
        self._worker: Optional[_ExportWorker] = None
        self.setWindowTitle("Export Gauge Data")
        self.resize(640, 400)

        self.from_combo = self._combo()
        self.to_combo = self._combo()
        for day in days:
            self.from_combo.addItem(day.isoformat())
            self.to_combo.addItem(day.isoformat())
        self.from_combo.setCurrentIndex(len(days) - 1)  # the whole record until narrowed
        self.drive_combo = self._combo()
        self.from_combo.currentIndexChanged.connect(lambda _: self._update())
        self.to_combo.currentIndexChanged.connect(lambda _: self._update())

        self.status_label = QLabel()
        self.status_label.setObjectName("monitorInfo")
        self.status_label.setWordWrap(True)
        self.close_button = QPushButton("Close")
        self.close_button.setFixedHeight(44)
        self.close_button.clicked.connect(self.reject)
        self.export_button = QPushButton("Export")
        self.export_button.setIcon(make_icon("save"))
        self.export_button.setFixedSize(170, 44)
        self.export_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.export_button.clicked.connect(self._export)

        top, top_layout = top_bar((10, 6, 10, 6))
        title = QLabel("Export Gauge Data")
        title.setObjectName("monitorInfo")
        top_layout.addWidget(title)
        top_layout.addStretch(1)

        card, body = make_card("Date range and drive")
        for text, combo in (("From date", self.from_combo), ("To date", self.to_combo), ("USB drive", self.drive_combo)):
            body.addLayout(row(QLabel(text), None, combo))
        hint = QLabel("One CSV file per day, in a new folder on the drive.")
        hint.setObjectName("hint")
        body.addWidget(hint)
        body.addStretch(1)
        body.addWidget(self.status_label)

        bottom, bottom_layout = bottom_bar()
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.close_button)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.export_button)

        content = QHBoxLayout()
        content.addWidget(card)
        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(content, 1)
        layout.addWidget(bottom)

        self._timer = QTimer(self)
        self._timer.setInterval(DRIVE_SCAN_MS)
        self._timer.timeout.connect(self._scan_drives)
        self._timer.start()
        self._update()
        self._scan_drives()

    @staticmethod
    def _combo() -> QComboBox:
        combo = QComboBox()
        combo.setFixedSize(COMBO_WIDTH, ROW_HEIGHT)
        return combo

    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _chosen_days(self) -> List[date]:
        """Oldest first; empty when From is after To."""
        first = self._days[self.from_combo.currentIndex()]
        last = self._days[self.to_combo.currentIndex()]
        return sorted(day for day in self._days if first <= day <= last)

    @Slot()
    def _scan_drives(self) -> None:
        """A stick can be plugged in or pulled out while this is open."""
        drives = usb_drives()
        if self.busy or drives == self._drives:
            return
        current = self.drive_combo.currentData()
        self._drives = drives
        self.drive_combo.clear()
        for drive in drives:
            self.drive_combo.addItem(drive.name, str(drive))
        index = self.drive_combo.findData(current)
        if index >= 0:
            self.drive_combo.setCurrentIndex(index)
        self._update()

    def _update(self) -> None:
        days = self._chosen_days()
        if not days:
            text = tr("From date is after To date.")
        elif not self._drives:
            text = tr("Plug in a USB drive.")
        else:
            text = tr("{} days, {} MB").format(len(days), f"{days_size(self._part, days) / 1e6:.1f}")
        self.status_label.setText(text)
        self.export_button.setEnabled(bool(days and self._drives))

    def _set_controls(self, enabled: bool) -> None:
        for widget in (self.from_combo, self.to_combo, self.drive_combo, self.close_button):
            widget.setEnabled(enabled)

    @Slot()
    def _export(self) -> None:
        self._set_controls(False)
        self.export_button.setEnabled(False)
        self.status_label.setText(tr("Copying to USB..."))
        self._worker = _ExportWorker(self._part, self._chosen_days(), Path(self.drive_combo.currentData()))
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._result = ""
        self._worker.start()

    @Slot(str, int)
    def _on_done(self, target: str, count: int) -> None:
        self._result = tr("Saved {} days to {}. You can remove the USB drive.").format(count, target)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._result = tr("Export failed: {}").format(message)

    @Slot()
    def _on_finished(self) -> None:
        self._set_controls(True)
        self._scan_drives()
        self._update()  # re-enables Export for the next copy
        self.status_label.setText(self._result)

    def reject(self) -> None:
        if self.busy:
            return  # stay until the copy is finished: pulling the stick now would lose it
        self._timer.stop()
        super().reject()
