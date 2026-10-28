from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import jobs
from ..context import AppContext
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon
from ..widgets.layout import bottom_bar, make_card, top_bar
from ..widgets.remove_job_dialog import RemoveJobDialog


class JobChangeTab(QWidget):
    """Grid of saved jobs; tapping one, or scanning a barcode, switches part."""

    JOB_GRID_COLUMNS = 4

    job_selected = Signal(str)
    scan_requested = Signal()

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx

        self.part_config_label = QLabel("No part scanned")
        self.remove_job_button = QPushButton("Remove Job...")
        self.remove_job_button.setIcon(make_icon("trash", "#e0e0e0"))
        self.remove_job_button.setFixedHeight(44)
        self.remove_job_button.clicked.connect(self._on_remove_job_clicked)

        self.scan_barcode_button = QPushButton("Scan Barcode...")
        self.scan_barcode_button.setIcon(make_icon("barcode", "#e0e0e0"))
        self.scan_barcode_button.setFixedHeight(44)
        self.scan_barcode_button.setMinimumWidth(200)
        self.scan_barcode_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.scan_barcode_button.clicked.connect(self.scan_requested)

        self._build_layout()
        self.refresh_job_buttons()
        ctx.part_changed.connect(self._on_part_changed)

    def _build_layout(self) -> None:
        top, top_layout = top_bar(margins=(10, 4, 10, 4))
        caption = QLabel("Current Part:")
        caption.setObjectName("monitorInfo")
        self.part_config_label.setObjectName("currentPart")
        top_layout.addWidget(caption)
        top_layout.addWidget(self.part_config_label)
        top_layout.addStretch(1)

        jobs_card, jobs_body = make_card("Saved Jobs", icon="folder")
        self.job_buttons_layout = QGridLayout()
        self.job_buttons_layout.setSpacing(10)
        grid_widget = QWidget()
        grid_widget.setObjectName("clearBox")
        grid_widget.setLayout(self.job_buttons_layout)
        self.job_buttons_container = QWidget()
        self.job_buttons_container.setObjectName("clearBox")
        container_layout = QVBoxLayout(self.job_buttons_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(grid_widget)
        container_layout.addStretch(1)

        jobs_scroll = QScrollArea()
        jobs_scroll.setWidgetResizable(True)
        jobs_scroll.setFrameShape(QFrame.NoFrame)
        jobs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        jobs_scroll.setWidget(self.job_buttons_container)
        jobs_scroll.viewport().setObjectName("clearBox")
        jobs_body.addWidget(jobs_scroll, 1)

        bottom, bottom_layout = bottom_bar(margins=(10, 4, 6, 4))
        bottom_layout.addWidget(self.remove_job_button)
        bottom_layout.addSpacing(10)
        hint = QLabel("Tap a job to switch  ·  Scan a barcode to load or create a job")
        hint.setObjectName("hint")
        bottom_layout.addWidget(hint)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.scan_barcode_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addWidget(jobs_card, 1)
        layout.addWidget(bottom)

    @Slot(object)
    def _on_part_changed(self, part_no: Optional[str]) -> None:
        self.part_config_label.setText(part_no if part_no is not None else "No part selected")
        self.refresh_job_buttons()

    @Slot()
    def _on_remove_job_clicked(self) -> None:
        saved = jobs.list_saved_jobs()
        if not saved:
            self._ctx.status.emit("No saved jobs to remove.")
            return
        dialog = RemoveJobDialog([j[0] for j in saved], self._ctx.current_part_no, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        part_no = dialog.selected_part()
        config_path = next((path for p, path, _ in saved if p == part_no), None)
        if config_path is None:
            return
        try:
            config_path.unlink()
        except OSError as exc:
            self._ctx.status.emit(f"Could not remove job '{part_no}': {exc}")
            return
        if part_no == self._ctx.current_part_no:
            self._ctx.set_current_part(None)
        self.refresh_job_buttons()
        self._ctx.status.emit(f"Removed job '{part_no}'.")

    def refresh_job_buttons(self) -> None:
        while self.job_buttons_layout.count() > 0:
            item = self.job_buttons_layout.takeAt(0)
            if item.widget():
                # Hide now: deleteLater only runs once control returns to the
                # event loop, and until then the old tile would still paint.
                item.widget().hide()
                item.widget().deleteLater()

        saved = jobs.list_saved_jobs()
        self.remove_job_button.setEnabled(bool(saved))
        if not saved:
            empty = QLabel("No saved jobs yet. Scan a barcode to create one.")
            empty.setObjectName("hint")
            self.job_buttons_layout.addWidget(empty, 0, 0)
            return

        position = 0
        for part_no, _, config in saved:
            model_path = config.get("model_path")
            model = Path(model_path).stem if model_path else "no model"
            # Tile shows the part and its model; the active job is highlighted.
            button = QPushButton(f"{part_no}\n{model}")
            button.setObjectName("jobButton")
            button.setCheckable(True)
            button.setChecked(part_no == self._ctx.current_part_no)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(lambda checked, p=part_no: self.job_selected.emit(p))
            row, col = divmod(position, self.JOB_GRID_COLUMNS)
            self.job_buttons_layout.addWidget(button, row, col)
            position += 1
        for col in range(self.JOB_GRID_COLUMNS):
            self.job_buttons_layout.setColumnStretch(col, 1)
