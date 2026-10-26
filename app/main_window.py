import sys
from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from . import jobs
from .context import AppContext
from .tabs.job_change_tab import JobChangeTab
from .tabs.log_tab import LogTab
from .tabs.monitor_tab import MonitorTab
from .tabs.settings_tab import SettingsTab
from .tabs.train_tab import TrainTab
from .theme import DARK_INDUSTRIAL_QSS
from .widgets.barcode_scan import BarcodeScanDialog
from .widgets.image_viewer import ImageOverlay


class MainWindow(QWidget):
    """Hosts the tabs and the flows that span several of them (switching
    jobs, barcode scanning, shutdown). Per-tab behaviour lives in app/tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Object Detection Concept")
        self.ctx = AppContext(self)
        self._barcode_dialog: Optional[BarcodeScanDialog] = None

        self.status_label = QLabel("Ready")
        self.ctx.status.connect(self.status_label.setText)
        self.ctx.worker.status.connect(self.status_label.setText)
        self.ctx.tower.message.connect(self.status_label.setText)

        self.monitor_tab = MonitorTab(self.ctx)
        self.settings_tab = SettingsTab(self.ctx)
        self.job_change_tab = JobChangeTab(self.ctx)
        self.train_tab = TrainTab(self.ctx)
        self.log_tab = LogTab()

        self.job_change_tab.job_selected.connect(self._on_job_selected)
        self.job_change_tab.scan_requested.connect(self._open_barcode_scan_dialog)
        self.monitor_tab.ng_saved.connect(self.log_tab.mark_dirty)
        self.log_tab.image_requested.connect(lambda path, caption: self.image_overlay.show_image(path, caption))

        self.tabs = QTabWidget()
        self.tabs.addTab(self.monitor_tab, "Monitor")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.job_change_tab, "Job Change")
        self.tabs.addTab(self.train_tab, "Train")
        self.log_tab_index = self.tabs.addTab(self.log_tab, "Log")
        self.image_overlay = ImageOverlay(self)

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.tabs, 1)
        root_layout.addWidget(self.status_label)
        self.setLayout(root_layout)

        self.settings_tab.scan_cameras()
        self.ctx.tower.connect_device()
        self._auto_load_last_config()

    # Monitor-tab actions, kept here as the window's public controls.

    def start_stream(self) -> None:
        self.monitor_tab.start_stream()

    def stop_stream(self) -> None:
        self.monitor_tab.stop_stream()

    # --- Switching jobs -------------------------------------------------

    def _apply_job(self, part_no: str, config: dict) -> None:
        model_ok = self.settings_tab.apply_config(config)
        self.ctx.set_current_part(part_no)
        if model_ok:
            self.ctx.status.emit(f"Applied config for '{part_no}'.")

    def _save_current_as_job(self, part_no: str) -> None:
        self.ctx.set_current_part(part_no)
        jobs.save_part_config(part_no, self.settings_tab.collect_config())
        self.job_change_tab.refresh_job_buttons()

    def _auto_load_last_config(self) -> None:
        part_no = jobs.read_last_part()
        if part_no is None:
            return
        config = jobs.load_part_config(part_no)
        if config is None:
            return
        model_ok = self.settings_tab.apply_config(config)
        self.ctx.set_current_part(part_no)
        if model_ok:
            self.ctx.status.emit(f"Auto-loaded last config: '{part_no}'.")

    @Slot(str)
    def _on_job_selected(self, part_no: str) -> None:
        config = jobs.load_part_config(part_no)
        if config is None:
            self.ctx.status.emit(f"Config for '{part_no}' no longer exists.")
            self.job_change_tab.refresh_job_buttons()
            return
        self._apply_job(part_no, config)

    # --- Barcode scanning -----------------------------------------------

    @Slot()
    def _open_barcode_scan_dialog(self) -> None:
        if self.ctx.worker.isRunning():
            self.ctx.status.emit("Stop the Monitor stream before scanning a barcode (camera is in use).")
            return
        index = self.ctx.camera_index
        if index is None:
            self.ctx.status.emit("Select a valid camera.")
            return

        self._barcode_dialog = BarcodeScanDialog(int(index), parent=self)
        self._barcode_dialog.load_requested.connect(self._on_part_load_requested)
        self._barcode_dialog.save_requested.connect(self._on_part_save_requested)
        self._barcode_dialog.finished.connect(self._on_barcode_dialog_finished)
        self._barcode_dialog.show()

    def _on_barcode_dialog_finished(self) -> None:
        self._barcode_dialog = None

    @Slot(str)
    def _on_part_load_requested(self, part_no: str) -> None:
        config = jobs.load_part_config(part_no)
        if config is None:
            answer = QMessageBox.question(
                self,
                "No Config Found",
                f"No config found for '{part_no}'. Save current settings as a new config for this part?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer == QMessageBox.Yes:
                self._save_current_as_job(part_no)
                self.ctx.status.emit(f"Saved new config for '{part_no}'.")
            return

        answer = QMessageBox.question(
            self,
            "Config Found",
            f"Found saved config for '{part_no}'. Apply?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._apply_job(part_no, config)

    @Slot(str)
    def _on_part_save_requested(self, part_no: str) -> None:
        self._save_current_as_job(part_no)
        self.ctx.status.emit(f"Saved config for '{part_no}'.")

    # --- Window events --------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "image_overlay") and self.image_overlay.isVisible():
            self.image_overlay.setGeometry(self.rect())

    def closeEvent(self, event) -> None:
        self.stop_stream()
        # Must fully join: destroying a QThread that's still running aborts
        # the whole process ("QThread: Destroyed while thread is still running").
        self.ctx.worker.wait()
        if self._barcode_dialog is not None:
            self._barcode_dialog.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_INDUSTRIAL_QSS)
    window = MainWindow()
    window.resize(1050, 720)
    window.showFullScreen()
    sys.exit(app.exec())
