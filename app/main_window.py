import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QLockFile, QSize, QStandardPaths, Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import jobs, ng_log
from .context import AppContext
from .tabs.job_change_tab import JobChangeTab
from .tabs.log_tab import LogTab
from .tabs.monitor_tab import MonitorTab
from .tabs.settings_tab import SettingsTab
from .tabs.tools_tab import ToolsTab
from .tabs.train_tab import TrainTab
from .i18n import LANGUAGES, tr, translator
from .theme import DARK_INDUSTRIAL_QSS, HintWrapper, badge_icon, make_flag_icon, make_icon
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

        # Status messages are shown in the chosen language (i18n patterns).
        self.status_label = QLabel("Ready")
        self.status_label.setProperty("_i18n_skip", True)
        self._status_text = "Ready"
        for signal in (self.ctx.status, self.ctx.worker.status, self.ctx.tower.message):
            signal.connect(self._show_status)

        self.monitor_tab = MonitorTab(self.ctx)
        self.settings_tab = SettingsTab(self.ctx)
        self.tools_tab = ToolsTab(self.ctx)
        self.job_change_tab = JobChangeTab(self.ctx)
        self.train_tab = TrainTab(self.ctx)
        self.log_tab = LogTab()

        self.job_change_tab.job_selected.connect(self._on_job_selected)
        self.job_change_tab.scan_requested.connect(self._open_barcode_scan_dialog)
        self.monitor_tab.ng_saved.connect(self.log_tab.mark_dirty)
        self.log_tab.image_requested.connect(lambda path, caption: self.image_overlay.show_image(path, caption))

        self.ctx.show_tab_requested.connect(self._show_tab)
        self.ctx.stop_stream_requested.connect(self.monitor_tab.stop_stream)
        self._resume_stream_after_scan = False

        self.tabs = QTabWidget()
        self.tabs.addTab(self.monitor_tab, "Monitor")
        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.tools_tab, "Tools")
        self.tabs.addTab(self.job_change_tab, "Job Change")
        self.tabs.addTab(self.train_tab, "Train")
        self.log_tab_index = self.tabs.addTab(self.log_tab, "Log")
        # Same badges as the panel headers: image-processing tools vs AI.
        self.tabs.setTabIcon(self.tabs.indexOf(self.tools_tab), badge_icon("vision"))
        self.tabs.setTabIcon(self.tabs.indexOf(self.train_tab), badge_icon("ai"))
        for tab, kind in ((self.monitor_tab, "monitor"), (self.settings_tab, "gear"),
                          (self.job_change_tab, "swap"), (self.log_tab, "list")):
            self.tabs.setTabIcon(self.tabs.indexOf(tab), make_icon(kind, "#e0e0e0"))
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.setCornerWidget(self._make_language_switch(), Qt.TopRightCorner)
        self._tools_started_camera = False
        self._prev_tab = self.tabs.currentWidget()
        self.image_overlay = ImageOverlay(self)

        root_layout = QVBoxLayout()
        root_layout.addWidget(self.tabs, 1)
        root_layout.addWidget(self.status_label)
        self.setLayout(root_layout)

        self.settings_tab.scan_cameras()
        self.ctx.tower.connect_device()
        self.ctx.relay.restart()
        self._trim_ng_log()
        self._auto_load_last_config()

    # Monitor-tab actions, kept here as the window's public controls.

    def start_stream(self) -> None:
        self.monitor_tab.start_stream()

    def stop_stream(self) -> None:
        self.monitor_tab.stop_stream()

    # --- Switching jobs -------------------------------------------------

    def confirm_unsaved(self) -> bool:
        """Settings has unsaved changes: Save / Discard / Cancel. False = Cancel."""
        changed = self.settings_tab.unsaved_changes()
        if not changed:
            return True
        box = QMessageBox(QMessageBox.Warning, "Unsaved Settings",
                          f"{len(changed)} setting(s) changed but not saved:\n\n{', '.join(changed)}\n\n"
                          "Save them?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, self)
        box.setDefaultButton(QMessageBox.Save)
        answer = box.exec()
        if answer == QMessageBox.Save:
            self.settings_tab.save()
        elif answer == QMessageBox.Discard:
            self.settings_tab.revert()
        return answer != QMessageBox.Cancel

    def _apply_job(self, part_no: str, config: dict) -> None:
        if not self.confirm_unsaved():
            return
        model_ok = self.settings_tab.apply_config(config)
        self.ctx.set_current_part(part_no)
        if model_ok:
            self.ctx.status.emit(f"Applied config for '{part_no}'.")

    def _save_current_as_job(self, part_no: str) -> None:
        self.ctx.set_current_part(part_no)
        jobs.save_part_config(part_no, self.settings_tab.collect_config())
        self.settings_tab.mark_job_saved()
        self.job_change_tab.refresh_job_buttons()

    def _auto_load_last_config(self) -> None:
        self._last_job_loaded = False
        part_no = jobs.read_last_part()
        if part_no is None:
            return
        config = jobs.load_part_config(part_no)
        if config is None:
            return
        model_ok = self.settings_tab.apply_config(config)
        self.ctx.set_current_part(part_no)
        self._last_job_loaded = True
        if model_ok:
            self.ctx.status.emit(f"Auto-loaded last config: '{part_no}'.")

    def auto_start(self) -> bool:
        """Called once the window is up: with the last job loaded and Start
        when the app opens on (Settings > Camera), start inspecting, as if the
        operator pressed Start. False when it did not start."""
        if not (self._last_job_loaded and self.ctx.settings["auto_start"]) or self.ctx.worker.isRunning():
            return False
        if not self.monitor_tab.start_stream():
            return False
        self.ctx.status.emit(f"Started job '{self.ctx.current_part_no}' automatically.")
        return True

    @Slot(str)
    def _on_job_selected(self, part_no: str) -> None:
        config = jobs.load_part_config(part_no)
        if config is None:
            self.ctx.status.emit(f"Config for '{part_no}' no longer exists.")
            self.job_change_tab.refresh_job_buttons()
            return
        self._apply_job(part_no, config)

    # --- Barcode scanning -----------------------------------------------

    @Slot(str)
    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        """The Tools and Settings tabs stream live: start the camera (as a
        setup preview, no counting or tower) when one opens, stop it again on
        leaving if that tab is what started it. Leaving Settings with unsaved
        changes asks first (Cancel stays on Settings)."""
        widget = self.tabs.widget(index)
        if self._prev_tab is self.settings_tab and widget is not self.settings_tab:
            if not self.confirm_unsaved():
                self.tabs.blockSignals(True)
                self.tabs.setCurrentWidget(self.settings_tab)
                self.tabs.blockSignals(False)
                return
        self._prev_tab = widget
        if widget is self.tools_tab or widget is self.settings_tab:
            frozen = widget is self.tools_tab and self.tools_tab.frozen
            if not frozen and not self.ctx.worker.isRunning():
                self._tools_started_camera = self.monitor_tab.ensure_live(
                    lambda: None, f"Starting camera for {self.tabs.tabText(index)}...")
        elif self._tools_started_camera:
            self._tools_started_camera = False
            self.monitor_tab.end_preview()

    def _show_tab(self, title: str) -> None:
        """`title` is the tab's English title (tab texts are translated)."""
        bar = self.tabs.tabBar()
        for i in range(self.tabs.count()):
            if translator().source(bar, f"tab{i}", self.tabs.tabText(i)) == title:
                self.tabs.setCurrentIndex(i)
                return

    @Slot()
    def _open_barcode_scan_dialog(self) -> None:
        if self._barcode_dialog is not None:
            return
        worker = self.ctx.worker
        if worker.isRunning():
            # The scanner needs the camera: stop monitoring, open the scanner
            # once the camera is free, resume when the scanner closes.
            self._resume_stream_after_scan = not self.monitor_tab.in_preview
            self.ctx.status.emit("Pausing monitoring for the barcode scanner...")

            def reopen() -> None:
                worker.finished.disconnect(reopen)
                self._open_barcode_scan_dialog()

            worker.finished.connect(reopen)
            self.monitor_tab.stop_stream()
            return
        index = self.ctx.camera_index
        if index is None:
            self.ctx.camera_scan_requested.emit()
            index = self.ctx.camera_index
        if index is None:
            self.ctx.status.emit("No camera found — check the USB cable.")
            self._resume_stream_after_scan = False
            return

        self._barcode_dialog = BarcodeScanDialog(int(index), parent=self)
        self._barcode_dialog.load_requested.connect(self._on_part_load_requested)
        self._barcode_dialog.save_requested.connect(self._on_part_save_requested)
        self._barcode_dialog.finished.connect(self._on_barcode_dialog_finished)
        self._barcode_dialog.show()

    def _on_barcode_dialog_finished(self) -> None:
        self._barcode_dialog = None
        if self._resume_stream_after_scan:
            self._resume_stream_after_scan = False
            self.monitor_tab.start_stream()

    @Slot(str)
    def _on_part_load_requested(self, part_no: str) -> None:
        if part_no == self.ctx.current_part_no:
            self.ctx.status.emit(f"'{part_no}' is already the current job.")
            return
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

    def _trim_ng_log(self) -> None:
        """At start, the NG log back under its limit (it may have been lowered,
        or images added by an older version without a limit)."""
        limit = int(self.ctx.settings["ng_log_max_gb"] * 1024 ** 3)
        if limit > 0 and ng_log.ng_log_size() > limit:
            deleted, freed = ng_log.prune_ng_log(int(limit * ng_log.PRUNE_TO))
            self.ctx.status.emit(f"NG log over its limit: deleted the {deleted} oldest NG images "
                                 f"({freed / 1024 ** 3:.1f} GB).")

    # --- Language -------------------------------------------------------

    def _make_language_switch(self) -> QWidget:
        """EN / TH flags in the tab bar's corner, on every tab."""
        box = QWidget()
        box.setObjectName("clearBox")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(4)
        self.language_group = QButtonGroup(self)
        self.language_buttons = {}
        for code, name in LANGUAGES:
            button = QToolButton()
            button.setObjectName("flagButton")
            button.setCheckable(True)
            button.setIcon(make_flag_icon(code))
            button.setIconSize(QSize(36, 24))
            button.setToolTip(name)
            button.setProperty("_i18n_skip", True)
            button.clicked.connect(lambda _=False, c=code: self.set_language(c))
            self.language_group.addButton(button)
            layout.addWidget(button)
            self.language_buttons[code] = button
        return box

    def set_language(self, code: str) -> None:
        """Switches the whole UI at once; remembered for the next start."""
        self.ctx.settings.set(language=code)
        translator().set_language(code)
        self.language_buttons[translator().language].setChecked(True)
        self._show_status(self._status_text)

    def _show_status(self, text: str) -> None:
        self._status_text = text
        self.status_label.setText(tr(text))

    # --- Window events --------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "image_overlay") and self.image_overlay.isVisible():
            self.image_overlay.setGeometry(self.rect())

    def closeEvent(self, event) -> None:
        if not self.confirm_unsaved():
            event.ignore()
            return
        self.stop_stream()
        self.ctx.relay.shutdown()  # outputs off, poll thread stopped
        # Must fully join: destroying a QThread that's still running aborts
        # the whole process ("QThread: Destroyed while thread is still running").
        self.ctx.worker.wait()
        if self._barcode_dialog is not None:
            self._barcode_dialog.close()
        super().closeEvent(event)


ALREADY_RUNNING_TEXT = (
    "UniversalV1 is already running.\n\n"
    "If it was started automatically, stop it first:\n"
    "systemctl --user stop universalv1"
)


def _show_already_running(app: QApplication) -> None:
    """Tells the user, then exits 0 (so systemd doesn't restart-loop). The
    box closes itself, so a copy started without anyone watching can't hang."""
    box = QMessageBox(QMessageBox.Warning, "UniversalV1", ALREADY_RUNNING_TEXT)
    QTimer.singleShot(10000, box.accept)
    box.exec()


def main() -> None:
    app = QApplication(sys.argv)
    # One copy only: two would fight over the camera, the tower light and the
    # relay (the boot service and a start from a terminal / VS Code).
    lock = QLockFile(str(Path(QStandardPaths.writableLocation(QStandardPaths.TempLocation)) / "universalv1-app.lock"))
    lock.setStaleLockTime(0)  # a lock left by a crashed copy is taken over (its process is gone)
    if not lock.tryLock(100):
        print(ALREADY_RUNNING_TEXT.replace("\n\n", " ").replace("\n", " "), file=sys.stderr)
        _show_already_running(app)
        sys.exit(0)
    app.setStyleSheet(DARK_INDUSTRIAL_QSS)
    hints = HintWrapper(app)
    app.installEventFilter(hints)
    window = MainWindow()
    window.resize(1050, 720)
    window.set_language(window.ctx.settings["language"])
    window.showFullScreen()
    QTimer.singleShot(1000, window.auto_start)  # after the window is up and the model warm
    sys.exit(app.exec())
