from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..camera import RESOLUTION_CHOICES, find_cameras, parse_resolution
from ..context import AppContext
from ..optional_deps import BUZZER_OFF, BUZZER_PATTERN_NAMES
from ..theme import make_icon
from ..widgets.layout import make_card, row, section_heading, tip_label


class SettingsTab(QWidget):
    """Model, camera, tower light/buzzer and detection filter settings."""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        settings = ctx.settings
        tower = ctx.tower

        self.model_label = QLabel("No model selected")
        self.load_model_button = QPushButton("Load .pt Model")
        self.load_model_button.setIcon(make_icon("folder", "#e0e0e0"))
        self.load_model_button.clicked.connect(self.load_model)
        ctx.model_pick_requested.connect(self.load_model)
        ctx.model_path_changed.connect(lambda path: self.model_label.setText(path.name))
        ctx.model_classes_changed.connect(self._populate_class_filters)

        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.scan_button = QPushButton("Scan Cameras")
        self.scan_button.setIcon(make_icon("refresh", "#e0e0e0"))
        self.scan_button.clicked.connect(self.scan_cameras)

        self.resolution_combo = QComboBox()
        for label, value in RESOLUTION_CHOICES:
            self.resolution_combo.addItem(label, value)
        index = self.resolution_combo.findData(settings["camera_resolution"])
        self.resolution_combo.setCurrentIndex(max(index, 0))
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.resolution_actual_label = QLabel("Actual: — (shown after Start)")
        self.resolution_actual_label.setObjectName("hint")
        ctx.worker.resolution_changed.connect(self._on_actual_resolution)
        ctx.worker.camera_index_changed.connect(self._on_camera_index_changed)
        ctx.camera_scan_requested.connect(self.scan_cameras)

        self.tower_status_label = QLabel("Tower Light: Not Connected")
        tower.connection_text_changed.connect(self.tower_status_label.setText)

        # Saved values are applied before signals are connected, so restoring
        # them at startup doesn't fire a preview beep or rewrite the file.
        self.buzzer_enabled_checkbox = QCheckBox("Enable Buzzer")
        self.buzzer_enabled_checkbox.setChecked(tower.buzzer_enabled)
        self.buzzer_enabled_checkbox.toggled.connect(tower.set_buzzer_enabled)
        tower.buzzer_enabled_changed.connect(self.buzzer_enabled_checkbox.setChecked)

        self.buzzer_pattern_combo = self._make_buzzer_style_combo(tower.buzzer_pattern)
        if self.buzzer_pattern_combo.currentData() != tower.buzzer_pattern:
            settings.set(buzzer_pattern=self.buzzer_pattern_combo.currentData())
        self.buzzer_pattern_combo.currentIndexChanged.connect(
            lambda _: tower.set_buzzer_pattern(self.buzzer_pattern_combo.currentData()))

        self.buzzer_test_button = QPushButton("Test")
        self.buzzer_test_button.setIcon(make_icon("speaker_on", "#e0e0e0"))
        self.buzzer_test_button.clicked.connect(tower.test_alarm)

        self.ok_beep_checkbox = QCheckBox("Beep on OK")
        self.ok_beep_checkbox.setChecked(settings["ok_beep_enabled"])
        self.ok_beep_checkbox.toggled.connect(tower.set_ok_beep_enabled)

        self.ok_beep_duration_spinbox = QSpinBox()
        self.ok_beep_duration_spinbox.setRange(20, 1000)
        self.ok_beep_duration_spinbox.setSingleStep(10)
        self.ok_beep_duration_spinbox.setSuffix(" ms")
        self.ok_beep_duration_spinbox.setValue(settings["ok_beep_ms"])
        if self.ok_beep_duration_spinbox.value() != settings["ok_beep_ms"]:
            settings.set(ok_beep_ms=self.ok_beep_duration_spinbox.value())
        self.ok_beep_duration_spinbox.valueChanged.connect(tower.set_ok_beep_ms)

        self.ok_beep_test_button = QPushButton("Test")
        self.ok_beep_test_button.setIcon(make_icon("speaker_on", "#e0e0e0"))
        self.ok_beep_test_button.clicked.connect(tower.test_ok_beep)

        self.state_hold_spinbox = QSpinBox()
        self.state_hold_spinbox.setRange(0, 5000)
        self.state_hold_spinbox.setSingleStep(100)
        self.state_hold_spinbox.setSuffix(" ms")
        self.state_hold_spinbox.setValue(settings["state_hold_ms"])
        if self.state_hold_spinbox.value() != settings["state_hold_ms"]:
            settings.set(state_hold_ms=self.state_hold_spinbox.value())
        self.state_hold_spinbox.valueChanged.connect(
            lambda value: settings.set(state_hold_ms=value))

        self.slow_tool_spinbox = QDoubleSpinBox()
        self.slow_tool_spinbox.setRange(0.5, 10.0)
        self.slow_tool_spinbox.setSingleStep(0.5)
        self.slow_tool_spinbox.setDecimals(1)
        self.slow_tool_spinbox.setSuffix(" Hz")
        self.slow_tool_spinbox.setValue(settings["slow_tool_hz"])
        self.slow_tool_spinbox.valueChanged.connect(self._on_slow_tool_hz_changed)

        self.autocapture_max_spinbox = self._make_setting_spinbox(
            "autocapture_max_images", 1, 100000, 50, " images")
        self.autocapture_threshold_spinbox = self._make_setting_spinbox(
            "autocapture_threshold", 0, 32, 1, " / 64 bits")
        self.autocapture_interval_spinbox = self._make_setting_spinbox(
            "autocapture_interval_ms", 0, 60000, 100, " ms")
        self.autocapture_wait_clear_checkbox = QCheckBox("Wait for part to leave before next capture")
        self.autocapture_wait_clear_checkbox.setChecked(settings["autocapture_wait_clear"])
        self.autocapture_wait_clear_checkbox.toggled.connect(
            lambda checked: settings.set(autocapture_wait_clear=checked))

        # Filter panel
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(0)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)

        self.confidence_label = QLabel("Confidence: 0%")

        self.max_detections_label = QLabel("Max Detections (0 = unlimited):")
        self.max_detections_spinbox = QSpinBox()
        self.max_detections_spinbox.setMinimum(0)
        self.max_detections_spinbox.setMaximum(100)
        self.max_detections_spinbox.setValue(0)
        self.max_detections_spinbox.valueChanged.connect(self.on_max_detections_changed)

        self.target_class_label = QLabel("Target Classes:")
        self.target_class_combo_1 = QComboBox()
        self.target_class_combo_1.addItem("-- None --", None)
        self.target_class_combo_1.currentIndexChanged.connect(self.on_target_class_1_changed)

        self.target_gate_combo = QComboBox()
        self.target_gate_combo.addItem("OR")
        self.target_gate_combo.addItem("AND")
        self.target_gate_combo.currentTextChanged.connect(self.on_target_gate_changed)

        self.target_class_combo_2 = QComboBox()
        self.target_class_combo_2.addItem("-- None --", None)
        self.target_class_combo_2.currentIndexChanged.connect(self.on_target_class_2_changed)

        # Scrollable class filters
        self.class_filters_scroll = QScrollArea()
        self.class_filters_scroll.setWidgetResizable(True)
        self.class_filters_container = QWidget()
        self.class_filters_layout = QVBoxLayout()
        self.class_filters_container.setLayout(self.class_filters_layout)
        self.class_filters_scroll.setWidget(self.class_filters_container)
        self.class_filters_scroll.setMinimumHeight(200)

        # Required marks: the classes a part of this job must show (NG if one is missing).
        self.required_container = QWidget()
        self.required_layout = QVBoxLayout(self.required_container)
        self.required_scroll = QScrollArea()
        self.required_scroll.setWidgetResizable(True)
        self.required_scroll.setWidget(self.required_container)
        self.required_scroll.setMinimumHeight(140)
        self.required_teach_button = QPushButton("Set from current part")
        self.required_teach_button.setIcon(make_icon("target"))
        self.required_teach_button.setToolTip(
            "With a good part on the Monitor, tick exactly the marks the AI finds on it now.")
        self.required_teach_button.clicked.connect(self.teach_required_marks)
        self.required_count_label = QLabel()
        self.required_count_label.setObjectName("hint")
        self.layout_check_checkbox = QCheckBox("Mark inspection: check layout")
        self.layout_check_checkbox.setToolTip(
            "Warn on the Monitor when marks are out of the usual AGC order: brand logo, AGC AUTOMOTIVE, "
            "E-mark, then M-code / TIS from top to bottom; 11T right-most, SNI on the left, short name "
            "left-most. A warning only: the OK / NG verdict does not change.")
        self.layout_check_checkbox.toggled.connect(self._toggle_layout_check)

        self._build_layout()

    def _build_layout(self) -> None:
        model_card, model_body = make_card("AI Model", badge="ai")
        model_body.addLayout(row(self.load_model_button, self.model_label))
        self.model_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        camera_card, camera_body = make_card("Camera", icon="camera")
        camera_body.addLayout(row(tip_label("Camera:", "Which camera to use. Tap Scan Cameras after plugging one in.", self.camera_combo), self.camera_combo, self.scan_button, None))
        camera_body.addLayout(row(tip_label("Resolution:", "Image size asked from the camera. Higher = finer detail but slower; Actual shows what the camera gave.", self.resolution_combo), self.resolution_combo, None))
        camera_body.addWidget(self.resolution_actual_label)

        tower_card, tower_body = make_card("Tower Light & Buzzer", icon="tower")
        tower_body.addWidget(self.tower_status_label)
        tower_body.addWidget(self.buzzer_enabled_checkbox)
        tower_body.addLayout(row(
            tip_label("NG alarm:", "How the buzzer sounds while the result is NG.", self.buzzer_pattern_combo), self.buzzer_pattern_combo, None, self.buzzer_test_button))
        tower_body.addLayout(row(
            self.ok_beep_checkbox, None, self.ok_beep_duration_spinbox, self.ok_beep_test_button))
        tower_body.addLayout(row(tip_label("State hold time:", "The shown result (and tower light) stays at least this long before it may change, so it does not flicker. Each change counts one part. 0 = off.", self.state_hold_spinbox), None, self.state_hold_spinbox))
        hold_hint = QLabel("Minimum time between STANDBY / OK / NG changes (0 = off)")
        hold_hint.setObjectName("hint")
        hold_hint.setWordWrap(True)
        tower_body.addWidget(hold_hint)
        tower_body.addLayout(row(tip_label("OCR / code rate:", "How many times a second OCR and code-reader tools run. They run in the background, so the live view stays smooth; higher = faster result, more CPU.", self.slow_tool_spinbox), None, self.slow_tool_spinbox))
        slow_hint = QLabel("OCR and code tools run this often in the background; the live view keeps its full frame rate")
        slow_hint.setObjectName("hint")
        slow_hint.setWordWrap(True)
        tower_body.addWidget(slow_hint)

        autocapture_card, autocapture_body = make_card("Auto Capture", icon="autocapture")
        autocapture_body.addLayout(row(tip_label("Max images per job:", "Auto Capture stops after saving this many images for the job.", self.autocapture_max_spinbox), None, self.autocapture_max_spinbox))
        autocapture_body.addLayout(row(tip_label("Match threshold:", "How different from a master a frame may be and still be saved. 0 = identical; 8-12 is typical. Higher saves more varied frames.", self.autocapture_threshold_spinbox), None, self.autocapture_threshold_spinbox))
        autocapture_body.addLayout(row(tip_label("Min interval:", "Shortest time between two saved images.", self.autocapture_interval_spinbox), None, self.autocapture_interval_spinbox))
        autocapture_body.addWidget(self.autocapture_wait_clear_checkbox)
        autocapture_hint = QLabel(
            "Saves a frame when it differs from a master by at most the threshold "
            "(0 = identical; 8-12 typical), at most once per interval. Waiting for the part "
            "to leave saves each part once; turn it off to shoot several angles of one part. "
            "Set masters on the Monitor tab.")
        autocapture_hint.setObjectName("hint")
        autocapture_hint.setWordWrap(True)
        autocapture_body.addWidget(autocapture_hint)

        filters_card, filters_body = make_card("AI Detection Filters", badge="ai")
        for widget, hint in (
            (self.max_detections_label, "Keep only this many detections (the most confident). 0 = all."),
            (self.max_detections_spinbox, "Keep only this many detections (the most confident). 0 = all."),
            (self.target_class_label, "OK needs these classes found: OR = either one, AND = both. -- None -- = not used."),
            (self.target_class_combo_1, "First class that must be found for OK."),
            (self.target_gate_combo, "OR: either class is enough.  AND: both must be found."),
            (self.target_class_combo_2, "Second class that must be found for OK."),
            (self.buzzer_enabled_checkbox, "Sound the buzzer on the tower light when the result is NG."),
            (self.ok_beep_checkbox, "Short beep each time a part becomes OK; set its length on the right."),
            (self.ok_beep_duration_spinbox, "Length of the OK beep."),
            (self.autocapture_wait_clear_checkbox, "Save each part once: wait until the part has left before the next save. "
                                                   "Turn off to shoot several angles of one part."),
            (self.load_model_button, "Open a trained YOLO model (.pt) for this job."),
            (self.scan_button, "Look again for connected cameras."),
            (self.buzzer_test_button, "Play the NG alarm once."),
            (self.ok_beep_test_button, "Play the OK beep once."),
        ):
            widget.setToolTip(hint)
        self.confidence_label.setStyleSheet("color: #ffcf73; font-weight: 700;")
        filters_body.addLayout(row(tip_label("Confidence threshold", "AI detections less sure than this are ignored. Raise it if wrong things are found; lower it if real ones are missed.", self.confidence_slider, self.confidence_label), None, self.confidence_label))
        filters_body.addWidget(self.confidence_slider)
        filters_body.addSpacing(6)
        self.max_detections_spinbox.setFixedWidth(120)
        filters_body.addLayout(row(self.max_detections_label, None, self.max_detections_spinbox))
        filters_body.addSpacing(6)
        filters_body.addWidget(self.target_class_label)
        filters_body.addLayout(row(
            self.target_class_combo_1, self.target_gate_combo, self.target_class_combo_2))
        filters_body.addSpacing(6)
        filters_body.addWidget(tip_label("Classes to show:", "Only the ticked classes are drawn and counted. None ticked = all classes."))
        filters_body.addWidget(self.class_filters_scroll, 1)

        required_card, required_body = make_card("Required Marks", badge="ai")
        required_hint = QLabel("NG when a ticked mark is not found on the part. None ticked = not used "
                               "(Target Classes decide).")
        required_hint.setObjectName("hint")
        required_hint.setWordWrap(True)
        required_body.addLayout(row(self.required_teach_button, None, self.required_count_label))
        required_body.addWidget(required_hint)
        required_body.addWidget(self.required_scroll, 1)
        required_body.addWidget(self.layout_check_checkbox)

        # Left: this station, the same for every job. Right: saved with the
        # job (the AI model and its filters), so it changes on Job Change.
        machine_heading, _ = section_heading("Machine", "Same for every job")
        left_column = QVBoxLayout()
        left_column.addWidget(machine_heading)
        left_column.addWidget(camera_card)
        left_column.addWidget(tower_card)
        left_column.addWidget(autocapture_card)
        left_column.addStretch(1)

        job_heading, self.job_note = section_heading("This Job")
        right_column = QVBoxLayout()
        right_column.addWidget(job_heading)
        right_column.addWidget(model_card)
        right_column.addWidget(filters_card, 1)
        right_column.addWidget(required_card, 1)

        columns = QHBoxLayout()
        columns.addLayout(left_column, 1)
        columns.addLayout(right_column, 1)

        content = QWidget()
        content.setLayout(columns)

        # Wrapped in a scroll area so the tab degrades gracefully if a panel
        # is smaller than the 1024x768 kiosk screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self._ctx.part_changed.connect(self._refresh_job_note)
        self._refresh_job_note()

    def _refresh_job_note(self, *_args) -> None:
        part = self._ctx.current_part_no
        self.job_note.setText(f"Saved with job {part}" if part else "Saved with the job (no job loaded)")

    def _make_setting_spinbox(self, key: str, low: int, high: int, step: int, suffix: str) -> QSpinBox:
        """Spinbox bound to an app setting, saved on every change."""
        settings = self._ctx.settings
        spinbox = QSpinBox()
        spinbox.setRange(low, high)
        spinbox.setSingleStep(step)
        spinbox.setSuffix(suffix)
        spinbox.setValue(settings[key])
        spinbox.valueChanged.connect(lambda value: settings.set(**{key: value}))
        return spinbox

    @staticmethod
    def _make_buzzer_style_combo(selected_pattern: int) -> QComboBox:
        combo = QComboBox()
        for name, value in BUZZER_PATTERN_NAMES:
            if value == BUZZER_OFF:
                continue
            combo.addItem(name, value)
        index = combo.findData(selected_pattern)
        if index >= 0:
            combo.setCurrentIndex(index)
        return combo

    # --- Camera and model -----------------------------------------------

    def scan_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = find_cameras()
        if not cameras:
            self.camera_combo.addItem("No camera", None)
            return
        for idx in cameras:
            self.camera_combo.addItem(f"Camera {idx}", idx)

    @Slot(float)
    def _on_slow_tool_hz_changed(self, hz: float) -> None:
        self._ctx.settings.set(slow_tool_hz=hz)
        self._ctx.worker.set_slow_hz(hz)

    @Slot(int)
    def _on_camera_changed(self, index: int) -> None:
        self._ctx.camera_index = self.camera_combo.currentData()

    @Slot(int)
    def _on_camera_index_changed(self, index: int) -> None:
        """The camera came back under another /dev/videoN after a replug."""
        combo_index = self.camera_combo.findData(index)
        if combo_index < 0:
            self.camera_combo.addItem(f"Camera {index}", index)
            combo_index = self.camera_combo.count() - 1
        self.camera_combo.setCurrentIndex(combo_index)

    @Slot(int)
    def _on_resolution_changed(self, index: int) -> None:
        value = self.resolution_combo.currentData()
        self._ctx.settings.set(camera_resolution=value)
        self._ctx.worker.set_resolution(parse_resolution(value))
        if self._ctx.worker.isRunning():
            self._ctx.status.emit("New resolution applies at the next Start.")

    @Slot(int, int)
    def _on_actual_resolution(self, width: int, height: int) -> None:
        wanted = parse_resolution(self.resolution_combo.currentData())
        if (width, height) == (0, 0):
            text, color = "Actual: no frame from camera", "#e57373"
        elif (width, height) != wanted:
            text, color = f"Actual: {width} × {height} (fallback; chosen size not available)", "#ffcf73"
        else:
            text, color = f"Actual: {width} × {height}", "#9aa5b1"
        self.resolution_actual_label.setText(text)
        self.resolution_actual_label.setStyleSheet(f"color: {color};")

    @Slot()
    def load_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO .pt Model",
            "",
            "PyTorch Model (*.pt)",
        )
        if not file_path:
            return
        self._ctx.load_model(Path(file_path))

    # --- Detection filters ----------------------------------------------

    @Slot(list)
    def _populate_class_filters(self, model_classes: list) -> None:
        """Create checkboxes for all model classes."""
        filters = self._ctx.filters
        # Clear existing checkboxes
        while self.class_filters_layout.count() > 0:
            item = self.class_filters_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        filters.selected_classes.clear()

        # Create checkbox for each class
        for class_name in sorted(model_classes):
            checkbox = QCheckBox(class_name)
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, cn=class_name: self._toggle_class_filter(cn, checked)
            )
            self.class_filters_layout.addWidget(checkbox)
            filters.selected_classes.add(class_name)

        # Repopulate target-class combos from the same source list
        for combo in (self.target_class_combo_1, self.target_class_combo_2):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("-- None --", None)
            for class_name in sorted(model_classes):
                combo.addItem(class_name, class_name)
            combo.blockSignals(False)
        filters.target_class_1 = None
        filters.target_class_2 = None

        # Required marks: same classes, none ticked until taught or chosen.
        while self.required_layout.count() > 0:
            item = self.required_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        filters.required_classes.clear()
        for class_name in sorted(model_classes):
            checkbox = QCheckBox(class_name)
            checkbox.toggled.connect(lambda checked, cn=class_name: self._toggle_required(cn, checked))
            self.required_layout.addWidget(checkbox)
        self.required_layout.addStretch(1)
        self._refresh_required_count()

    def _required_checkboxes(self) -> list:
        return [w for w in (self.required_layout.itemAt(i).widget() for i in range(self.required_layout.count()))
                if isinstance(w, QCheckBox)]

    def _toggle_required(self, class_name: str, checked: bool) -> None:
        if checked:
            self._ctx.filters.required_classes.add(class_name)
        else:
            self._ctx.filters.required_classes.discard(class_name)
        self._refresh_required_count()

    def _toggle_layout_check(self, checked: bool) -> None:
        self._ctx.filters.check_layout = checked

    def _refresh_required_count(self) -> None:
        n = len(self._ctx.filters.required_classes)
        self.required_count_label.setText(f"{n} required" if n else "not used")

    def set_required_marks(self, classes) -> None:
        """Ticks exactly `classes` (unknown names are ignored)."""
        wanted = set(classes)
        for checkbox in self._required_checkboxes():
            checkbox.setChecked(checkbox.text() in wanted)

    def teach_required_marks(self) -> None:
        """Required marks = the classes the AI finds on the part now."""
        detections = self._ctx.last_detections
        found = sorted({d.get("class_name") for d in detections or []} & set(self._ctx.model_classes))
        if not found:
            QMessageBox.information(
                self, "Set Required Marks",
                "No AI marks found yet. Start the camera on the Monitor tab with a good part in view, "
                "then tap this again.")
            return
        answer = QMessageBox.question(
            self, "Set Required Marks",
            f"Require these {len(found)} marks for this job?\n\n{', '.join(found)}\n\n"
            "A part missing any of them will be NG.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self.set_required_marks(found)
            self._ctx.status.emit(f"Required marks set: {', '.join(found)}.")

    def _toggle_class_filter(self, class_name: str, checked: bool) -> None:
        if checked:
            self._ctx.filters.selected_classes.add(class_name)
        else:
            self._ctx.filters.selected_classes.discard(class_name)

    @Slot(int)
    def on_confidence_changed(self, value: int) -> None:
        self._ctx.filters.confidence_threshold = value / 100.0
        self.confidence_label.setText(f"Confidence: {value}%")

    @Slot(int)
    def on_max_detections_changed(self, value: int) -> None:
        self._ctx.filters.max_detections = value

    @Slot(int)
    def on_target_class_1_changed(self, index: int) -> None:
        self._ctx.filters.target_class_1 = self.target_class_combo_1.currentData()

    @Slot(int)
    def on_target_class_2_changed(self, index: int) -> None:
        self._ctx.filters.target_class_2 = self.target_class_combo_2.currentData()

    @Slot(str)
    def on_target_gate_changed(self, text: str) -> None:
        self._ctx.filters.target_gate = text

    # --- Saved job configs ----------------------------------------------

    def collect_config(self) -> dict:
        ctx = self._ctx
        filters = ctx.filters
        return {
            "part_no": ctx.current_part_no,
            "model_path": str(ctx.model_path) if ctx.model_path else None,
            "confidence_threshold": filters.confidence_threshold,
            "max_detections": filters.max_detections,
            "target_class_1": filters.target_class_1,
            "target_class_2": filters.target_class_2,
            "target_gate": filters.target_gate,
            "selected_classes": sorted(filters.selected_classes),
            "required_classes": sorted(filters.required_classes),
            "check_layout": filters.check_layout,
            "perspective": filters.perspective,
            "roi": list(filters.roi) if filters.roi else None,
            "autocapture_masters": ctx.master_entries(),
            "tools": ctx.tool_entries(),
            "tools_gate": ctx.tools_gate,
            "mm_per_px": ctx.mm_per_px,
        }

    def apply_config(self, config: dict) -> bool:
        """Applies a saved config to the UI. Returns False if the config's model
        file was missing on disk (filters still get applied to whatever model is
        currently loaded), True otherwise."""
        model_ok = True
        model_path_str = config.get("model_path")
        if model_path_str:
            model_path = Path(model_path_str)
            current = self._ctx.model_path
            if model_path.exists() and current != model_path:
                self._ctx.load_model(model_path)
            elif not model_path.exists():
                model_ok = False
                self._ctx.status.emit(
                    f"Config model not found on disk: {model_path.name} — filters applied to current model."
                )

        # Older jobs have no "perspective" / "roi" keys: camera view, whole frame.
        # Top view first: the ROI's fractions refer to it.
        self._ctx.set_perspective(config.get("perspective"))
        self._ctx.set_roi(config.get("roi"))
        self._ctx.set_masters(config.get("autocapture_masters"))
        self._ctx.load_tools(config)  # older jobs have no "tools": none
        self.confidence_slider.setValue(int(round(config.get("confidence_threshold", 0.0) * 100)))
        self.max_detections_spinbox.setValue(int(config.get("max_detections", 0)))

        idx1 = self.target_class_combo_1.findData(config.get("target_class_1"))
        self.target_class_combo_1.setCurrentIndex(idx1 if idx1 >= 0 else 0)
        idx2 = self.target_class_combo_2.findData(config.get("target_class_2"))
        self.target_class_combo_2.setCurrentIndex(idx2 if idx2 >= 0 else 0)
        self.target_gate_combo.setCurrentText(config.get("target_gate", "OR"))

        selected = set(config.get("selected_classes", []))
        for i in range(self.class_filters_layout.count()):
            widget = self.class_filters_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(widget.text() in selected)
        self.set_required_marks(config.get("required_classes", []))  # older jobs: none
        self.layout_check_checkbox.setChecked(bool(config.get("check_layout", False)))
        self._ctx.filters.check_layout = self.layout_check_checkbox.isChecked()

        return model_ok
