from pathlib import Path
from typing import Optional

from PySide6.QtCore import QRectF, QRegularExpression, Qt, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from .. import jobs
from .. import paths
from ..camera import RESOLUTION_CHOICES, find_cameras, parse_resolution
from ..context import AppContext
from ..i18n import tr, translator
from ..marking.short_name import valid_short_name
from ..optional_deps import BUZZER_OFF, BUZZER_PATTERN_NAMES
from ..roi import roi_to_pixels
from ..theme import AI_COLOR, NG_RED, OK_GREEN, RUN_BUTTON_STYLES, STANDBY_YELLOW, action_button_qss, badge_icon, make_icon
from ..widgets.layout import bottom_bar, make_card, row, tip_label, top_bar
from ..widgets.tool_overlay import draw_label, overlay_pen
from ..widgets.video_view import VideoView
from .settings.image_card import make_image_card
from .settings.relay_card import make_relay_card

DEFAULT_MARK_CONF = 40  # % for a newly ticked mark (the standard-mark model is checked at 0.4)
JUDGE_MODES = (("detection", "AI Detection"), ("required", "Required Marks"))
# Nav rail: (group, [(key, title, icon)]). Machine = this station, the same for
# every job (app settings); This Job = saved with the current job.
SECTIONS = (
    ("Machine", (("camera", "Camera", "camera"), ("tower", "Tower & Buzzer", "tower"),
                 ("relay", "Relay I/O", "relay"), ("autocapture", "Auto Capture", "autocapture"))),
    ("This Job", (("model", "AI Model", "ai"), ("marks", "Required Marks", "mark"))),
)
MACHINE_SECTIONS = {k for k, _, _ in SECTIONS[0][1]}
NAV_WIDTH = 186
FORM_WIDTH = 436
# The job config keys this tab owns; Save writes these into the job file.
JOB_KEYS = ("model_path", "confidence_threshold", "max_detections", "target_class_1", "target_class_2",
            "target_gate", "selected_classes", "judge_mode", "required_classes", "mark_conf", "check_layout",
            "short_name_expected")


class SettingsTab(QWidget):
    """Model, camera, tower light/buzzer and detection filter settings."""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        settings = ctx.settings
        tower = ctx.tower
        # Every change works at once; Save writes it (machine settings to
        # app_settings.json, this job's to its config), Revert goes back.
        self._applying = False
        self._job_baseline: dict = {}  # JOB_KEYS as last saved / loaded
        self._machine_reloaders: list = []  # sections' widget reloads after Revert
        ctx.model_path_changed.connect(lambda _: self._job_edited())

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
        tower.connection_text_changed.connect(lambda text: self.tower_status_label.setText(tr(text)))

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

        # What decides OK / NG for the AI: Target Classes or the Required Marks list.
        self.judge_mode_group = QButtonGroup(self)
        self.judge_mode_buttons = {}
        for mode, text in JUDGE_MODES:
            button = QPushButton(text)
            button.setIcon(badge_icon("mark" if mode == "required" else "ai"))
            button.setCheckable(True)
            button.setMinimumHeight(44)
            self.judge_mode_group.addButton(button)
            self.judge_mode_buttons[mode] = button
        self.judge_mode_buttons["detection"].setChecked(True)
        self.judge_mode_buttons["detection"].setToolTip(
            "OK / NG from the Target Classes below (OR / AND).")
        self.judge_mode_buttons["required"].setToolTip(
            "OK only when every ticked mark is found at or above its own confidence; a missing mark is NG.")
        self.judge_mode_group.buttonToggled.connect(self._on_judge_mode_toggled)

        # Required marks: the classes a part of this job must show (NG if one is missing),
        # each with its own minimum confidence.
        self.required_table = QTableWidget(0, 2)
        self.required_table.setHorizontalHeaderLabels(["Mark", "Min conf"])
        self.required_table.verticalHeader().setVisible(False)
        self.required_table.verticalHeader().setDefaultSectionSize(44)
        self.required_table.setSelectionMode(QTableWidget.NoSelection)
        self.required_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.required_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.required_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.required_table.setColumnWidth(1, 150)
        self.required_table.setMinimumHeight(260)
        self._mark_conf_spins = {}
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
        # The short name (58, U8, T1...) must read as this; empty = only check it is there.
        self.short_name_edit = QLineEdit()
        self.short_name_edit.setMaxLength(2)
        self.short_name_edit.setFixedWidth(110)
        self.short_name_edit.setPlaceholderText("e.g. T1 or 58")
        self.short_name_edit.setValidator(QRegularExpressionValidator(QRegularExpression("[A-Za-z0-9][0-9]?")))
        self.short_name_edit.setToolTip(
            "What the short name must read (OCR): an uppercase letter and a digit, or two digits. "
            "A different or unreadable short name is NG. Empty = only check that it is there.")
        self.short_name_edit.textEdited.connect(self._on_short_name_edited)
        self.short_name_edit.setEnabled(False)
        self.short_name_status = QLabel()
        self.short_name_status.setObjectName("hint")

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
        filters_body.addLayout(row(
            tip_label("Judge by:", "What decides OK / NG for the AI: Target Classes, or the Required Marks list "
                      "(missing mark = NG, each mark with its own confidence)."),
            self.judge_mode_buttons["detection"], self.judge_mode_buttons["required"]))
        filters_body.addSpacing(6)
        filters_body.addLayout(row(tip_label("Confidence threshold", "AI detections less sure than this are ignored. Raise it if wrong things are found; lower it if real ones are missed.", self.confidence_slider, self.confidence_label), None, self.confidence_label))
        filters_body.addWidget(self.confidence_slider)
        filters_body.addSpacing(6)
        self.max_detections_spinbox.setFixedWidth(120)
        filters_body.addLayout(row(self.max_detections_label, None, self.max_detections_spinbox))
        filters_body.addSpacing(6)
        self.target_box = QWidget()
        self.target_box.setObjectName("clearBox")
        target_layout = QVBoxLayout(self.target_box)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.target_class_label)
        target_layout.addLayout(row(
            self.target_class_combo_1, self.target_gate_combo, self.target_class_combo_2))
        filters_body.addWidget(self.target_box)
        filters_body.addSpacing(6)
        filters_body.addWidget(tip_label("Classes to show:", "Only the ticked classes are drawn and counted. None ticked = all classes."))
        filters_body.addWidget(self.class_filters_scroll, 1)

        self.required_card, required_body = make_card("Required Marks", badge="mark")
        required_hint = QLabel("OK when every ticked mark is found at or above its Min conf; a missing "
                               "or weaker mark is NG. None ticked = Target Classes decide.")
        required_hint.setObjectName("hint")
        required_hint.setWordWrap(True)
        required_body.addLayout(row(self.required_teach_button, None, self.required_count_label))
        required_body.addWidget(required_hint)
        required_body.addWidget(self.required_table, 1)
        required_body.addLayout(row(tip_label("Short name (OCR):", self.short_name_edit.toolTip(), self.short_name_edit),
                                    self.short_name_edit, self.short_name_status, None))
        required_body.addWidget(self.layout_check_checkbox)

        # Three bands like the other tabs: what is being set up on top, a nav
        # rail + the section's form + the live camera in the middle (every
        # change shows on the image at once), Revert / Save at the bottom.
        top, top_layout = top_bar()
        self.section_title = QLabel()
        self.section_title.setObjectName("currentPart")
        self.job_note = QLabel()
        self.job_note.setObjectName("monitorInfo")
        self.job_note.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top_layout.addWidget(self.section_title)
        top_layout.addSpacing(16)
        top_layout.addWidget(self.job_note, 1)

        self.pages = QStackedWidget()
        self.nav_group = QButtonGroup(self)
        self.nav_buttons = {}
        nav = QFrame()
        nav.setObjectName("settingsNav")
        nav.setFixedWidth(NAV_WIDTH)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(6, 6, 6, 6)
        nav_layout.setSpacing(6)
        for card in (filters_card, self.required_card):
            card.setProperty("fillPage", True)
        self.section_cards = {
            "camera": [camera_card],
            "tower": [tower_card],
            "autocapture": [autocapture_card],
            "model": [model_card, filters_card],
            "marks": [self.required_card],
        }
        self.marks_off_hint = QLabel("Judge by is set to AI Detection. Choose Required Marks on the "
                                     "AI Model page to use this list.")
        self.marks_off_hint.setObjectName("hint")
        self.marks_off_hint.setWordWrap(True)
        self.section_cards["marks"].insert(0, self.marks_off_hint)
        for group, sections in SECTIONS:
            heading = QLabel(group)
            heading.setObjectName("navGroup")
            nav_layout.addWidget(heading)
            for key, title, icon in sections:
                button = QPushButton(title.replace("&", "&&"))  # a lone & would be a shortcut mark
                button.setObjectName("navTile")
                button.setCheckable(True)
                button.setIcon(badge_icon(icon) if icon in ("ai", "mark") else make_icon(icon, "#e0e0e0"))
                button.setMinimumHeight(48)
                button.clicked.connect(lambda _=False, k=key: self.show_section(k))
                self.nav_group.addButton(button)
                nav_layout.addWidget(button)
                self.nav_buttons[key] = button
                self.pages.addWidget(self._make_page(self.section_cards.get(key, [])))
        nav_layout.addStretch(1)

        self.preview = VideoView()
        self.preview.setMinimumSize(320, 240)
        self.preview_info = QLabel("Live camera: opens with this tab")
        self.preview_info.setObjectName("hint")
        self.preview_info.setWordWrap(True)
        preview_frame = QFrame()
        preview_frame.setObjectName("videoFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(2, 2, 2, 2)
        preview_layout.addWidget(self.preview, 1)
        preview_column = QVBoxLayout()
        preview_column.addWidget(preview_frame, 1)
        preview_column.addWidget(self.preview_info)
        self.pages.setFixedWidth(FORM_WIDTH)

        content = QHBoxLayout()
        content.addWidget(nav)
        content.addWidget(self.pages)
        content.addLayout(preview_column, 1)

        bottom, bottom_layout = bottom_bar()
        self.unsaved_label = QLabel()
        self.unsaved_label.setObjectName("unsavedChip")
        self.revert_button = QPushButton("Revert")
        self.revert_button.setIcon(make_icon("refresh", "#e0e0e0"))
        self.revert_button.setMinimumHeight(44)
        self.revert_button.clicked.connect(self.revert)
        self.save_button = QPushButton("Save")
        self.save_button.setIcon(make_icon("save", "#ffffff"))
        self.save_button.setMinimumSize(170, 44)
        self.save_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.save_button.clicked.connect(self.save)
        hint = QLabel("Changes work at once on the live image; Save keeps them")
        hint.setObjectName("hint")
        bottom_layout.addWidget(hint, 1)
        bottom_layout.addWidget(self.unsaved_label)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.revert_button)
        bottom_layout.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(content, 1)
        layout.addWidget(bottom)

        self._show_judge_mode()
        self._ctx.part_changed.connect(self._on_part_changed)
        self._ctx.worker.frame_ready.connect(self._on_preview_frame)
        self._ctx.settings.changed.connect(lambda _: self._refresh_dirty())
        self.image_card, reload_image = make_image_card(self._ctx)
        self.add_section_widget("camera", self.image_card)
        self._machine_reloaders.append(reload_image)
        self.relay_card, reload_relay = make_relay_card(self._ctx, self)
        self.add_section_widget("relay", self.relay_card)
        self._machine_reloaders.append(reload_relay)
        self.show_section("camera")
        self._refresh_job_note()
        translator().language_changed.connect(self._on_language_changed)

    @staticmethod
    def _make_page(widgets: list) -> QScrollArea:
        body = QWidget()
        body.setObjectName("clearBox")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 6, 0)
        # A card marked fillPage takes the page's spare height (its list grows
        # to the bottom); otherwise the cards sit at the top.
        fill = any(w.property("fillPage") for w in widgets)
        for widget in widgets:
            body_layout.addWidget(widget, 1 if widget.property("fillPage") else 0)
        if not fill:
            body_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        scroll.viewport().setObjectName("clearBox")
        return scroll

    def add_section_widget(self, key: str, widget: QWidget) -> None:
        """Adds a card to a section page (later features: image adjust, relay I/O)."""
        index = list(self.nav_buttons).index(key)
        body = self.pages.widget(index).widget()
        body.layout().insertWidget(body.layout().count() - 1, widget)

    def show_section(self, key: str) -> None:
        self.current_section = key
        self.nav_buttons[key].setChecked(True)
        self.pages.setCurrentIndex(list(self.nav_buttons).index(key))
        title = next(t for _, sections in SECTIONS for k, t, _ in sections if k == key)
        self.section_title.setText(tr(title))
        self._refresh_job_note()

    def _on_language_changed(self, _language: str) -> None:
        self.show_section(self.current_section)
        self._refresh_dirty()
        self._refresh_required_count()
        self._sync_short_name()
        self.on_confidence_changed(self.confidence_slider.value())
        if self._ctx.tower._connection_text:
            self.tower_status_label.setText(tr(self._ctx.tower._connection_text))

    def _on_part_changed(self, *_args) -> None:
        self._refresh_job_note()
        self._refresh_dirty()

    def _refresh_job_note(self, *_args) -> None:
        if getattr(self, "current_section", "camera") in MACHINE_SECTIONS:
            self.job_note.setText(tr("Machine: the same for every job"))
            return
        part = self._ctx.current_part_no
        self.job_note.setText(tr("Saved with job {}").format(part) if part else tr("No job loaded: not saved to a job"))

    # --- Live preview ---------------------------------------------------

    def _on_preview_frame(self, image: QImage, detections: list, tool_results: list) -> None:
        if not self.isVisible():
            return
        filters = self._ctx.filters
        section = getattr(self, "current_section", "camera")
        shown = filters.apply(detections) if section in ("model", "marks") else []
        marks = {m["class_name"]: m for m in filters.mark_results(detections)} if section == "marks" else {}
        self.preview_info.setText(tr(f"Live camera  ·  {image.width()} × {image.height()}"
                                     + (f"  ·  {len(shown)} detections" if section in ("model", "marks") else "")))

        def overlay(painter: QPainter) -> None:
            if section in ("model", "marks") and filters.roi:
                x1, y1, x2, y2 = roi_to_pixels(filters.roi, image.width(), image.height())
                painter.setPen(overlay_pen(painter, "#ffd54f", 2, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            for det in shown:
                name = det.get("class_name", "")
                label = f"{name} {det.get('conf', 0):.2f}"
                if section == "marks":
                    if name not in marks:
                        continue
                    ok = marks[name]["ok"]
                    color = QColor(STANDBY_YELLOW if ok is None else OK_GREEN if ok else NG_RED)
                    if "read" in marks[name]:  # the short name as read by OCR
                        label += f"  →  {marks[name]['read'] or '?'}"
                else:
                    color = QColor(AI_COLOR).lighter(130)
                painter.setPen(overlay_pen(painter, color))
                painter.setBrush(Qt.NoBrush)
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                draw_label(painter, x1, y1, label, color)

        self.preview.set_frame(QPixmap.fromImage(image), overlay)

    # --- Save / Revert --------------------------------------------------

    def job_settings(self) -> dict:
        """The part of the job's config this tab owns (JOB_KEYS); tools, ROI
        and masters are saved by their own tabs."""
        ctx, filters = self._ctx, self._ctx.filters
        return {
            "model_path": str(ctx.model_path) if ctx.model_path else None,
            "confidence_threshold": filters.confidence_threshold,
            "max_detections": filters.max_detections,
            "target_class_1": filters.target_class_1,
            "target_class_2": filters.target_class_2,
            "target_gate": filters.target_gate,
            "selected_classes": sorted(filters.selected_classes),
            "judge_mode": filters.judge_mode,
            "required_classes": sorted(filters.required_classes),
            "mark_conf": {c: filters.mark_conf[c] for c in sorted(filters.required_classes) if c in filters.mark_conf},
            "check_layout": filters.check_layout,
            "short_name_expected": filters.short_name_expected,
        }

    def unsaved_changes(self) -> list:
        """Names of the settings that differ from what is saved."""
        changed = list(self._ctx.settings.unsaved_keys())
        if self._ctx.current_part_no:
            current = self.job_settings()
            changed += [k for k in JOB_KEYS if current[k] != self._job_baseline.get(k)]
        return changed

    def _refresh_dirty(self) -> None:
        if not hasattr(self, "save_button"):
            return
        n = len(self.unsaved_changes())
        self.unsaved_label.setText(tr(f"●  Unsaved changes ({n})" if n else "✓  All saved"))
        self.unsaved_label.setProperty("dirty", bool(n))
        self.unsaved_label.style().unpolish(self.unsaved_label)
        self.unsaved_label.style().polish(self.unsaved_label)
        self.save_button.setEnabled(bool(n))
        self.revert_button.setEnabled(bool(n))

    def save(self) -> None:
        """Writes the machine settings and this job's settings to disk."""
        self._ctx.settings.commit()
        part_no = self._ctx.current_part_no
        if part_no:
            config = jobs.load_part_config(part_no)
            if config is None:
                config = self.collect_config()
            config.update(self.job_settings())
            jobs.save_part_config(part_no, config)
            self._job_baseline = self.job_settings()
        self._ctx.status.emit(f"Settings saved{f' (job {part_no})' if part_no else ''}.")
        self._refresh_dirty()

    def mark_job_saved(self) -> None:
        """The job file was just written with the current settings (new job)."""
        self._job_baseline = self.job_settings()
        self._refresh_dirty()

    def revert(self) -> None:
        """Back to the saved machine settings and the job as saved."""
        self._ctx.settings.revert()
        self._load_machine_widgets()
        if self._ctx.current_part_no and self._job_baseline:
            self.apply_config({**self.job_settings(), **self._job_baseline})
        self._ctx.status.emit("Changes reverted.")
        self._refresh_dirty()

    def _load_machine_widgets(self) -> None:
        """Machine widgets back to the values in settings (after Revert)."""
        settings = self._ctx.settings
        self.resolution_combo.setCurrentIndex(max(self.resolution_combo.findData(settings["camera_resolution"]), 0))
        self.buzzer_enabled_checkbox.setChecked(settings["buzzer_enabled"])
        index = self.buzzer_pattern_combo.findData(settings["buzzer_pattern"])
        if index >= 0:
            self.buzzer_pattern_combo.setCurrentIndex(index)
        self.ok_beep_checkbox.setChecked(settings["ok_beep_enabled"])
        self.ok_beep_duration_spinbox.setValue(settings["ok_beep_ms"])
        self.state_hold_spinbox.setValue(settings["state_hold_ms"])
        self.slow_tool_spinbox.setValue(settings["slow_tool_hz"])
        self.autocapture_max_spinbox.setValue(settings["autocapture_max_images"])
        self.autocapture_threshold_spinbox.setValue(settings["autocapture_threshold"])
        self.autocapture_interval_spinbox.setValue(settings["autocapture_interval_ms"])
        self.autocapture_wait_clear_checkbox.setChecked(settings["autocapture_wait_clear"])
        for reload in self._machine_reloaders:
            reload()

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
        self.resolution_actual_label.setText(tr(text))
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
        filters.required_classes.clear()
        filters.mark_conf.clear()
        self.required_table.setRowCount(0)
        self._mark_conf_spins = {}
        for row_index, class_name in enumerate(sorted(model_classes)):
            checkbox = QCheckBox(class_name)
            spin = QSpinBox()
            spin.setRange(1, 100)
            spin.setSingleStep(5)
            spin.setSuffix(" %")
            spin.setValue(DEFAULT_MARK_CONF)
            spin.setEnabled(False)
            spin.setToolTip(f"{class_name} counts as found only at or above this confidence.")
            filters.mark_conf[class_name] = DEFAULT_MARK_CONF / 100.0
            checkbox.toggled.connect(lambda checked, cn=class_name: self._toggle_required(cn, checked))
            spin.valueChanged.connect(lambda value, cn=class_name: self._on_mark_conf_changed(cn, value))
            self.required_table.insertRow(row_index)
            self.required_table.setCellWidget(row_index, 0, checkbox)
            self.required_table.setCellWidget(row_index, 1, spin)
            self._mark_conf_spins[class_name] = spin
        self._refresh_required_count()
        self._sync_short_name()

    def _job_edited(self) -> None:
        if not self._applying:
            self._refresh_dirty()

    def _load_mark_model(self) -> bool:
        """Required Marks judge with the standard-mark model; loads it unless it is
        already loaded. False when the file is missing."""
        path = paths.MARK_MODEL_PATH
        if not path.exists():
            self._ctx.status.emit(f"Mark model not found: {path.name}")
            return False
        if self._ctx.model_path != path:
            self._ctx.load_model(path)
        return True

    def _required_checkboxes(self) -> list:
        return [self.required_table.cellWidget(r, 0) for r in range(self.required_table.rowCount())]

    def _toggle_required(self, class_name: str, checked: bool) -> None:
        if checked:
            self._ctx.filters.required_classes.add(class_name)
        else:
            self._ctx.filters.required_classes.discard(class_name)
        self._mark_conf_spins[class_name].setEnabled(checked)
        self._refresh_required_count()
        self._sync_short_name()
        self._job_edited()

    def _on_short_name_edited(self, text: str) -> None:
        upper = text.upper()
        if upper != text:
            self.short_name_edit.setText(upper)
        self._ctx.filters.short_name_expected = upper if valid_short_name(upper) else ""
        self._sync_short_name()
        self._job_edited()

    def set_short_name(self, text: str) -> None:
        self.short_name_edit.setText(text or "")
        self._on_short_name_edited(self.short_name_edit.text())

    def _sync_short_name(self) -> None:
        """The worker reads the short name only while the job checks it."""
        filters = self._ctx.filters
        self.short_name_edit.setEnabled("short_name" in filters.required_classes)
        text = self.short_name_edit.text()
        if not text:
            status = tr("presence only")
        elif not valid_short_name(text):
            status = tr("letter + digit, or 2 digits")
        else:
            status = tr("read by OCR")
        self.short_name_status.setText(status)
        self._ctx.worker.set_short_name_expected(filters.short_name_expected if filters.checks_short_name else "")

    def _on_mark_conf_changed(self, class_name: str, value: int) -> None:
        self._ctx.filters.mark_conf[class_name] = value / 100.0
        self._job_edited()

    def set_mark_conf(self, conf: dict) -> None:
        """Min conf per mark, 0-1 (marks not in `conf` keep theirs)."""
        for class_name, value in conf.items():
            spin = self._mark_conf_spins.get(class_name)
            if spin is not None:
                spin.setValue(int(round(value * 100)))

    def set_judge_mode(self, mode: str) -> None:
        self.judge_mode_buttons.get(mode, self.judge_mode_buttons["detection"]).setChecked(True)

    def _on_judge_mode_toggled(self, button, checked: bool) -> None:
        if checked:
            self._ctx.filters.judge_mode = next(m for m, b in self.judge_mode_buttons.items() if b is button)
            if self._ctx.filters.by_marks and not self._applying:
                self._load_mark_model()
            self._sync_short_name()
            self._show_judge_mode()
            self._job_edited()

    def _show_judge_mode(self) -> None:
        """Only the controls that decide OK / NG in this mode are shown."""
        by_marks = self._ctx.filters.by_marks
        self.target_box.setVisible(not by_marks)
        self.required_card.setVisible(by_marks)
        if hasattr(self, "marks_off_hint"):
            self.marks_off_hint.setVisible(not by_marks)

    def _toggle_layout_check(self, checked: bool) -> None:
        self._ctx.filters.check_layout = checked
        self._job_edited()

    def _refresh_required_count(self) -> None:
        n = len(self._ctx.filters.required_classes)
        self.required_count_label.setText(tr(f"{n} required" if n else "not used"))

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
        self._job_edited()

    @Slot(int)
    def on_confidence_changed(self, value: int) -> None:
        self._ctx.filters.confidence_threshold = value / 100.0
        self.confidence_label.setText(tr(f"Confidence: {value}%"))
        self._job_edited()

    @Slot(int)
    def on_max_detections_changed(self, value: int) -> None:
        self._ctx.filters.max_detections = value
        self._job_edited()

    @Slot(int)
    def on_target_class_1_changed(self, index: int) -> None:
        self._ctx.filters.target_class_1 = self.target_class_combo_1.currentData()
        self._job_edited()

    @Slot(int)
    def on_target_class_2_changed(self, index: int) -> None:
        self._ctx.filters.target_class_2 = self.target_class_combo_2.currentData()
        self._job_edited()

    @Slot(str)
    def on_target_gate_changed(self, text: str) -> None:
        self._ctx.filters.target_gate = text
        self._job_edited()

    # --- Saved job configs ----------------------------------------------

    def collect_config(self) -> dict:
        ctx = self._ctx
        filters = ctx.filters
        return {
            "part_no": ctx.current_part_no,
            **self.job_settings(),
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
        self._applying = True
        try:
            return self._apply_config(config)
        finally:
            self._applying = False
            self._job_baseline = self.job_settings()  # as loaded = as saved
            self._refresh_dirty()

    def _apply_config(self, config: dict) -> bool:
        model_ok = True
        required = config.get("required_classes", [])
        # Older jobs: Required Marks decided whenever any were ticked.
        judge_mode = config.get("judge_mode") or ("required" if required else "detection")
        model_path_str = config.get("model_path")
        saved_model = model_path_str
        if judge_mode == "required" and paths.MARK_MODEL_PATH.exists():
            model_path_str = str(paths.MARK_MODEL_PATH)  # always the current mark model
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
        if not selected or model_path_str != saved_model:
            selected = set(self._ctx.model_classes)  # nothing saved for this model: show all
        for i in range(self.class_filters_layout.count()):
            widget = self.class_filters_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(widget.text() in selected)
        self.set_required_marks(required)  # older jobs: none
        # Older jobs: no own confidence per mark (the slider's).
        threshold = config.get("confidence_threshold", 0.0)
        self.set_mark_conf({c: threshold for c in required})
        self.set_mark_conf(config.get("mark_conf", {}))
        self.set_judge_mode(judge_mode)
        self.set_short_name(config.get("short_name_expected", ""))  # older jobs: presence only
        self.layout_check_checkbox.setChecked(bool(config.get("check_layout", False)))
        self._ctx.filters.check_layout = self.layout_check_checkbox.isChecked()

        return model_ok
