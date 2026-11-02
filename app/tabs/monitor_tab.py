import time
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QRect, QRectF, QSize, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..autocapture import (
    AutoCaptureGate,
    autocapture_images_dir,
    autocapture_part_dir,
    capture_stem,
    count_images,
    image_hash,
    qimage_to_array,
)
from ..context import AppContext
from ..i18n import tr, translator, trf
from ..jobs import inspection_types
from ..marking.short_name import pad_box
from ..judgement import InspectionCycle
from ..ng_log import save_ng_image
from ..paths import CAPTURES_IMAGES_DIR, CAPTURES_LABELS_DIR, NO_PART_NAME
from ..roi import detections_in_crop, roi_to_pixels
from ..theme import (
    NG_RED,
    OK_GREEN,
    RUN_BUTTON_STYLES,
    STANDBY_YELLOW,
    action_button_qss,
    gloss_gradient,
    inspection_title,
    make_icon,
    set_badge_title,
)
from ..vision.total import total_state
from ..widgets.layout import SIDE_PANEL_WIDTH, VIDEO_MIN_SIZE, bottom_bar, top_bar
from ..widgets.master_dialog import MasterDialog
from ..widgets.master_strip import MasterStrip
from ..widgets.perspective_dialog import PerspectiveDialog
from ..widgets.tool_overlay import draw_label, draw_overlay, overlay_pen, tool_value_text
from ..widgets.tool_params import kind_hint
from ..widgets.video_view import VideoView
from ..yolo_labels import pixel_box_to_yolo, write_yolo_label_file

TOOLBAR_HEIGHT = 40
ICON_SIZE = QSize(20, 20)


class MonitorTab(QWidget):
    """Live camera view, OK/NG judgement, cycle counters and Start/Stop."""

    ng_saved = Signal()

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._class_counts = {}
        self._class_colors = {}
        self._current_frame_image: Optional[QImage] = None
        self._current_frame_detections: list = []
        self._current_tool_results: list = []
        self._judge_mode_shown: Optional[bool] = None
        # NG short name, zoomed on the image until the result is no longer NG:
        # {"crop": QImage, "box": (x1, y1, x2, y2), "text": caption}.
        self._short_name_zoom: Optional[dict] = None
        self._kinds_shown: Optional[list] = None
        # Held judgement, trigger mode and the Confirm NG latch (relay inputs).
        self._latch = InspectionCycle()
        self._pending_ng_save: bool = False
        self._stop_requested: bool = False
        self._camera_lost: bool = False
        self._tower_lost: bool = False
        self._autocapture_gate: Optional[AutoCaptureGate] = None
        # Setup preview: camera started only so the user can set something up
        # (masters, ROI, top view, a capture). Frames show, but nothing is
        # judged or counted and the tower stays off.
        self._preview: bool = False
        self._live: bool = False  # a frame arrived since the stream started
        self._on_first_frame: list = []  # callables run on the first frame
        ctx.worker.finished.connect(self._on_stream_finished)

        self.run_button = QPushButton()
        self.run_button.setIconSize(ICON_SIZE)
        self.run_button.setFixedSize(170, 44)
        self.run_button.clicked.connect(self.toggle_stream)
        self.run_status_label = QLabel()
        self.run_status_label.setMinimumWidth(120)
        # Follow the worker's real state, so the button also flips back if the
        # stream ends on its own (camera open failure, lost frames).
        ctx.worker.started.connect(self._refresh_run_button)
        ctx.worker.finished.connect(self._refresh_run_button)
        ctx.worker.frame_ready.connect(self.on_frame)
        ctx.worker.camera_connected.connect(self._on_camera_connected)
        ctx.worker.finished.connect(self._clear_camera_lost)
        ctx.tower.connection_text_changed.connect(self._on_tower_connection_text)
        ctx.relay.connection_changed.connect(self._refresh_monitor_info)
        ctx.relay.input_pressed.connect(self._on_relay_input)
        translator().language_changed.connect(self._on_language_changed)
        self._refresh_run_button()

        self.capture_button = QPushButton("Capture")
        self.capture_button.setIcon(make_icon("camera", "#e0e0e0"))
        self.capture_button.setIconSize(ICON_SIZE)
        self.capture_button.setFixedHeight(TOOLBAR_HEIGHT)
        self.capture_button.clicked.connect(self.capture_frame)

        # Quick-access on/off toggle, mirroring the Settings tab checkbox.
        self.buzzer_toggle_button = QPushButton()
        self.buzzer_toggle_button.setIconSize(ICON_SIZE)
        self.buzzer_toggle_button.setFixedSize(140, TOOLBAR_HEIGHT)
        self.buzzer_toggle_button.setCheckable(True)
        self.buzzer_toggle_button.setChecked(ctx.tower.buzzer_enabled)
        self.buzzer_toggle_button.toggled.connect(ctx.tower.set_buzzer_enabled)
        ctx.tower.buzzer_enabled_changed.connect(self._on_buzzer_enabled_changed)
        self._refresh_buzzer_toggle_button()

        self.judgement_banner = QLabel()
        self.judgement_banner.setObjectName("judgementBanner")
        self.judgement_banner.setAlignment(Qt.AlignCenter)
        self.judgement_banner.setFixedSize(130, 36)

        # Required marks the AI did not find (Settings > Required Marks).
        self.missing_label = QLabel()
        self.missing_label.setObjectName("missingMarks")
        self.missing_label.setStyleSheet(f"color: {QColor(NG_RED).lighter(150).name()}; font-weight: 800;")
        self.missing_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.missing_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.missing_label.setToolTip("Marks this job requires that the AI did not find on the part.")
        # Mark inspection warnings (Settings > Required Marks > check layout); amber, not NG.
        self.layout_label = QLabel()
        self.layout_label.setObjectName("layoutWarnings")
        self.layout_label.setStyleSheet(f"color: {STANDBY_YELLOW}; font-weight: 800;")
        self.layout_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.layout_label.setToolTip("Marks printed out of the usual layout. A warning: the verdict is not changed.")

        # Short name check: the preset value, big; green when read right, red when not.
        self.short_name_tile = QLabel()
        self.short_name_tile.setObjectName("shortNameTile")
        self.short_name_tile.setAlignment(Qt.AlignCenter)
        self.short_name_tile.setFixedHeight(60)  # the side panel's full width
        self.short_name_tile.hide()
        self._set_short_name_tile(None)

        # Trigger mode / Confirm NG (relay inputs): what the cycle waits for.
        self.cycle_label = QLabel()
        self.cycle_label.setObjectName("cycleState")
        self.cycle_label.setStyleSheet(f"color: {STANDBY_YELLOW}; font-weight: 800;")
        self.cycle_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cycle_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.confirm_button = QPushButton("Confirm NG")
        self.confirm_button.setIcon(make_icon("check", "#ffffff"))
        self.confirm_button.setMinimumHeight(44)
        self.confirm_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["stop"]))
        self.confirm_button.setToolTip("The NG part was taken aside: clear the NG and go on with the next part.")
        self.confirm_button.clicked.connect(self.confirm_ng)
        self.confirm_button.hide()
        for label in (self.missing_label, self.layout_label, self.cycle_label):
            label.hide()  # shown only while they have something to say

        # Part and model large enough to read from the line, one per row; the
        # rest of the info (kind of inspection, top view, tower) at normal size.
        self.part_label = QLabel()
        self.model_label = QLabel()
        for label in (self.part_label, self.model_label):
            label.setObjectName("monitorPartModel")
            label.setTextFormat(Qt.PlainText)
            label.setMinimumWidth(1)  # full text when it fits, clipped (never widening the window) when not
        self.monitor_info_label = QLabel()
        self.monitor_info_label.setObjectName("monitorInfo")
        self.monitor_info_label.setTextFormat(Qt.RichText)
        self._refresh_monitor_info()
        ctx.part_changed.connect(self._refresh_monitor_info)
        ctx.model_path_changed.connect(self._refresh_monitor_info)

        self.video_label = VideoView()
        self.video_label.setMinimumSize(*VIDEO_MIN_SIZE)
        self.video_label.roi_drawn.connect(self._on_roi_drawn)

        self.roi_button = QPushButton("Set ROI")
        self.roi_button.setIcon(make_icon("roi", "#e0e0e0"))
        self.roi_button.setIconSize(ICON_SIZE)
        self.roi_button.setFixedHeight(TOOLBAR_HEIGHT)
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(self._on_roi_button_toggled)
        self.roi_clear_button = QPushButton("Clear ROI")
        self.roi_clear_button.setIcon(make_icon("close", "#e0e0e0"))
        self.roi_clear_button.setFixedHeight(TOOLBAR_HEIGHT)
        self.roi_clear_button.clicked.connect(lambda: self._ctx.set_roi(None, save_to_job=True))
        ctx.roi_changed.connect(self._on_roi_changed)
        self.roi_clear_button.setEnabled(False)

        self.top_view_button = QPushButton("Top View")
        self.top_view_button.setIcon(make_icon("perspective", "#e0e0e0"))
        self.top_view_button.setIconSize(ICON_SIZE)
        self.top_view_button.setFixedHeight(TOOLBAR_HEIGHT)
        self.top_view_button.clicked.connect(self._open_top_view_dialog)
        ctx.perspective_changed.connect(self._refresh_monitor_info)

        self.masters_button = QPushButton("Masters")
        self.masters_button.setIcon(make_icon("master", "#e0e0e0"))
        self.masters_button.setIconSize(ICON_SIZE)
        self.masters_button.setFixedHeight(TOOLBAR_HEIGHT)
        self.masters_button.clicked.connect(self._open_master_dialog)

        self.autocapture_button = QPushButton()
        self.autocapture_button.setIconSize(ICON_SIZE)
        self.autocapture_button.setFixedHeight(TOOLBAR_HEIGHT)
        self.autocapture_button.setCheckable(True)
        self.autocapture_button.toggled.connect(self._on_autocapture_toggled)
        # Off whenever the stream stops or the job changes: the count and
        # masters belong to one job's run.
        ctx.worker.finished.connect(lambda: self.autocapture_button.setChecked(False))
        ctx.part_changed.connect(lambda _: self.autocapture_button.setChecked(False))
        ctx.masters_changed.connect(lambda _: self._refresh_autocapture_button())
        self.master_strip = MasterStrip()
        self.master_strip.set_masters(ctx.masters)
        self.master_strip.edit_requested.connect(self._open_master_dialog)
        ctx.masters_changed.connect(self.master_strip.set_masters)
        ctx.worker.finished.connect(lambda: self.master_strip.set_match(None, False))
        self._refresh_autocapture_button()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Class", "Count"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumWidth(220)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Judge by Required Marks: one row per ticked mark, like the tool table;
        # shown instead of the class counts.
        self.mark_table = QTableWidget(0, 4)
        self.mark_table.setHorizontalHeaderLabels(["Mark", "Conf", "Min", "Result"])
        self.mark_table.verticalHeader().setVisible(False)
        self.mark_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.mark_table.setSelectionMode(QTableWidget.NoSelection)
        self.mark_table.setAlternatingRowColors(True)
        self.mark_table.setMinimumWidth(220)
        self.mark_table.setMinimumHeight(60)
        header = self.mark_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)

        # Rule-based vision tools (app/vision): one row per tool, hidden when
        # the job has none.
        self.tool_header = QLabel()
        self.tool_header.setObjectName("panelHeader")
        set_badge_title(self.tool_header, "vision", "Image Tools")
        self.tool_table = QTableWidget(0, 3)
        self.tool_table.setHorizontalHeaderLabels(["Tool", "Value", "Result"])
        self.tool_table.verticalHeader().setVisible(False)
        self.tool_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tool_table.setSelectionMode(QTableWidget.NoSelection)
        self.tool_table.setAlternatingRowColors(True)
        self.tool_table.setMinimumWidth(220)
        self.tool_table.setMaximumHeight(170)
        self.tool_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._set_tool_panel_visible(False)
        ctx.tools_changed.connect(self._on_tools_changed)

        self._build_layout()
        self._update_judgement_banner("YELLOW")

    def _build_layout(self) -> None:
        for widget, hint in (
            (self.capture_button, "Save the current image (and its AI boxes as labels) for training."),
            (self.buzzer_toggle_button, "Turn the NG buzzer on or off."),
            (self.top_view_button, "Straighten a camera that looks at the part at an angle (top-view correction)."),
            (self.roi_button, "Drag a box on the image: AI detection only looks inside it."),
            (self.roi_clear_button, "Remove the AI area box: detect on the whole image."),
            (self.masters_button, "Choose the master images Auto Capture compares with."),
            (self.run_button, "Start or stop the camera and the inspection."),
        ):
            widget.setToolTip(hint)
        # Layout follows the Keyence reference: info strip + judgement pill on
        # top, framed image with a light side panel, action bar at the bottom
        # with Run on the far right.
        top, top_layout = top_bar()
        info_grid = QGridLayout()
        info_grid.setVerticalSpacing(0)
        info_grid.addWidget(self.part_label, 0, 0)
        info_grid.addWidget(self.monitor_info_label, 0, 1, Qt.AlignRight)
        info_grid.addWidget(self.model_label, 1, 0)
        warnings = QVBoxLayout()
        warnings.addWidget(self.missing_label)
        warnings.addWidget(self.layout_label)
        warnings.addWidget(self.cycle_label)
        info_grid.addLayout(warnings, 1, 1)
        # Right under the NG pill, where the operator is already looking.
        info_grid.addWidget(self.confirm_button, 1, 3)
        info_grid.setColumnStretch(1, 1)
        top_layout.addLayout(info_grid, 1)
        info_grid.addWidget(self.judgement_banner, 0, 3, Qt.AlignRight)


        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(2, 2, 2, 2)
        video_layout.addWidget(self.video_label)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_panel.setFixedWidth(SIDE_PANEL_WIDTH)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        # The preset short name, big, above the marks (only when it is checked).
        side_layout.addWidget(self.short_name_tile)
        self.ai_header = QLabel()
        self.ai_header.setObjectName("panelHeader")
        side_layout.addWidget(self.ai_header)
        side_layout.addWidget(self.table, 1)
        side_layout.addWidget(self.mark_table, 1)
        self._show_judge_mode()
        side_layout.addWidget(self.tool_header)
        side_layout.addWidget(self.tool_table)
        # In the side panel, not the bottom bar: that must fit the 1024-wide
        # kiosk screen and is already full.
        autocapture_header = QLabel("Auto Capture")
        autocapture_header.setObjectName("panelHeader")
        side_layout.addWidget(autocapture_header)
        side_layout.addWidget(self.master_strip)
        autocapture_row = QHBoxLayout()
        autocapture_row.setContentsMargins(6, 6, 6, 0)
        autocapture_row.setSpacing(6)
        autocapture_row.addWidget(self.autocapture_button, 1)
        autocapture_row.addWidget(self.masters_button)
        side_layout.addLayout(autocapture_row)
        side_layout.addWidget(self._build_cycle_counters())

        content_row = QHBoxLayout()
        content_row.addWidget(video_frame, 1)
        content_row.addWidget(side_panel)

        bottom, bottom_layout = bottom_bar()
        bottom_layout.addWidget(self.capture_button)
        bottom_layout.addWidget(self.buzzer_toggle_button)
        bottom_layout.addWidget(self.top_view_button)
        bottom_layout.addWidget(self.roi_button)
        bottom_layout.addWidget(self.roi_clear_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.run_status_label)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.run_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(content_row, 1)
        layout.addWidget(bottom)

    # --- Cycle counters -------------------------------------------------

    @staticmethod
    def _make_counter_tile(title: str, top: str, bottom: str, digits: str):
        tile = QFrame()
        tile.setObjectName("counterTile")
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(0, 0, 0, 0)
        tile_layout.setSpacing(0)
        title_label = QLabel(title)
        title_label.setObjectName("counterTitle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            f"QLabel#counterTitle {{ background: {gloss_gradient(top, bottom, bottom)}; }}"
        )
        value_label = QLabel("0")
        value_label.setObjectName("counterValue")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet(f"QLabel#counterValue {{ color: {digits}; }}")
        tile_layout.addWidget(title_label)
        tile_layout.addWidget(value_label, 1)
        return tile, value_label

    def _build_cycle_counters(self) -> QWidget:
        ok_tile, self.ok_count_label = self._make_counter_tile(
            "OK", "#5cbf60", "#2e7d32", "#4cd964")
        ng_tile, self.ng_count_label = self._make_counter_tile(
            "NG", "#e0685a", "#b83225", "#ff5a4a")

        self.counter_reset_button = QPushButton("Reset Count")
        self.counter_reset_button.setIcon(make_icon("refresh", "#e0e0e0"))
        self.counter_reset_button.setToolTip("Set the OK and NG counters back to 0.")
        self.counter_reset_button.setObjectName("counterReset")
        self.counter_reset_button.clicked.connect(self.on_reset_counters_clicked)

        tiles_row = QHBoxLayout()
        tiles_row.setSpacing(6)
        tiles_row.addWidget(ok_tile, 1)
        tiles_row.addWidget(ng_tile, 1)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        reset_row.addWidget(self.counter_reset_button)

        box = QWidget()
        box.setObjectName("counterBox")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addLayout(tiles_row)
        layout.addLayout(reset_row)
        self._refresh_cycle_counters()
        return box

    @property
    def ok_count(self) -> int:
        return self._ctx.settings["ok_count"]

    @property
    def ng_count(self) -> int:
        return self._ctx.settings["ng_count"]

    def _refresh_cycle_counters(self) -> None:
        self.ok_count_label.setText(str(self.ok_count))
        self.ng_count_label.setText(str(self.ng_count))

    def _count_cycle(self, state: str) -> None:
        if state == "GREEN":
            self._ctx.settings.set(ok_count=self.ok_count + 1)
        elif state == "RED":
            self._ctx.settings.set(ng_count=self.ng_count + 1)
            # The frame is annotated later in on_frame; save it there.
            self._pending_ng_save = True
        else:
            return
        self._refresh_cycle_counters()

    @Slot()
    def on_reset_counters_clicked(self) -> None:
        answer = QMessageBox.question(
            self, "Reset Count",
            f"Reset cycle counters?\n\nOK: {self.ok_count}    NG: {self.ng_count}",
        )
        if answer != QMessageBox.Yes:
            return
        self._ctx.settings.set(ok_count=0, ng_count=0)
        self._refresh_cycle_counters()

    # --- Info strip, judgement and buzzer toggle ------------------------

    def _refresh_monitor_info(self, *_) -> None:
        part = self._ctx.current_part_no or "—"
        model_path = self._ctx.model_path
        model = Path(model_path).stem if model_path else "—"
        # Rich text (for the red tower warning) collapses plain spaces.
        sep = "&nbsp;&nbsp;&nbsp;·&nbsp;&nbsp;&nbsp;"
        # Values formatted in after translating: a part number is never translated.
        self.part_label.setText(tr("Part: {}").format(part))
        self.model_label.setText(tr("Model: {}").format(model))
        parts = []
        if self._ctx.filters.perspective:
            parts.append(tr("Top View"))
        self._kinds_shown = self._inspection_types()
        if self._kinds_shown:
            parts.append(inspection_title(self._kinds_shown))
        relay = self._ctx.relay
        if self._ctx.settings["relay_enabled"]:
            color = "#7ee08a" if relay.connected else "#ff5a4a"
            parts.append(f'<span style="color:{color};">Relay {"●" if relay.connected else "○"}</span>')
        if self._tower_lost:
            parts.append(f'<span style="color:#ff5a4a;">{tr("Tower light disconnected")}</span>')
        self.monitor_info_label.setText(sep.join(parts))

    def _inspection_types(self) -> list:
        ctx, filters = self._ctx, self._ctx.filters
        return inspection_types({"model_path": ctx.model_path, "judge_mode": filters.judge_mode,
                                 "required_classes": filters.required_classes, "tools": ctx.tools})

    # --- Hardware connection (camera / tower light cable) ----------------

    @Slot(bool)
    def _on_camera_connected(self, connected: bool) -> None:
        self._camera_lost = not connected
        if connected:
            self.video_label.set_overlay_text(None)
        else:
            # No frames: never leave a stale OK/NG on the banner or tower.
            self._reset_judgement()
            self._ctx.tower.update_state("YELLOW")
            self.video_label.set_overlay_text(
                "CAMERA DISCONNECTED\nCheck the USB cable — reconnecting automatically")
        self._refresh_run_button()

    @Slot()
    def _clear_camera_lost(self) -> None:
        """Stream ended: drop the camera-lost warning with it."""
        self._camera_lost = False
        self.video_label.set_overlay_text(None)
        self._refresh_run_button()

    @Slot(str)
    def _on_tower_connection_text(self, text: str) -> None:
        self._tower_lost = self._ctx.tower.reconnecting
        self._refresh_monitor_info()

    def _update_judgement_banner(self, state: str) -> None:
        text, bg, fg = {
            "GREEN": ("OK", OK_GREEN, "#ffffff"),
            "RED": ("NG", NG_RED, "#ffffff"),
            "YELLOW": ("STANDBY", STANDBY_YELLOW, "#1a1a1a"),
        }[state]
        self.judgement_banner.setText(tr(text))
        gloss = gloss_gradient(QColor(bg).lighter(135).name(), bg, QColor(bg).darker(125).name())
        self.judgement_banner.setStyleSheet(
            f"QLabel#judgementBanner {{ background: {gloss}; color: {fg}; }}"
        )

    def _reset_judgement(self) -> None:
        self._latch.reset()
        self._short_name_zoom = None
        self._set_short_name_tile(None)
        self._update_judgement_banner("YELLOW")
        self._refresh_cycle_prompt()

    def _on_language_changed(self, _language: str) -> None:
        self._judge_mode_shown = None  # re-title the side panel
        self._set_short_name_tile("RED" if self._short_name_zoom else None)
        self._show_judge_mode()
        self._refresh_monitor_info()
        self._refresh_run_button()
        self._refresh_cycle_prompt()
        self._update_judgement_banner(self._latch.state or "YELLOW")

    @staticmethod
    def _set_warning(label: QLabel, text: str) -> None:
        """Empty warning lines are hidden, so the top strip keeps its height."""
        label.setText(text)
        label.setVisible(bool(text))

    def _show_state(self, state: str) -> None:
        """Banner, tower light and relay outputs follow the held state."""
        self._update_judgement_banner(state)
        self._ctx.tower.update_state(state)
        self._ctx.relay.update_state(state)
        self._refresh_cycle_prompt()

    def _refresh_cycle_prompt(self) -> None:
        cycle, relay = self._latch, self._ctx.relay
        self.confirm_button.setVisible(cycle.awaiting_confirm)
        if cycle.awaiting_confirm:
            text = "NG: confirm to continue"
        elif relay.has_input("trigger") and self._ctx.worker.isRunning() and not self._preview:
            text = "Judging…" if cycle.armed else "Waiting for trigger"
        else:
            text = ""
        self._set_warning(self.cycle_label, tr(text))

    @Slot(str)
    def _on_relay_input(self, role: str) -> None:
        if self._preview or not self._ctx.worker.isRunning():
            return
        if role == "confirm_ng":
            self.confirm_ng()
        elif role == "trigger":
            if not self._latch.trigger():
                self._ctx.status.emit("Trigger ignored: confirm the NG first.")
            self._refresh_cycle_prompt()

    @Slot()
    def confirm_ng(self) -> None:
        if self._latch.confirm():
            self._ctx.status.emit("NG confirmed.")
            self._show_state("YELLOW")

    @Slot(bool)
    def _on_buzzer_enabled_changed(self, enabled: bool) -> None:
        self.buzzer_toggle_button.setChecked(enabled)
        self._refresh_buzzer_toggle_button()

    def _refresh_buzzer_toggle_button(self) -> None:
        # Neutral button; the speaker icon carries the on/muted state, so green
        # stays reserved for the Start action.
        if self._ctx.tower.buzzer_enabled:
            self.buzzer_toggle_button.setText("Buzzer On")
            self.buzzer_toggle_button.setIcon(make_icon("speaker_on", "#e0e0e0"))
            self.buzzer_toggle_button.setStyleSheet("")
        else:
            self.buzzer_toggle_button.setText("Buzzer Off")
            self.buzzer_toggle_button.setIcon(make_icon("speaker_off", "#e57373"))
            self.buzzer_toggle_button.setStyleSheet("color: #9aa5b1;")

    # --- Top view (perspective correction) ------------------------------

    @Slot()
    def _open_top_view_dialog(self) -> None:
        self.ensure_live(self._open_top_view_now, "Starting camera for Top View...")

    def _open_top_view_now(self) -> None:
        frame = self._ctx.worker.last_raw_frame
        if frame is None:
            return
        self.roi_button.setChecked(False)
        dialog = PerspectiveDialog(frame.copy(), self._ctx.filters.perspective, parent=self)
        accepted = dialog.exec() == PerspectiveDialog.Accepted
        self.end_preview()
        if not accepted:
            return
        self._ctx.set_perspective(dialog.quad, save_to_job=True)
        saved = f" (saved to job {self._ctx.current_part_no})" if self._ctx.current_part_no else ""
        state = "set" if dialog.quad else "removed"
        self._ctx.status.emit(f"Top View {state}{saved}.")

    # --- Detection area (ROI) -------------------------------------------

    @Slot(bool)
    def _on_roi_button_toggled(self, checked: bool) -> None:
        if not checked:
            self.video_label.set_roi_mode(False)
            self.end_preview()
            return

        def enter_roi_mode() -> None:
            if self.roi_button.isChecked():
                self.video_label.set_roi_mode(True)
                self._ctx.status.emit("Drag on the image to draw the detection area.")

        if not self.ensure_live(enter_roi_mode, "Starting camera for the ROI..."):
            self.roi_button.setChecked(False)

    @Slot(tuple)
    def _on_roi_drawn(self, roi: tuple) -> None:
        self.roi_button.setChecked(False)  # also ends a setup preview
        self._ctx.set_roi(roi, save_to_job=True)
        saved = f" (saved to job {self._ctx.current_part_no})" if self._ctx.current_part_no else ""
        self._ctx.status.emit(f"Detection area set{saved}.")

    @Slot(object)
    def _on_roi_changed(self, roi) -> None:
        self.roi_clear_button.setEnabled(roi is not None)
        # While stopped, redraw the last frame so the new area shows at once.
        if not self._ctx.worker.isRunning() and self._current_frame_image is not None:
            self._current_frame_detections = []
            self._show_frame(self._current_frame_image, [])

    def _roi_pixels(self, image: QImage) -> Optional[tuple]:
        roi = self._ctx.filters.roi
        return roi_to_pixels(roi, image.width(), image.height()) if roi else None

    def _draw_roi(self, painter: QPainter, image: QImage) -> None:
        box = self._roi_pixels(image)
        if box is None:
            return
        x1, y1, x2, y2 = box
        # Dim everything outside the area, then outline it.
        shade = QColor(0, 0, 0, 120)
        w, h = image.width(), image.height()
        for rect in (QRect(0, 0, w, y1), QRect(0, y2, w, h - y2),
                     QRect(0, y1, x1, y2 - y1), QRect(x2, y1, w - x2, y2 - y1)):
            painter.fillRect(rect, shade)
        painter.setPen(overlay_pen(painter, "#ffd54f", 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        draw_label(painter, x1, y1, "ROI", "#ffd54f", "inside")

    # --- Stream control -------------------------------------------------

    @Slot()
    def toggle_stream(self) -> None:
        # During a setup preview the button reads Start: it turns the
        # preview into a real run.
        if self._ctx.worker.isRunning() and not self._preview:
            self.stop_stream()
        else:
            self.start_stream()

    @property
    def in_preview(self) -> bool:
        return self._preview and self._ctx.worker.isRunning()

    def ensure_live(self, then: Callable[[], None], message: str = "Starting camera...",
                    preview: bool = True) -> bool:
        """Runs `then` once live frames are flowing, starting the camera if
        it is stopped (as a setup preview unless preview=False). Returns
        False if the camera could not be started (the reason is shown)."""
        worker = self._ctx.worker
        if worker.isRunning() and self._stop_requested:
            # Still shutting down from a Stop: start again once it has.
            def retry() -> None:
                worker.finished.disconnect(retry)
                self.ensure_live(then, message, preview)
            worker.finished.connect(retry)
            self._ctx.status.emit(message)
            return True
        if worker.isRunning():
            if not preview and self._preview:
                self._end_preview_keep_running()
            if self._live:
                then()
            else:
                self._on_first_frame.append(then)
            return True
        self._on_first_frame = [then]
        self._preview = preview
        if not self.start_stream(keep_preview=True):
            self._on_first_frame = []
            self._preview = False
            return False
        self._ctx.status.emit(message)
        return True

    def end_preview(self) -> None:
        """Stops the camera if it was only started for a setup step."""
        if self.in_preview and self._autocapture_gate is None:
            self.stop_stream()

    def _end_preview_keep_running(self) -> None:
        """Setup preview -> real run, without restarting the camera."""
        self._preview = False
        self._reset_judgement()
        self._refresh_run_button()
        if not self._ctx.tower.connected:
            self._ctx.tower.connect_device()

    @Slot()
    def _on_stream_finished(self) -> None:
        self._preview = False
        self._live = False
        self._on_first_frame = []
        self._refresh_run_button()

    def _refresh_run_button(self) -> None:
        running = self._ctx.worker.isRunning()
        if not running:
            self._stop_requested = False
        stopping = running and self._stop_requested

        if stopping:
            text, icon, style = "Stopping…", "stop", RUN_BUTTON_STYLES["stop"]
            status, dot = "Stopping…", "#f4d03f"
        elif running and self._camera_lost:
            text, icon, style = "Stop", "stop", RUN_BUTTON_STYLES["stop"]
            status, dot = "No camera", "#f4d03f"
        elif running and self._preview:
            text, icon, style = "Start", "play", RUN_BUTTON_STYLES["start"]
            status, dot = "Preview", "#f0a020"
        elif running:
            text, icon, style = "Stop", "stop", RUN_BUTTON_STYLES["stop"]
            status, dot = "Running", "#2ecc71"
        else:
            text, icon, style = "Start", "play", RUN_BUTTON_STYLES["start"]
            status, dot = "Stopped", "#9aa5b1"

        self.run_button.setEnabled(not stopping)
        self.run_button.setText(tr(text))
        status = tr(status)
        self.run_button.setIcon(make_icon(icon))
        self.run_button.setStyleSheet(action_button_qss(*style))
        self.run_status_label.setText(
            f'<span style="color:{dot}; font-size:16pt;">●</span>'
            f'&nbsp;<span style="color:{dot}; font-weight:600;">{status}</span>'
        )

    @Slot()
    def start_stream(self, keep_preview: bool = False) -> bool:
        """Starts monitoring; returns False (reason shown) if it can't."""
        ctx = self._ctx
        if ctx.worker.isRunning():
            if self._preview and not keep_preview:
                self._end_preview_keep_running()
                ctx.status.emit("Running.")
                return True
            ctx.status.emit("Already running.")
            return True
        if ctx.training_active:
            ctx.status.emit("Training in progress. Wait for it to finish.")
            return False
        if ctx.camera_index is None:
            ctx.camera_scan_requested.emit()  # maybe plugged in since boot
        if ctx.camera_index is None:
            ctx.status.emit("No camera found — check the USB cable.")
            return False
        if not keep_preview:
            self._preview = False
        self._live = False
        ctx.worker.set_camera_index(int(ctx.camera_index))
        self._reset_judgement()
        ctx.worker.start()
        self._refresh_run_button()
        if not ctx.tower.connected and not self._preview:
            ctx.tower.connect_device()
        return True

    @Slot()
    def stop_stream(self) -> None:
        # Non-blocking: the worker can take seconds to notice (first inference
        # warms up the GPU), so the button shows "Stopping..." until the
        # worker's finished signal flips it back to Start.
        if self._ctx.worker.isRunning():
            self._stop_requested = True
            self._ctx.worker.stop()
        self._ctx.tower.shutdown()
        self._ctx.relay.update_state(None)
        self._reset_judgement()
        self._refresh_run_button()

    @Slot()
    def capture_frame(self) -> None:
        self.ensure_live(self._capture_now, "Starting camera to capture...")

    def _capture_now(self) -> None:
        try:
            self._save_capture()
        finally:
            self.end_preview()

    def _save_capture(self) -> None:
        CAPTURES_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        CAPTURES_LABELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = self._ctx.model_path
        model_stem = Path(model_path).stem if model_path else "capture"
        stem = f"{model_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        image_path = CAPTURES_IMAGES_DIR / f"{stem}.jpg"
        label_path = CAPTURES_LABELS_DIR / f"{stem}.txt"

        image, boxes = self._crop_and_labels(self._current_frame_image, self._current_frame_detections)
        pixmap = QPixmap.fromImage(image)
        if not pixmap.save(str(image_path), "JPG"):
            self._ctx.status.emit("Frame save failed.")
            return
        write_yolo_label_file(label_path, boxes)

        self._ctx.status.emit(f"Captured: {image_path.name} ({len(boxes)} boxes)")

    def _crop_and_labels(self, image: QImage, detections: list):
        """(image, YOLO boxes) as saved for training. With an ROI, that is
        what the model sees (the area only), so training images match
        inference input."""
        box = self._roi_pixels(image)
        if box is not None:
            x1, y1, x2, y2 = box
            image = image.copy(x1, y1, x2 - x1, y2 - y1)
            detections = detections_in_crop(detections, box)
        boxes = []
        for det in detections:
            yolo = pixel_box_to_yolo(det["x1"], det["y1"], det["x2"], det["y2"], image.width(), image.height())
            yolo["cls_id"] = det["cls_id"]
            boxes.append(yolo)
        return image, boxes

    def _roi_crop(self, image: QImage) -> QImage:
        """The frame cut to the ROI (what the model and masters see)."""
        box = self._roi_pixels(image)
        if box is None:
            return image
        x1, y1, x2, y2 = box
        return image.copy(x1, y1, x2 - x1, y2 - y1)

    def _current_roi_crop(self) -> Optional[QImage]:
        """Latest live frame cut to the ROI, None while not live."""
        image = self._current_frame_image
        if image is None or not self._live:
            return None
        return self._roi_crop(image)

    # --- Auto capture (masters + hash similarity) -----------------------

    @Slot()
    def _open_master_dialog(self) -> bool:
        """Returns True if masters were saved."""
        part_no = self._ctx.current_part_no
        if not part_no:
            self._ctx.show_tab_requested.emit("Job Change")
            self._ctx.status.emit("Pick or scan a job — masters are saved per job.")
            return False
        dialog = MasterDialog(self._ctx.masters, part_no, self._current_roi_crop,
                              lambda then: self.ensure_live(then, "Starting camera for masters..."),
                              parent=self)
        accepted = dialog.exec() == MasterDialog.Accepted
        if accepted:
            self._ctx.set_masters(dialog.entries, save_to_job=True)
            self._ctx.status.emit(f"Masters saved to job {part_no}: {len(self._ctx.masters)} set.")
        self.end_preview()
        return accepted

    def _autocapture_dir(self) -> Path:
        return autocapture_part_dir(self._ctx.current_part_no)

    @Slot(bool)
    def _on_autocapture_toggled(self, checked: bool) -> None:
        if checked:
            settings = self._ctx.settings
            gate = AutoCaptureGate(
                threshold=settings["autocapture_threshold"],
                interval_s=settings["autocapture_interval_ms"] / 1000,
                max_images=settings["autocapture_max_images"],
                count=count_images(self._autocapture_dir()),
                require_clear=settings["autocapture_wait_clear"],
            )
            if not self._ctx.masters or not self._live or self._preview:
                # Missing a step: uncheck for now, do the step, then come back.
                self.autocapture_button.blockSignals(True)
                self.autocapture_button.setChecked(False)
                self.autocapture_button.blockSignals(False)
                if not self._ctx.masters:
                    if self._open_master_dialog() and self._ctx.masters:
                        self.autocapture_button.setChecked(True)
                else:
                    self.ensure_live(lambda: self.autocapture_button.setChecked(True),
                                     "Starting camera for Auto Capture...", preview=False)
                return
            problem = None
            if gate.full:
                problem = (f"Auto capture folder is full ({gate.count}/{gate.max_images}). "
                           "Raise the limit in Settings or move the images.")
            if problem:
                self._ctx.status.emit(problem)
                self.autocapture_button.setChecked(False)
                return
            self._autocapture_gate = gate
            self._ctx.status.emit(
                f"Auto capture on: saving to {autocapture_images_dir(self._ctx.current_part_no)}")
        elif self._autocapture_gate is not None:
            self._ctx.status.emit(f"Auto capture off ({self._autocapture_gate.count} images).")
            self._autocapture_gate = None
        self._refresh_autocapture_button()

    def _refresh_autocapture_button(self) -> None:
        gate = self._autocapture_gate
        if gate is not None:
            self.autocapture_button.setText(f"Auto {gate.count}/{gate.max_images}")
            self.autocapture_button.setToolTip(
                "Saved — waiting for the part to leave" if gate.waiting else "Auto capture on")
            self.autocapture_button.setIcon(make_icon("autocapture", "#1a1a1a"))
            self.autocapture_button.setStyleSheet(
                f"QPushButton {{ background: {gloss_gradient('#ffcf73', '#f0a020', '#c07a10')};"
                " color: #1a1a1a; font-weight: 700; }")
        else:
            self.autocapture_button.setText("Auto")
            self.autocapture_button.setToolTip("Auto capture: save frames that match a master")
            self.autocapture_button.setIcon(make_icon("autocapture", "#e0e0e0"))
            self.autocapture_button.setStyleSheet("")

    def _match_masters(self, image: QImage, detections: list) -> None:
        """Per frame while the job has masters: outline the nearest master
        thumbnail, and run auto capture if it is on."""
        crop = self._roi_crop(image)
        frame_hash = image_hash(qimage_to_array(crop))
        masters = self._ctx.masters
        nearest, distance = AutoCaptureGate.nearest(frame_hash, masters)
        threshold = self._ctx.settings["autocapture_threshold"]
        index = next(i for i, m in enumerate(masters) if m is nearest)
        self.master_strip.set_match(index, distance <= threshold)
        if self._autocapture_gate is not None:
            self._autocapture_step(crop, frame_hash)

    def _autocapture_step(self, crop: QImage, frame_hash: int) -> None:
        gate = self._autocapture_gate
        now = time.monotonic()
        waiting = gate.waiting
        master = gate.check(frame_hash, self._ctx.masters, now)
        if waiting and not gate.waiting:
            self._refresh_autocapture_button()  # part left: tooltip back to armed
        if master is None:
            return
        number = next(i for i, m in enumerate(self._ctx.masters) if m is master) + 1
        images_dir = autocapture_images_dir(self._ctx.current_part_no)
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            # No label file: boxes are drawn by hand in the Train tab.
            if not crop.save(str(images_dir / f"{capture_stem()}_m{number}.jpg"), "JPG", 95):
                raise OSError("image write failed")
        except OSError as exc:
            self._ctx.status.emit(f"Auto capture save failed: {exc}")
            self.autocapture_button.setChecked(False)
            return
        gate.record_saved(frame_hash, now, master)
        next_hint = " — remove part for next" if gate.waiting else ""
        self._ctx.status.emit(f"Auto: master {number} · {gate.count}/{gate.max_images}{next_hint}")
        if gate.full:
            self.autocapture_button.setChecked(False)
            self._ctx.status.emit(f"Auto capture full ({gate.max_images}).")
        else:
            self._refresh_autocapture_button()

    # --- Per-frame processing -------------------------------------------

    @Slot(QImage, list)
    def on_frame(self, image: QImage, detections: list, tool_results: Optional[list] = None) -> None:
        marks = self._ctx.filters.mark_results(detections)  # before filtering: weak marks show their conf
        detections = self._ctx.filters.apply(detections)  # Use filtered for drawing and table
        tool_results = tool_results or []
        self._current_frame_image = image
        self._current_frame_detections = detections
        self._current_tool_results = tool_results
        self._ctx.last_detections = detections
        self._live = True
        filters = self._ctx.filters
        missing = filters.missing(detections)
        issue = filters.short_name_issue(detections)
        warning = [tr(f"Missing: {', '.join(missing)}")] if missing else []
        if issue:
            warning.append(tr(issue))
        self._set_warning(self.missing_label, "   ·   ".join(warning))
        layout = self._ctx.filters.layout_warnings(detections)
        self._set_warning(self.layout_label, tr(f"Layout: {'; '.join(layout)}") if layout else "")

        if not self._preview:
            # One cycle = each time the held judgement becomes OK or NG, or
            # one per trigger in trigger mode.
            relay = self._ctx.relay
            ocr = filters.short_name_ocr(detections)
            state, counted = self._latch.update(
                self._raw_state(detections, tool_results), self._ctx.settings["state_hold_ms"],
                trigger_mode=relay.has_input("trigger"), confirm_ng=relay.has_input("confirm_ng"),
                pending=ocr is not None and ocr["state"] == "pending")
            if counted:
                self._count_cycle(counted)
            self._keep_short_name_zoom(image, detections, ocr, issue, state)
            if self._short_name_zoom is not None:
                self._set_short_name_tile("RED")  # wrong / unreadable: red while the NG lasts
            else:
                self._set_short_name_tile("GREEN" if ocr is not None and ocr["state"] == "ok" else None)
            self._show_state(state)

        self._show_frame(image, detections, tool_results)
        self._show_judge_mode()
        self._update_class_table(detections)
        self._update_mark_table(marks)
        self._update_tool_table(tool_results)
        if self._ctx.masters:
            self._match_masters(image, detections)
        if self._pending_ng_save:
            self._pending_ng_save = False
            self._save_ng_image(self._render_frame(image, detections, tool_results))
        if self._on_first_frame:
            pending, self._on_first_frame = self._on_first_frame, []
            for then in pending:
                then()

    def _raw_state(self, detections: list, tool_results: list) -> str:
        """YOLO's state as before when the job has no tools; otherwise the
        AND / OR of the tools and YOLO (a lost part gives STANDBY)."""
        yolo_state = self._ctx.filters.judge(detections)
        if not tool_results:
            return yolo_state
        if self._ctx.model_path is None:
            yolo_state = None  # no model: the tools alone decide
        position_found = all(r.ok for r in tool_results if r.kind == "position")
        return total_state(tool_results, yolo_state, self._ctx.tools_gate, position_found)

    def _set_short_name_tile(self, state: Optional[str]) -> None:
        """Shown while the job checks the short name: its preset value on
        green (read right), red (wrong / unreadable) or neutral (no part yet,
        still reading)."""
        filters = self._ctx.filters
        self.short_name_tile.setVisible(filters.checks_short_name)
        if not filters.checks_short_name:
            return
        bg, fg = {"GREEN": (OK_GREEN, "#ffffff"), "RED": (NG_RED, "#ffffff")}.get(state, ("#3a4048", "#e8ecf0"))
        caption = tr("Short name")
        self.short_name_tile.setText(
            f'<span style="font-size:13pt; font-weight:700;">{caption}&nbsp;&nbsp;</span>'
            f'<span style="font-size:34pt; font-weight:800;">{filters.short_name_expected}</span>')
        self.short_name_tile.setStyleSheet(
            f"QLabel#shortNameTile {{ background: {bg}; color: {fg}; border: 2px solid #0b0d10; border-radius: 0; }}")

    def _keep_short_name_zoom(self, image: QImage, detections: list, ocr: Optional[dict], issue: str,
                              state: str) -> None:
        """While the short name is wrong / unreadable, remember a crop of it to
        show zoomed; it stays as long as the result is NG (e.g. waiting for
        Confirm NG after the part was taken away)."""
        if issue:
            det = max((d for d in detections if d.get("class_name") == "short_name"), key=lambda d: d.get("conf", 0.0))
            box = tuple(int(det[k]) for k in ("x1", "y1", "x2", "y2"))
            x1, y1, x2, y2 = pad_box(box, image.width(), image.height())
            if ocr["state"] == "unreadable":
                caption = trf("Unreadable  ·  expected {}", ocr["expected"])
            else:
                caption = trf("Read {}  ≠  expected {}", ocr["text"], ocr["expected"])
            self._short_name_zoom = {"crop": image.copy(QRect(x1, y1, x2 - x1, y2 - y1)), "box": box, "text": caption}
        elif state != "RED":
            self._short_name_zoom = None

    def _draw_short_name_zoom(self, painter: QPainter, image: QImage) -> None:
        """The NG short name magnified (at least 3x, ~38 % of the frame wide) in
        the corner furthest from it, framed red, with what was read."""
        zoom = self._short_name_zoom
        if zoom is None or zoom["crop"].isNull():
            return
        crop, (x1, y1, x2, y2) = zoom["crop"], zoom["box"]
        w, h = image.width(), image.height()
        scale = max(3.0, 0.38 * w / max(1, crop.width()))
        scale = min(scale, 0.45 * h / max(1, crop.height()), 0.6 * w / max(1, crop.width()))
        zw, zh = crop.width() * scale, crop.height() * scale
        margin = 0.02 * w
        zx = margin if (x1 + x2) / 2 > w / 2 else w - zw - margin
        zy = h - zh - margin if (y1 + y2) / 2 < h / 2 else margin + 0.05 * h
        target = QRectF(zx, zy, zw, zh)
        painter.drawImage(target, crop)
        painter.setPen(overlay_pen(painter, NG_RED, 4))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(target)
        painter.setPen(overlay_pen(painter, NG_RED, 2, Qt.DashLine))
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))  # where it is on the part
        draw_label(painter, zx, zy, zoom["text"], NG_RED)

    def _render_frame(self, image: QImage, detections: list, tool_results: Optional[list] = None) -> QPixmap:
        """Frame with the overlay burned in: what the NG log saves."""
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        self._paint_overlay(painter, image, detections, tool_results)
        painter.end()
        return pixmap

    def _show_frame(self, image: QImage, detections: list, tool_results: Optional[list] = None) -> None:
        """Live view: the plain frame, with the overlay painted over it at screen size."""
        self.video_label.set_frame(
            QPixmap.fromImage(image),
            lambda painter: self._paint_overlay(painter, image, detections, tool_results))

    def _paint_overlay(self, painter: QPainter, image: QImage, detections: list,
                       tool_results: Optional[list] = None) -> None:
        """ROI, tool overlays and detection boxes, in frame pixels."""
        self._draw_roi(painter, image)
        for result in tool_results or []:
            draw_overlay(painter, result.overlay)
        for det in detections:
            # Each class keeps its own color.
            class_name = det.get("class_name", "unknown")
            if class_name not in self._class_colors:
                hue = (len(self._class_colors) * 37) % 360
                self._class_colors[class_name] = QColor.fromHsv(hue, 200, 255)
            color = self._class_colors[class_name]
            det["color"] = color
            painter.setPen(overlay_pen(painter, color))
            painter.setBrush(Qt.NoBrush)
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            draw_label(painter, x1, y1, det["label"], color)
        self._draw_short_name_zoom(painter, image)

    def _save_ng_image(self, pixmap: QPixmap) -> None:
        try:
            save_ng_image(pixmap, self._ctx.current_part_no or NO_PART_NAME)
        except OSError as exc:
            self._ctx.status.emit(f"NG image save failed: {exc}")
            return
        self.ng_saved.emit()

    def _update_class_table(self, detections: list) -> None:
        self._class_counts.clear()
        for det in detections:
            class_name = det.get("class_name", "unknown")
            self._class_counts[class_name] = self._class_counts.get(class_name, 0) + 1

        self.table.setRowCount(len(self._class_counts))
        for row, (class_name, count) in enumerate(sorted(self._class_counts.items())):
            color = self._class_colors.get(class_name, Qt.white)

            class_item = QTableWidgetItem(class_name)
            count_item = QTableWidgetItem(str(count))
            brush = QBrush(color)
            class_item.setForeground(brush)
            count_item.setForeground(brush)

            self.table.setItem(row, 0, class_item)
            self.table.setItem(row, 1, count_item)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._show_judge_mode()  # the switch may have changed on the Settings tab
        self._set_short_name_tile("RED" if self._short_name_zoom else None)  # the preset may have changed

    def _show_judge_mode(self) -> None:
        """Required Marks mode: the mark table instead of the class counts."""
        if self._inspection_types() != self._kinds_shown:
            self._refresh_monitor_info()
        by_marks = self._ctx.filters.by_marks
        if self._judge_mode_shown == by_marks:
            return
        self._judge_mode_shown = by_marks
        set_badge_title(self.ai_header, "mark" if by_marks else "ai", tr("Required Marks" if by_marks else "AI Detections"))
        self.table.setVisible(not by_marks)
        self.mark_table.setVisible(by_marks)
        if by_marks:
            self._update_mark_table(self._ctx.filters.mark_results(self._current_frame_detections))

    def _update_mark_table(self, marks: list) -> None:
        self.mark_table.setRowCount(len(marks))
        for row, mark in enumerate(marks):
            conf = mark["conf"]
            name = QTableWidgetItem(mark["class_name"])
            if "read" in mark:  # the short name as read by OCR
                shown = {"ok": mark["read"], "ng": f"{mark['read']} ≠ {mark['expected']}",
                         "unreadable": "?", "pending": "…"}[mark["ocr_state"]]
                name.setText(f"{mark['class_name']}  {shown}")
                name.setToolTip(trf("Expected {}", mark["expected"]))
            value = QTableWidgetItem("—" if conf is None else f"{conf:.2f}")
            value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            minimum = QTableWidgetItem(f"{mark['min_conf']:.2f}")
            minimum.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if mark["ok"] is None:
                verdict = QTableWidgetItem("—")
            else:
                verdict = QTableWidgetItem("OK" if mark["ok"] else "NG")
                verdict.setBackground(QBrush(QColor(OK_GREEN if mark["ok"] else NG_RED)))
                verdict.setForeground(QBrush(QColor("#ffffff")))
                verdict.setToolTip(tr("Found at or above Min" if mark["ok"] else
                                      "Not found" if conf is None else "Found below Min"))
            verdict.setTextAlignment(Qt.AlignCenter)
            font = verdict.font()
            font.setBold(True)
            verdict.setFont(font)
            for col, item in enumerate((name, value, minimum, verdict)):
                self.mark_table.setItem(row, col, item)

    # --- Vision tools ---------------------------------------------------

    def _set_tool_panel_visible(self, visible: bool) -> None:
        self.tool_header.setVisible(visible)
        self.tool_table.setVisible(visible)

    def _on_tools_changed(self, tools: list) -> None:
        self._refresh_monitor_info()
        self._set_tool_panel_visible(bool(tools))
        if not tools:
            self._current_tool_results = []
            self.tool_table.setRowCount(0)

    def _update_tool_table(self, tool_results: list) -> None:
        if not tool_results:
            self.tool_table.setRowCount(0)
            return
        self._set_tool_panel_visible(True)
        mm = bool(self._ctx.mm_per_px)
        self.tool_table.setRowCount(len(tool_results))
        for row, result in enumerate(tool_results):
            if result.extra.get("pending"):
                verdict = QTableWidgetItem("…")
                verdict.setBackground(QBrush(QColor(STANDBY_YELLOW)))
                verdict.setForeground(QBrush(QColor("#1a1a1a")))
            else:
                verdict = QTableWidgetItem("OK" if result.ok else "NG")
                verdict.setBackground(QBrush(QColor(OK_GREEN if result.ok else NG_RED)))
                verdict.setForeground(QBrush(QColor("#ffffff")))
            verdict.setTextAlignment(Qt.AlignCenter)
            font = verdict.font()
            font.setBold(True)
            verdict.setFont(font)
            value = QTableWidgetItem(tool_value_text(result, mm))
            value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setToolTip(result.text)
            name = QTableWidgetItem(result.name)
            name.setToolTip(kind_hint(result.kind))
            self.tool_table.setItem(row, 0, name)
            self.tool_table.setItem(row, 1, value)
            self.tool_table.setItem(row, 2, verdict)
