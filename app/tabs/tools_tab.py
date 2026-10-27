"""Tools tab: build a job's rule-based vision tools on a frozen camera frame.

The camera streams live here, with every tool's area and result drawn on it.
Tap Snapshot to freeze the frame (tap it again, now "Resume Live", to go back
to the stream), Add Tool to pick a kind and draw its area, then Teach / set
the limits against the value. Every
change is written to the job through `ctx.set_tools(..., save_to_job=True)`.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..autocapture import qimage_to_array
from ..context import AppContext
from ..perspective import PerspectiveWarp
from ..roi import normalize_roi, roi_to_pixels
from ..theme import NG_RED, OK_GREEN, RUN_BUTTON_STYLES, STANDBY_YELLOW, action_button_qss, gloss_gradient
from ..vision.base import Tool, ToolResult, create
from ..vision.calibration import calibrate_from_width
from ..vision.runner import run_tools
from ..vision.total import total_state
from ..widgets.layout import bottom_bar, make_card, top_bar
from ..widgets.remove_job_dialog import RemoveJobDialog
from ..widgets.tool_overlay import draw_overlay, tool_value_text
from ..widgets.tool_params import HSV_KINDS, TOUCH_HEIGHT, AddToolDialog, ToolParamsForm, kind_title
from ..widgets.video_view import VideoView

DEFAULT_ROI = (0.3, 0.3, 0.7, 0.7)
# Kinds that cannot work before they are taught; they learn from their ROI
# right after it is drawn when a frame is frozen.
AUTO_TEACH = {"position", "color_area", "color_average", "outline"}
SELECTED_COLOR = QColor("#f0a330")
IDLE_COLOR = QColor("#9aa3ad")
LIMIT_PERCENT = 10.0


class ToolsTab(QWidget):
    """Frozen frame with every tool's area on the left; tool list, buttons,
    limits and parameters on the right."""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._frame: Optional[np.ndarray] = None  # frozen BGR frame; None = live stream
        self._last_image: Optional[QImage] = None  # newest frame the worker delivered
        self._live_results: List[ToolResult] = []  # the worker's tool results for that frame
        self._results: Dict[str, ToolResult] = {}
        self._mode: Optional[str] = None  # add / set_roi / teach_box / calibrate / pick_color
        self._pending_kind: Optional[str] = None
        self._pending_mm: float = 0.0
        self._busy = False  # ignore our own tools_changed echo
        self._loading = False  # filling widgets from a tool, not user edits

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self.save)

        self._build_widgets()
        self._build_layout()

        ctx.worker.frame_ready.connect(self._on_worker_frame)
        ctx.tools_changed.connect(self._on_tools_changed)
        ctx.part_changed.connect(lambda _: self._refresh_info())
        ctx.mm_per_px_changed.connect(lambda _: self._refresh_all())
        self._rebuild_list()
        self._refresh_all()

    # --- Widgets --------------------------------------------------------

    def _button(self, text: str, handler, height: int = TOUCH_HEIGHT) -> QPushButton:
        button = QPushButton(text)
        button.setMinimumHeight(height)
        button.clicked.connect(handler)
        return button

    def _build_widgets(self) -> None:
        self.info_label = QLabel()
        self.info_label.setObjectName("monitorInfo")
        self.total_pill = QLabel("STANDBY")
        self.total_pill.setObjectName("judgementBanner")
        self.total_pill.setAlignment(Qt.AlignCenter)
        self.total_pill.setFixedSize(130, 36)

        self.view = VideoView()
        self.view.setMinimumSize(520, 300)
        self.view.roi_drawn.connect(self._on_roi_drawn)
        self.view.point_picked.connect(self._on_point_picked)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Tool", "Kind", "Value", "Result"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(TOUCH_HEIGHT)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(190)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.currentCellChanged.connect(lambda row, *_: self._on_selected(row))

        self.teach_button = self._button("Teach", self._on_teach_clicked)
        self.roi_button = self._button("Draw ROI", self._on_draw_roi_clicked)
        self.roi_button.setCheckable(True)
        self.pick_color_button = self._button("Pick Color", self._on_pick_color_clicked)
        self.pick_color_button.setCheckable(True)
        self.delete_button = self._button("Delete", self._on_delete_clicked)
        self.delete_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["stop"]))
        self.gate_button = self._button("", self._on_gate_clicked)
        self.calibrate_button = self._button("Calibrate mm", self._on_calibrate_clicked)

        self.live_label = QLabel("—")
        self.live_label.setObjectName("currentPart")
        self.low_check, self.low_spin = self._limit_widgets("Low", self._on_limit_changed)
        self.high_check, self.high_spin = self._limit_widgets("High", self._on_limit_changed)
        self.percent_button = self._button(f"Set from current ±{LIMIT_PERCENT:.0f}%", self._on_percent_clicked)
        self.enabled_check = QCheckBox("Enabled")
        self.follow_check = QCheckBox("Follow position")
        for check in (self.enabled_check, self.follow_check):
            check.setMinimumHeight(TOUCH_HEIGHT)
            check.toggled.connect(self._on_flags_changed)
        self.form = ToolParamsForm()
        self.form.changed.connect(self._on_params_changed)

        self.snapshot_button = self._button("Snapshot", self.toggle_snapshot, height=44)
        self.snapshot_button.setMinimumWidth(150)
        self.add_button = self._button("Add Tool", self._on_add_clicked, height=44)
        self.add_button.setMinimumWidth(170)
        self.add_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.hint_label = QLabel()
        self.hint_label.setObjectName("hint")
        self.hint_label.setWordWrap(True)

    def _limit_widgets(self, caption: str, handler):
        check = QCheckBox(caption)
        check.setMinimumHeight(TOUCH_HEIGHT)
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-1e9, 1e9)
        spin.setMinimumHeight(TOUCH_HEIGHT)
        check.toggled.connect(handler)
        spin.valueChanged.connect(handler)
        return check, spin

    def _build_layout(self) -> None:
        top, top_layout = top_bar()
        top_layout.addWidget(self.info_label)
        top_layout.addStretch(1)
        top_layout.addWidget(self.total_pill)

        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(2, 2, 2, 2)
        video_layout.addWidget(self.view)

        list_panel = QFrame()
        list_panel.setObjectName("sidePanel")
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(0)
        header = QLabel("Tools")
        header.setObjectName("panelHeader")
        list_layout.addWidget(header)
        list_layout.addWidget(self.table)

        buttons = QGridLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.teach_button, 0, 0)
        buttons.addWidget(self.roi_button, 0, 1)
        buttons.addWidget(self.pick_color_button, 1, 0)
        buttons.addWidget(self.delete_button, 1, 1)
        buttons.addWidget(self.gate_button, 2, 0)
        buttons.addWidget(self.calibrate_button, 2, 1)

        limits_card, limits_body = make_card("Limits (OK range)")
        limits_body.addWidget(self.live_label)
        for check, spin in ((self.low_check, self.low_spin), (self.high_check, self.high_spin)):
            limit_row = QHBoxLayout()
            limit_row.addWidget(check)
            limit_row.addWidget(spin, 1)
            limits_body.addLayout(limit_row)
        limits_body.addWidget(self.percent_button)

        params_card, params_body = make_card("Parameters")
        flag_row = QHBoxLayout()
        flag_row.addWidget(self.enabled_check)
        flag_row.addWidget(self.follow_check)
        params_body.addLayout(flag_row)
        params_body.addWidget(self.form)

        right = QWidget()
        right.setObjectName("clearBox")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 6, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(list_panel)
        right_layout.addLayout(buttons)
        right_layout.addWidget(limits_card)
        right_layout.addWidget(params_card)
        right_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(right)
        scroll.viewport().setObjectName("clearBox")
        scroll.setFixedWidth(430)

        content = QHBoxLayout()
        content.addWidget(video_frame, 1)
        content.addWidget(scroll)

        bottom, bottom_layout = bottom_bar()
        bottom_layout.addWidget(self.snapshot_button)
        bottom_layout.addSpacing(10)
        bottom_layout.addWidget(self.hint_label, 1)
        bottom_layout.addWidget(self.add_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(content, 1)
        layout.addWidget(bottom)

    # --- Frame ----------------------------------------------------------

    @Slot(QImage, list, list)
    def _on_worker_frame(self, image: QImage, detections: list, tool_results: list) -> None:
        first = self._last_image is None
        self._last_image = image
        self._live_results = tool_results
        if first:
            self._refresh_info()  # hint: "Waiting for the camera" -> "Live"
        # Live: show the worker's own results; the tools are not run twice.
        if self._frame is None and self.isVisible():
            self._show_results(tool_results)
            self._render()

    @property
    def frozen(self) -> bool:
        return self._frame is not None

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        """Freezes a BGR frame (the Monitor's, after the top-view warp);
        None goes back to the live stream."""
        self._frame = frame
        self.snapshot_button.setText("Resume Live" if frame is not None else "Snapshot")
        self.snapshot_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]) if frame is not None else "")
        self._refresh_all()

    @Slot()
    def toggle_snapshot(self) -> None:
        if self.frozen:
            self.resume_live()
        else:
            self.snapshot()

    def resume_live(self) -> None:
        self._exit_mode()
        self.set_frame(None)

    @Slot()
    def snapshot(self) -> None:
        """Freezes the newest frame of the running camera."""
        if self._last_image is not None:
            frame = cv2.cvtColor(qimage_to_array(self._last_image), cv2.COLOR_RGB2BGR)
        else:
            raw = self._ctx.worker.last_raw_frame
            if raw is None:
                self._ctx.status.emit("No camera frame yet: start the camera on the Monitor tab first.")
                return
            quad = self._ctx.filters.perspective
            frame = PerspectiveWarp(quad).apply(raw) if quad else raw.copy()
        self.set_frame(frame)

    # --- Tools ----------------------------------------------------------

    @property
    def tools(self) -> List[Tool]:
        return self._ctx.tools

    def selected_index(self) -> int:
        return self.table.currentRow()

    def selected_tool(self) -> Optional[Tool]:
        i = self.selected_index()
        return self.tools[i] if 0 <= i < len(self.tools) else None

    def _unique_name(self, base: str) -> str:
        names = {t.name for t in self.tools}
        n = 1
        while f"{base} {n}" in names:
            n += 1
        return f"{base} {n}"

    def add_tool(self, kind: str, roi=None, name: Optional[str] = None) -> Tool:
        """Adds a tool, learns it from its ROI when it needs that, and saves."""
        roi = normalize_roi(roi) or DEFAULT_ROI
        tool = create(kind, name=name or self._unique_name(kind_title(kind)), roi=roi)
        if self._frame is not None and kind in AUTO_TEACH:
            tool.teach(self._frame)
        self._set_tools(self.tools + [tool])
        self.table.selectRow(len(self.tools) - 1)
        return tool

    def delete_selected(self) -> None:
        i = self.selected_index()
        if 0 <= i < len(self.tools):
            self._set_tools(self.tools[:i] + self.tools[i + 1:])

    def teach_selected(self, **kw) -> bool:
        tool = self.selected_tool()
        if tool is not None and self._frame is None and self._last_image is not None:
            self.snapshot()  # teach on a still frame, then show it frozen
        if tool is None or self._frame is None:
            self._ctx.status.emit("Select a tool, with the camera running, first.")
            return False
        try:
            tool.teach(self._frame, **kw)
        except Exception as exc:
            self._ctx.status.emit(f"Teach failed: {exc}")
            return False
        self._set_tools(list(self.tools))
        return True

    def set_roi_selected(self, roi) -> None:
        tool, roi = self.selected_tool(), normalize_roi(roi)
        if tool is not None and roi is not None:
            tool.roi = roi
            self._set_tools(list(self.tools))

    def set_limits(self, low: Optional[float], high: Optional[float]) -> None:
        tool = self.selected_tool()
        if tool is None:
            return
        tool.low, tool.high = low, high
        self._load_limits(tool)
        self._set_tools(list(self.tools))

    def calibrate(self, roi, known_mm: float, direction: Optional[str] = None) -> Optional[float]:
        """mm per px from a width of `known_mm` inside `roi` on the frozen frame."""
        roi = normalize_roi(roi)
        if self._frame is None or roi is None:
            return None
        if direction is None:
            direction = "x" if (roi[2] - roi[0]) * self._frame.shape[1] >= (roi[3] - roi[1]) * self._frame.shape[0] else "y"
        mm = calibrate_from_width(self._frame, roi, known_mm, direction)
        if mm is None:
            self._ctx.status.emit("Calibration failed: no width found in that area.")
            return None
        self._ctx.set_mm_per_px(mm, save_to_job=True)
        self._ctx.status.emit(f"Calibrated: {mm:.5f} mm/px")
        return mm

    def _set_tools(self, tools: List[Tool]) -> None:
        """Every structural change: tell the context (and the job file) now."""
        self._save_timer.stop()
        self._busy = True
        try:
            self._ctx.set_tools(tools, save_to_job=True)
        finally:
            self._busy = False
        self._rebuild_list()
        self._refresh_all()

    @Slot()
    def save(self) -> None:
        """Writes the current tools (the debounced path for spin-box edits)."""
        self._busy = True
        try:
            self._ctx.set_tools(list(self.tools), save_to_job=True)
        finally:
            self._busy = False

    def _schedule_save(self) -> None:
        self._save_timer.start()

    @Slot(list)
    def _on_tools_changed(self, tools: list) -> None:
        if self._busy:
            return
        self._rebuild_list()
        self._refresh_all()

    # --- List and selection --------------------------------------------

    def _rebuild_list(self) -> None:
        keep = self.selected_index()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.tools))
        for row, tool in enumerate(self.tools):
            self.table.setItem(row, 0, QTableWidgetItem(tool.name))
            self.table.setItem(row, 1, QTableWidgetItem(kind_title(tool.kind)))
            self.table.setItem(row, 2, QTableWidgetItem("—"))
            self.table.setItem(row, 3, QTableWidgetItem("—"))
        if self.tools:
            self.table.selectRow(min(max(keep, 0), len(self.tools) - 1))
        self.table.blockSignals(False)
        self._on_selected(self.selected_index())

    def _on_selected(self, row: int) -> None:
        tool = self.selected_tool()
        self._loading = True
        try:
            self.form.set_tool(tool)
            if tool is not None:
                self.enabled_check.setChecked(tool.enabled)
                self.follow_check.setChecked(tool.follow_position)
                self._load_limits(tool)
        finally:
            self._loading = False
        has = tool is not None
        for widget in (self.teach_button, self.roi_button, self.delete_button, self.percent_button,
                       self.low_check, self.high_check, self.low_spin, self.high_spin,
                       self.enabled_check, self.follow_check):
            widget.setEnabled(has)
        self.pick_color_button.setVisible(has and tool.kind in HSV_KINDS)
        self.pick_color_button.setEnabled(has)
        self._exit_mode()
        self._render()

    def _load_limits(self, tool: Tool) -> None:
        was, self._loading = self._loading, True
        try:
            for check, spin, value in ((self.low_check, self.low_spin, tool.low),
                                       (self.high_check, self.high_spin, tool.high)):
                check.setChecked(value is not None)
                spin.setEnabled(value is not None)
                spin.setValue(value if value is not None else 0.0)
        finally:
            self._loading = was

    # --- Running the tools on the frozen frame -------------------------

    def _refresh_all(self) -> None:
        self._refresh_info()
        self._run()
        self._render()

    def _run(self) -> None:
        """Frozen: runs the tools on the frame. Live: shows the worker's
        latest results (it already runs the tools on every frame)."""
        if self._frame is None:
            self._show_results(self._live_results if self._last_image is not None else [])
            return
        results: List[ToolResult] = []
        if self.tools:
            results = run_tools(self._frame, self.tools, self._ctx.mm_per_px, self._ctx.current_part_no)
        self._show_results(results)

    def _show_results(self, results: List[ToolResult]) -> None:
        self._results = {r.name: r for r in results}
        mm = bool(self._ctx.mm_per_px)
        for row, tool in enumerate(self.tools):
            res = self._results.get(tool.name)
            value = QTableWidgetItem(tool_value_text(res, mm) if res else "—")
            value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pending = res is not None and res.extra.get("pending")
            verdict = QTableWidgetItem("—" if res is None else "…" if pending else ("OK" if res.ok else "NG"))
            verdict.setTextAlignment(Qt.AlignCenter)
            if pending:
                verdict.setBackground(QBrush(QColor(STANDBY_YELLOW)))
                verdict.setForeground(QBrush(QColor("#1a1a1a")))
            elif res is not None:
                verdict.setBackground(QBrush(QColor(OK_GREEN if res.ok else NG_RED)))
                verdict.setForeground(QBrush(QColor("#ffffff")))
            self.table.setItem(row, 2, value)
            self.table.setItem(row, 3, verdict)
        tool = self.selected_tool()
        res = self._results.get(tool.name) if tool else None
        self.live_label.setText("Live value: —" if res is None else f"Live value: {tool_value_text(res, bool(self._ctx.mm_per_px))}"
                                + (f"   {res.text}" if res.text and res.kind not in ("ocr", "code") else ""))
        self._update_total(results)

    def _update_total(self, results: List[ToolResult]) -> None:
        if not results:
            state = "YELLOW"
        else:
            found = all(r.ok for r in results if r.kind == "position")
            state = total_state(results, None, self._ctx.tools_gate, found)
        text, bg, fg = {"GREEN": ("OK", OK_GREEN, "#ffffff"), "RED": ("NG", NG_RED, "#ffffff"),
                        "YELLOW": ("STANDBY", STANDBY_YELLOW, "#1a1a1a")}[state]
        gloss = gloss_gradient(QColor(bg).lighter(135).name(), bg, QColor(bg).darker(125).name())
        self.total_pill.setText(text)
        self.total_pill.setStyleSheet(f"QLabel#judgementBanner {{ background: {gloss}; color: {fg}; }}")

    def _refresh_info(self) -> None:
        mm = self._ctx.mm_per_px
        calibration = f"{mm:.4f} mm/px" if mm else "not calibrated"
        part = self._ctx.current_part_no or "—"
        self.info_label.setText(f"Part: {part}   ·   Tools: {len(self.tools)}   ·   Scale: {calibration}")
        self.gate_button.setText(f"Gate: {self._ctx.tools_gate}")
        if self._mode is None:
            if self._frame is not None:
                hint = "Frozen frame  ·  Select a tool to teach it, set its limits or edit its parameters"
            elif self._last_image is None:
                hint = "Waiting for the camera  ·  Add Tool, then drag its area"
            else:
                hint = "Live  ·  Snapshot freezes the frame  ·  Add Tool, then drag its area"
            if any(t.SLOW for t in self.tools) and not any(t.kind == "position" for t in self.tools):
                hint += "  ·  Add a Position tool so OCR / code results are matched to each part"
            self.hint_label.setText(hint)

    def _render(self) -> None:
        if self._frame is not None:
            h, w = self._frame.shape[:2]
            rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
            pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        elif self._last_image is not None:
            pixmap = QPixmap.fromImage(self._last_image)
            w, h = pixmap.width(), pixmap.height()
        else:
            self.view.setPixmap(QPixmap())
            return
        painter = QPainter(pixmap)
        for res in self._results.values():
            draw_overlay(painter, res.overlay)
        selected = self.selected_tool()
        for tool in self.tools:
            x1, y1, x2, y2 = roi_to_pixels(tool.roi, w, h)
            is_sel = tool is selected
            pen = QPen(SELECTED_COLOR if is_sel else IDLE_COLOR)
            pen.setWidth(3 if is_sel else 1)
            pen.setStyle(Qt.SolidLine if is_sel else Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1 + 4, y2 - 5, tool.name)
        painter.end()
        self.view.setPixmap(pixmap)

    # --- Edits ----------------------------------------------------------

    def _edited(self) -> None:
        """A value changed in the form: re-run now, save shortly."""
        self._run()
        self._render()
        self._schedule_save()

    def _on_params_changed(self) -> None:
        if not self._loading:
            self._edited()

    def _on_flags_changed(self, _=None) -> None:
        tool = self.selected_tool()
        if tool is None or self._loading:
            return
        tool.enabled = self.enabled_check.isChecked()
        tool.follow_position = self.follow_check.isChecked()
        self._edited()

    def _on_limit_changed(self, _=None) -> None:
        tool = self.selected_tool()
        if tool is None or self._loading:
            return
        self.low_spin.setEnabled(self.low_check.isChecked())
        self.high_spin.setEnabled(self.high_check.isChecked())
        tool.low = self.low_spin.value() if self.low_check.isChecked() else None
        tool.high = self.high_spin.value() if self.high_check.isChecked() else None
        self._edited()

    def _on_percent_clicked(self) -> None:
        tool = self.selected_tool()
        res = self._results.get(tool.name) if tool else None
        if res is None:
            self._ctx.status.emit("No value yet: start the camera or take a snapshot first.")
            return
        a, b = res.value * (1 - LIMIT_PERCENT / 100), res.value * (1 + LIMIT_PERCENT / 100)
        self.set_limits(min(a, b), max(a, b))

    # --- Modes: drawing on the image ----------------------------------

    def _enter_mode(self, mode: str, hint: str, point: bool = False) -> bool:
        if self._frame is None and self._last_image is not None:
            self.snapshot()  # drawing and teaching need a still frame: freeze the live one
        if self._frame is None:
            self._ctx.status.emit("No camera frame yet: start the camera on the Monitor tab first.")
            self.roi_button.setChecked(False)
            self.pick_color_button.setChecked(False)
            return False
        self._exit_mode()
        self._mode = mode
        self.hint_label.setText(hint)
        for name, button in (("set_roi", self.roi_button), ("pick_color", self.pick_color_button)):
            button.blockSignals(True)
            button.setChecked(mode == name)
            button.blockSignals(False)
        if point:
            self.view.set_point_mode(True)
        else:
            self.view.set_roi_mode(True, hint)
        return True

    def _exit_mode(self) -> None:
        self._mode = None
        self._pending_kind = None
        self.view.set_roi_mode(False)
        self.view.set_point_mode(False)
        for button in (self.roi_button, self.pick_color_button):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        self._refresh_info()

    def _on_add_clicked(self) -> None:
        dialog = AddToolDialog(self)
        if dialog.exec() != QDialog.Accepted or dialog.selected_kind is None:
            return
        self.begin_add(dialog.selected_kind)

    def begin_add(self, kind: str) -> None:
        """Asks the user to draw the new tool's area on the frozen frame."""
        if self._enter_mode("add", f"Drag the {kind_title(kind)} area on the image"):
            self._pending_kind = kind

    def _on_draw_roi_clicked(self, checked: bool) -> None:
        if not checked:
            self._exit_mode()
        elif self.selected_tool() is not None:
            if not self._enter_mode("set_roi", "Drag the new area for this tool on the image"):
                self.roi_button.setChecked(False)

    def _on_teach_clicked(self) -> None:
        tool = self.selected_tool()
        if tool is None:
            return
        if tool.kind == "position" and tool.mode == "2d":
            self._enter_mode("teach_box", "Drag the template box (a feature of the part) inside the search area")
        else:
            self.teach_selected()

    def _on_pick_color_clicked(self, checked: bool) -> None:
        if not checked:
            self._exit_mode()
        elif not self._enter_mode("pick_color", "Tap the color to learn on the image", point=True):
            self.pick_color_button.setChecked(False)

    def _on_calibrate_clicked(self) -> None:
        mm, ok = QInputDialog.getDouble(self, "Calibrate mm", "Known size of the feature (mm):", 5.0, 0.01, 10000.0, 2)
        if ok and mm > 0:
            self._pending_mm = mm
            self._enter_mode("calibrate", "Drag across the feature of known size (edge to edge)")

    def _on_roi_drawn(self, roi: tuple) -> None:
        mode, kind = self._mode, self._pending_kind
        self._exit_mode()
        if mode == "add" and kind:
            self.add_tool(kind, roi)
        elif mode == "set_roi":
            self.set_roi_selected(roi)
        elif mode == "teach_box":
            self.teach_selected(template_roi=roi)
        elif mode == "calibrate":
            self.calibrate(roi, self._pending_mm)

    def _on_point_picked(self, point: tuple) -> None:
        if self._mode != "pick_color" or self._frame is None:
            return
        self._exit_mode()
        h, w = self._frame.shape[:2]
        self.teach_selected(point=(int(point[0] * w), int(point[1] * h)))

    def _on_gate_clicked(self) -> None:
        self._ctx.tools_gate = "OR" if self._ctx.tools_gate == "AND" else "AND"
        self._set_tools(list(self.tools))

    def _on_delete_clicked(self) -> None:
        tool = self.selected_tool()
        if tool is None:
            return
        dialog = RemoveJobDialog(
            [t.name for t in self.tools], tool.name, parent=self, title="Delete Tool", item_label="Tool",
            warning=lambda name: f"This removes the tool \"{name}\" and its taught data from the job. "
                                 "This cannot be undone.")
        if dialog.exec() != QDialog.Accepted:
            return
        name = dialog.selected_part()
        self.table.selectRow(next((i for i, t in enumerate(self.tools) if t.name == name), self.selected_index()))
        self.delete_selected()
