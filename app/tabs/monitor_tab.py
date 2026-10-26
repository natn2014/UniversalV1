import html
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QRect, QSize, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..context import AppContext
from ..judgement import JudgementLatch
from ..ng_log import save_ng_image
from ..paths import CAPTURES_IMAGES_DIR, CAPTURES_LABELS_DIR, NO_PART_NAME
from ..roi import detections_in_crop, roi_to_pixels
from ..theme import RUN_BUTTON_STYLES, action_button_qss, gloss_gradient, make_icon
from ..widgets.layout import bottom_bar, top_bar
from ..widgets.perspective_dialog import PerspectiveDialog
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
        self._current_pixmap = None
        self._current_frame_image: Optional[QImage] = None
        self._current_frame_detections: list = []
        self._latch = JudgementLatch()
        self._pending_ng_save: bool = False
        self._stop_requested: bool = False
        self._camera_lost: bool = False
        self._tower_lost: bool = False

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

        self.monitor_info_label = QLabel()
        self.monitor_info_label.setObjectName("monitorInfo")
        self.monitor_info_label.setTextFormat(Qt.RichText)
        self._refresh_monitor_info()
        ctx.part_changed.connect(self._refresh_monitor_info)
        ctx.model_path_changed.connect(self._refresh_monitor_info)

        self.video_label = VideoView()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.roi_drawn.connect(self._on_roi_drawn)

        self.roi_button = QPushButton("Set ROI")
        self.roi_button.setIcon(make_icon("roi", "#e0e0e0"))
        self.roi_button.setIconSize(ICON_SIZE)
        self.roi_button.setFixedHeight(TOOLBAR_HEIGHT)
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(self._on_roi_button_toggled)
        self.roi_clear_button = QPushButton("Clear ROI")
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

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Class", "Count"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumWidth(220)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self._build_layout()
        self._update_judgement_banner("YELLOW")

    def _build_layout(self) -> None:
        # Layout follows the Keyence reference: info strip + judgement pill on
        # top, framed image with a light side panel, action bar at the bottom
        # with Run on the far right.
        top, top_layout = top_bar()
        top_layout.addWidget(self.monitor_info_label)
        top_layout.addStretch(1)
        top_layout.addWidget(self.judgement_banner)

        video_frame = QFrame()
        video_frame.setObjectName("videoFrame")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(2, 2, 2, 2)
        video_layout.addWidget(self.video_label)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        panel_header = QLabel("Detections")
        panel_header.setObjectName("panelHeader")
        side_layout.addWidget(panel_header)
        side_layout.addWidget(self.table, 1)
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
        parts = [f"Part: {html.escape(part)}", f"Model: {html.escape(model)}"]
        if self._ctx.filters.perspective:
            parts.append("Top View")
        if self._tower_lost:
            parts.append('<span style="color:#ff5a4a;">Tower light disconnected</span>')
        self.monitor_info_label.setText(sep.join(parts))

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
            "GREEN": ("OK", "#2e7d32", "#ffffff"),
            "RED": ("NG", "#c0392b", "#ffffff"),
            "YELLOW": ("STANDBY", "#f4d03f", "#1a1a1a"),
        }[state]
        self.judgement_banner.setText(text)
        gloss = gloss_gradient(QColor(bg).lighter(135).name(), bg, QColor(bg).darker(125).name())
        self.judgement_banner.setStyleSheet(
            f"QLabel#judgementBanner {{ background: {gloss}; color: {fg}; }}"
        )

    def _reset_judgement(self) -> None:
        self._latch.reset()
        self._update_judgement_banner("YELLOW")

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
        frame = self._ctx.worker.last_raw_frame
        if frame is None or not self._ctx.worker.isRunning():
            self._ctx.status.emit("Start the camera first, then set the Top View corners.")
            return
        self.roi_button.setChecked(False)
        dialog = PerspectiveDialog(frame.copy(), self._ctx.filters.perspective, parent=self)
        if dialog.exec() != PerspectiveDialog.Accepted:
            return
        self._ctx.set_perspective(dialog.quad, save_to_job=True)
        saved = f" (saved to job {self._ctx.current_part_no})" if self._ctx.current_part_no else ""
        state = "set" if dialog.quad else "removed"
        self._ctx.status.emit(f"Top View {state}{saved}.")

    # --- Detection area (ROI) -------------------------------------------

    @Slot(bool)
    def _on_roi_button_toggled(self, checked: bool) -> None:
        if checked and self._current_frame_image is None:
            self._ctx.status.emit("Start the camera first, then draw the ROI on the image.")
            self.roi_button.setChecked(False)
            return
        self.video_label.set_roi_mode(checked)

    @Slot(tuple)
    def _on_roi_drawn(self, roi: tuple) -> None:
        self.roi_button.setChecked(False)
        self._ctx.set_roi(roi, save_to_job=True)
        saved = f" (saved to job {self._ctx.current_part_no})" if self._ctx.current_part_no else ""
        self._ctx.status.emit(f"Detection area set{saved}.")

    @Slot(object)
    def _on_roi_changed(self, roi) -> None:
        self.roi_clear_button.setEnabled(roi is not None)
        # While stopped, redraw the last frame so the new area shows at once.
        if not self._ctx.worker.isRunning() and self._current_frame_image is not None:
            self._current_frame_detections = []
            self.video_label.setPixmap(self._render_frame(self._current_frame_image, []))

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
        pen = QPen(QColor("#ffd54f"))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        painter.drawText(x1 + 4, y1 + 14, "ROI")

    # --- Stream control -------------------------------------------------

    @Slot()
    def toggle_stream(self) -> None:
        if self._ctx.worker.isRunning():
            self.stop_stream()
        else:
            self.start_stream()

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
        elif running:
            text, icon, style = "Stop", "stop", RUN_BUTTON_STYLES["stop"]
            status, dot = "Running", "#2ecc71"
        else:
            text, icon, style = "Start", "play", RUN_BUTTON_STYLES["start"]
            status, dot = "Stopped", "#9aa5b1"

        self.run_button.setEnabled(not stopping)
        self.run_button.setText(text)
        self.run_button.setIcon(make_icon(icon))
        self.run_button.setStyleSheet(action_button_qss(*style))
        self.run_status_label.setText(
            f'<span style="color:{dot}; font-size:16pt;">●</span>'
            f'&nbsp;<span style="color:{dot}; font-weight:600;">{status}</span>'
        )

    @Slot()
    def start_stream(self) -> None:
        ctx = self._ctx
        if ctx.worker.isRunning():
            ctx.status.emit("Already running.")
            return
        if ctx.training_active:
            ctx.status.emit("Training in progress. Wait for it to finish.")
            return
        if ctx.camera_index is None:
            ctx.status.emit("Select a valid camera.")
            return
        ctx.worker.set_camera_index(int(ctx.camera_index))
        self._reset_judgement()
        ctx.worker.start()
        self._refresh_run_button()
        if not ctx.tower.connected:
            ctx.tower.connect_device()

    @Slot()
    def stop_stream(self) -> None:
        # Non-blocking: the worker can take seconds to notice (first inference
        # warms up the GPU), so the button shows "Stopping..." until the
        # worker's finished signal flips it back to Start.
        if self._ctx.worker.isRunning():
            self._stop_requested = True
            self._ctx.worker.stop()
        self._ctx.tower.shutdown()
        self._reset_judgement()
        self._refresh_run_button()

    @Slot()
    def capture_frame(self) -> None:
        if self._current_frame_image is None:
            self._ctx.status.emit("No frame to capture.")
            return

        CAPTURES_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        CAPTURES_LABELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = self._ctx.model_path
        model_stem = Path(model_path).stem if model_path else "capture"
        stem = f"{model_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        image_path = CAPTURES_IMAGES_DIR / f"{stem}.jpg"
        label_path = CAPTURES_LABELS_DIR / f"{stem}.txt"

        # With an ROI, save what the model sees (the area only), so training
        # images match inference input.
        image = self._current_frame_image
        detections = self._current_frame_detections
        box = self._roi_pixels(image)
        if box is not None:
            x1, y1, x2, y2 = box
            image = image.copy(x1, y1, x2 - x1, y2 - y1)
            detections = detections_in_crop(detections, box)

        pixmap = QPixmap.fromImage(image)
        if not pixmap.save(str(image_path), "JPG"):
            self._ctx.status.emit("Frame save failed.")
            return

        img_w = image.width()
        img_h = image.height()
        boxes = []
        for det in detections:
            box = pixel_box_to_yolo(det["x1"], det["y1"], det["x2"], det["y2"], img_w, img_h)
            box["cls_id"] = det["cls_id"]
            boxes.append(box)
        write_yolo_label_file(label_path, boxes)

        self._ctx.status.emit(f"Captured: {image_path.name} ({len(boxes)} boxes)")

    # --- Per-frame processing -------------------------------------------

    @Slot(QImage, list)
    def on_frame(self, image: QImage, detections: list) -> None:
        detections = self._ctx.filters.apply(detections)  # Use filtered for drawing and table
        self._current_frame_image = image
        self._current_frame_detections = detections

        # One cycle = each time the held judgement becomes OK or NG.
        raw_state = self._ctx.filters.judge(detections)
        if self._latch.update(raw_state, self._ctx.settings["state_hold_ms"]):
            self._count_cycle(raw_state)
        state = self._latch.state
        self._update_judgement_banner(state)
        self._ctx.tower.update_state(state)

        pixmap = self._render_frame(image, detections)
        self._current_pixmap = pixmap
        self.video_label.setPixmap(pixmap)
        self._update_class_table(detections)
        if self._pending_ng_save:
            self._pending_ng_save = False
            self._save_ng_image(pixmap)

    def _render_frame(self, image: QImage, detections: list) -> QPixmap:
        """Frame with the ROI and detection boxes drawn on it (also what the
        NG log saves)."""
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        self._draw_roi(painter, image)
        if detections:
            # Assign colors to each detection based on class name
            for det in detections:
                class_name = det.get("class_name", "unknown")
                if class_name not in self._class_colors:
                    hue = (len(self._class_colors) * 37) % 360
                    self._class_colors[class_name] = QColor.fromHsv(hue, 200, 255)
                det["color"] = self._class_colors[class_name]

            pen = QPen()
            pen.setWidth(2)
            for det in detections:
                color = det.get("color", Qt.green)
                pen.setColor(color)
                painter.setPen(pen)
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.drawText(x1, max(0, y1 - 6), det["label"])
        painter.end()
        return pixmap

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
