"""Record demo GIFs of the app's functions for the user manuals.

Drives the real app on the kiosk display (live camera + tower light) with
scripted taps, grabs the whole screen (so popups and drop-downs are included),
draws finger markers where it taps, and writes docs/gifs/*.gif.

All data the app writes goes to scratch copies (NG log, captures, jobs,
settings), so production data and counters are untouched.

Usage (stop the kiosk service first, it holds the camera and tower light):
    systemctl --user stop wire-harness-detector.service
    DISPLAY=:0 .venv/bin/python tools/record_demo_gifs.py [log settings train]
    systemctl --user start wire-harness-detector.service
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
from app import main_window, paths  # noqa: E402
from app.ng_log import collect_ng_log, ng_stamp_text  # noqa: E402
from app.theme import DARK_INDUSTRIAL_QSS  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QMessageBox, QScrollArea, QStyle, QStyleOptionSlider, QToolButton,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import code39  # noqa: E402

OUT_DIR = ROOT / "docs" / "gifs"
MOCK_PART = "15K45(40)-MM-4FYMX-C3(LF)"
# Leading "P": the app's AIM Code 39 decoder drops the first character of an
# extended-code scan (label-prefix convention from Hioki_data_logger).
MOCK_PAYLOAD = "P" + code39.to_extended(MOCK_PART)
FPS = 8
SCALE = 0.75          # 1024x768 -> 768x576
FINGER_MS = 450       # how long a tap marker stays visible


# ------------------------------------------------------------ data sandbox ---
def redirect_paths(**values: Path) -> None:
    """The app modules import these constants by name, so rebind each copy."""
    for name, module in list(sys.modules.items()):
        if name == "app" or name.startswith("app."):
            for attr, value in values.items():
                if hasattr(module, attr):
                    setattr(module, attr, value)


def sandbox_data() -> Path:
    """Point the app at scratch copies of everything it reads or writes."""
    tmp = Path(tempfile.mkdtemp(prefix="demo_gifs_"))
    for name in ("ng_logs", "captures", "part_configs"):
        if (ROOT / name).exists():
            shutil.copytree(ROOT / name, tmp / name)
    if (ROOT / "app_settings.json").exists():
        shutil.copy(ROOT / "app_settings.json", tmp / "app_settings.json")
    redirect_paths(
        APP_SETTINGS_FILE=tmp / "app_settings.json",
        PART_CONFIGS_DIR=tmp / "part_configs",
        LAST_PART_FILE=tmp / "part_configs" / ".last_part",
        NG_LOGS_DIR=tmp / "ng_logs",
        CAPTURES_DIR=tmp / "captures",
        CAPTURES_IMAGES_DIR=tmp / "captures" / "images",
        CAPTURES_LABELS_DIR=tmp / "captures" / "labels",
    )
    return tmp


# ---------------------------------------------------------------- recorder ---
class Recorder:
    def __init__(self, app: QApplication) -> None:
        self.app = app
        self.screen = QGuiApplication.primaryScreen()
        self.frames: list = []            # [(QImage, bytes-key)]
        self.taps: list = []              # [(QPointF global, t_end)]
        self.touches: list = []           # live finger points during gestures
        self.timer = QTimer()
        self.timer.timeout.connect(self._grab)
        self.recording = False

    # recording
    def start(self) -> None:
        self.frames = []
        self.recording = True
        self.timer.start(int(1000 / FPS))

    def stop(self) -> list:
        self.timer.stop()
        self.recording = False
        return self.frames

    def _grab(self) -> None:
        image = self.screen.grabWindow(0).toImage().convertToFormat(QImage.Format_RGB888)
        now = time.monotonic()
        self.taps = [t for t in self.taps if t[1] > now]
        points = [t[0] for t in self.taps] + list(self.touches)
        if points:
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            for p in points:
                painter.setPen(QPen(QColor(255, 255, 255, 230), 3))
                painter.setBrush(QColor(255, 207, 115, 110))
                painter.drawEllipse(p, 22, 22)
            painter.end()
        image = image.scaled(int(image.width() * SCALE), int(image.height() * SCALE),
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.frames.append(image)

    # event pumping
    def pump(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.app.processEvents()
            time.sleep(0.01)

    # actions
    def _mark(self, widget, local: QPoint) -> None:
        self.taps.append((QPointF(widget.mapToGlobal(local)), time.monotonic() + FINGER_MS / 1000))

    def tap(self, widget, local: QPoint = None, after: float = 0.6) -> None:
        local = local if local is not None else widget.rect().center()
        self._mark(widget, local)
        self.pump(0.15)                    # marker visible just before the press
        QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, local)
        self.pump(after)

    def mark_only(self, widget, local: QPoint = None, after: float = 0.3) -> None:
        """Show a tap marker without a real click (e.g. a native file dialog
        we don't want to open); the caller performs the effect directly."""
        self._mark(widget, local if local is not None else widget.rect().center())
        self.pump(after)

    def tap_tab(self, tabs, index: int, after: float = 1.0) -> None:
        bar = tabs.tabBar()
        self.tap(bar, bar.tabRect(index).center(), after)

    def slide(self, slider, target: int, seconds: float = 1.0) -> None:
        opt = QStyleOptionSlider()
        slider.initStyleOption(opt)
        start = slider.value()
        steps = max(1, int(seconds * 25))
        for i in range(1, steps + 1):
            slider.setValue(round(start + (target - start) * i / steps))
            slider.initStyleOption(opt)
            handle = slider.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, slider)
            self.touches = [QPointF(slider.mapToGlobal(handle.center()))]
            self.pump(seconds / steps)
        self.touches = []
        self.pump(0.4)

    def spin_step(self, spin, up: bool, times: int = 1) -> None:
        local = QPoint(spin.width() - 18, 8 if up else spin.height() - 8)
        for _ in range(times):
            self.tap(spin, local, after=0.35)

    def choose(self, combo, index: int) -> None:
        self.tap(combo, after=0.6)         # opens the drop-down
        view = combo.view()
        rect = view.visualRect(view.model().index(index, 0))
        self._mark(view.viewport(), rect.center())
        self.pump(0.3)
        QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())
        if combo.currentIndex() != index:  # style-dependent popups: fall back
            combo.setCurrentIndex(index)
            combo.hidePopup()
        self.pump(0.6)

    # touch gestures (real QTouchEvents, so ZoomImageView's pinch path runs)
    def pinch(self, widget, center: QPoint, start_gap: int, end_gap: int, seconds: float) -> None:
        device = QTest.createTouchDevice()
        pts = lambda g: (center - QPoint(g // 2, 0), center + QPoint(g // 2, 0))
        a, b = pts(start_gap)
        QTest.touchEvent(widget, device).press(0, a).press(1, b)
        steps = max(2, int(seconds * 25))
        for i in range(1, steps + 1):
            gap = int(start_gap + (end_gap - start_gap) * i / steps)
            a, b = pts(gap)
            QTest.touchEvent(widget, device).move(0, a).move(1, b)
            self.touches = [QPointF(widget.mapToGlobal(a)), QPointF(widget.mapToGlobal(b))]
            self.pump(seconds / steps)
        QTest.touchEvent(widget, device).release(0, a).release(1, b)
        self.touches = []
        self.pump(0.3)

    def drag(self, widget, start: QPoint, delta: QPoint, seconds: float) -> None:
        device = QTest.createTouchDevice()
        QTest.touchEvent(widget, device).press(0, start)
        steps = max(2, int(seconds * 25))
        for i in range(1, steps + 1):
            p = start + QPoint(int(delta.x() * i / steps), int(delta.y() * i / steps))
            QTest.touchEvent(widget, device).move(0, p)
            self.touches = [QPointF(widget.mapToGlobal(p))]
            self.pump(seconds / steps)
        QTest.touchEvent(widget, device).release(0, p)
        self.touches = []
        self.pump(0.3)

    def double_tap(self, widget, local: QPoint) -> None:
        device = QTest.createTouchDevice()
        for _ in range(2):
            self._mark(widget, local)
            QTest.touchEvent(widget, device).press(0, local)
            self.pump(0.05)
            QTest.touchEvent(widget, device).release(0, local)
            self.pump(0.12)
        self.pump(0.8)


# ------------------------------------------------------------------ output ---
def save_gif(frames: list, path: Path, hold_last_ms: int = 1500) -> None:
    """Merge identical consecutive frames (static UI stretches) and write."""
    images, durations, last_key = [], [], None
    frame_ms = int(1000 / FPS)
    for qimage in frames:
        key = bytes(qimage.constBits())
        if key == last_key:
            durations[-1] += frame_ms
            continue
        last_key = key
        pil = Image.frombuffer("RGB", (qimage.width(), qimage.height()), key,
                               "raw", "RGB", qimage.bytesPerLine(), 1)
        images.append(pil.quantize(colors=256, method=Image.Quantize.FASTOCTREE))
        durations.append(frame_ms)
    durations[-1] += hold_last_ms
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True)
    print(f"saved {path.relative_to(ROOT)}: {len(images)} frames, "
          f"{sum(durations) / 1000:.1f}s, {path.stat().st_size / 1e6:.1f} MB")


# ------------------------------------------------------------------ scenes ---
def detection_focus(image_path: Path):
    """(score, (x, y)) for a saved NG frame: detection boxes are drawn in
    saturated class colours, so count/locate strongly saturated pixels."""
    image = Image.open(image_path).convert("HSV").resize((160, 120))
    xs, ys = [], []
    for i, (h, s, v) in enumerate(image.getdata()):
        if s > 150 and v > 150:
            xs.append(i % 160)
            ys.append(i // 160)
    if not xs:
        return 0, (0.5, 0.5)
    return len(xs), (sum(xs) / len(xs) / 160, sum(ys) / len(ys) / 120)


def prepare_log(rec: Recorder, w) -> None:
    """Off camera: run the live inspection briefly so a fresh NG frame (real
    harness, current timestamp) lands in the scratch NG log."""
    before = w.monitor_tab.ng_count
    w.start_stream()
    deadline = time.monotonic() + 40
    while w.monitor_tab.ng_count == before and time.monotonic() < deadline:
        rec.pump(0.3)
    rec.pump(1.0)
    w.stop_stream()
    while w.ctx.worker.isRunning():
        rec.pump(0.2)
    w.log_tab._dirty = True
    print("fresh NG frame:", "yes" if w.monitor_tab.ng_count > before else "no (using existing log)")


def scene_log(rec: Recorder, w) -> None:
    rec.tap_tab(w.tabs, w.log_tab_index, after=1.5)
    thumbs = [b for b in w.tabs.widget(w.log_tab_index).findChildren(QToolButton)
              if not b.isHidden()]
    # Of the thumbnails on screen, open the NG picture with the most visible
    # detections (the newest one may be an almost empty frame).
    shown = []
    for part_name, thumb_paths, _ in collect_ng_log():
        for thumb in thumb_paths[:paths.NG_THUMBS_PER_PART]:
            full = thumb.with_name(thumb.name[: -len(paths.NG_THUMB_SUFFIX)] + ".jpg")
            shown.append((detection_focus(full), ng_stamp_text(thumb), part_name))
    # Prefer frames with a sane timestamp (1970 = saved before the clock synced).
    dated = [item for item in shown if not item[1].startswith("1970")] or shown
    # A frame recorded just now (prepare_log) wins if it shows detections.
    newest = max(dated, key=lambda item: item[1])
    if newest[0][0] > 20:
        (_, focus), stamp, _ = newest
    else:
        (_, focus), stamp, _ = max(dated, key=lambda item: item[0][0])
    target = next(b for b in thumbs if b.text() == stamp)
    w.tabs.widget(w.log_tab_index).findChild(QScrollArea).ensureWidgetVisible(target)
    rec.pump(0.4)
    rec.tap(target, after=1.4)
    view = w.image_overlay.view
    rect = view._image_rect()
    anchor = QPoint(int(rect.x() + focus[0] * rect.width()), int(rect.y() + focus[1] * rect.height()))
    rec.pinch(view, anchor, 90, 230, 1.4)          # zoom in on the detections
    rec.pump(0.8)
    center = view.rect().center()
    to_center = QPoint(int((center.x() - anchor.x()) * 0.7), int((center.y() - anchor.y()) * 0.7))
    rec.drag(view, anchor, to_center, 1.0)         # bring them to the middle
    rec.pump(0.8)
    rec.pinch(view, center, 220, 150, 0.8)         # zoom out a bit
    rec.pump(0.6)
    rec.double_tap(view, view.rect().center())     # back to fit
    rec.pump(0.6)
    rec.tap(w.image_overlay.close_button, after=1.2)


def scene_settings(rec: Recorder, w) -> None:
    rec.tap_tab(w.tabs, 1, after=1.2)
    rec.slide(w.settings_tab.confidence_slider, 75, 1.0)
    rec.slide(w.settings_tab.confidence_slider, 60, 0.8)
    rec.spin_step(w.settings_tab.max_detections_spinbox, up=True, times=2)
    rec.spin_step(w.settings_tab.max_detections_spinbox, up=False, times=2)
    if w.ctx.tower.connected:                         # skip if the tower is unplugged
        rec.tap(w.settings_tab.buzzer_test_button, after=1.2)   # plays one NG alarm cycle
        rec.tap(w.settings_tab.ok_beep_test_button, after=0.8)
    rec.spin_step(w.settings_tab.state_hold_spinbox, up=True, times=2)
    rec.spin_step(w.settings_tab.state_hold_spinbox, up=False, times=2)
    rec.tap_tab(w.tabs, 0, after=2.0)              # back to the live view


def scene_train(rec: Recorder, w) -> None:
    train_index = 3
    rec.tap_tab(w.tabs, train_index, after=1.0)
    # The real button opens a native folder dialog; show the tap, load directly.
    rec.mark_only(w.train_tab.train_select_folder_button, after=0.2)
    w.train_tab.scan_train_folder(paths.CAPTURES_IMAGES_DIR)
    rec.pump(1.0)
    lst = w.train_tab.train_list_widget
    rec.tap(lst.viewport(), lst.visualItemRect(lst.item(2)).center(), after=1.0)

    widget = w.train_tab.train_image_widget
    box = w.train_tab._train_boxes[0]
    draw = widget._draw_rect
    scale = draw.width() / widget.pixmap().width()
    local = QPoint(int(draw.x() + (box["x1"] + box["x2"]) / 2 * scale),
                   int(draw.y() + (box["y1"] + box["y2"]) / 2 * scale))
    rec.tap(widget, local, after=1.0)              # select a box

    combo = w.train_tab.train_class_combo
    current = combo.currentIndex()
    rec.choose(combo, (current + 1) % combo.count())
    rec.tap(w.train_tab.train_confirm_button, after=1.2)     # label file updated
    rec.choose(combo, current)                     # put the class back
    rec.tap(w.train_tab.train_confirm_button, after=1.0)
    rec.tap(w.train_tab.train_next_button, after=0.9)
    rec.tap(w.train_tab.train_next_button, after=1.2)


class MockLabel:
    """Slides a printed Code 39 label into the real camera picture, on the red
    guide line, so the app's own zbar pipeline reads it."""

    def __init__(self) -> None:
        self.t_in = None
        self.t_out = None
        self.image = None

    def show(self) -> None:
        self.t_in, self.t_out = time.monotonic(), None

    def hide(self) -> None:
        self.t_out = time.monotonic()

    def composite(self, frame):
        if self.t_in is None:
            return frame
        fh, fw = frame.shape[:2]
        if self.image is None:
            # Largest whole-pixel module that fits ~90% of the frame: bars stay
            # crisp (no resampling), like a label held close to the lens.
            narrow = max(1, int(fw * 0.9) // (code39.units(MOCK_PAYLOAD) + 24))
            label = code39.render(MOCK_PAYLOAD, MOCK_PART, narrow=narrow,
                                  height=max(60, fh // 6), font_size=max(14, 10 * narrow + 6))
            self.image = np.array(label)[:, :, ::-1].copy()      # RGB -> BGR
        lh, lw = self.image.shape[:2]
        now = time.monotonic()
        slide = min(1.0, (now - self.t_in) / 1.0)
        if self.t_out is not None:
            slide = max(0.0, 1.0 - (now - self.t_out) / 0.8)
        center_x = (fw - lw) // 2
        x = int(fw + (center_x - fw) * (1 - (1 - slide) ** 3))   # ease-out
        y = fh // 2 - (16 + max(60, fh // 6) // 2)               # bars on the red line
        x0, x1 = max(0, x), min(fw, x + lw)
        if x1 > x0:
            frame = frame.copy()
            frame[y:y + lh, x0:x1] = self.image[:, x0 - x:x1 - x]
        return frame


LABEL = MockLabel()


class LabelCamera:
    """cv2.VideoCapture wrapper that adds the mock label to every frame."""

    def __init__(self, capture) -> None:
        self._capture = capture

    def read(self):
        ok, frame = self._capture.read()
        return (ok, LABEL.composite(frame)) if ok else (ok, frame)

    def __getattr__(self, name):
        return getattr(self._capture, name)


def answer_next_question(rec: Recorder, button=QMessageBox.Yes, after: float = 2.0) -> None:
    """Tap a button on the next QMessageBox once it has been readable for a
    moment. Polls from timers, which keep running inside the box's own loop."""
    seen = {"at": None}

    def poll():
        box = QApplication.activeModalWidget()
        if isinstance(box, QMessageBox) and box.button(button) is not None:
            seen["at"] = seen["at"] or time.monotonic()
            if time.monotonic() - seen["at"] >= after:
                target = box.button(button)
                rec._mark(target, target.rect().center())
                QTimer.singleShot(180, target.click)
                return
        QTimer.singleShot(100, poll)

    poll()


def scene_barcode(rec: Recorder, w) -> None:
    real_capture = cv2.VideoCapture
    cv2.VideoCapture = lambda *a, **k: LabelCamera(real_capture(*a, **k))
    try:
        rec.tap_tab(w.tabs, 2, after=1.0)                     # Job Change
        rec.tap(w.job_change_tab.scan_barcode_button, after=2.5)             # live camera opens
        dialog = w._barcode_dialog
        answer_next_question(rec, QMessageBox.Yes, after=2.2)  # "save as new job?" -> Yes
        LABEL.show()
        deadline = time.monotonic() + 15
        while w.ctx.current_part_no != MOCK_PART and time.monotonic() < deadline:
            rec.pump(0.2)
        rec.pump(1.2)
        LABEL.hide()
        rec.pump(1.2)
        rec.tap(dialog.close_button, after=2.2)               # new job tile is active
    finally:
        cv2.VideoCapture = real_capture


PREPARE = {"log": prepare_log}

SCENES = {
    # name: (scene, needs live stream)
    "log": (scene_log, False),
    "settings": (scene_settings, True),
    "train": (scene_train, False),
    "barcode": (scene_barcode, False),
}


def main_entry() -> None:
    wanted = [a for a in sys.argv[1:] if a in SCENES] or list(SCENES)
    scratch = sandbox_data()
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_INDUSTRIAL_QSS)
    window = main_window.MainWindow()
    window.showFullScreen()
    rec = Recorder(app)
    rec.pump(2.0)

    try:
        for name in wanted:
            scene, needs_stream = SCENES[name]
            window.tabs.setCurrentIndex(0)
            if name in PREPARE:
                PREPARE[name](rec, window)
            if needs_stream:
                frames = [0]
                window.ctx.worker.frame_ready.connect(lambda *a: frames.__setitem__(0, frames[0] + 1))
                window.start_stream()
                deadline = time.monotonic() + 60
                while frames[0] < 20 and time.monotonic() < deadline:
                    rec.pump(0.3)
            rec.pump(0.8)
            rec.start()
            rec.pump(1.2 if needs_stream else 0.6)
            scene(rec, window)
            captured = rec.stop()
            if needs_stream:
                window.stop_stream()
                while window.ctx.worker.isRunning():
                    rec.pump(0.2)
            save_gif(captured, OUT_DIR / f"{name}.gif")
    finally:
        window.stop_stream()
        window.ctx.worker.wait()
        window.ctx.tower.shutdown()
        window.close()
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main_entry()
