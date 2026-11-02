import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Seconds a frame grab may wait before failing. OpenCV's default (10 s)
# would make an unplugged camera take 10 s per attempt to notice. Must be
# set before OpenCV opens a camera; main.py sets it before any import too.
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "2")

import cv2  # noqa: E402
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .image_adjust import ImageAdjust
from .marking.short_name import ShortNameReader, pad_box, valid_short_name
from .optional_deps import YOLO
from .perspective import PerspectiveWarp, Quad
from .roi import Roi, offset_detections, roi_to_pixels
from .vision.runner import run_tools_with_offset
from .vision.ocr import get_engine
from .vision.slow_runner import SlowToolRunner


ALLOWED_FPS = [24, 30, 60]

# Capture resolutions offered in Settings, as (label, "WxH").
RESOLUTION_CHOICES = [
    ("1920 × 1080 (Full HD)", "1920x1080"),
    ("1280 × 720 (HD)", "1280x720"),
    ("640 × 480", "640x480"),
    ("3840 × 2160 (4K, needs USB 3)", "3840x2160"),
]
DEFAULT_RESOLUTION = "1920x1080"
# Used when the camera can't deliver the chosen resolution.
FALLBACK_RESOLUTION = (640, 480)
# OpenCV threads while streaming. Ultralytics sets cv2.setNumThreads(0) on
# import (for its training DataLoader), leaving the per-frame top-view warp
# and colour conversion on one core: 1080p warp 29 ms -> 7 ms with 4.
STREAM_CV_THREADS = 4
# Camera lost: this many failed grabs in a row, or no frame for this long.
MAX_READ_FAILURES = 5
LOST_AFTER_S = 1.5
# Wait between reconnect attempts after the camera is lost.
RECONNECT_INTERVAL_S = 1.0
# Indices searched when the camera comes back under a new /dev/videoN.
MAX_CAMERA_INDEX = 10

# CAP_DSHOW is Windows-only; use V4L2 on Linux (Jetson) and let OpenCV
# auto-pick elsewhere.
CAMERA_BACKEND = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY


def find_cameras(max_index: int = 10) -> List[int]:
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, CAMERA_BACKEND)
        if cap.isOpened():
            available.append(idx)
        cap.release()
    return available


def parse_resolution(text: str) -> Tuple[int, int]:
    """"1920x1080" -> (1920, 1080); anything unreadable -> the default."""
    try:
        width, height = (int(v) for v in str(text).lower().split("x"))
        if width > 0 and height > 0:
            return width, height
    except ValueError:
        pass
    return parse_resolution(DEFAULT_RESOLUTION)


def apply_resolution(cap, width: int, height: int) -> Tuple[int, int]:
    """Asks the camera for width x height at 30 fps and returns the size of
    a frame it actually delivers ((0, 0) if none). UVC cameras such as the
    MX Brio only offer sizes above 640x480 as MJPG, so that is requested
    there; the format must be set before the size."""
    fourcc = "MJPG" if width * height > 640 * 480 else "YUYV"
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    return (frame.shape[1], frame.shape[0]) if ok else (0, 0)


def nearest_allowed_fps(value: float) -> int:
    if value <= 1:
        return 30
    return min(ALLOWED_FPS, key=lambda x: abs(x - value))


class VideoWorker(QThread):
    frame_ready = Signal(QImage, list, list)  # image, detections, vision tool results
    status = Signal(str)
    resolution_changed = Signal(int, int)  # what the camera actually delivers
    camera_connected = Signal(bool)  # False while lost / waiting, True once frames flow
    camera_index_changed = Signal(int)  # camera came back under another index

    def __init__(self) -> None:
        super().__init__()
        self._running = False
        self._camera_index: Optional[int] = None
        self._model_path: Optional[Path] = None
        self._model = None
        self._target_fps: int = 30
        self._roi: Optional[Roi] = None
        self._warp: Optional[PerspectiveWarp] = None
        self._tools: list = []
        self._mm_per_px: Optional[float] = None
        self._part_no: Optional[str] = None
        self._slow = SlowToolRunner()
        self._slow_hz = 2.0
        # Goes up each time a part arrives (Position Adjustment lost -> found)
        # or the tools change; a slow result only counts within its epoch.
        self._part_epoch = 0
        self._part_was_found = False
        # Latest unwarped frame, for the Top View corner editor's snapshot.
        self.last_raw_frame = None
        self._camera_ok: Optional[bool] = None  # last camera_connected value sent
        self._resolution: Tuple[int, int] = parse_resolution(DEFAULT_RESOLUTION)
        self._adjust = ImageAdjust()
        # Required Marks short name check: read in its own thread (OCR is slow).
        self._short_name_expected = ""
        self._short_name = ShortNameReader(get_engine)

    def set_camera_index(self, index: Optional[int]) -> None:
        self._camera_index = index

    def set_model_path(self, path: Optional[Path]) -> None:
        self._model_path = path
        self._model = None

    def set_resolution(self, resolution: Tuple[int, int]) -> None:
        """Takes effect at the next start()."""
        self._resolution = resolution

    def set_perspective(self, quad: Optional[Quad]) -> None:
        """Top view: frames are straightened before anything else sees them.
        Read once per frame, so it takes effect on the next frame."""
        self._warp = PerspectiveWarp(quad) if quad else None

    def set_image_adjust(self, adjust: ImageAdjust) -> None:
        """Mirror / flip / colour, applied to each frame as it is read. Read
        once per frame, so it takes effect on the next frame."""
        self._adjust = adjust

    def set_short_name_expected(self, expected: str) -> None:
        """The value the detected short name must read as; "" = don't read it."""
        self._short_name_expected = expected if valid_short_name(expected) else ""

    def _read_short_name(self, frame, detections: list) -> None:
        """Puts this part's short-name read (see ShortNameReader) into the
        most confident short_name detection, and hands the reader a new crop
        when it is due."""
        expected = self._short_name_expected
        boxes = [d for d in detections if d["class_name"] == "short_name"]
        if not expected or not boxes:
            return
        det = max(boxes, key=lambda d: d["conf"])
        now = time.monotonic()
        reader = self._short_name
        reader.seen(now)
        if reader.due(now, self._slow_hz):
            x1, y1, x2, y2 = pad_box((det["x1"], det["y1"], det["x2"], det["y2"]), frame.shape[1], frame.shape[0])
            reader.submit(frame[y1:y2, x1:x2].copy(), expected, now)
        det["ocr"] = reader.result(expected, now)

    def set_roi(self, roi: Optional[Roi]) -> None:
        """Detection area; the model only sees this part of the frame.
        Read once per frame, so it takes effect on the next frame."""
        self._roi = roi

    def set_tools(self, tools: list) -> None:
        """Rule-based vision tools (app/vision). Replaces the list reference,
        like set_roi; read once per frame."""
        self._tools = list(tools)
        self._slow.clear()
        self._part_epoch += 1

    def set_slow_hz(self, hz: float) -> None:
        """How often the slow tools (OCR, codes) may run, in their own thread."""
        self._slow_hz = max(0.1, float(hz))

    def set_mm_per_px(self, value: Optional[float]) -> None:
        self._mm_per_px = value

    def set_part_no(self, part_no: Optional[str]) -> None:
        self._part_no = part_no

    def start(self) -> None:
        # Set before the thread begins, not inside run(): a stop() arriving
        # during camera open or model load must not be overwritten.
        self._running = True
        super().start()

    def stop(self) -> None:
        self._running = False

    def _load_model(self) -> None:
        if self._model_path is None:
            return
        if YOLO is None:
            self.status.emit("Ultralytics not available. Install requirements.")
            return
        try:
            self._model = YOLO(str(self._model_path))
            self.status.emit(f"Model loaded: {self._model_path.name}")
        except Exception as exc:
            self.status.emit(f"Model load failed: {exc}")
            self._model = None

    def _run_vision(self, frame, tools: list) -> list:
        """Fast tools on this frame. Slow tools (OCR, codes) get a copy in
        their own thread at most slow_hz times a second; their latest result
        for this part is joined in, or "pending" until there is one."""
        results, offset = run_tools_with_offset(frame, tools, self._mm_per_px, self._part_no, slow=False)
        if offset.found and not self._part_was_found:
            self._part_epoch += 1
        self._part_was_found = offset.found
        slow_tools = [t for t in tools if t.enabled and t.SLOW]
        if not slow_tools or not offset.found:
            return results
        now = time.monotonic()
        hz = self._slow_hz
        if self._slow.due(now, hz):
            self._slow.submit(frame.copy(), offset, tools, self._part_epoch, now,
                              self._mm_per_px, self._part_no)
        # Room for a few missed slots, and for a run slower than the interval.
        max_age = 3.0 / hz + 2.0 * self._slow.last_run_s
        return results + self._slow.results_for(slow_tools, self._part_epoch, now, max_age)

    def _extract_detections(self, results) -> List[dict]:
        detections: List[dict] = []
        if results is None:
            return detections
        boxes = results.boxes
        names = results.names
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            class_name = str(names.get(cls_id, cls_id))
            label = f"{class_name} {conf:.2f}"
            x1, y1, x2, y2 = xyxy.tolist()
            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                    "class_name": class_name,
                    "conf": conf,
                    "cls_id": cls_id,
                }
            )
        return detections

    def run(self) -> None:
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(STREAM_CV_THREADS)
        self._part_was_found = False
        try:
            self._run_stream()
        finally:
            cv2.setNumThreads(previous_threads)

    def _set_camera_ok(self, ok: bool) -> None:
        """Emits camera_connected only when the state actually changes."""
        if ok != self._camera_ok:
            self._camera_ok = ok
            self.camera_connected.emit(ok)

    def _configure(self, cap) -> Tuple[int, int]:
        """Applies the chosen resolution (falling back to 640x480); returns
        the delivered size, (0, 0) if the camera gives no frames."""
        width, height = self._resolution
        actual = apply_resolution(cap, width, height)
        if actual != (width, height) and (width, height) != FALLBACK_RESOLUTION:
            actual = apply_resolution(cap, *FALLBACK_RESOLUTION)
            if actual != (0, 0):
                self.status.emit(
                    f"Camera can't deliver {width}x{height}; using {actual[0]}x{actual[1]}.")
        return actual

    def _try_index(self, index: int):
        cap = cv2.VideoCapture(index, CAMERA_BACKEND)
        if cap.isOpened():
            actual = self._configure(cap)
            if actual != (0, 0):
                self.resolution_changed.emit(*actual)
                return cap
        cap.release()
        return None

    def _open_camera(self):
        """Opens the camera, retrying every RECONNECT_INTERVAL_S until it
        delivers frames or stop() is called (then returns None). The chosen
        index is tried first; after a replug the camera can come back as
        another /dev/videoN, so the other indices are searched too."""
        announced = False
        while self._running:
            others = [i for i in range(MAX_CAMERA_INDEX) if i != self._camera_index]
            for index in [self._camera_index] + others:
                cap = self._try_index(index)
                if cap is not None:
                    if index != self._camera_index:
                        self.status.emit(f"Camera found at index {index}.")
                        self._camera_index = index
                        self.camera_index_changed.emit(index)
                    return cap
                if not self._running:
                    return None
            if not announced:
                announced = True
                self._set_camera_ok(False)
                self.status.emit("Waiting for camera... check the USB cable.")
            deadline = time.monotonic() + RECONNECT_INTERVAL_S
            while self._running and time.monotonic() < deadline:
                time.sleep(0.05)
        return None

    def _run_stream(self) -> None:
        if self._camera_index is None:
            self.status.emit("Select a camera.")
            return

        cap = None
        self._camera_ok = None
        reconnecting = False
        frame_period = 1.0 / self._target_fps
        failures = 0
        last_ok = time.monotonic()

        while self._running:
            if cap is None:
                cap = self._open_camera()
                if cap is None:
                    break
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                self._target_fps = nearest_allowed_fps(fps)
                frame_period = 1.0 / self._target_fps
                if reconnecting:
                    self.status.emit("Camera reconnected.")
                else:
                    self.status.emit(f"Camera FPS: {fps:.2f} -> Target FPS: {self._target_fps}")
                self._set_camera_ok(True)
                failures = 0
                last_ok = time.monotonic()
                if self._running and self._model_path is not None and self._model is None:
                    self._load_model()

            start_time = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                failures += 1
                if failures >= MAX_READ_FAILURES or time.monotonic() - last_ok > LOST_AFTER_S:
                    # Unplugged (or hung): drop the handle so the device node
                    # is freed, then keep trying to open it again.
                    cap.release()
                    cap = None
                    reconnecting = True
                    self.last_raw_frame = None
                    self._set_camera_ok(False)
                    self.status.emit("Camera disconnected — check the USB cable. Reconnecting...")
                continue
            failures = 0
            last_ok = time.monotonic()
            adjust = self._adjust
            if not adjust.is_neutral:
                frame = adjust.apply(frame)
            self.last_raw_frame = frame
            warp = self._warp
            if warp is not None:
                frame = warp.apply(frame)

            tool_results = []
            tools = self._tools
            if tools:
                tool_results = self._run_vision(frame, tools)

            detections = []
            if self._model is not None:
                # Crop to the detection area: objects outside it are never
                # seen, and the model gets more pixels per object.
                roi = self._roi
                x1 = y1 = 0
                model_input = frame
                if roi is not None:
                    x1, y1, x2, y2 = roi_to_pixels(roi, frame.shape[1], frame.shape[0])
                    model_input = frame[y1:y2, x1:x2]
                try:
                    results = self._model(model_input, verbose=False)[0]
                except Exception as exc:
                    self.status.emit(f"Inference error: {exc}")
                    results = None
                detections = self._extract_detections(results)
                offset_detections(detections, x1, y1)
                self._read_short_name(frame, detections)

            elapsed = time.perf_counter() - start_time

            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
            else:
                frames_to_skip = int(elapsed / frame_period) - 1
                for _ in range(max(0, frames_to_skip)):
                    cap.grab()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            # .copy() forces a deep copy so the QImage doesn't reference a
            # numpy buffer that gets overwritten/freed on the next iteration.
            image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_ready.emit(image, detections, tool_results)

        if cap is not None:
            cap.release()
        self._slow.stop()
        self.status.emit("Stopped.")
