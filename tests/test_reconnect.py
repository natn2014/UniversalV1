"""Cable pull / replug recovery for the tower light and the camera, with
fake hardware."""

import time
import unittest

import numpy as np
from PySide6.QtCore import QCoreApplication, QTimer

import app.camera as camera
import app.tower as tower
from app.settings import AppSettings

APP = QCoreApplication.instance() or QCoreApplication([])


class FakeTowerLight:
    """Stands in for patlite_lr6.TowerLight; fails like a pulled cable."""

    plugged = True
    opened = 0
    commands = []

    def __init__(self, backend="auto"):
        self.backend_name = "fake"

    def open(self):
        if not FakeTowerLight.plugged:
            raise OSError("device not found")
        FakeTowerLight.opened += 1

    def _send(self, what):
        if not FakeTowerLight.plugged:
            raise OSError("write failed")
        FakeTowerLight.commands.append(what)

    def set_buzzer(self, pattern, limit=0):
        self._send(("buzzer", pattern))

    def set_tower(self, red=0, yellow=0, green=0):
        self._send(("tower", "red" if red else "yellow" if yellow else "green"))

    def reset(self):
        pass

    def close(self):
        pass


class TowerReconnect(unittest.TestCase):
    def setUp(self):
        self._saved = (tower.TowerLight, tower.device_present)
        tower.TowerLight = FakeTowerLight
        tower.device_present = lambda: FakeTowerLight.plugged
        FakeTowerLight.plugged, FakeTowerLight.opened, FakeTowerLight.commands = True, 0, []
        settings = AppSettings.__new__(AppSettings)
        settings._values = {"buzzer_enabled": False, "buzzer_pattern": 1,
                            "ok_beep_enabled": False, "ok_beep_ms": 50}
        settings.set = lambda **v: settings._values.update(v)
        self.ctl = tower.TowerController(settings)
        self.texts = []
        self.ctl.connection_text_changed.connect(self.texts.append)

    def tearDown(self):
        self.ctl.shutdown()
        tower.TowerLight, tower.device_present = self._saved

    def test_pull_and_replug_restores_current_state(self):
        self.ctl.connect_device()
        self.ctl.update_state("RED")
        self.assertIn(("tower", "red"), FakeTowerLight.commands)

        FakeTowerLight.plugged = False
        self.ctl._check_connection()  # watchdog tick sees it gone
        self.assertFalse(self.ctl.connected)
        self.assertIn("Disconnected", self.texts[-1])

        self.ctl.update_state("GREEN")  # judgement moves on while unplugged
        self.ctl._check_connection()  # still unplugged: no crash, no open
        self.assertEqual(FakeTowerLight.opened, 1)

        FakeTowerLight.commands.clear()
        FakeTowerLight.plugged = True
        self.ctl._check_connection()
        self.assertTrue(self.ctl.connected)
        self.assertEqual(FakeTowerLight.commands[-1], ("tower", "green"))  # current, not stale RED

    def test_write_error_drops_then_recovers(self):
        self.ctl.connect_device()
        FakeTowerLight.plugged = False
        self.ctl.update_state("RED")  # write fails mid-cycle
        self.assertFalse(self.ctl.connected)
        FakeTowerLight.plugged = True
        self.ctl._check_connection()
        self.assertEqual(FakeTowerLight.commands[-1], ("tower", "red"))

    def test_not_plugged_at_start_connects_later(self):
        FakeTowerLight.plugged = False
        self.ctl.connect_device()
        self.assertFalse(self.ctl.connected)
        self.assertTrue(self.ctl._watchdog.isActive())
        FakeTowerLight.plugged = True
        self.ctl._check_connection()
        self.assertTrue(self.ctl.connected)

    def test_still_reconnecting_after_later_ticks(self):
        self.ctl.connect_device()
        FakeTowerLight.plugged = False
        self.ctl._check_connection()
        for _ in range(3):
            self.ctl._check_connection()
        self.assertTrue(self.ctl.reconnecting)
        self.assertEqual(len([t for t in self.texts if "retrying" in t]), 2)  # no repeats per tick

    def test_found_but_cannot_open_says_so(self):
        tower.device_present = lambda: True
        FakeTowerLight.plugged = False  # open() fails although the device is present
        self.ctl.connect_device()
        self.assertIn("can't open", self.texts[-1])

    def test_shutdown_stops_watchdog(self):
        self.ctl.connect_device()
        self.ctl.shutdown()
        self.assertFalse(self.ctl._watchdog.isActive())


class FakeBus:
    """Which /dev/videoN the fake camera is at; None = unplugged."""
    index = 0


class FakeCapture:
    def __init__(self, index, backend=None):
        self.index = index
        self.alive = FakeBus.index == index

    def isOpened(self):
        return self.alive

    def set(self, *args):
        return True

    def get(self, prop):
        return 30

    def read(self):
        if not self.alive or FakeBus.index != self.index:
            self.alive = False
            return False, None
        time.sleep(0.01)
        return True, np.zeros((48, 64, 3), np.uint8)

    def grab(self):
        return True

    def release(self):
        self.alive = False


class CameraReconnect(unittest.TestCase):
    def setUp(self):
        self._saved = (camera.cv2.VideoCapture, camera.RECONNECT_INTERVAL_S)
        camera.cv2.VideoCapture = FakeCapture
        camera.RECONNECT_INTERVAL_S = 0.05
        FakeBus.index = 0

    def tearDown(self):
        camera.cv2.VideoCapture, camera.RECONNECT_INTERVAL_S = self._saved

    def run_worker(self, script, start_index=0, timeout_ms=8000):
        """Runs the worker while `script` [(ms, action)] changes the bus."""
        w = camera.VideoWorker()
        w.set_camera_index(start_index)
        events, frames = [], []
        w.camera_connected.connect(lambda ok: events.append(("connected", ok)))
        w.camera_index_changed.connect(lambda i: events.append(("index", i)))
        w.frame_ready.connect(lambda img, d: frames.append(time.monotonic()))
        for ms, action in script:
            QTimer.singleShot(ms, action)
        QTimer.singleShot(timeout_ms, w.stop)
        w.finished.connect(APP.quit)
        w.start()
        APP.exec()
        w.wait()
        return w, events, frames

    def test_unplug_and_replug_on_new_index(self):
        marks = {}

        def unplug():
            marks["unplug"] = time.monotonic()
            FakeBus.index = None

        def replug():
            marks["replug"] = time.monotonic()
            FakeBus.index = 2  # came back as /dev/video2

        w, events, frames = self.run_worker([(600, unplug), (1500, replug)], timeout_ms=2600)
        self.assertEqual(events, [("connected", True), ("connected", False),
                                  ("index", 2), ("connected", True)])
        self.assertEqual(w._camera_index, 2)
        self.assertTrue(any(t > marks["replug"] for t in frames), "frames resume after replug")
        self.assertFalse(any(marks["unplug"] + 0.1 < t < marks["replug"] for t in frames))

    def test_waits_for_camera_plugged_in_after_start(self):
        FakeBus.index = None

        def plug():
            FakeBus.index = 0

        w, events, frames = self.run_worker([(500, plug)], timeout_ms=1500)
        self.assertEqual(events, [("connected", False), ("connected", True)])
        self.assertTrue(frames)

    def test_stop_while_waiting_returns_promptly(self):
        FakeBus.index = None
        start = time.monotonic()
        w, events, frames = self.run_worker([], timeout_ms=300)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual(frames, [])


if __name__ == "__main__":
    unittest.main()
