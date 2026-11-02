"""Opening the app: the last job is loaded and, with "Start inspecting when the
app opens" on, inspection starts by itself. Hardware mocked, temp settings / jobs."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import app.jobs as jobs_mod  # noqa: E402
import app.ng_log as ng_log  # noqa: E402
import app.paths as paths_mod  # noqa: E402
import app.settings as settings_mod  # noqa: E402
import app.tabs.settings_tab as settings_tab_mod  # noqa: E402

APP = QApplication.instance() or QApplication([])


class AutoStart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.jobs = root / "jobs"
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", root / "app_settings.json"),
            mock.patch.object(paths_mod, "MARK_MODEL_PATH", root / "missing.pt"),
            mock.patch.object(jobs_mod, "PART_CONFIGS_DIR", self.jobs),
            mock.patch.object(jobs_mod, "LAST_PART_FILE", self.jobs / ".last_part"),
            mock.patch.object(ng_log, "NG_LOGS_DIR", root / "ng"),
            mock.patch.object(settings_tab_mod, "find_cameras", return_value=[]),
            mock.patch("app.tower.TowerController.connect_device"),
            mock.patch("app.tabs.monitor_tab.MonitorTab.start_stream", return_value=True),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def window(self, last_job: bool = True, auto: bool = True):
        if last_job:
            jobs_mod.save_part_config("P-100", {"part_no": "P-100", "confidence_threshold": 0.4})
            jobs_mod.write_last_part("P-100")
        settings_mod.save_app_settings({"auto_start": auto})
        from app.main_window import MainWindow
        win = MainWindow()
        self.addCleanup(win.deleteLater)
        self.addCleanup(win.ctx.relay.shutdown)
        return win

    def test_starts_the_last_job(self):
        win = self.window()
        self.assertEqual(win.ctx.current_part_no, "P-100")
        self.assertTrue(win.auto_start())
        win.monitor_tab.start_stream.assert_called_once()

    def test_setting_off(self):
        win = self.window(auto=False)
        self.assertEqual(win.ctx.current_part_no, "P-100")  # the job is still loaded
        self.assertFalse(win.auto_start())
        win.monitor_tab.start_stream.assert_not_called()
        self.assertFalse(win.settings_tab.auto_start_checkbox.isChecked())

    def test_no_previous_job(self):
        win = self.window(last_job=False)
        self.assertFalse(win.auto_start())
        win.monitor_tab.start_stream.assert_not_called()

    def test_already_running(self):
        win = self.window()
        with mock.patch.object(type(win.ctx.worker), "isRunning", return_value=True):
            self.assertFalse(win.auto_start())
        win.monitor_tab.start_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
