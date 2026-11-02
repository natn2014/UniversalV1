"""Settings draft: changes work at once, Save writes them, Revert goes back,
and the main window asks before leaving with unsaved changes. Offscreen,
with settings and job files in a temp dir."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

import app.jobs as jobs_mod  # noqa: E402
import app.paths as paths_mod  # noqa: E402
import app.settings as settings_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from app.tabs.settings_tab import SECTIONS, SettingsTab  # noqa: E402

APP = QApplication.instance() or QApplication([])


class Draft(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.file = root / "app_settings.json"
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", self.file),
            mock.patch.object(paths_mod, "MARK_MODEL_PATH", root / "missing.pt"),
            mock.patch.object(jobs_mod, "PART_CONFIGS_DIR", root / "part_configs"),
            mock.patch.object(jobs_mod, "LAST_PART_FILE", root / "part_configs" / ".last_part"),
        ]
        for p in self.patches:
            p.start()
        self.ctx = AppContext()
        self.settings = SettingsTab(self.ctx)
        self.ctx.model_classes = ["e_mark", "tis_mark"]
        self.ctx.model_classes_changed.emit(self.ctx.model_classes)

    def tearDown(self):
        self.settings.deleteLater()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def on_disk(self):
        return json.loads(self.file.read_text()) if self.file.exists() else {}

    def test_machine_change_is_live_but_saved_only_on_save(self):
        self.settings.state_hold_spinbox.setValue(2500)
        self.assertEqual(self.ctx.settings["state_hold_ms"], 2500)  # works at once
        self.assertNotIn("state_hold_ms", self.on_disk())
        self.assertEqual(self.settings.unsaved_changes(), ["state_hold_ms"])
        self.assertTrue(self.settings.save_button.isEnabled())
        self.assertIn("(1)", self.settings.unsaved_label.text())
        self.settings.save()
        self.assertEqual(self.on_disk()["state_hold_ms"], 2500)
        self.assertEqual(self.settings.unsaved_changes(), [])
        self.assertFalse(self.settings.save_button.isEnabled())

    def test_revert_restores_values_and_widgets(self):
        before = self.ctx.settings["state_hold_ms"]
        self.settings.state_hold_spinbox.setValue(before + 500)
        self.settings.revert()
        self.assertEqual(self.ctx.settings["state_hold_ms"], before)
        self.assertEqual(self.settings.state_hold_spinbox.value(), before)
        self.assertEqual(self.settings.unsaved_changes(), [])

    def test_change_and_undo_is_not_unsaved(self):
        before = self.ctx.settings["state_hold_ms"]
        self.settings.state_hold_spinbox.setValue(before + 100)
        self.settings.state_hold_spinbox.setValue(before)
        self.assertEqual(self.settings.unsaved_changes(), [])

    def test_counters_are_written_at_once(self):
        self.ctx.settings.set(ok_count=7)
        self.assertEqual(self.on_disk()["ok_count"], 7)
        self.assertEqual(self.settings.unsaved_changes(), [])

    def test_job_change_saved_into_the_job_file_keeping_tools(self):
        self.ctx.set_current_part("P1")
        jobs_mod.save_part_config("P1", {"part_no": "P1", "tools": [{"kind": "width"}], "roi": [0, 0, 1, 1]})
        self.settings.apply_config(jobs_mod.load_part_config("P1"))
        self.settings.confidence_slider.setValue(55)
        self.assertEqual(self.settings.unsaved_changes(), ["confidence_threshold"])
        self.settings.save()
        saved = jobs_mod.load_part_config("P1")
        self.assertEqual(saved["confidence_threshold"], 0.55)
        self.assertEqual(saved["tools"], [{"kind": "width"}])  # other tabs' parts kept

    def test_job_revert(self):
        self.ctx.set_current_part("P1")
        self.settings.apply_config({"confidence_threshold": 0.3})
        self.settings.confidence_slider.setValue(80)
        self.settings.revert()
        self.assertEqual(self.ctx.filters.confidence_threshold, 0.3)
        self.assertEqual(self.settings.unsaved_changes(), [])

    def test_no_job_loaded_job_edits_are_not_counted(self):
        self.settings.confidence_slider.setValue(80)
        self.assertEqual(self.settings.unsaved_changes(), [])
        self.settings.show_section("model")
        self.assertIn("No job loaded", self.settings.job_note.text())

    def test_nav_shows_each_section(self):
        for _, sections in SECTIONS:
            for key, title, _ in sections:
                self.settings.show_section(key)
                self.assertEqual(self.settings.section_title.text(), title)
                self.assertTrue(self.settings.nav_buttons[key].isChecked())


class LeaveGuard(unittest.TestCase):
    """MainWindow.confirm_unsaved on a stand-in window (no camera / tower)."""

    def window(self, changed):
        tab = mock.Mock()
        tab.unsaved_changes.return_value = changed
        win = QWidget()
        win.settings_tab = tab
        return win, tab

    def test_nothing_unsaved_passes(self):
        win, tab = self.window([])
        self.assertTrue(MainWindow.confirm_unsaved(win))

    def test_save_discard_cancel(self):
        for answer, result, called in ((QMessageBox.Save, True, "save"), (QMessageBox.Discard, True, "revert"),
                                       (QMessageBox.Cancel, False, None)):
            win, tab = self.window(["state_hold_ms"])
            with mock.patch.object(QMessageBox, "exec", return_value=answer):
                self.assertEqual(MainWindow.confirm_unsaved(win), result)
            for name in ("save", "revert"):
                self.assertEqual(getattr(tab, name).called, name == called, (answer, name))


if __name__ == "__main__":
    unittest.main()
