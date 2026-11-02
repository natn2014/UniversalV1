"""NG log size limit: the oldest images (any part) are deleted first, down to
90 % of the limit; saves keep the log under it; the Settings value. Temp dirs."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import app.ng_log as ng_log  # noqa: E402
import app.paths as paths_mod  # noqa: E402
import app.settings as settings_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.paths import NG_THUMB_SUFFIX  # noqa: E402
from app.tabs.settings_tab import SettingsTab  # noqa: E402

APP = QApplication.instance() or QApplication([])


class Limit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ng_logs"
        self.patch = mock.patch.object(ng_log, "NG_LOGS_DIR", self.root)
        self.patch.start()
        ng_log._total_bytes = None

    def tearDown(self):
        self.patch.stop()
        ng_log._total_bytes = None
        self.tmp.cleanup()

    def image(self, part, stamp, size=1000):
        d = self.root / part
        d.mkdir(parents=True, exist_ok=True)
        (d / ".part_name").write_text(part)
        (d / f"{stamp}.jpg").write_bytes(b"x" * size)
        (d / f"{stamp}{NG_THUMB_SUFFIX}").write_bytes(b"x" * (size // 10))

    def stamps(self):
        return sorted(p.stem for p in self.root.rglob("*.jpg") if not p.name.endswith(NG_THUMB_SUFFIX))

    def test_oldest_first_across_parts(self):
        self.image("A", "20261101_080000_000")
        self.image("B", "20261101_090000_000")
        self.image("A", "20261102_080000_000")
        self.image("B", "20261102_090000_000")
        self.assertEqual(ng_log.ng_log_size(), 4 * 1100)
        deleted, freed = ng_log.prune_ng_log(2 * 1100)
        self.assertEqual((deleted, freed), (2, 2200))
        self.assertEqual(self.stamps(), ["20261102_080000_000", "20261102_090000_000"])
        self.assertFalse(list(self.root.rglob(f"20261101*{NG_THUMB_SUFFIX}")))  # thumbnails go with them

    def test_empty_part_folder_removed(self):
        self.image("OLD", "20261101_080000_000")
        self.image("NEW", "20261102_080000_000")
        ng_log.prune_ng_log(1100)
        self.assertFalse((self.root / "OLD").exists())
        self.assertTrue((self.root / "NEW").exists())

    def test_under_the_limit_nothing_deleted(self):
        self.image("A", "20261101_080000_000")
        self.assertEqual(ng_log.prune_ng_log(10_000), (0, 0))
        self.assertEqual(len(self.stamps()), 1)

    def test_save_keeps_the_log_under_the_limit(self):
        for i in range(10):
            self.image("A", f"20261101_08{i:02d}00_000", size=5000)
        pixmap = QPixmap(64, 48)
        pixmap.fill(QColor("#c0392b"))
        limit = 30_000
        deleted = ng_log.save_ng_image(pixmap, "B", limit)
        self.assertGreater(deleted, 0)
        self.assertLessEqual(ng_log.ng_log_size(), int(limit * ng_log.PRUNE_TO))
        self.assertTrue(list((self.root / "B").glob(f"*{NG_THUMB_SUFFIX}")))  # the new one is kept
        self.assertEqual(ng_log.save_ng_image(pixmap, "B", 0), 0)  # 0 = no limit


class SettingsCard(unittest.TestCase):
    def test_limit_setting(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(settings_mod, "APP_SETTINGS_FILE", Path(tmp) / "s.json"), \
                mock.patch.object(paths_mod, "MARK_MODEL_PATH", Path(tmp) / "x.pt"), \
                mock.patch.object(ng_log, "NG_LOGS_DIR", Path(tmp) / "ng"):
            ctx = AppContext()
            tab = SettingsTab(ctx)
            self.assertEqual(ctx.settings["ng_log_max_gb"], 5.0)
            tab.ng_log_limit_spinbox.setValue(2.5)
            self.assertEqual(ctx.settings["ng_log_max_gb"], 2.5)
            self.assertIn("ng_log_max_gb", tab.unsaved_changes())
            tab.revert()
            self.assertEqual(tab.ng_log_limit_spinbox.value(), 5.0)
            tab.show_section("autocapture")
            self.assertIn("0.00 GB", tab.ng_log_size_label.text())
            tab.deleteLater()


if __name__ == "__main__":
    unittest.main()
