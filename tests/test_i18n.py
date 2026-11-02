"""EN / TH language: tr() and the patterns, the Thai catalogue's shape, and the
whole window switched to Thai (every visible text translated, back to English
again, tab lookup, flags, remembered choice). Offscreen, hardware mocked,
settings and jobs in a temp dir."""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QLabel, QTableWidget  # noqa: E402

import app.jobs as jobs_mod  # noqa: E402
import app.paths as paths_mod  # noqa: E402
import app.settings as settings_mod  # noqa: E402
import app.tabs.settings_tab as settings_tab_mod  # noqa: E402
from app.i18n import translator, tr  # noqa: E402
from app.i18n_th import FORMATS, TH  # noqa: E402
from app.marking.classes import STANDARD_CLASSES  # noqa: E402

APP = QApplication.instance() or QApplication([])
THAI = re.compile(r"[฀-๿]")
# Names, not words: shown as they are in both languages.
KEEP = set(STANDARD_CLASSES) | {"English", "Modbus RTU over TCP", "Modbus TCP", "1280 × 720 (HD)",
                                 "1920 × 1080 (Full HD)", "MarkTest"}


class Strings(unittest.TestCase):
    def setUp(self):
        self.t = translator()
        self.before = self.t.language
        self.t.language = "th"

    def tearDown(self):
        self.t.language = self.before

    def test_exact_patterns_and_rich_text(self):
        self.assertEqual(tr("Save"), "บันทึก")
        self.assertEqual(tr("Missing: short_name, tis_mark"), "ขาด: short_name, tis_mark")
        self.assertEqual(tr("●  Unsaved changes (3)"), "●  ยังไม่บันทึก (3)")
        self.assertEqual(tr("Settings saved (job P1)."), "บันทึกการตั้งค่าแล้ว (งาน P1)")
        self.assertEqual(tr("Master 2 — tap to edit masters"), "ภาพต้นแบบ 2 — แตะเพื่อแก้ไข")  # longest pattern wins
        self.assertEqual(tr('<img src="a.svg">&nbsp;&nbsp;Required Marks'), '<img src="a.svg">&nbsp;&nbsp;เครื่องหมายบังคับ')
        self.assertEqual(tr("Part: {}").format("Monitor"), "ชิ้นงาน: Monitor")  # values formatted in stay as they are

    def test_names_and_unknown_text_unchanged(self):
        for text in ("e_mark", "standard_marks_v8.pt", "5403010-ER51", "", "something new"):
            self.assertEqual(tr(text), text)

    def test_english_is_identity(self):
        self.t.language = "en"
        self.assertEqual(tr("Save"), "Save")
        self.assertEqual(tr("Missing: e_mark"), "Missing: e_mark")

    def test_catalogue_shape(self):
        for key, value in FORMATS.items():
            self.assertEqual(key.count("{}"), value.count("{}"), key)
        for key, value in TH.items():
            self.assertEqual(key.count("\n"), value.count("\n"), key)  # tiles keep their line breaks
            self.assertTrue(value.strip(), key)


class Window(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.settings_file = root / "app_settings.json"
        cls.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", cls.settings_file),
            mock.patch.object(paths_mod, "MARK_MODEL_PATH", root / "missing.pt"),
            mock.patch.object(jobs_mod, "PART_CONFIGS_DIR", root / "jobs"),
            mock.patch.object(jobs_mod, "LAST_PART_FILE", root / "jobs" / ".last_part"),
            mock.patch.object(settings_tab_mod, "find_cameras", return_value=[]),
            mock.patch("app.tower.TowerController.connect_device"),
            mock.patch("app.main_window.MainWindow.confirm_unsaved", return_value=True),
            mock.patch("app.tabs.monitor_tab.MonitorTab.ensure_live", return_value=False),
        ]
        for p in cls.patches:
            p.start()
        from app.main_window import MainWindow
        cls.win = MainWindow()
        cls.win.ctx.model_classes = list(STANDARD_CLASSES)
        cls.win.ctx.model_classes_changed.emit(cls.win.ctx.model_classes)
        cls.win.ctx.set_current_part("MarkTest")
        cls.win.settings_tab.set_judge_mode("required")
        cls.win.resize(1024, 768)
        cls.win.show()
        APP.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.win.set_language("en")
        cls.win.ctx.relay.shutdown()
        cls.win.hide()
        cls.win.deleteLater()
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def visible_english(self):
        left = set()
        for w in APP.allWidgets():
            if not w.isVisible():
                continue
            texts = [w.toolTip()]
            if isinstance(w, (QLabel, QAbstractButton)):
                texts.append(w.text())
            if isinstance(w, QComboBox):
                texts += [w.itemText(i) for i in range(w.count())]
            if isinstance(w, QTableWidget):
                texts += [w.horizontalHeaderItem(i).text() for i in range(w.columnCount()) if w.horizontalHeaderItem(i)]
            for text in texts:
                plain = re.sub(r"<[^>]+>|&nbsp;", " ", text or "").strip()
                if re.search(r"[A-Za-z]{3,}", plain) and not THAI.search(plain) and plain not in KEEP \
                        and not plain.endswith(".pt"):
                    left.add(plain)
        return left

    def each_page(self):
        win = self.win
        for i in range(win.tabs.count()):
            win.tabs.setCurrentIndex(i)
            APP.processEvents()
            if win.tabs.widget(i) is win.settings_tab:
                for _, sections in settings_tab_mod.SECTIONS:
                    for key, _, _ in sections:
                        win.settings_tab.show_section(key)
                        APP.processEvents()
                        yield f"Settings/{key}"
            else:
                yield win.tabs.tabText(i)
        win.tabs.setCurrentIndex(0)

    def test_thai_everywhere_then_english_again(self):
        self.win.set_language("th")
        self.assertTrue(self.win.language_buttons["th"].isChecked())
        self.assertEqual(json.loads(self.settings_file.read_text())["language"], "th")  # remembered at once
        for page in self.each_page():
            self.assertEqual(self.visible_english(), set(), page)
        self.assertEqual(self.win.tabs.tabText(0), "หน้าตรวจ")
        self.win._show_tab("Job Change")  # found by its English title
        self.assertIs(self.win.tabs.currentWidget(), self.win.job_change_tab)
        self.win.set_language("en")
        self.assertEqual(self.win.tabs.tabText(0), "Monitor")
        self.assertEqual(self.win.settings_tab.save_button.text(), "Save")
        self.assertEqual(self.win.monitor_tab.judgement_banner.text(), "STANDBY")
        for page in self.each_page():
            for w in APP.allWidgets():
                if not w.isVisible() or w.property("_i18n_skip"):
                    continue  # the flags name their language in it
                texts = [w.toolTip()]
                if isinstance(w, (QLabel, QAbstractButton)):
                    texts.append(w.text())
                if isinstance(w, QComboBox):
                    texts += [w.itemText(i) for i in range(w.count())]
                if isinstance(w, QTableWidget):
                    texts += [w.horizontalHeaderItem(i).text() for i in range(w.columnCount()) if w.horizontalHeaderItem(i)]
                for text in texts:
                    self.assertIsNone(THAI.search(text or ""), (page, text))


if __name__ == "__main__":
    unittest.main()
