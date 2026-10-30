"""Mark inspection: layout rules on detections, the job setting and the
Monitor's amber warning, offscreen with settings in a temp dir."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import app.settings as settings_mod  # noqa: E402
import app.tabs.monitor_tab as monitor_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.judgement import DetectionFilters  # noqa: E402
from app.marking.classes import STANDARD_CLASSES  # noqa: E402
from app.marking.layout import check_layout  # noqa: E402
from app.tabs.monitor_tab import MonitorTab  # noqa: E402
from app.tabs.settings_tab import SettingsTab  # noqa: E402
from app.yolo_labels import read_yolo_label_file  # noqa: E402

APP = QApplication.instance() or QApplication([])
REAL = Path(__file__).resolve().parents[1] / "datasets" / "real_marks"


def box(cls, x1, y1, x2, y2):
    return {"class_name": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": 0.9}


def good_layout():
    """A drawing-like AGC marking (pixels, y down), right-aligned at x 400."""
    return [
        box("logo_honda", 180, 10, 280, 80),
        box("logo_agc", 100, 100, 400, 125),
        box("glass_class", 40, 150, 70, 180),
        box("e_mark", 80, 135, 400, 200),
        box("m_code", 330, 210, 400, 235),
        box("tis_mark", 80, 245, 400, 320),
        box("letter_l", 160, 290, 180, 320),
        box("plant_code", 360, 295, 400, 320),
        box("lamisafe", 250, 330, 400, 355),
        box("short_name", 20, 330, 70, 365),
        box("sni_mark", 90, 330, 140, 380),
        box("date_code", 220, 390, 400, 460),
    ]


def moved(dets, cls, dx=0, dy=0):
    return [dict(d, x1=d["x1"] + dx, x2=d["x2"] + dx, y1=d["y1"] + dy, y2=d["y2"] + dy)
            if d["class_name"] == cls else d for d in dets]


class Rules(unittest.TestCase):
    def test_good_layout(self):
        self.assertEqual(check_layout(good_layout()), [])

    def test_plant_code_not_right_most(self):
        self.assertEqual(check_layout(moved(good_layout(), "plant_code", dx=-200)), ["plant_code not right-most"])

    def test_sni_not_on_left(self):
        self.assertEqual(check_layout(moved(good_layout(), "sni_mark", dx=200)), ["sni_mark not on left"])

    def test_short_name_not_left_most(self):
        self.assertEqual(check_layout(moved(good_layout(), "short_name", dx=150)), ["short_name not left-most"])

    def test_vertical_order(self):
        dets = moved(good_layout(), "logo_agc", dy=-100)  # AGC printed above the brand logo
        self.assertEqual(check_layout(dets), ["logo_agc not below logo_honda"])
        dets = moved(good_layout(), "m_code", dy=-90)  # M-code above the E-mark
        self.assertEqual(check_layout(dets), ["m_code not below e_mark"])

    def test_missing_marks_skip_their_rule(self):
        dets = [d for d in good_layout() if d["class_name"] not in ("logo_agc", "plant_code", "sni_mark")]
        self.assertEqual(check_layout(dets), [])

    @unittest.skipUnless((REAL / "labels").is_dir(), "real marking drawings not in datasets/")
    def test_real_drawings_pass(self):
        for image in sorted((REAL / "images").glob("*.png")):
            h, w = cv2.imread(str(image)).shape[:2]
            dets = [box(STANDARD_CLASSES[b["cls_id"]], (b["cx"] - b["w"] / 2) * w, (b["cy"] - b["h"] / 2) * h,
                        (b["cx"] + b["w"] / 2) * w, (b["cy"] + b["h"] / 2) * h)
                    for b in read_yolo_label_file(REAL / "labels" / f"{image.stem}.txt")]
            self.assertEqual(check_layout(dets), [], image.stem)


class Filters(unittest.TestCase):
    def test_off_by_default_and_no_effect_on_verdict(self):
        bad = moved(good_layout(), "plant_code", dx=-200)
        f = DetectionFilters(required_classes={"e_mark", "tis_mark"})
        self.assertEqual(f.layout_warnings(bad), [])
        f.check_layout = True
        self.assertEqual(f.layout_warnings(bad), ["plant_code not right-most"])
        self.assertEqual(f.layout_warnings([]), [])
        self.assertEqual(f.judge(bad), "GREEN")


class Tabs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", Path(self.tmp.name) / "app_settings.json"),
            mock.patch.object(monitor_mod, "save_ng_image", mock.Mock()),
        ]
        for p in self.patches:
            p.start()
        self.ctx = AppContext()
        self.settings = SettingsTab(self.ctx)
        self.ctx.model_classes = sorted({d["class_name"] for d in good_layout()})
        self.ctx.model_classes_changed.emit(self.ctx.model_classes)

    def tearDown(self):
        self.settings.deleteLater()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_saved_with_the_job(self):
        self.settings.layout_check_checkbox.setChecked(True)
        config = self.settings.collect_config()
        self.assertTrue(config["check_layout"])
        self.settings.layout_check_checkbox.setChecked(False)
        self.settings.apply_config(config)
        self.assertTrue(self.ctx.filters.check_layout)
        self.settings.apply_config({k: v for k, v in config.items() if k != "check_layout"})
        self.assertFalse(self.ctx.filters.check_layout)  # older job: off

    def test_monitor_shows_warning_verdict_unchanged(self):
        tab = MonitorTab(self.ctx)
        image = QImage(500, 500, QImage.Format_RGB888)
        image.fill(QColor("#202020"))
        self.ctx.filters.selected_classes = set(self.ctx.model_classes)
        self.settings.set_required_marks(["e_mark", "tis_mark"])
        self.settings.layout_check_checkbox.setChecked(True)
        tab._reset_judgement()
        tab.on_frame(image, moved(good_layout(), "plant_code", dx=-200), [])
        self.assertEqual(tab.layout_label.text(), "Layout: plant_code not right-most")
        self.assertEqual(tab._latch.state, "GREEN")
        tab.on_frame(image, good_layout(), [])
        self.assertEqual(tab.layout_label.text(), "")
        tab.deleteLater()


if __name__ == "__main__":
    unittest.main()
