"""Short name OCR check: the reader on real drawings and synthetic print,
the background reader's pending / unreadable / new-part handling, the
judgement, the Settings field and the Monitor's NG zoom. Offscreen, settings
in a temp dir; the OCR tests are skipped without Tesseract."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import app.paths as paths_mod  # noqa: E402
import app.settings as settings_mod  # noqa: E402
import app.tabs.monitor_tab as monitor_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.judgement import DetectionFilters, InspectionCycle  # noqa: E402
from app.marking.classes import STANDARD_CLASSES  # noqa: E402
from app.marking.short_name import ShortNameReader, pad_box, read_short_name, valid_short_name  # noqa: E402
from app.tabs.monitor_tab import MonitorTab  # noqa: E402
from app.tabs.settings_tab import SettingsTab  # noqa: E402
from app.theme import NG_RED  # noqa: E402
from app.vision.ocr import get_engine  # noqa: E402
from app.yolo_labels import read_yolo_label_file  # noqa: E402

APP = QApplication.instance() or QApplication([])
ENGINE = get_engine()
REAL = Path(__file__).resolve().parents[1] / "datasets" / "real_marks"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# Short names printed on the drawings that are hardest to read: a red frame
# line through a character (2, 4, 11, 22, 23, 25), a tail on the 9, a Y.
HARD = {2: "29", 4: "U7", 9: "98", 11: "U4", 22: "T1", 23: "Y0", 25: "Q0", 26: "Y2"}


def drawing_crop(n: int) -> np.ndarray:
    img = cv2.imread(str(REAL / "images" / f"Mark_image{n}.png"))
    h, w = img.shape[:2]
    sn = STANDARD_CLASSES.index("short_name")
    b = next(b for b in read_yolo_label_file(REAL / "labels" / f"Mark_image{n}.txt") if b["cls_id"] == sn)
    box = ((b["cx"] - b["w"] / 2) * w, (b["cy"] - b["h"] / 2) * h, (b["cx"] + b["w"] / 2) * w, (b["cy"] + b["h"] / 2) * h)
    x1, y1, x2, y2 = pad_box(box, w, h)
    return img[y1:y2, x1:x2]


def printed(text: str, red_line: bool = False, blur: int = 0) -> np.ndarray:
    """Bold black print on white, like the drawings, optionally with the red frame line through it."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (120, 70), "white")
    ImageDraw.Draw(img).text((14, 8), text, font=ImageFont.truetype(FONT, 46), fill="black")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if red_line:
        cv2.line(bgr, (32, 0), (32, 69), (0, 0, 220), 4)
    if blur:
        bgr = cv2.GaussianBlur(bgr, (blur, blur), 0)
    return bgr


class Format(unittest.TestCase):
    def test_letter_digit_or_two_digits(self):
        for good in ("T1", "U8", "58", "00", "Y0"):
            self.assertTrue(valid_short_name(good), good)
        for bad in ("", "T", "1T", "t1", "TT", "581", "T-"):
            self.assertFalse(valid_short_name(bad), bad)

    def test_pad_box_grows_and_clips(self):
        self.assertEqual(pad_box((100, 100, 140, 120), 1000, 1000), (93, 93, 147, 127))
        self.assertEqual(pad_box((0, 0, 40, 20), 30, 15), (0, 0, 30, 15))


@unittest.skipIf(ENGINE is None, "Tesseract not installed")
class Reading(unittest.TestCase):
    def test_synthetic_print(self):
        for text in ("T1", "58", "U8", "Y2"):
            for red_line, blur in ((False, 0), (True, 0), (True, 3)):
                read = read_short_name(printed(text, red_line, blur), text, ENGINE)
                self.assertEqual((read.text, read.state), (text, "ok"), (text, red_line, blur))

    def test_wrong_name_is_ng(self):
        self.assertEqual(read_short_name(printed("T1"), "T2", ENGINE).state, "ng")
        self.assertEqual(read_short_name(printed("20"), "Q0", ENGINE).state, "ng")  # a digit never passes as a letter
        self.assertEqual(read_short_name(printed("U8"), "58", ENGINE).state, "ng")

    def test_blank_is_unreadable(self):
        blank = np.full((60, 110, 3), 255, np.uint8)
        self.assertEqual(read_short_name(blank, "T1", ENGINE).state, "unreadable")
        self.assertEqual(read_short_name(printed("T1"), "", ENGINE).state, "unreadable")  # no valid preset

    @unittest.skipUnless((REAL / "labels").is_dir(), "real marking drawings not in datasets/")
    def test_real_drawings(self):
        names = sorted(set(HARD.values()))
        for n, truth in HARD.items():
            crop = drawing_crop(n)
            read = read_short_name(crop, truth, ENGINE)
            self.assertEqual((read.text, read.state), (truth, "ok"), n)
            other = names[(names.index(truth) + 1) % len(names)]  # someone else's short name
            self.assertNotEqual(read_short_name(crop, other, ENGINE).state, "ok", (n, other))


class FakeEngine:
    """Reads whatever text it is told to, per character."""

    def __init__(self, text="T1", conf=90.0):
        self.text, self.conf = text, conf

    def read(self, image, psm=7, whitelist=""):
        return self.text, self.conf


class Reader(unittest.TestCase):
    def wait_for(self, reader, expected, now, state, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if reader.result(expected, now)["state"] == state:
                return True
            time.sleep(0.01)
        return False

    def test_pending_then_read_then_new_part(self):
        engine = FakeEngine()
        reader = ShortNameReader(lambda: engine)
        with mock.patch("app.marking.short_name.read_short_name",
                        side_effect=lambda crop, exp, eng: __import__("app.marking.short_name", fromlist=["x"])
                        .ShortNameRead(eng.text, 90.0, "ok" if eng.text == exp else "ng")):
            reader.seen(10.0)
            self.assertEqual(reader.result("T1", 10.0)["state"], "pending")
            self.assertTrue(reader.due(10.0, 2.0))
            reader.submit(np.zeros((5, 5, 3), np.uint8), "T1", 10.0)
            self.assertFalse(reader.due(10.1, 2.0))
            self.assertTrue(self.wait_for(reader, "T1", 10.1, "ok"))
            self.assertEqual(reader.result("T2", 10.2)["state"], "ng")  # preset changed: compared again
            reader.seen(11.0)  # gone 0.5 s+ before this: a new part starts over
            self.assertEqual(reader.result("T1", 11.0)["state"], "pending")

    def test_unreadable_after_timeout(self):
        reader = ShortNameReader(lambda: None)  # no OCR engine: never a clear read
        reader.seen(0.0)
        self.assertEqual(reader.result("T1", 1.0)["state"], "pending")
        self.assertEqual(reader.result("T1", 1.6)["state"], "unreadable")


def det(cls, conf=0.9, ocr=None, box=(20, 20, 60, 50)):
    d = {"class_name": cls, "conf": conf, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "label": cls}
    if ocr:
        d["ocr"] = ocr
    return d


def ocr(state, text="T1", expected="T1"):
    return {"text": text, "conf": 90.0, "expected": expected, "state": state}


class Judging(unittest.TestCase):
    def filters(self, expected="T1"):
        return DetectionFilters(judge_mode="required", selected_classes={"e_mark", "short_name"},
                                required_classes={"e_mark", "short_name"}, short_name_expected=expected)

    def test_states(self):
        f = self.filters()
        part = lambda o: [det("e_mark"), det("short_name", ocr=o)]  # noqa: E731
        self.assertEqual(f.judge(part(ocr("ok"))), "GREEN")
        self.assertEqual(f.judge(part(ocr("ng", "TA"))), "RED")
        self.assertEqual(f.judge(part(ocr("unreadable", ""))), "RED")
        self.assertEqual(f.judge(part(ocr("pending", ""))), "YELLOW")
        self.assertEqual(f.judge(part(None)), "YELLOW")  # box without a read yet
        self.assertEqual(f.judge([det("e_mark")]), "RED")  # short name missing
        self.assertEqual(f.short_name_issue(part(ocr("ng", "TA"))), "Short name: read TA ≠ T1")
        self.assertEqual(f.short_name_issue(part(ocr("unreadable", ""))), "Short name: unreadable")
        self.assertEqual(f.short_name_issue(part(ocr("ok"))), "")

    def test_stale_preset_read_is_pending(self):
        f = self.filters("T2")
        self.assertEqual(f.judge([det("e_mark"), det("short_name", ocr=ocr("ok", "T1", "T1"))]), "YELLOW")

    def test_check_off(self):
        for f in (self.filters(""), self.filters("T1x"),
                  DetectionFilters(judge_mode="detection", selected_classes={"short_name"},
                                   required_classes={"short_name"}, short_name_expected="T1", target_class_1="short_name")):
            self.assertFalse(f.checks_short_name)
            self.assertEqual(f.judge([det("e_mark"), det("short_name", ocr=ocr("ng", "TA"))]),
                             "GREEN" if f.by_marks or f.target_class_1 else "RED")

    def test_mark_table_row(self):
        f = self.filters()
        rows = {r["class_name"]: r for r in f.mark_results([det("e_mark"), det("short_name", ocr=ocr("ng", "TA"))])}
        self.assertEqual((rows["short_name"]["read"], rows["short_name"]["ok"]), ("TA", False))
        rows = {r["class_name"]: r for r in f.mark_results([det("e_mark"), det("short_name", ocr=ocr("pending", ""))])}
        self.assertIsNone(rows["short_name"]["ok"])

    def test_trigger_waits_for_the_read(self):
        c = InspectionCycle()
        c.trigger()
        self.assertEqual(c.update("YELLOW", 0, trigger_mode=True, pending=True), ("YELLOW", None))
        self.assertTrue(c.armed)  # still waiting for this part's read
        self.assertEqual(c.update("RED", 0, trigger_mode=True), ("RED", "RED"))


class Tabs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", Path(self.tmp.name) / "app_settings.json"),
            mock.patch.object(paths_mod, "MARK_MODEL_PATH", Path(self.tmp.name) / "missing.pt"),
            mock.patch.object(monitor_mod, "save_ng_image", mock.Mock()),
        ]
        for p in self.patches:
            p.start()
        self.ctx = AppContext()
        self.settings = SettingsTab(self.ctx)
        self.ctx.model_classes = ["e_mark", "short_name"]
        self.ctx.model_classes_changed.emit(self.ctx.model_classes)
        self.settings.set_judge_mode("required")

    def tearDown(self):
        self.settings.deleteLater()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_settings_field(self):
        edit = self.settings.short_name_edit
        self.assertFalse(edit.isEnabled())  # short_name not ticked
        self.settings.set_required_marks(["e_mark", "short_name"])
        self.assertTrue(edit.isEnabled())
        edit.setFocus()
        edit.textEdited.emit("t1")
        edit.setText("t1")
        self.settings._on_short_name_edited("t1")
        self.assertEqual(edit.text(), "T1")
        self.assertEqual(self.ctx.filters.short_name_expected, "T1")
        self.assertEqual(self.ctx.worker._short_name_expected, "T1")
        self.settings.set_short_name("T")  # not complete: check off
        self.assertEqual(self.ctx.filters.short_name_expected, "")
        self.assertEqual(self.ctx.worker._short_name_expected, "")
        self.settings.set_short_name("58")
        self.assertEqual(self.settings.collect_config()["short_name_expected"], "58")
        self.settings.set_required_marks(["e_mark"])
        self.assertEqual(self.ctx.worker._short_name_expected, "")  # not checked when short_name is not required
        self.settings.apply_config({"judge_mode": "required", "required_classes": ["short_name"], "short_name_expected": "U8"})
        self.assertEqual((self.ctx.filters.short_name_expected, self.ctx.worker._short_name_expected), ("U8", "U8"))
        self.settings.apply_config({"judge_mode": "required", "required_classes": ["short_name"]})
        self.assertEqual(self.settings.short_name_edit.text(), "")  # older job: presence only

    def test_monitor_ng_zoom(self):
        self.settings.set_required_marks(["e_mark", "short_name"])
        self.settings.set_short_name("T1")
        self.ctx.filters.selected_classes = {"e_mark", "short_name"}
        tab = MonitorTab(self.ctx)
        image = QImage(400, 300, QImage.Format_RGB888)
        image.fill(QColor("#ffffff"))
        tab._reset_judgement()
        bad = [det("e_mark", box=(200, 40, 380, 100)), det("short_name", ocr=ocr("ng", "TA"), box=(30, 200, 80, 240))]
        tab.on_frame(image, bad, [])
        self.assertEqual(tab.judgement_banner.text(), "NG")
        self.assertIn("Short name: read TA ≠ T1", tab.missing_label.text())
        self.assertIsNotNone(tab._short_name_zoom)
        self.assertIn("TA", tab._short_name_zoom["text"])
        # The saved NG image has the zoom in the corner away from the box (top right here).
        shot = tab._render_frame(image, tab._ctx.filters.apply(bad), []).toImage()
        self.assertEqual(QColor(shot.pixel(int(400 * 0.98) - 2, int(300 * 0.07) + 30)).name(), QColor(NG_RED).name())
        tab._reset_judgement()
        good = [det("e_mark", box=(200, 40, 380, 100)), det("short_name", ocr=ocr("ok"), box=(30, 200, 80, 240))]
        tab.on_frame(image, good, [])
        self.assertEqual(tab.judgement_banner.text(), "OK")
        self.assertIsNone(tab._short_name_zoom)
        tab.deleteLater()


if __name__ == "__main__":
    unittest.main()
