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
from app.theme import NG_RED, OK_GREEN  # noqa: E402
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
        self.assertEqual(pad_box((100, 100, 140, 120), 1000, 1000), (90, 94, 150, 126))  # 0.5 h sideways, 0.3 h up / down
        self.assertEqual(pad_box((0, 0, 40, 20), 30, 15), (0, 0, 30, 15))

    def test_pad_box_stops_at_neighbours(self):
        short = (100, 100, 140, 120)
        sni = (146, 98, 180, 124)  # SNI just right of it
        slashes = (80, 102, 96, 118)  # // just left
        below = (100, 123, 140, 140)  # type code just below
        self.assertEqual(pad_box(short, 1000, 1000, [sni, slashes, below]), (96, 94, 146, 123))
        self.assertEqual(pad_box(short, 1000, 1000, [(50, 50, 300, 300)]), (90, 94, 150, 126))  # frame round it: ignored
        self.assertEqual(pad_box(short, 1000, 1000, [(400, 400, 450, 450)]), (90, 94, 150, 126))  # far away


@unittest.skipIf(ENGINE is None, "Tesseract not installed")
class Reading(unittest.TestCase):
    def test_synthetic_print(self):
        for text in ("T1", "58", "U8", "Y2"):
            for red_line, blur in ((False, 0), (True, 0), (True, 3)):
                read = read_short_name(printed(text, red_line, blur), text, ENGINE)
                self.assertEqual((read.text, read.state), (text, "ok"), (text, red_line, blur))

    def test_tilted_print_is_straightened(self):
        # Tightly set T1 turned the other way is still weak (the T and the 1 merge).
        for text, angles in (("T1", (10, 20)), ("58", (-15, 10, 20))):
            for deg in angles:
                crop = printed(text, red_line=True)
                h, w = crop.shape[:2]
                m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
                tilted = cv2.warpAffine(cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)),
                                        cv2.getRotationMatrix2D((w / 2 + 20, h / 2 + 20), deg, 1.0), (w + 40, h + 40),
                                        borderValue=(255, 255, 255))
                read = read_short_name(tilted, text, ENGINE)
                self.assertEqual((read.text, read.state), (text, "ok"), (text, deg))
                self.assertEqual(read_short_name(tilted, "U8", ENGINE).state != "ok", True)

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
            reader.seen(10.9)  # still the same part: the good read stays
            self.assertEqual(reader.result("T1", 10.9)["state"], "ok")
            reader.seen(12.0)  # nothing seen for over a second before this: a new part starts over
            self.assertEqual(reader.result("T1", 12.0)["state"], "pending")

    def test_blinking_box_keeps_the_read(self):
        """Worker: frames where only the short name's box is missing (other marks
        in view) don't start a new part, so the next box gets the good read at once."""
        from app.camera import VideoWorker
        from app.marking.short_name import ShortNameRead
        worker = VideoWorker()
        worker.set_short_name_expected("48")
        frame = np.full((300, 400, 3), 255, np.uint8)
        clock = [0.0]
        with mock.patch("app.camera.time.monotonic", side_effect=lambda: clock[0]), \
             mock.patch("app.marking.short_name.read_short_name", return_value=ShortNameRead("48", 90.0, "ok")), \
             mock.patch.object(worker._short_name, "_engine_factory", lambda: FakeEngine()):
            box = lambda: det("short_name", box=(30, 200, 80, 240))  # noqa: E731
            other = lambda: det("e_mark", box=(200, 40, 380, 100))  # noqa: E731
            worker._read_short_name(frame, [other(), box()])
            self.assertTrue(self.wait_for(worker._short_name, "48", 0.0, "ok"))
            for t in (0.4, 0.8, 1.2, 1.6):  # the box blinks out for 1.6 s; the marking stays
                clock[0] = t
                worker._read_short_name(frame, [other()])
            clock[0] = 1.7
            dets = [other(), box()]
            worker._read_short_name(frame, dets)
            self.assertEqual(dets[1]["ocr"]["state"], "ok")  # straight away, not pending

    def test_stale_good_read_does_not_stand_in(self):
        """The marking stays in view while another print is shown: an old good
        read stands in for unclear new reads only for a second."""
        from app.marking.short_name import ShortNameRead
        reads = iter([ShortNameRead("48", 90.0, "ok"), ShortNameRead("", 0.0, "unreadable")])
        reader = ShortNameReader(lambda: FakeEngine())
        with mock.patch("app.marking.short_name.read_short_name", side_effect=lambda *a: next(reads)):
            reader.seen(0.0)
            reader.submit(np.zeros((5, 5, 3), np.uint8), "48", 0.0)
            self.assertTrue(self.wait_for(reader, "48", 0.0, "ok"))
            reader.seen(0.9)
            reader.submit(np.zeros((5, 5, 3), np.uint8), "48", 0.9)  # unclear read of the new print
            time.sleep(0.2)
            self.assertEqual(reader.result("48", 0.9)["state"], "ok")  # the good read is fresh
            reader.seen(1.6)
            self.assertNotEqual(reader.result("48", 1.6)["state"], "ok")  # stale: not passed

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
        self.assertEqual(f.judge(part(ocr("unreadable", ""))), "YELLOW")  # not read: STANDBY, never OK
        self.assertEqual(f.judge(part(ocr("pending", ""))), "YELLOW")
        self.assertEqual(f.judge(part(None)), "YELLOW")  # box without a read yet
        self.assertEqual(f.judge([det("e_mark")]), "RED")  # short name missing
        self.assertEqual(f.short_name_issue(part(ocr("ng", "TA"))), "Short name: read TA ≠ T1")
        self.assertEqual(f.short_name_issue(part(ocr("unreadable", ""))), "")  # not an NG
        self.assertEqual(f.short_name_issue(part(ocr("ok"))), "")

    def test_never_ok_without_its_read(self):
        """The job that showed OK with X6 under the camera: Max Detections 3
        (left from an older job) dropped the short name's box, so its row took
        the raw confidence and passed without OCR."""
        f = self.filters("48")
        f.selected_classes |= {"tis_mark", "m_code", "sni_mark"}
        f.max_detections = 3  # not used when judging by Required Marks
        part = [det("tis_mark", 0.92), det("m_code", 0.92), det("sni_mark", 0.93), det("e_mark", 0.88),
                det("short_name", 0.80, ocr=ocr("ng", "X6", "48"))]
        self.assertEqual(len(f.apply(part)), 5)
        self.assertEqual(f.judge(f.apply(part)), "RED")
        rows = {r["class_name"]: r for r in f.mark_results(part)}
        self.assertEqual((rows["short_name"]["ok"], rows["short_name"]["read"]), (False, "X6"))
        no_read = [det("e_mark"), det("short_name", 0.8)]  # found, but no OCR result with it
        rows = {r["class_name"]: r for r in f.mark_results(no_read)}
        self.assertIsNone(rows["short_name"]["ok"])  # STANDBY, never OK
        self.assertEqual(f.judge(f.apply(no_read)), "YELLOW")
        f.judge_mode = "detection"
        self.assertEqual(len(f.apply(part)), 3)  # the cap still works for AI Detection jobs

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
        self.ctx.settings.set(mark_hold_ms=0)  # judged per frame (the hold is tested in test_required_marks)
        self.ctx.settings.commit()
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

    def test_monitor_holds_a_good_read_for_a_second(self):
        self.ctx.settings.set(mark_hold_ms=1000, state_hold_ms=0)
        self.settings.set_required_marks(["e_mark", "short_name"])
        self.settings.set_short_name("T1")
        self.ctx.filters.selected_classes = {"e_mark", "short_name"}
        tab = MonitorTab(self.ctx)
        image = QImage(400, 300, QImage.Format_RGB888)
        image.fill(QColor("#ffffff"))
        tab._reset_judgement()
        part = lambda o: [det("e_mark", box=(200, 40, 380, 100)), det("short_name", ocr=o, box=(30, 200, 80, 240))]  # noqa: E731
        clock = [50.0]
        with mock.patch.object(monitor_mod.time, "monotonic", side_effect=lambda: clock[0]):
            tab.on_frame(image, part(ocr("ok")), [])
            self.assertEqual(tab.judgement_banner.text(), "OK")
            for t, read in ((50.3, ocr("ng", "TA")), (50.6, ocr("pending", "")), (50.9, ocr("ng", "T4"))):
                clock[0] = t  # misreads / a new read pending within the second: still OK
                tab.on_frame(image, part(read), [])
                self.assertEqual(tab.judgement_banner.text(), "OK", t)
                self.assertIsNone(tab._short_name_zoom)
                self.assertIn(QColor(OK_GREEN).name(), tab.short_name_tile.styleSheet())
                self.assertEqual(tab.missing_label.text(), "")
            clock[0] = 51.2  # wrong for more than a second since the last good read
            tab.on_frame(image, part(ocr("ng", "TA")), [])
            self.assertEqual(tab.judgement_banner.text(), "NG")
            self.assertIn("Short name: read TA ≠ T1", tab.missing_label.text())
            self.assertIsNotNone(tab._short_name_zoom)
            self.assertIn(QColor(NG_RED).name(), tab.short_name_tile.styleSheet())
            clock[0] = 51.3
            tab.on_frame(image, part(ocr("ok")), [])  # read right again
            self.assertEqual(tab.judgement_banner.text(), "OK")
        tab.deleteLater()

    def test_tower_green_only_after_a_correct_read(self):
        """The tower light (and relay) follow the verdict: never OK before the
        short name has read right; NG once it has read wrong for over a second."""
        self.ctx.settings.set(mark_hold_ms=1000, state_hold_ms=0)
        self.settings.set_required_marks(["e_mark", "short_name"])
        self.settings.set_short_name("48")
        self.ctx.filters.selected_classes = {"e_mark", "short_name"}
        tab = MonitorTab(self.ctx)
        tower, relay = [], []
        self.ctx.tower.update_state = tower.append
        self.ctx.relay.update_state = lambda st, short_name_ng=False: relay.append((st, short_name_ng))
        image = QImage(400, 300, QImage.Format_RGB888)
        image.fill(QColor("#ffffff"))
        tab._reset_judgement()
        part = lambda o: [det("e_mark", box=(200, 40, 380, 100)), det("short_name", ocr=o, box=(30, 200, 80, 240))]  # noqa: E731
        clock = [0.0]
        with mock.patch.object(monitor_mod.time, "monotonic", side_effect=lambda: clock[0]):
            for t, o in ((0.0, ocr("pending", "", "48")), (0.5, ocr("ng", "43", "48")), (1.2, ocr("ng", "43", "48"))):
                clock[0] = t
                tab.on_frame(image, part(o), [])
                self.assertNotEqual(tower[-1], "GREEN", t)
            self.assertEqual((tower[-1], relay[-1]), ("RED", ("RED", True)))  # NG from the short name alone
            clock[0] = 1.4
            tab.on_frame(image, part(ocr("ok", "48", "48")), [])
            self.assertEqual((tower[-1], relay[-1]), ("GREEN", ("GREEN", False)))
        tab.deleteLater()

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
        tile = tab.short_name_tile
        self.assertFalse(tile.isHidden())
        self.assertIn("T1", tile.text())  # the preset, big
        self.assertIn(QColor(NG_RED).name(), tile.styleSheet())
        # The saved NG image has the zoom in the corner away from the box (top right here).
        shot = tab._render_frame(image, tab._ctx.filters.apply(bad), []).toImage()
        self.assertEqual(QColor(shot.pixel(int(400 * 0.98) - 2, int(300 * 0.07) + 30)).name(), QColor(NG_RED).name())
        tab._reset_judgement()
        good = [det("e_mark", box=(200, 40, 380, 100)), det("short_name", ocr=ocr("ok"), box=(30, 200, 80, 240))]
        tab.on_frame(image, good, [])
        self.assertEqual(tab.judgement_banner.text(), "OK")
        self.assertIsNone(tab._short_name_zoom)
        self.assertIn(QColor(OK_GREEN).name(), tab.short_name_tile.styleSheet())
        tab._reset_judgement()
        self.assertNotIn(QColor(OK_GREEN).name(), tab.short_name_tile.styleSheet())  # no part: neutral
        self.settings.set_short_name("")
        tab._set_short_name_tile(None)
        self.assertTrue(tab.short_name_tile.isHidden())  # presence only: no tile
        tab.deleteLater()


if __name__ == "__main__":
    unittest.main()
