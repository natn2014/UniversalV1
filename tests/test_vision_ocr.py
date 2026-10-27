import unittest
from datetime import datetime

import cv2
import numpy as np

from app.optional_deps import TESSERACT
from app.vision import ocr
from app.vision.ocr import OcrTool, estimate_skew, prepare_for_ocr, verify
from app.vision.preprocess import apply_chain, to_gray


def label(text="LOT 2026", angle=0.0, size=(110, 420)):
    """Dark text on a brightness ramp, optionally tilted (BGR)."""
    h, w = size
    img = np.tile(np.linspace(120, 230, w, dtype=np.float32), (h, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.putText(img, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 30, 3, cv2.LINE_AA)
    if angle:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def pipeline(img, **kw):
    gray = apply_chain(to_gray(img), ocr.DEFAULT_OCR_PRE)
    return prepare_for_ocr(gray, **kw)


class Prepare(unittest.TestCase):
    def test_binarized_text_is_dark_on_light(self):
        out = pipeline(label())
        self.assertGreater(np.count_nonzero(out == 255), out.size / 2)
        self.assertEqual(set(np.unique(out)), {0, 255})

    def test_inverts_light_text(self):
        img = 255 - label()
        out = pipeline(img)
        self.assertGreater(np.count_nonzero(out == 255), out.size / 2)

    def test_deskew_removes_5_degree_tilt(self):
        tilted = label(angle=-5.0)
        before = estimate_skew(ocr._binarize_dark_on_light(apply_chain(to_gray(tilted), ocr.DEFAULT_OCR_PRE)))
        self.assertGreater(abs(before), 3)  # the tilt is really seen
        after = estimate_skew(pipeline(tilted))
        self.assertLess(abs(after), 1.0)

    def test_deskew_other_direction(self):
        self.assertLess(abs(estimate_skew(pipeline(label(angle=5.0)))), 1.0)

    def test_small_print_is_scaled_up(self):
        small = np.full((40, 200), 220, np.uint8)
        cv2.putText(small, "AB12", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 20, 1, cv2.LINE_AA)
        out = prepare_for_ocr(small)
        self.assertGreater(out.shape[0], small.shape[0] * 1.5)

    def test_blank_does_not_raise(self):
        prepare_for_ocr(np.full((50, 50), 100, np.uint8))


class FakeEngine:
    def __init__(self, text="ABC", conf=88.0):
        self.text, self.conf, self.calls = text, conf, []

    def read(self, binary, psm=7, whitelist=""):
        self.calls.append((binary.shape, psm, whitelist))
        return self.text, self.conf


class ToolWithFakeEngine(unittest.TestCase):
    def tool(self, **kw):
        t = OcrTool(**kw)
        t.engine = FakeEngine()
        return t

    def test_value_is_confidence_and_text_is_read(self):
        res = self.tool().run(label())
        self.assertEqual((res.value, res.text, res.ok), (88.0, "ABC", True))

    def test_low_confidence_is_ng(self):
        t = self.tool()
        t.engine.conf = 40.0
        self.assertFalse(t.run(label()).ok)

    def test_params_reach_engine(self):
        t = self.tool(psm=6, whitelist="ABC")
        t.run(label())
        self.assertEqual(t.engine.calls[0][1:], (6, "ABC"))

    def test_default_pre_and_round_trip(self):
        t = OcrTool()
        self.assertEqual([s["op"] for s in t.pre], ["flat_field", "clahe"])
        self.assertEqual(t.low, 60.0)

    def test_missing_engine(self):
        t = OcrTool()
        t.engine = None
        saved = ocr.get_engine
        ocr.get_engine = lambda: None
        try:
            res = t.run(label())
        finally:
            ocr.get_engine = saved
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "OCR engine not installed")


NOW = datetime(2026, 10, 27)


class Verify(unittest.TestCase):
    def test_expected(self):
        self.assertTrue(verify("LOT 2026", 90, expected="LOT 2026")[0])
        self.assertTrue(verify("  LOT 2026\n", 90, expected="LOT 2026")[0])
        self.assertFalse(verify("LOT 2025", 90, expected="LOT 2026")[0])

    def test_pattern_fullmatch(self):
        self.assertTrue(verify("LOT 2026", 90, pattern=r"LOT \d{4}")[0])
        self.assertFalse(verify("LOT 2026 X", 90, pattern=r"LOT \d{4}")[0])
        self.assertFalse(verify("LOT 20", 90, pattern=r"LOT \d{4}")[0])
        self.assertFalse(verify("x", 90, pattern="(")[0])

    def test_date_placeholder(self):
        self.assertTrue(verify("EXP 27/10/26", 90, pattern=r"EXP {date}", now=NOW)[0])
        self.assertFalse(verify("EXP 28/10/26", 90, pattern=r"EXP {date}", now=NOW)[0])
        self.assertTrue(verify("27-10-2026", 90, expected="{date}", date_format="%d-%m-%Y", now=NOW)[0])

    def test_char_count_ignores_spaces(self):
        self.assertTrue(verify("AB CD", 90, char_count=4)[0])
        self.assertFalse(verify("AB CDE", 90, char_count=4)[0])
        self.assertTrue(verify("anything", 90, char_count=0)[0])

    def test_low_confidence(self):
        ok, reason = verify("LOT 2026", 40, expected="LOT 2026", low=60)
        self.assertFalse(ok)
        self.assertEqual(reason, "low confidence")

    def test_all_checks_must_pass(self):
        self.assertFalse(verify("LOT 2026", 90, expected="LOT 2026", char_count=9)[0])

    def test_tool_applies_checks_with_fake_engine(self):
        t = OcrTool(expected="ABC", char_count=3)
        t.engine = FakeEngine("ABC", 90)
        self.assertTrue(t.run(label()).ok)
        t.engine = FakeEngine("ABD", 90)
        res = t.run(label())
        self.assertFalse(res.ok)
        self.assertEqual(res.extra["reason"], "text differs from expected")

    def test_orientation_with_fake_engine(self):
        class Flipper:
            """Reads garbage for the original patch and the right text for the flipped one."""
            def __init__(self):
                self.n = 0

            def read(self, binary, psm=7, whitelist=""):
                self.n += 1
                return ("XYZ", 55.0) if self.n == 1 else ("LOT 2026", 85.0)

        t = OcrTool(expected="LOT 2026", check_orientation=True, low=50)
        t.engine = Flipper()
        res = t.run(label())
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "upside down")


@unittest.skipUnless(TESSERACT, "tesseract not installed")
class WithTesseract(unittest.TestCase):
    def test_reads_lot_2026(self):
        res = OcrTool(psm=7).run(label())
        self.assertEqual(res.text.strip(), "LOT 2026")
        self.assertGreater(res.value, 60)
        self.assertTrue(res.ok)

    def test_reads_tilted_label(self):
        self.assertEqual(OcrTool(psm=7).run(label(angle=4.0)).text.strip(), "LOT 2026")

    def test_upside_down_label_is_flagged(self):
        flipped = cv2.rotate(label(), cv2.ROTATE_180)
        res = OcrTool(psm=7, expected="LOT 2026", check_orientation=True).run(flipped)
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "upside down")

    def test_upright_label_passes_orientation_check(self):
        res = OcrTool(psm=7, expected="LOT 2026", check_orientation=True).run(label())
        self.assertTrue(res.ok)
        self.assertEqual(res.text.strip(), "LOT 2026")

    def test_whitelist(self):
        res = OcrTool(psm=7, whitelist="0123456789").run(label("2026"))
        self.assertEqual(res.text.strip(), "2026")


if __name__ == "__main__":
    unittest.main()
