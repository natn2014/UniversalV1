import unittest

import cv2
import numpy as np

from app.vision.base import Offset
from app.vision.color import (
    ColorAreaTool,
    ColorAverageTool,
    ColorProhibitTool,
    delta_e_2000,
    merge_hsv_range,
    widen_hsv_range,
)

RED = (0, 0, 220)
BLUE = (200, 40, 20)
GRAY = (90, 90, 90)


def frame_with(color, frac=1.0, bg=GRAY, size=(200, 300)):
    """Frame whose left `frac` of the columns is `color`."""
    img = np.full((*size, 3), bg, np.uint8)
    img[:, : int(size[1] * frac)] = color
    return img


class ColorArea(unittest.TestCase):
    def test_quarter_patch(self):
        t = ColorAreaTool()
        res = t.run(frame_with(RED, 0.25))
        self.assertAlmostEqual(res.value, 25.0, delta=0.5)
        self.assertEqual(res.overlay[0]["type"], "rect")
        self.assertEqual(res.overlay[1]["type"], "region")

    def test_limits_decide_ok(self):
        self.assertTrue(ColorAreaTool(low=20, high=30).run(frame_with(RED, 0.25)).ok)
        self.assertFalse(ColorAreaTool(low=30).run(frame_with(RED, 0.25)).ok)

    def test_hue_wrap_both_sides_of_zero(self):
        hsv = np.zeros((100, 100, 3), np.uint8)
        hsv[:, :50] = (2, 200, 200)    # just above 0
        hsv[:, 50:] = (177, 200, 200)  # just below 179
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertGreater(ColorAreaTool().run(img).value, 95)
        self.assertLess(ColorAreaTool(h_lo=100, h_hi=130).run(img).value, 1)

    def test_teach_blue_then_run(self):
        t = ColorAreaTool()
        t.teach(frame_with(BLUE))
        self.assertGreater(t.run(frame_with(BLUE)).value, 98)
        self.assertLess(t.run(frame_with(RED)).value, 2)

    def test_teach_point_and_roi(self):
        img = frame_with(RED, 0.5, bg=BLUE)
        t = ColorAreaTool(roi=(0.0, 0.0, 1.0, 1.0))
        t.teach(img, point=(10, 10))
        self.assertAlmostEqual(t.run(img).value, 50, delta=1)
        t2 = ColorAreaTool()
        t2.teach(img, teach_roi=(0.6, 0.1, 0.9, 0.9))  # the blue half
        self.assertAlmostEqual(t2.run(img).value, 50, delta=1)

    def test_taught_range_survives_20_percent_brighter(self):
        base = frame_with((120, 40, 20))
        t = ColorAreaTool()
        t.teach(base)
        brighter = np.clip(base.astype(float) * 1.2, 0, 255).astype(np.uint8)
        self.assertGreaterEqual(t.run(brighter).value, 0.9 * t.run(base).value)

    def test_teach_circular_hue_around_red(self):
        rng = np.random.default_rng(0)
        hsv = np.zeros((60, 60, 3), np.uint8)
        hsv[..., 0] = (rng.integers(-5, 6, (60, 60)) % 180)
        hsv[..., 1] = 200
        hsv[..., 2] = 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        t = ColorAreaTool(h_lo=60, h_hi=70)
        t.teach(img)
        self.assertGreater(t.h_lo, t.h_hi)  # wraps through 0
        self.assertGreater(t.run(img).value, 95)

    def test_open_removes_specks(self):
        img = frame_with(GRAY)
        img[::7, ::7] = RED
        self.assertGreater(ColorAreaTool().run(img).value, 0)
        self.assertEqual(ColorAreaTool(open_px=3).run(img).value, 0)

    def test_no_match_has_no_region(self):
        res = ColorAreaTool().run(frame_with(BLUE))
        self.assertEqual(res.value, 0)
        self.assertEqual([o["type"] for o in res.overlay], ["rect"])

    def test_follows_position_offset(self):
        img = frame_with(GRAY)
        img[100:150, 150:250] = RED
        t = ColorAreaTool(roi=(0.2, 0.3, 0.5, 0.7))
        self.assertEqual(t.run(img, Offset()).value, 0)
        self.assertGreater(t.run(img, Offset(dx=100, dy=40)).value, 0)

    def test_gray_input_does_not_raise(self):
        self.assertTrue(ColorAreaTool().run(np.zeros((100, 100), np.uint8)).value == 0)


class ColorAverage(unittest.TestCase):
    def test_sharma_pairs(self):
        pairs = [
            ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
            ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
            ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
            ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
            ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ]
        for lab1, lab2, expected in pairs:
            self.assertAlmostEqual(delta_e_2000(lab1, lab2), expected, delta=0.01)
            self.assertAlmostEqual(delta_e_2000(lab2, lab1), expected, delta=0.01)

    def test_same_patch_is_near_zero(self):
        img = frame_with((60, 120, 180))
        t = ColorAverageTool()
        t.teach(img)
        res = t.run(img)
        self.assertLess(res.value, 0.5)
        self.assertTrue(res.ok)

    def test_different_color_fails(self):
        t = ColorAverageTool()
        t.teach(frame_with((60, 120, 180)))
        res = t.run(frame_with((180, 120, 60)))
        self.assertGreater(res.value, 10)
        self.assertFalse(res.ok)

    def test_not_taught(self):
        self.assertFalse(ColorAverageTool().run(frame_with(RED)).ok)

    def test_ref_lab_is_json_safe(self):
        import json
        t = ColorAverageTool()
        t.teach(frame_with(RED))
        self.assertEqual(len(json.loads(json.dumps(t.to_dict()))["ref_lab"]), 3)


class ColorProhibit(unittest.TestCase):
    def test_clean_roi_ok(self):
        res = ColorProhibitTool().run(frame_with(GRAY))
        self.assertEqual(res.value, 0)
        self.assertTrue(res.ok)

    def test_spot_is_ng(self):
        img = frame_with(GRAY)
        img[50:60, 100:110] = RED
        res = ColorProhibitTool().run(img)
        self.assertAlmostEqual(res.value, 100, delta=10)
        self.assertFalse(res.ok)
        self.assertEqual([o["type"] for o in res.overlay], ["rect", "region", "rect"])
        self.assertFalse(res.overlay[2]["ok"])
        region = res.overlay[1]
        self.assertEqual(region["color"], "red")
        self.assertEqual(int(region["mask"].max()), 255)
        self.assertEqual(cv2.countNonZero(region["mask"]), res.value)  # just the spot

    def test_scattered_noise_removed(self):
        img = frame_with(GRAY)
        img[::5, ::5] = RED
        res = ColorProhibitTool().run(img)
        self.assertEqual(res.value, 0)
        self.assertTrue(res.ok)

    def test_small_blob_below_min_is_ok(self):
        img = frame_with(GRAY)
        img[50:55, 100:105] = RED  # 25 px
        res = ColorProhibitTool().run(img)
        self.assertGreater(res.value, 0)
        self.assertTrue(res.ok)
        self.assertEqual(len(res.overlay), 1)

    def test_default_high_follows_min_blob(self):
        self.assertEqual(ColorProhibitTool().high, 29)
        self.assertEqual(ColorProhibitTool(min_blob_px=50).high, 49)
        self.assertEqual(ColorProhibitTool(high=None).high, None)



class EyedropperRange(unittest.TestCase):
    def rng(self, h_lo, h_hi, s=(80, 200), v=(50, 200)):
        return {"h_lo": h_lo, "h_hi": h_hi, "s_lo": s[0], "s_hi": s[1], "v_lo": v[0], "v_hi": v[1]}

    def test_merge_across_the_red_wrap(self):
        out = merge_hsv_range(self.rng(170, 5), self.rng(0, 15, s=(60, 255)))
        self.assertEqual((out["h_lo"], out["h_hi"], out["s_lo"], out["s_hi"]), (170, 15, 60, 255))

    def test_merge_takes_the_shorter_arc(self):
        out = merge_hsv_range(self.rng(10, 20), self.rng(100, 110))
        self.assertEqual((out["h_lo"], out["h_hi"]), (10, 110))
        out = merge_hsv_range(self.rng(10, 20), self.rng(160, 170))
        self.assertEqual((out["h_lo"], out["h_hi"]), (160, 20))

    def test_merge_with_full_hue_stays_full(self):
        out = merge_hsv_range(self.rng(0, 179), self.rng(30, 40))
        self.assertEqual((out["h_lo"], out["h_hi"]), (0, 179))

    def test_widen_and_narrow(self):
        r = self.rng(175, 5)
        wide = widen_hsv_range(r, 1)
        self.assertEqual((wide["h_lo"], wide["h_hi"], wide["s_lo"], wide["s_hi"]), (172, 8, 70, 210))
        narrow = widen_hsv_range(r, -20)
        self.assertGreaterEqual((narrow["h_hi"] - narrow["h_lo"]) % 180, 8)
        self.assertLessEqual(narrow["s_lo"], narrow["s_hi"])
        self.assertEqual(widen_hsv_range(r, 40)["h_hi"], 179)

    def test_tool_pick_then_add(self):
        img = frame_with(GRAY)
        img[50:100, 100:200] = RED
        img[150:200, 100:200] = BLUE
        t = ColorAreaTool()
        t.pick(img, (150, 75))
        red = t.run(img).value
        t.pick(img, (150, 175), add=True)
        self.assertGreater(t.run(img).value, red * 1.5)
        t.pick(img, (150, 75))  # without add: starts again from red only
        self.assertAlmostEqual(t.run(img).value, red, delta=0.5)


if __name__ == "__main__":
    unittest.main()
