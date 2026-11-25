import unittest

import cv2
import numpy as np

from app.vision.base import Offset
from app.vision.search import PatternSearchTool

W, H = 480, 320


def mark(angle=0.0, size=60):
    """A round "E" approval mark, dark on light, turned by `angle` (CCW)."""
    m = np.full((size, size), 220, np.uint8)
    c = size // 2
    cv2.circle(m, (c, c), c - 5, 40, 3, cv2.LINE_AA)
    cv2.putText(m, "E", (c - 11, c + 11), cv2.FONT_HERSHEY_DUPLEX, 1.0, 40, 2, cv2.LINE_AA)
    rot = cv2.getRotationMatrix2D((c - 0.5, c - 0.5), angle, 1.0)
    return cv2.warpAffine(m, rot, (size, size), borderValue=220)


def scene(placements, seed=0):
    """Light label with some noise and the mark at each (cx, cy, angle)."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(220, 3, (H, W)), 0, 255).astype(np.uint8)
    cv2.putText(img, "LOT 24", (300, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 60, 2)  # other print
    for cx, cy, angle in placements:
        m = mark(angle)
        s = m.shape[0] // 2
        img[cy - s:cy + s, cx - s:cx + s] = np.minimum(img[cy - s:cy + s, cx - s:cx + s], m)
    return cv2.merge([img] * 3)


def taught(**kw):
    """A tool taught on the mark at (90, 90), searching the whole frame."""
    t = PatternSearchTool(roi=(0, 0, 1, 1), **kw)
    t.teach(scene([(90, 90, 0)]), template_roi=(60 / W, 60 / H, 120 / W, 120 / H))
    return t


class Teach(unittest.TestCase):
    def test_template_is_the_dragged_box_exactly(self):
        t = taught()
        self.assertEqual(t.template.shape, (60, 60))  # not grown to the 32 px ROI minimum

    def test_small_box_is_kept_small(self):
        t = PatternSearchTool(roi=(0, 0, 1, 1))
        t.teach(scene([(90, 90, 0)]), template_roi=(80 / W, 80 / H, 92 / W, 94 / H))
        self.assertEqual(t.template.shape, (14, 12))

    def test_not_taught(self):
        res = PatternSearchTool().run(scene([]))
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "not taught")


class Search(unittest.TestCase):
    def test_finds_each_copy_at_its_angle(self):
        places = [(90, 90, 0), (240, 120, 35), (390, 90, -70), (160, 240, 150)]
        res = taught().run(scene(places))
        self.assertEqual(res.value, 4)
        self.assertTrue(res.ok)
        for cx, cy, angle in places:
            m = min(res.extra["matches"], key=lambda f: (f["x"] - cx) ** 2 + (f["y"] - cy) ** 2)
            self.assertLess(abs(m["x"] - cx) + abs(m["y"] - cy), 4, (cx, cy))
            self.assertLess(abs((m["angle"] - angle + 180) % 360 - 180), 6, (cx, cy, m["angle"]))
            self.assertGreater(m["score"], 0.88)

    def test_limited_angle_range_misses_a_turned_copy(self):
        res = taught(angle_range=20).run(scene([(90, 90, 0), (300, 150, 90)]))
        self.assertEqual(res.value, 1)

    def test_nothing_there_is_ng_by_default(self):
        res = taught().run(scene([]))
        self.assertEqual(res.value, 0)
        self.assertFalse(res.ok)  # DEFAULT_LOW = 1: at least one copy

    def test_count_limits(self):
        t = taught()
        t.low, t.high = 3, 3
        self.assertTrue(t.run(scene([(90, 90, 0), (240, 120, 35), (390, 90, -70)])).ok)
        self.assertFalse(t.run(scene([(90, 90, 0), (240, 120, 35)])).ok)

    def test_max_count(self):
        res = taught(max_count=2).run(scene([(90, 90, 0), (240, 120, 35), (390, 90, -70)]))
        self.assertEqual(res.value, 2)

    def test_overlay_draws_each_copy_in_frame_pixels(self):
        t = taught()
        t.roi = (0.25, 0.1, 0.75, 0.9)  # search area starts at x = 120
        res = t.run(scene([(240, 160, 30)]))
        self.assertEqual(res.value, 1)
        region = next(o for o in res.overlay if o["type"] == "region")
        size = region["mask"].shape[0]
        self.assertAlmostEqual(region["pts"][0] + size / 2, 240, delta=3)
        self.assertAlmostEqual(region["pts"][1] + size / 2, 160, delta=3)
        self.assertEqual(region["color"], "lines")
        self.assertTrue(any(o["type"] == "text" and o["label"].startswith("1:") for o in res.overlay))

    def test_search_area_follows_position(self):
        t = taught()
        t.roi = (0.0, 0.0, 0.4, 0.5)
        moved = scene([(90 + 150, 90 + 80, 0)])
        self.assertEqual(t.run(moved).value, 0)
        self.assertEqual(t.run(moved, Offset(dx=150, dy=80)).value, 1)

    def test_is_a_slow_tool(self):
        self.assertTrue(PatternSearchTool.SLOW)

    def test_survives_a_save_round_trip(self):
        t = taught()
        clone = PatternSearchTool.from_dict(t.to_dict())
        clone.template = t.template
        self.assertEqual(clone.run(scene([(300, 200, 60)])).value, 1)


if __name__ == "__main__":
    unittest.main()
