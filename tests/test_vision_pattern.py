import unittest

import cv2
import numpy as np

from app.vision.pattern import OutlineTool

H, W = 240, 320


def part(shape="rect", notch=False, angle=0.0, dx=0, dy=0):
    """BGR frame with a bright part on a dark background."""
    img = np.full((H, W), 25, np.uint8)
    cx, cy = W // 2 + dx, H // 2 + dy
    if shape == "rect":
        cv2.rectangle(img, (cx - 70, cy - 45), (cx + 70, cy + 45), 210, -1)
        cv2.circle(img, (cx - 30, cy), 12, 25, -1)
    elif shape == "square":
        cv2.rectangle(img, (cx - 60, cy - 60), (cx + 60, cy + 60), 210, -1)
    elif shape == "circle":
        cv2.circle(img, (cx, cy), 60, 210, -1)
    if notch:
        cv2.rectangle(img, (cx + 40, cy - 50), (cx + 75, cy - 5), 25, -1)
    if angle:
        m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=25)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class OutlineTemplate(unittest.TestCase):
    def taught(self, **kw):
        t = OutlineTool(roi=(0.05, 0.05, 0.95, 0.95), **kw)
        t.teach(part())
        return t

    def test_same_part_scores_high(self):
        self.assertGreaterEqual(self.taught().run(part()).value, 95)

    def test_small_shift_still_matches(self):
        res = self.taught().run(part(dx=5, dy=-4))
        self.assertGreaterEqual(res.value, 90)

    def test_notch_scores_lower(self):
        t = self.taught()
        full, cut = t.run(part()).value, t.run(part(notch=True)).value
        self.assertLessEqual(cut, full - 10)

    def test_untaught_and_blank(self):
        t = OutlineTool()
        self.assertFalse(t.run(part()).ok)
        blank = np.full((H, W, 3), 25, np.uint8)
        self.assertEqual(self.taught().run(blank).value, 0)

    def test_overlay_rect(self):
        res = self.taught().run(part())
        self.assertEqual(res.overlay[1]["type"], "rect")

    def test_roi_size_change_asks_for_reteach(self):
        t = self.taught()
        t.roi = (0.1, 0.1, 0.9, 0.9)
        self.assertEqual(t.run(part()).text, "ROI size changed: teach again")


class OutlineShape(unittest.TestCase):
    def taught(self, shape="square"):
        t = OutlineTool(mode="shape", roi=(0.05, 0.05, 0.95, 0.95))
        t.teach(part(shape))
        return t

    def test_same_shape(self):
        self.assertGreaterEqual(self.taught().run(part("square")).value, 95)

    def test_circle_vs_square_is_low(self):
        self.assertLess(self.taught("square").run(part("circle")).value, 60)

    def test_rotation_invariant(self):
        self.assertGreaterEqual(self.taught("rect").run(part("rect", angle=30)).value, 90)

    def test_template_round_trip(self):
        t = self.taught()
        again = OutlineTool.from_dict(t.to_dict())
        again.template = t.template.copy()
        self.assertGreaterEqual(again.run(part("square")).value, 95)


if __name__ == "__main__":
    unittest.main()
