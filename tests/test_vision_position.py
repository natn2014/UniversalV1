import time
import unittest

import cv2
import numpy as np

from app.vision.base import Offset, Tool
from app.vision.position import PositionTool, offset_from_result

W, H = 640, 480


def part_frame(dx=0, dy=0, angle=0.0, noise=0):
    """Textured background with a white L-shaped part (centre ~ (320, 240))."""
    rng = np.random.default_rng(3)
    img = np.full((H, W), 20, np.uint8)
    part = np.zeros((H, W), np.uint8)
    cv2.rectangle(part, (280, 190), (310, 290), 255, -1)
    cv2.rectangle(part, (280, 260), (380, 290), 255, -1)
    cv2.circle(part, (345, 215), 9, 255, -1)
    m = cv2.getRotationMatrix2D((320, 240), angle, 1.0)
    m[:, 2] += (dx, dy)
    part = cv2.warpAffine(part, m, (W, H), flags=cv2.INTER_LINEAR)
    img = np.maximum(img, part)
    if noise:
        img = np.clip(img.astype(int) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def taught(**kw):
    t = PositionTool(roi=(0.25, 0.2, 0.75, 0.8), **kw)
    t.teach(part_frame(), template_roi=(0.4, 0.37, 0.62, 0.64))
    return t


class Position2D(unittest.TestCase):
    def test_translation(self):
        t = taught()
        res = t.run(part_frame(12, -7))
        off = offset_from_result(res)
        self.assertTrue(res.ok and off.found)
        self.assertAlmostEqual(off.dx, 12, delta=1)
        self.assertAlmostEqual(off.dy, -7, delta=1)
        self.assertIs(t.offset(), t._last)

    def test_same_position_is_zero(self):
        off = offset_from_result(taught().run(part_frame()))
        self.assertAlmostEqual(off.dx, 0, delta=0.5)
        self.assertAlmostEqual(off.dy, 0, delta=0.5)
        self.assertGreater(off.score, 0.95)

    def test_rotation(self):
        t = taught(angle_range=10, angle_step=1.0)
        off = offset_from_result(t.run(part_frame(angle=5.0)))
        self.assertTrue(off.found)
        self.assertAlmostEqual(off.angle, 5.0, delta=1.0)

    def test_blank_frame_not_found(self):
        t = taught()
        res = t.run(np.full((H, W, 3), 20, np.uint8))
        self.assertFalse(res.ok)
        self.assertFalse(offset_from_result(res).found)

    def test_unrelated_content_not_found(self):
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        self.assertFalse(offset_from_result(taught().run(noise)).found)

    def test_not_taught_and_empty(self):
        t = PositionTool()
        self.assertFalse(t.run(part_frame()).ok)
        self.assertFalse(t.run(np.zeros((0, 0, 3), np.uint8)).ok)

    def test_overlay_shapes(self):
        res = taught().run(part_frame(5, 5))
        kinds = [o["type"] for o in res.overlay]
        self.assertEqual(kinds, ["rect", "poly", "line", "line"])

    def test_noisy_frame(self):
        off = offset_from_result(taught().run(part_frame(8, 4, noise=10)))
        self.assertAlmostEqual(off.dx, 8, delta=1)
        self.assertAlmostEqual(off.dy, 4, delta=1)

    def test_round_trip_keeps_taught_values(self):
        t = taught()
        again = Tool.from_dict(t.to_dict())
        self.assertEqual(again.taught_center, t.taught_center)
        self.assertEqual(again.template_roi, t.template_roi)

    def test_speed_1280x720(self):
        frame = np.full((720, 1280, 3), 20, np.uint8)
        cv2.rectangle(frame, (600, 300), (700, 400), 255, -1)
        cv2.circle(frame, (650, 330), 12, 20, -1)
        t = PositionTool(roi=(0.35, 0.3, 0.66, 0.72))
        t.teach(frame, template_roi=(0.44, 0.38, 0.56, 0.58))
        moved = np.roll(frame, (6, 10), axis=(0, 1))
        best = 1e9
        for _ in range(20):
            t0 = time.perf_counter()
            res = t.run(moved)
            best = min(best, time.perf_counter() - t0)
        self.assertTrue(res.ok)
        self.assertLess(best * 1000, 15)  # target 5 ms on the Orin; slack for a busy CI box
        print(f"position 2d 1280x720: {best * 1000:.2f} ms")


class PositionAxis(unittest.TestCase):
    def step_frame(self, x):
        img = np.full((H, W, 3), 30, np.uint8)
        img[:, x:] = 200
        return img

    def test_x_mode_finds_9px_shift(self):
        t = PositionTool(roi=(0.2, 0.2, 0.8, 0.8), mode="x")
        t.teach(self.step_frame(300))
        off = offset_from_result(t.run(self.step_frame(309)))
        self.assertTrue(off.found)
        self.assertAlmostEqual(off.dx, 9, delta=0.5)
        self.assertEqual(off.dy, 0)

    def test_y_mode(self):
        def rows(y):
            img = np.full((H, W, 3), 30, np.uint8)
            img[y:, :] = 200
            return img
        t = PositionTool(roi=(0.2, 0.2, 0.8, 0.8), mode="y")
        t.teach(rows(240))
        off = offset_from_result(t.run(rows(250)))
        self.assertAlmostEqual(off.dy, 10, delta=0.5)
        self.assertEqual(off.dx, 0)

    def test_flat_frame_not_found(self):
        t = PositionTool(roi=(0.2, 0.2, 0.8, 0.8), mode="x")
        t.teach(self.step_frame(300))
        self.assertFalse(offset_from_result(t.run(np.full((H, W, 3), 90, np.uint8))).found)


if __name__ == "__main__":
    unittest.main()
