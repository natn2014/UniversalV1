import unittest

import cv2
import numpy as np

from app.vision.brightness import AreaTool, BrightnessAverageTool, EdgePixelsTool


def square_frame(frac=0.3, fg=230, bg=10, size=(200, 300)):
    """ROI-sized frame with a bright block covering `frac` of the area."""
    img = np.full(size, bg, np.uint8)
    img[:, : int(size[1] * frac)] = fg
    return cv2.merge([img] * 3)


class Area(unittest.TestCase):
    def test_fixed_and_otsu_bright(self):
        for method in ("fixed", "otsu"):
            res = AreaTool(method=method).run(square_frame(0.3))
            self.assertAlmostEqual(res.value, 30, delta=0.5, msg=method)

    def test_dark_polarity(self):
        for method in ("fixed", "otsu"):
            res = AreaTool(method=method, polarity="dark").run(square_frame(0.3))
            self.assertAlmostEqual(res.value, 70, delta=0.5, msg=method)

    def test_limits(self):
        self.assertTrue(AreaTool(low=25, high=35).run(square_frame(0.3)).ok)
        self.assertFalse(AreaTool(low=40).run(square_frame(0.3)).ok)

    def test_uniform_otsu_does_not_raise(self):
        res = AreaTool(method="otsu").run(np.full((100, 100, 3), 90, np.uint8))
        self.assertIn(res.value, (0.0, 100.0))

    def test_gray_input_and_mask_box(self):
        res = AreaTool().run(square_frame(0.3)[:, :, 0])
        self.assertEqual(res.overlay[1]["pts"], [0, 0, 90, 200])


class BrightnessAverage(unittest.TestCase):
    def test_uniform(self):
        res = BrightnessAverageTool().run(np.full((100, 100, 3), 100, np.uint8))
        self.assertAlmostEqual(res.value, 100, delta=0.5)
        self.assertAlmostEqual(res.extra["std"], 0, delta=0.01)

    def test_half_and_half(self):
        img = np.zeros((100, 100), np.uint8)
        img[:, 50:] = 200
        res = BrightnessAverageTool().run(img)
        self.assertAlmostEqual(res.value, 100, delta=0.5)
        self.assertAlmostEqual(res.extra["std"], 100, delta=0.5)
        self.assertAlmostEqual(BrightnessAverageTool(use_std=True).run(img).value, 100, delta=0.5)

    def test_limits(self):
        img = np.full((100, 100), 100, np.uint8)
        self.assertTrue(BrightnessAverageTool(low=90, high=110).run(img).ok)
        self.assertFalse(BrightnessAverageTool(low=110).run(img).ok)


class EdgePixels(unittest.TestCase):
    def square(self):
        img = np.zeros((200, 200), np.uint8)
        img[75:125, 75:125] = 255
        return img

    def test_flat_roi(self):
        res = EdgePixelsTool().run(np.full((100, 100, 3), 120, np.uint8))
        self.assertLess(res.value, 5)

    def test_square_outline(self):
        res = EdgePixelsTool().run(self.square())
        self.assertAlmostEqual(res.value, 200, delta=20)
        self.assertAlmostEqual(res.extra["density"], res.value / 200 / 200 * 100, places=6)

    def test_scratch_adds_about_its_length(self):
        base = EdgePixelsTool().run(self.square()).value
        img = self.square()
        cv2.line(img, (10, 20), (150, 20), 255, 1)  # 140 px long, 1 px wide
        extra = EdgePixelsTool().run(img).value - base
        self.assertGreater(extra, 100)
        self.assertLess(extra, 320)  # a thin line has an edge on each side


if __name__ == "__main__":
    unittest.main()
