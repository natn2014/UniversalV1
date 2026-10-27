import unittest

import cv2
import numpy as np

from app.vision.measure import WidthTool


def bar_frame(width=40, x=100, size=(100, 300), noise=0.0, seed=0):
    img = np.full(size, 40.0)
    img[:, x:x + width] = 200.0
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class Width(unittest.TestCase):
    def test_bar_width(self):
        res = WidthTool().run(bar_frame())
        self.assertTrue(res.ok)  # no limits set
        self.assertAlmostEqual(res.value, 40, delta=0.5)
        self.assertEqual(len(res.overlay), 4)

    def test_noise(self):
        for seed in range(5):
            res = WidthTool().run(bar_frame(noise=5, seed=seed))
            self.assertAlmostEqual(res.value, 40, delta=1, msg=f"seed {seed}")

    def test_mm(self):
        tool = WidthTool()
        tool.mm_per_px = 0.1
        res = tool.run(bar_frame())
        self.assertAlmostEqual(res.value, 4.0, delta=0.05)
        self.assertIn("mm", res.text)

    def test_blank_roi_not_found(self):
        res = WidthTool().run(np.full((100, 300, 3), 90, np.uint8))
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "edge not found")

    def test_y_direction(self):
        img = bar_frame().transpose(1, 0, 2).copy()
        self.assertAlmostEqual(WidthTool(direction="y").run(img).value, 40, delta=0.5)

    def test_limits(self):
        self.assertTrue(WidthTool(low=39, high=41).run(bar_frame()).ok)
        self.assertFalse(WidthTool(low=45).run(bar_frame()).ok)

    def test_edge_pair_modes(self):
        img = bar_frame(40, 100)
        img[:, 220:224] = 200  # a thin extra bar to the right
        self.assertAlmostEqual(WidthTool(edge_pair="outer").run(img).value, 124, delta=1)
        self.assertAlmostEqual(WidthTool(edge_pair="first_rising_falling").run(img).value, 40, delta=0.5)
        self.assertAlmostEqual(WidthTool(edge_pair="strongest").run(img).value, 40, delta=4)

    def test_polarity_requirements(self):
        self.assertAlmostEqual(WidthTool(polarity_1="rising", polarity_2="falling").run(bar_frame()).value, 40, delta=0.5)
        self.assertFalse(WidthTool(polarity_1="falling", polarity_2="rising").run(bar_frame()).ok)


if __name__ == "__main__":
    unittest.main()
