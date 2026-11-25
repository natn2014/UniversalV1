import unittest

import cv2
import numpy as np

from app.vision.measure import DiameterTool, EdgePresenceTool, PitchTool, WidthTool


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


class EdgePresence(unittest.TestCase):
    def test_step_is_ok(self):
        res = EdgePresenceTool().run(bar_frame())
        self.assertTrue(res.ok)
        self.assertGreater(res.value, 20)

    def test_flat_is_ng(self):
        res = EdgePresenceTool().run(np.full((100, 300, 3), 90, np.uint8))
        self.assertFalse(res.ok)
        self.assertAlmostEqual(res.value, 0, delta=0.5)

    def test_wrong_polarity_is_ng(self):
        img = np.full((100, 300), 40, np.uint8)
        img[:, 150:] = 200  # only a rising edge
        self.assertTrue(EdgePresenceTool(polarity="rising").run(img).ok)
        self.assertFalse(EdgePresenceTool(polarity="falling").run(img).ok)

    def test_custom_low(self):
        self.assertFalse(EdgePresenceTool(low=500).run(bar_frame()).ok)


def bars(n=8, pitch=20, skip=None, size=(60, 240)):
    img = np.full(size, 30, np.uint8)
    for i in range(n):
        if i != skip:
            img[:, 20 + i * pitch:20 + i * pitch + 10] = 200
    return img


class Pitch(unittest.TestCase):
    def test_eight_bars(self):
        res = PitchTool().run(bars())
        self.assertEqual(res.value, 8)
        self.assertAlmostEqual(res.extra["mean"], 20, delta=0.3)
        self.assertEqual(len(res.overlay), 9)

    def test_one_bar_missing(self):
        res = PitchTool().run(bars(skip=3))
        self.assertEqual(res.value, 7)
        self.assertAlmostEqual(res.extra["max"], 40, delta=0.5)
        self.assertAlmostEqual(res.extra["min"], 20, delta=0.5)

    def test_measure_and_mm(self):
        tool = PitchTool(measure="mean")
        tool.mm_per_px = 0.5
        self.assertAlmostEqual(tool.run(bars()).value, 10, delta=0.2)
        self.assertAlmostEqual(PitchTool(measure="max").run(bars(skip=3)).value, 40, delta=0.5)

    def test_count_limits(self):
        self.assertTrue(PitchTool(low=8, high=8).run(bars()).ok)
        self.assertFalse(PitchTool(low=8, high=8).run(bars(skip=3)).ok)

    def test_too_few_edges(self):
        res = PitchTool(measure="mean").run(np.full((60, 240), 90, np.uint8))
        self.assertFalse(res.ok)
        self.assertEqual(PitchTool().run(np.full((60, 240), 90, np.uint8)).value, 0)


def disc(r=30, size=(120, 140), center=(65, 58)):
    img = np.full(size, 20, np.uint8)
    cv2.circle(img, center, r, 220, -1)
    return img


class Diameter(unittest.TestCase):
    def test_filled_circle(self):
        res = DiameterTool().run(disc())
        self.assertAlmostEqual(res.value, 60, delta=1)
        self.assertAlmostEqual(res.extra["center"][0], 65, delta=1)
        self.assertAlmostEqual(res.extra["center"][1], 58, delta=1)
        self.assertEqual(res.overlay[1]["type"], "circle")

    def test_outliers(self):
        img = disc()
        for x, y in [(10, 10), (125, 15), (20, 105), (118, 108), (70, 8)]:
            img[y - 1:y + 2, x - 1:x + 2] = 255  # isolated specks make stray edge points
        res = DiameterTool().run(img)
        self.assertAlmostEqual(res.value, 60, delta=1.5)

    def test_blank_roi(self):
        res = DiameterTool().run(np.full((100, 100, 3), 90, np.uint8))
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "circle not found")

    def test_mm_and_limits(self):
        tool = DiameterTool(low=5.9, high=6.1)
        tool.mm_per_px = 0.1
        res = tool.run(disc())
        self.assertAlmostEqual(res.value, 6.0, delta=0.1)
        self.assertTrue(res.ok)


if __name__ == "__main__":
    unittest.main()
