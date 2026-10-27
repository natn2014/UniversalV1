import unittest

import cv2
import numpy as np

from app.vision.brightness import AreaTool


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


if __name__ == "__main__":
    unittest.main()
