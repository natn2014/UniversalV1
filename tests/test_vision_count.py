import unittest

import cv2
import numpy as np

from app.vision.count import BlobCountTool


def dots(n=7, r=8, size=(100, 400)):
    img = np.zeros(size, np.uint8)
    for i in range(n):
        cv2.circle(img, (30 + i * 50, 50), r, 255, -1)
    return img


class BlobCount(unittest.TestCase):
    def test_seven_dots(self):
        res = BlobCountTool().run(dots())
        self.assertEqual(res.value, 7)
        self.assertEqual(len(res.extra["blobs"]), 7)
        self.assertEqual([o["label"] for o in res.overlay[1:]], [str(i) for i in range(1, 8)])

    def test_area_min_drops_small_dots(self):
        img = dots()
        for i in (1, 4):  # make two of them small (area about 28)
            cv2.circle(img, (30 + i * 50, 50), 8, 0, -1)
            cv2.circle(img, (30 + i * 50, 50), 3, 255, -1)
        self.assertEqual(BlobCountTool().run(img).value, 7)
        self.assertEqual(BlobCountTool(area_min=100).run(img).value, 5)
        self.assertEqual(BlobCountTool(area_min=10, area_max=60).run(img).value, 2)

    def test_circularity_keeps_circles_drops_bars(self):
        img = dots(4)
        for i in range(3):
            cv2.rectangle(img, (230 + i * 50 - 30, 20), (230 + i * 50 - 25, 80), 255, -1)  # thin bars
        self.assertEqual(BlobCountTool().run(img).value, 7)
        self.assertEqual(BlobCountTool(circ_min=0.8).run(img).value, 4)

    def test_aspect_ratio(self):
        img = dots(3)
        cv2.rectangle(img, (200, 40), (260, 50), 255, -1)  # 60x10 bar
        self.assertEqual(BlobCountTool(aspect_min=3).run(img).value, 1)
        self.assertEqual(BlobCountTool(aspect_max=2).run(img).value, 3)

    def test_polarity_dark(self):
        self.assertEqual(BlobCountTool(polarity="dark").run(255 - dots()).value, 7)

    def test_color_source(self):
        img = np.full((100, 400, 3), 90, np.uint8)
        for i in range(5):
            cv2.circle(img, (30 + i * 60, 50), 9, (0, 0, 220), -1)
        self.assertEqual(BlobCountTool(source="color").run(img).value, 5)
        self.assertEqual(BlobCountTool(source="color", h_lo=100, h_hi=130).run(img).value, 0)

    def test_open_removes_specks_and_limits(self):
        img = dots(3)
        img[5, 5] = 255
        self.assertEqual(BlobCountTool(area_min=1, open_px=1).run(img).value, 4)
        self.assertEqual(BlobCountTool(area_min=1, open_px=3).run(img).value, 3)
        self.assertTrue(BlobCountTool(low=3, high=3).run(dots(3)).ok)
        self.assertFalse(BlobCountTool(low=3, high=3).run(dots(2)).ok)

    def test_empty_roi_is_zero(self):
        self.assertEqual(BlobCountTool().run(np.zeros((100, 100), np.uint8)).value, 0)


if __name__ == "__main__":
    unittest.main()
