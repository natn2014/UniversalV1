import random
import unittest

import cv2
import numpy as np

from app.perspective import PerspectiveWarp, normalize_quad, order_corners, output_size, quad_to_pixels

# A tilted work area as an angled camera sees it: narrower at the top.
TRAPEZOID = [[0.3, 0.2], [0.7, 0.2], [0.9, 0.8], [0.1, 0.8]]


class Corners(unittest.TestCase):
    def test_any_order_becomes_tl_tr_br_bl(self):
        for _ in range(20):
            pts = TRAPEZOID[:]
            random.shuffle(pts)
            self.assertEqual(normalize_quad(pts), TRAPEZOID)

    def test_rotated_square(self):
        diamond = [[0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]]
        ordered = order_corners(np.array(diamond))
        self.assertEqual(len({tuple(p) for p in ordered}), 4)
        self.assertIsNotNone(normalize_quad(diamond))

    def test_rejects(self):
        cases = {
            "none": None,
            "three points": [[0, 0], [1, 0], [1, 1]],
            "text": [["a", 0], [1, 0], [1, 1], [0, 1]],
            "tiny": [[0.5, 0.5], [0.52, 0.5], [0.52, 0.52], [0.5, 0.52]],
            "concave": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.2], [0.5, 0.9]],
            "three in a line": [[0.1, 0.1], [0.5, 0.1], [0.9, 0.1], [0.5, 0.9]],
        }
        for name, pts in cases.items():
            self.assertIsNone(normalize_quad(pts), name)

    def test_clamps_into_frame(self):
        quad = normalize_quad([[-0.2, -0.1], [1.3, 0.0], [1.0, 1.2], [0.0, 1.0]])
        self.assertEqual(quad, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


class Size(unittest.TestCase):
    def test_uses_longer_opposite_edges(self):
        px = quad_to_pixels(TRAPEZOID, 1000, 1000)  # top 400, bottom 800, sides ~632
        w, h = output_size(px, max_side=5000)
        self.assertEqual(w, 800)
        self.assertEqual(h, 632)

    def test_long_side_capped(self):
        w, h = output_size(quad_to_pixels(TRAPEZOID, 1000, 1000), max_side=400)
        self.assertEqual(max(w, h), 400)
        self.assertAlmostEqual(w / h, 800 / 632, places=2)


class Warp(unittest.TestCase):
    def test_corners_land_on_rectangle_corners(self):
        frame = np.zeros((600, 800, 3), np.uint8)
        px = quad_to_pixels(TRAPEZOID, 800, 600).astype(int)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        for (x, y), color in zip(px, colors):
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
        warp = PerspectiveWarp(TRAPEZOID)
        out = warp.apply(frame)
        h, w = out.shape[:2]
        self.assertEqual((w, h), warp.size)
        for (x, y), color in zip([(2, 2), (w - 3, 2), (w - 3, h - 3), (2, h - 3)], colors):
            self.assertEqual(tuple(int(c) for c in out[y, x]), color)

    def test_straightens_a_tilted_rectangle(self):
        # Draw a white rectangle "seen at an angle" (inverse warp), then
        # check the corrected view shows an axis-aligned rectangle.
        out_w, out_h = 400, 300
        flat = np.zeros((out_h, out_w), np.uint8)
        cv2.rectangle(flat, (100, 75), (300, 225), 255, -1)
        src = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
        dst = quad_to_pixels(TRAPEZOID, 800, 600)
        angled = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(src, dst), (800, 600))
        warp = PerspectiveWarp(TRAPEZOID)
        corrected = warp.apply(angled)
        x, y, w, h = cv2.boundingRect(cv2.findNonZero((corrected > 127).astype(np.uint8)))
        filled = np.count_nonzero(corrected[y:y + h, x:x + w] > 127) / (w * h)
        self.assertGreater(filled, 0.97)  # a filled, axis-aligned rectangle
        cw, ch = warp.size
        self.assertAlmostEqual(x / cw, 0.25, delta=0.02)
        self.assertAlmostEqual((x + w) / cw, 0.75, delta=0.02)

    def test_matrix_rebuilt_for_new_frame_size(self):
        warp = PerspectiveWarp(TRAPEZOID)
        warp.apply(np.zeros((480, 640, 3), np.uint8))
        small = warp.size
        warp.apply(np.zeros((1080, 1920, 3), np.uint8))
        self.assertGreater(warp.size[0], small[0])


if __name__ == "__main__":
    unittest.main()
