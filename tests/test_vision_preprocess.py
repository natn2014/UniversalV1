import unittest

import cv2
import numpy as np

from app.vision import preprocess as pp


def ramp_with_square():
    ramp = np.tile(np.linspace(0, 200, 400, dtype=np.float32), (300, 1))
    img = ramp.copy()
    img[140:160, 190:210] *= 0.25  # a 20 px dark square, small next to the 51 px kernel
    return ramp.astype(np.uint8), img.astype(np.uint8)


class FlatField(unittest.TestCase):
    def test_removes_gradient_keeps_feature(self):
        ramp, img = ramp_with_square()
        out = pp.flat_field(img, 51)
        # Background = the rows without the square, away from the border.
        bg_before = img[20:100, 60:340].astype(float)
        bg_after = out[20:100, 60:340].astype(float)
        self.assertLess(bg_after.std(), 0.3 * bg_before.std())
        contrast = out[20:100, 150:250].mean() - out[145:155, 195:205].mean()
        self.assertGreater(contrast, 40)

    def test_ignore_dark_keeps_background_next_to_dense_print_flat(self):
        img = np.full((80, 400), 200, np.uint8)
        for x in range(20, 380, 9):
            img[25:55, x:x + 5] = 40  # 5 px strokes close together, like a line of bold text
        plain = pp.flat_field(img, 51)
        ignored = pp.flat_field(img, 51, ignore_dark=True)
        # Plain: the band darkens the blur, so the paper next to it is lifted.
        self.assertGreater(int(plain[20, 200]) - int(plain[2, 200]), 15)
        self.assertLess(abs(int(ignored[20, 200]) - int(ignored[2, 200])), 5)

    def test_bgr_and_dtype(self):
        _, img = ramp_with_square()
        out = pp.flat_field(cv2.merge([img] * 3), 31)
        self.assertEqual((out.dtype, out.shape), (np.uint8, img.shape + (3,)))


class Morph(unittest.TestCase):
    def test_top_hat_finds_bright_dot(self):
        ramp, _ = ramp_with_square()
        img = ramp.copy()
        img[100:103, 100:103] = 255
        out = pp.top_hat(img, 15)
        self.assertGreater(out[101, 101], 40)
        self.assertLess(out[200, 300], 5)

    def test_black_hat_finds_dark_dot(self):
        img = np.full((100, 100), 180, np.uint8)
        img[50:53, 50:53] = 20
        self.assertGreater(pp.black_hat(img, 15)[51, 51], 100)


class Spaces(unittest.TestCase):
    def test_gray_passthrough_and_conversion(self):
        g = np.full((8, 8), 7, np.uint8)
        self.assertIs(pp.to_gray(g), g)
        self.assertEqual(pp.to_gray(cv2.merge([g] * 3)).shape, (8, 8))

    def test_hsv_lab_shapes(self):
        bgr = np.zeros((8, 8, 3), np.uint8)
        bgr[..., 2] = 255
        self.assertEqual(pp.to_hsv(bgr)[0, 0].tolist(), [0, 255, 255])
        self.assertEqual(pp.to_lab(bgr).shape, (8, 8, 3))
        self.assertEqual(pp.to_hsv(bgr[..., 0]).shape, (8, 8, 3))


class Chain(unittest.TestCase):
    def test_matches_manual_calls(self):
        _, img = ramp_with_square()
        steps = [{"op": "flat_field", "kernel": 31}, {"op": "clahe", "clip": 3.0, "tiles": 4}]
        manual = pp.clahe(pp.flat_field(img, 31), 3.0, 4)
        self.assertTrue(np.array_equal(pp.apply_chain(img, steps), manual))

    def test_gray_and_bgr_input(self):
        _, img = ramp_with_square()
        steps = [{"op": "top_hat", "kernel": 9}]
        self.assertEqual(pp.apply_chain(img, steps).shape, img.shape)
        self.assertEqual(pp.apply_chain(cv2.merge([img] * 3), steps).shape, img.shape + (3,))

    def test_unknown_op_and_empty(self):
        img = np.zeros((10, 10), np.uint8)
        with self.assertRaises(ValueError):
            pp.apply_chain(img, [{"op": "nope"}])
        self.assertIs(pp.apply_chain(img, []), img)


if __name__ == "__main__":
    unittest.main()
