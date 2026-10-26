import unittest

from app.roi import detections_in_crop, normalize_roi, offset_detections, roi_to_pixels


def det(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_name": "Lead"}


class NormalizeRoi(unittest.TestCase):
    def test_orders_and_clamps(self):
        self.assertEqual(normalize_roi((0.8, 0.9, -0.2, 0.1)), (0.0, 0.1, 0.8, 0.9))

    def test_none_cases(self):
        for roi in (None, (0, 0, 1, 1), (0.5, 0.5, 0.51, 0.9), ("a", 0, 1, 1), (0.1, 0.2)):
            self.assertIsNone(normalize_roi(roi), roi)

    def test_list_from_json(self):
        self.assertEqual(normalize_roi([0.25, 0.25, 0.75, 0.5]), (0.25, 0.25, 0.75, 0.5))


class Pixels(unittest.TestCase):
    def test_scales_with_resolution(self):
        roi = (0.25, 0.5, 0.75, 1.0)
        self.assertEqual(roi_to_pixels(roi, 640, 480), (160, 240, 480, 480))
        self.assertEqual(roi_to_pixels(roi, 1920, 1080), (480, 540, 1440, 1080))

    def test_minimum_crop_stays_inside_frame(self):
        x1, y1, x2, y2 = roi_to_pixels((0.99, 0.99, 1.0, 1.0), 640, 480)
        self.assertEqual((x2 - x1, y2 - y1), (32, 32))
        self.assertEqual((x2, y2), (640, 480))


class Detections(unittest.TestCase):
    def test_offset_back_to_frame(self):
        dets = [det(10, 20, 30, 40)]
        offset_detections(dets, 100, 200)
        self.assertEqual(dets[0]["x1"], 110)
        self.assertEqual(dets[0]["y2"], 240)

    def test_crop_clips_and_drops(self):
        out = detections_in_crop([det(90, 90, 150, 150), det(0, 0, 50, 50)], (100, 100, 300, 300))
        self.assertEqual([(d["x1"], d["y1"], d["x2"], d["y2"]) for d in out], [(0, 0, 50, 50)])
        self.assertEqual(out[0]["class_name"], "Lead")


if __name__ == "__main__":
    unittest.main()
