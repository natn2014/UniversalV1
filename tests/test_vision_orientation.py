"""Orientation (AI) tool: angle math, judging, teaching and the failure cases,
with a fake pose predictor (no model or GPU needed)."""

import unittest

import numpy as np

from app.vision import create
from app.vision.base import Tool
from app.vision.orientation import (
    OrientationTool,
    angle_difference,
    direction_angle,
    resolve_model_path,
)

FRAME = np.zeros((200, 300, 3), np.uint8)


def part(head, tail, conf=0.9, kpt_conf=0.9):
    """A detection whose keypoint 0 is the head and 1 the tail."""
    return conf, np.array([[head[0], head[1], kpt_conf], [tail[0], tail[1], kpt_conf]], np.float32)


class FakePredictor:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def __call__(self, image, conf):
        self.calls.append((image.shape, conf))
        return self.detections


def tool_with(detections, **kw) -> OrientationTool:
    tool = create("orientation", **kw)
    tool.predictor = FakePredictor(detections)
    return tool


class AngleMath(unittest.TestCase):
    def test_directions(self):
        c = (100.0, 100.0)
        self.assertAlmostEqual(direction_angle(c, (150, 100)), 0.0)
        self.assertAlmostEqual(direction_angle(c, (100, 50)), 90.0)   # up: image y is down
        self.assertAlmostEqual(abs(direction_angle(c, (50, 100))), 180.0)
        self.assertAlmostEqual(direction_angle(c, (100, 150)), -90.0)
        self.assertAlmostEqual(direction_angle(c, (150, 50)), 45.0)

    def test_difference_wraps(self):
        self.assertAlmostEqual(angle_difference(95, 90), 5)
        self.assertAlmostEqual(angle_difference(-175, 175), 10)
        self.assertAlmostEqual(angle_difference(175, -175), -10)
        self.assertAlmostEqual(abs(angle_difference(-90, 90)), 180)

    def test_resolve_model_path(self):
        self.assertIsNone(resolve_model_path(""))
        self.assertIsNone(resolve_model_path("does/not/exist.pt"))


class Judging(unittest.TestCase):
    def test_right_way_up_is_ok(self):
        tool = tool_with([part(head=(150, 60), tail=(150, 140))])  # pointing up
        res = tool.run(FRAME)
        self.assertTrue(res.ok)
        self.assertAlmostEqual(res.value, 0.0, places=3)
        self.assertAlmostEqual(res.extra["angle"], 90.0, places=3)
        self.assertEqual(res.text, "90.0°")
        self.assertEqual(res.extra["count"], 1)

    def test_reversed_is_ng(self):
        tool = tool_with([part(head=(150, 140), tail=(150, 60))])  # pointing down
        res = tool.run(FRAME)
        self.assertFalse(res.ok)
        self.assertAlmostEqual(abs(res.value), 180.0, places=3)

    def test_small_turn_inside_limits(self):
        tool = tool_with([part(head=(160, 60), tail=(150, 140))])  # ~7° right of up
        res = tool.run(FRAME)
        self.assertTrue(res.ok)
        self.assertLess(res.value, 0)  # clockwise = negative
        tool.high, tool.low = 3.0, -3.0
        self.assertFalse(tool.run(FRAME).ok)

    def test_worst_part_decides(self):
        tool = tool_with([part((100, 60), (100, 140)), part((200, 140), (200, 60))], max_count=0)
        res = tool.run(FRAME)
        self.assertFalse(res.ok)
        self.assertEqual(res.extra["count"], 2)
        self.assertAlmostEqual(abs(res.value), 180.0, places=3)

    def test_too_many_parts(self):
        tool = tool_with([part((100, 60), (100, 140)), part((200, 60), (200, 140))])
        res = tool.run(FRAME)
        self.assertFalse(res.ok)
        self.assertIn("2 parts", res.text)
        tool.max_count = 2
        self.assertTrue(tool.run(FRAME).ok)

    def test_custom_keypoints(self):
        # A 3-point model: arrow from point 2 to point 1.
        kpts = np.array([[0, 0, 0.9], [200, 100, 0.9], [100, 100, 0.9]], np.float32)
        tool = tool_with([(0.9, kpts)], from_kpt=2, to_kpt=1, expected_angle=0.0)
        self.assertTrue(tool.run(FRAME).ok)
        tool.to_kpt = 5
        res = tool.run(FRAME)
        self.assertFalse(res.ok)
        self.assertIn("3 keypoints", res.text)


class Failures(unittest.TestCase):
    def test_no_part(self):
        res = tool_with([]).run(FRAME)
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "no part")

    def test_unsure_keypoints(self):
        res = tool_with([part((150, 60), (150, 140), kpt_conf=0.2)]).run(FRAME)
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "keypoints unsure")

    def test_no_model_chosen(self):
        res = create("orientation").run(FRAME)
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "choose a pose model")

    def test_missing_model_file(self):
        res = create("orientation", model="nowhere/best.pt").run(FRAME)
        self.assertFalse(res.ok)
        self.assertIn("model not found", res.text)


class TeachAndSave(unittest.TestCase):
    def test_teach_sets_expected_angle(self):
        tool = tool_with([part(head=(200, 100), tail=(100, 100))])  # pointing right
        tool.teach(FRAME)
        self.assertEqual(tool.expected_angle, 0.0)
        self.assertTrue(tool.run(FRAME).ok)

    def test_teach_without_part_raises(self):
        with self.assertRaises(RuntimeError):
            tool_with([]).teach(FRAME)

    def test_runs_on_roi_and_shifts_overlay(self):
        tool = tool_with([part((50, 10), (50, 90))], roi=(0.5, 0.0, 1.0, 1.0))
        res = tool.run(FRAME)
        self.assertEqual(tool.predictor.calls[0][0][:2], (200, 150))  # the ROI patch only
        line = next(item for item in res.overlay if item["type"] == "line")
        self.assertEqual(line["pts"][0], 50 + 150)

    def test_round_trip(self):
        tool = create("orientation", name="Dir", model="runs/pose/weights/best.pt",
                      expected_angle=-90.0, from_kpt=2, to_kpt=0, low=-5.0, high=5.0)
        again = Tool.from_dict(tool.to_dict())
        self.assertIsInstance(again, OrientationTool)
        self.assertEqual(again.params(), tool.params())
        self.assertEqual((again.low, again.high), (-5.0, 5.0))
        self.assertTrue(OrientationTool.SLOW)


if __name__ == "__main__":
    unittest.main()
