"""Part Check (AI seg) tool: side + item count judging and the failure cases,
with a fake segmentation predictor (no model or GPU needed)."""

import unittest
from unittest import mock

import numpy as np

from app.vision import create
from app.vision.base import Tool
from app.vision.segment import SegmentTool, inside, poly_centroid
from app.vision.total import total_state

FRAME = np.zeros((400, 600, 3), np.uint8)
GLASS = np.array([[50, 350], [550, 350], [400, 50]], np.float32)  # triangle part


def square(cx, cy, r=10):
    return np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]], np.float32)


SPONGES_ON_GLASS = [square(110, 330), square(500, 330), square(400, 90)]


class FakePredictor:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def __call__(self, image, conf):
        self.calls.append((image.shape, conf))
        return self.detections


def tool_with(detections, **kw) -> SegmentTool:
    kw.setdefault("both_sides", False)  # one image judged alone; BothSides tests the sequence
    tool = create("segment", **kw)
    tool.predictor = FakePredictor(detections)
    return tool


def back(n=3, conf=0.9, side_conf=0.95):
    return [("glass_back", side_conf, GLASS)] + [("sponge", conf, p) for p in SPONGES_ON_GLASS[:n]]


class Geometry(unittest.TestCase):
    def test_centroid_and_inside(self):
        cx, cy = poly_centroid(square(100, 200))
        self.assertAlmostEqual(cx, 100, places=3)
        self.assertAlmostEqual(cy, 200, places=3)
        self.assertTrue(inside(GLASS, (300, 300), 0))
        self.assertFalse(inside(GLASS, (20, 20), 0))
        self.assertTrue(inside(GLASS, (300, 355), 10))  # just below the edge, within margin


class Judging(unittest.TestCase):
    def test_back_with_three_sponges_is_ok(self):
        res = tool_with(back(3)).run(FRAME)
        self.assertTrue(res.ok)
        self.assertEqual(res.value, 3)
        self.assertEqual(res.text, "BACK 3/3")
        self.assertEqual(res.extra["side"], "back")

    def test_missing_sponge_is_ng(self):
        for n in (0, 1, 2):
            res = tool_with(back(n)).run(FRAME)
            self.assertFalse(res.ok, n)
            self.assertEqual(res.value, n)
            self.assertEqual(res.extra["reason"], "item count")

    def test_front_when_back_expected_is_ng(self):
        res = tool_with([("glass_front", 0.9, GLASS)]).run(FRAME)
        self.assertFalse(res.ok)
        self.assertEqual(res.extra["reason"], "wrong side")
        self.assertIn("FRONT", res.text)

    def test_front_expected_ignores_count(self):
        res = tool_with([("glass_front", 0.9, GLASS)], expected_side="front").run(FRAME)
        self.assertTrue(res.ok)
        self.assertEqual(res.text, "FRONT")

    def test_any_side(self):
        self.assertTrue(tool_with([("glass_front", 0.9, GLASS)], expected_side="any").run(FRAME).ok)
        self.assertTrue(tool_with(back(3), expected_side="any").run(FRAME).ok)
        self.assertFalse(tool_with(back(2), expected_side="any").run(FRAME).ok)

    def test_sponge_off_the_glass_is_not_counted(self):
        dets = back(2) + [("sponge", 0.9, square(30, 30))]  # in the parts bin
        res = tool_with(dets).run(FRAME)
        self.assertEqual(res.value, 2)
        self.assertEqual(res.extra["outside"], 1)
        self.assertFalse(res.ok)

    def test_unsure_sponge_is_not_counted(self):
        dets = back(2) + [("sponge", 0.2, SPONGES_ON_GLASS[2])]
        self.assertEqual(tool_with(dets).run(FRAME).value, 2)

    def test_best_glass_decides_the_side(self):
        dets = [("glass_front", 0.6, GLASS), ("glass_back", 0.9, GLASS)] + back(3)[1:]
        res = tool_with(dets).run(FRAME)
        self.assertEqual(res.extra["side"], "back")
        self.assertTrue(res.ok)

    def test_count_range_from_low_high(self):
        self.assertTrue(tool_with(back(2), low=2, high=2).run(FRAME).ok)
        self.assertEqual(tool_with(back(2), low=2, high=2).run(FRAME).text, "BACK 2/2")

    def test_overlay_has_glass_and_items(self):
        res = tool_with(back(3)).run(FRAME)
        kinds = [o["type"] for o in res.overlay]
        self.assertEqual(kinds.count("poly"), 4)
        self.assertEqual(kinds.count("circle"), 3)
        fills = [o.get("fill") for o in res.overlay if o["type"] == "poly"]
        self.assertEqual(fills, ["green", "blue", "blue", "blue"])  # glass green, sponges blue


FRONT = [("glass_front", 0.9, GLASS)]


class BothSides(unittest.TestCase):
    """Default rule: one glass is OK after its front and its back (3 sponges)."""

    def setUp(self):
        self.tool = create("segment")
        self.fake = FakePredictor([])
        self.tool.predictor = self.fake
        self.clock = [100.0]
        patcher = mock.patch("app.vision.segment.time.monotonic", lambda: self.clock[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def show(self, dets, dt=0.2):
        self.clock[0] += dt
        self.fake.detections = dets
        return self.tool.run(FRAME)

    def test_on_by_default(self):
        self.assertTrue(create("segment").both_sides)

    def test_back_then_front_is_ok(self):
        r = self.show(back(3))
        self.assertFalse(r.ok)
        self.assertTrue(r.extra["waiting"])
        self.assertIn("turn over: FRONT", r.text)
        self.assertEqual(total_state([r], None), "YELLOW")  # STANDBY, not NG
        r = self.show(FRONT)
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "FRONT ✓ BACK ✓")
        self.assertEqual(total_state([r], None), "GREEN")

    def test_front_then_back_is_ok(self):
        self.assertIn("turn over: BACK", self.show(FRONT).text)
        self.assertTrue(self.show(back(3)).ok)

    def test_missing_sponge_is_ng_at_once(self):
        self.show(FRONT)
        r = self.show(back(2))
        self.assertFalse(r.ok)
        self.assertFalse(r.extra["waiting"])
        self.assertEqual(total_state([r], None), "RED")
        self.assertFalse(self.show(FRONT).ok)  # the back must be shown again, fixed

    def test_only_one_side_never_ok(self):
        for _ in range(5):
            self.assertFalse(self.show(back(3)).ok)

    def test_next_part_starts_fresh_after_ok(self):
        self.show(back(3)); self.assertTrue(self.show(FRONT).ok)
        self.show([])  # the glass is taken away
        self.assertFalse(self.show(FRONT).ok)

    def test_half_checked_part_forgotten_after_reset_time(self):
        self.show(back(3))
        self.show([], dt=1.0)
        self.assertTrue(self.show(FRONT).ok)  # short gap (turning over): still the same part
        self.tool.reset_sides()
        self.show(back(3))
        self.show([], dt=5.0)  # gone longer than reset_s
        self.assertFalse(self.show([], dt=0.1).ok)
        self.assertFalse(self.show(FRONT).ok)


class Failures(unittest.TestCase):
    def test_no_part(self):
        res = tool_with([("sponge", 0.9, SPONGES_ON_GLASS[0])]).run(FRAME)
        self.assertFalse(res.ok)
        self.assertEqual(res.text, "no part")

    def test_no_model_chosen(self):
        res = create("segment").run(FRAME)
        self.assertFalse(res.ok)
        self.assertIn("segmentation model", res.text)

    def test_model_missing(self):
        res = create("segment", model="nowhere/best.pt").run(FRAME)
        self.assertFalse(res.ok)
        self.assertIn("not found", res.text)


class Saving(unittest.TestCase):
    def test_round_trip(self):
        tool = create("segment", name="Sponge", model="weights/d92_quarter_glass_seg.pt",
                      expected_side="front", conf=0.6)
        again = Tool.from_dict(tool.to_dict())
        self.assertIsInstance(again, SegmentTool)
        self.assertEqual((again.model, again.expected_side, again.conf),
                         ("weights/d92_quarter_glass_seg.pt", "front", 0.6))
        self.assertEqual((again.low, again.high), (3.0, 3.0))
        self.assertTrue(SegmentTool.SLOW)


if __name__ == "__main__":
    unittest.main()
