import unittest

import cv2
import numpy as np
from PySide6.QtCore import QObject

from app.camera import VideoWorker
from app.context import AppContext
from app.vision.base import Offset, Tool, register
from app.vision.measure import WidthTool
from app.vision.position import PositionTool
from app.vision.runner import run_tools

W, H = 640, 480


def scene(dx=0):
    """L-shaped locator part plus a 40 px wide bar that moves with it."""
    img = np.full((H, W), 20, np.uint8)
    cv2.rectangle(img, (100 + dx, 100), (130 + dx, 220), 255, -1)
    cv2.rectangle(img, (100 + dx, 190), (200 + dx, 220), 255, -1)
    cv2.circle(img, (165 + dx, 125), 9, 255, -1)
    img[330:400, 300 + dx:340 + dx] = 200
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def job_tools():
    pos = PositionTool(name="pos", roi=(0.05, 0.1, 0.8, 0.7))
    pos.teach(scene(), template_roi=(0.12, 0.17, 0.34, 0.5))
    width = WidthTool(name="w", roi=(0.4, 0.62, 0.7, 0.88), low=38, high=42)
    return pos, width


@register("_boom")
class Boom(Tool):
    def process(self, patch, origin):
        raise ValueError("kaboom")


class Runner(unittest.TestCase):
    def test_width_roi_follows_shifted_part(self):
        pos, width = job_tools()
        # Without the position tool the bar has left the 300 px wide ROI at dx = 120.
        moved = scene(dx=120)
        self.assertFalse(run_tools(moved, [width])[0].ok)
        results = run_tools(moved, [width, pos])
        self.assertEqual([r.kind for r in results], ["position", "width"])
        self.assertTrue(results[0].ok)
        self.assertAlmostEqual(results[0].extra["dx"], 120, delta=1)
        self.assertTrue(results[1].ok, results[1].text)
        self.assertAlmostEqual(results[1].value, 40, delta=1)

    def test_follow_position_false_keeps_roi(self):
        pos, width = job_tools()
        width.follow_position = False
        self.assertFalse(run_tools(scene(dx=120), [pos, width])[1].ok)

    def test_lost_part_returns_only_position(self):
        pos, width = job_tools()
        results = run_tools(np.full((H, W, 3), 20, np.uint8), [pos, width])
        self.assertEqual([r.kind for r in results], ["position"])
        self.assertFalse(results[0].ok)

    def test_raising_tool_is_ng_not_exception(self):
        results = run_tools(scene(), [Boom(name="bad"), WidthTool(roi=(0.4, 0.62, 0.7, 0.88))])
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].text, "kaboom")
        self.assertTrue(results[1].ok)

    def test_disabled_tools_skipped_and_timed(self):
        _, width = job_tools()
        off = WidthTool(name="off", enabled=False)
        results = run_tools(scene(), [off, width])
        self.assertEqual([r.name for r in results], ["w"])
        self.assertGreaterEqual(results[0].extra["ms"], 0)

    def test_mm_per_px_and_part_no_reach_tools(self):
        _, width = job_tools()
        res = run_tools(scene(), [width], mm_per_px=0.1, part_no="P1")[0]
        self.assertAlmostEqual(res.value, 4.0, delta=0.1)
        self.assertEqual(width.part_no, "P1")

    def test_no_tools(self):
        self.assertEqual(run_tools(scene(), []), [])


class WorkerAndContext(unittest.TestCase):
    def test_worker_setters_replace_values(self):
        w = VideoWorker()
        tools = [WidthTool()]
        w.set_tools(tools)
        w.set_mm_per_px(0.2)
        w.set_part_no("P")
        self.assertEqual((w._tools, w._mm_per_px, w._part_no), (tools, 0.2, "P"))
        self.assertIsNot(w._tools, tools)

    def test_context_pushes_to_worker(self):
        ctx = AppContext.__new__(AppContext)
        QObject.__init__(ctx)
        ctx.worker = VideoWorker()
        ctx.tools, ctx.current_part_no, ctx.mm_per_px = [], None, None
        tool = WidthTool()
        ctx.set_tools([tool])
        ctx.set_mm_per_px(0.5)
        self.assertEqual(ctx.worker._tools, [tool])
        self.assertEqual(ctx.worker._mm_per_px, 0.5)

    def test_frame_ready_has_three_arguments(self):
        got = []
        w = VideoWorker()
        w.frame_ready.connect(lambda img, dets, tools: got.append((dets, tools)))
        from PySide6.QtGui import QImage
        w.frame_ready.emit(QImage(), [], [1])
        self.assertEqual(got, [([], [1])])


if __name__ == "__main__":
    unittest.main()
