import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PySide6.QtCore import QObject

from app import context as context_mod
from app import jobs
from app.camera import VideoWorker
from app.context import AppContext
from app.vision.calibration import calibrate_from_width
from app.vision.measure import WidthTool
from app.vision.store import set_runtime


def bar(width=50):
    img = np.full((100, 300), 40, np.uint8)
    img[:, 100:100 + width] = 200
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class Calibration(unittest.TestCase):
    def test_bar_50px_is_5mm(self):
        mm = calibrate_from_width(bar(), (0.0, 0.2, 1.0, 0.8), 5.0)
        self.assertAlmostEqual(mm, 0.1, delta=0.002)

    def test_width_tool_reports_mm_after_calibration(self):
        mm = calibrate_from_width(bar(), (0.0, 0.2, 1.0, 0.8), 5.0)
        tool = WidthTool()
        set_runtime([tool], mm)
        res = tool.run(bar())
        self.assertAlmostEqual(res.value, 5.0, delta=0.1)
        self.assertIn("mm", res.text)

    def test_failures_return_none(self):
        blank = np.full((100, 300, 3), 90, np.uint8)
        self.assertIsNone(calibrate_from_width(blank, (0.0, 0.2, 1.0, 0.8), 5.0))
        self.assertIsNone(calibrate_from_width(bar(), (0.0, 0.2, 1.0, 0.8), 0))

    def test_set_runtime_sets_both(self):
        t = WidthTool()
        set_runtime([t], 0.25, "P1")
        self.assertEqual((t.mm_per_px, t.part_no), (0.25, "P1"))
        self.assertNotIn("mm_per_px", t.to_dict())


class Persistence(unittest.TestCase):
    def make_ctx(self):
        ctx = AppContext.__new__(AppContext)
        QObject.__init__(ctx)
        ctx.worker = VideoWorker()  # only stores the values here; never started
        ctx.tools, ctx.tools_gate, ctx.current_part_no, ctx.mm_per_px = [], "AND", "JOB1", None
        return ctx

    def test_survives_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(jobs, "PART_CONFIGS_DIR", root), \
                    mock.patch.object(context_mod, "TOOLS_DIR", root / "tools"):
                jobs.save_part_config("JOB1", {"part_no": "JOB1"})
                ctx = self.make_ctx()
                ctx.set_mm_per_px(0.1, save_to_job=True)
                self.assertEqual(jobs.load_part_config("JOB1")["mm_per_px"], 0.1)
                ctx.set_tools([WidthTool()], save_to_job=True)  # does not drop it
                config = jobs.load_part_config("JOB1")
                self.assertEqual(config["mm_per_px"], 0.1)

                other = self.make_ctx()
                other.load_tools(config)
                self.assertEqual(other.mm_per_px, 0.1)

    def test_old_job_has_no_calibration(self):
        ctx = self.make_ctx()
        ctx.mm_per_px = 0.3
        ctx.load_tools({"part_no": "OLD"})
        self.assertIsNone(ctx.mm_per_px)


if __name__ == "__main__":
    unittest.main()
