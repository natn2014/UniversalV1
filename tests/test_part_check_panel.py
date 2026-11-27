"""Part Check card: the checklist rows and banner for each Part Check result,
driven through the real SegmentTool with a fake predictor."""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

APP = QApplication.instance() or QApplication([])

from app.vision import create  # noqa: E402
from app.widgets.part_check_panel import PartCheckPanel  # noqa: E402
from tests.test_vision_segment import FRAME, FRONT, FakePredictor, back  # noqa: E402


class Panel(unittest.TestCase):
    def setUp(self):
        self.tool = create("segment")
        self.fake = FakePredictor([])
        self.tool.predictor = self.fake
        self.panel = PartCheckPanel()
        self.clock = [100.0]
        patcher = mock.patch("app.vision.segment.time.monotonic", lambda: self.clock[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def show(self, dets, dt=0.2):
        self.clock[0] += dt
        self.fake.detections = dets
        res = self.tool.run(FRAME)
        self.panel.show_result(res, "sponge")
        return res

    def test_empty(self):
        self.panel.show_result(None)
        self.assertEqual(self.panel.banner.text(), "NO GLASS")
        self.assertEqual([c for _, c in self.panel.rows()], ["—"] * 5)

    def test_all_correct_is_pass(self):
        self.show(back(3))
        self.assertEqual(self.panel.banner.text(), "TURN OVER")
        self.assertEqual(self.panel.rows(), [
            ("Glass front side", "—"), ("Glass back side", "✓ OK"),
            ("Clip seal sponge 1/3", "✓ OK"), ("Clip seal sponge 2/3", "✓ OK"), ("Clip seal sponge 3/3", "✓ OK")])
        self.assertIn("front", self.panel.hint.text())
        self.show(FRONT)
        self.assertEqual(self.panel.banner.text(), "PASS")
        self.assertEqual([c for _, c in self.panel.rows()], ["✓ OK"] * 5)

    def test_missing_sponges_are_ng_rows(self):
        self.show(back(1))
        self.assertEqual(self.panel.banner.text(), "NG")
        self.assertEqual([c for _, c in self.panel.rows()], ["—", "✓ OK", "✓ OK", "✗ NG", "✗ NG"])

    def test_no_sponge_at_all(self):
        self.show(back(0))
        self.assertEqual([c for _, c in self.panel.rows()][2:], ["✗ NG"] * 3)
        self.assertEqual(self.panel.banner.text(), "NG")

    def test_checked_back_kept_while_turning_over(self):
        self.show(back(3))
        self.show([], dt=1.0)  # no glass for a moment while it is turned
        self.assertEqual(self.panel.banner.text(), "TURN OVER")
        self.assertEqual(self.panel.rows()[1], ("Glass back side", "✓ OK"))

    def test_resets_for_next_glass(self):
        self.show(back(3)); self.show(FRONT)
        self.show([])
        self.assertEqual(self.panel.banner.text(), "NO GLASS")


if __name__ == "__main__":
    unittest.main()
