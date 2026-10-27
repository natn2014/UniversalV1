"""Monitor tab with vision tools, offscreen. Settings and NG-log writes are
redirected to a temp dir; no camera or tower light is touched."""

import os
import tempfile
import unittest

import numpy as np
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import app.settings as settings_mod  # noqa: E402
import app.tabs.monitor_tab as monitor_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.tabs.monitor_tab import MonitorTab  # noqa: E402
from app.vision.base import ToolResult  # noqa: E402
from app.widgets.tool_overlay import OVERLAY_NG, OVERLAY_OK, tool_value_text  # noqa: E402

APP = QApplication.instance() or QApplication([])


def det(cls="Lead", conf=0.9):
    return {"class_name": cls, "conf": conf, "x1": 10, "y1": 10, "x2": 40, "y2": 40, "label": cls}


def res(name, ok, kind="area", value=1.0, overlay=None, text=""):
    return ToolResult(name, kind, value, ok, overlay or [], text)


class MonitorTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", Path(self.tmp.name) / "app_settings.json"),
            mock.patch.object(monitor_mod, "save_ng_image", mock.Mock()),
        ]
        for p in self.patches:
            p.start()
        self.ctx = AppContext()
        self.ctx.filters.selected_classes = {"Lead", "Other"}
        self.ctx.filters.target_class_1 = "Lead"
        self.tab = MonitorTab(self.ctx)
        self.image = QImage(200, 100, QImage.Format_RGB888)
        self.image.fill(QColor("#202020"))

    def tearDown(self):
        self.tab.deleteLater()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def state(self, detections, results=None):
        self.tab._reset_judgement()
        self.tab.on_frame(self.image, detections, results)
        return self.tab._latch.state

    def test_no_tools_behaves_as_yolo_only(self):
        self.assertEqual(self.state([det("Lead")]), "GREEN")
        self.assertEqual(self.state([det("Other")]), "RED")
        self.assertEqual(self.state([]), "YELLOW")
        self.assertEqual(self.state([det("Lead")], []), "GREEN")  # explicit empty list too
        self.assertEqual(self.tab.judgement_banner.text(), "OK")
        self.assertTrue(self.tab.tool_table.isHidden() and self.tab.tool_header.isHidden())
        self.assertEqual(self.tab.tool_table.rowCount(), 0)

    def test_old_two_argument_signal_still_works(self):
        self.tab._reset_judgement()
        self.tab.on_frame(self.image, [det("Lead")])
        self.assertEqual(self.tab._latch.state, "GREEN")

    def test_tools_alone_decide_without_a_model(self):
        self.assertEqual(self.state([], [res("a", True), res("b", True)]), "GREEN")
        self.assertEqual(self.state([], [res("a", True), res("b", False)]), "RED")
        self.ctx.tools_gate = "OR"
        self.assertEqual(self.state([], [res("a", True), res("b", False)]), "GREEN")

    def test_yolo_is_one_more_input_with_a_model(self):
        self.ctx.worker.set_model_path(Path("/nonexistent/model.pt"))
        self.assertEqual(self.state([det("Lead")], [res("a", True)]), "GREEN")
        self.assertEqual(self.state([det("Other")], [res("a", True)]), "RED")
        self.assertEqual(self.state([], [res("a", True)]), "YELLOW")  # YOLO sees no part
        self.assertEqual(self.state([det("Lead")], [res("a", False)]), "RED")

    def test_lost_position_is_standby(self):
        results = [res("pos", False, kind="position"), ]
        self.assertEqual(self.state([], results), "YELLOW")
        results = [res("pos", True, kind="position"), res("w", False, kind="width")]
        self.assertEqual(self.state([], results), "RED")

    def test_tool_table_rows_and_verdicts(self):
        self.ctx.mm_per_px = 0.1
        self.state([], [res("Width A", True, "width", 4.0), res("Area B", False, "area", 12.345)])
        t = self.tab.tool_table
        self.assertFalse(t.isHidden())
        self.assertEqual(t.rowCount(), 2)
        self.assertEqual([t.item(0, c).text() for c in range(3)], ["Width A", "4.00 mm", "OK"])
        self.assertEqual([t.item(1, c).text() for c in range(3)], ["Area B", "12.35", "NG"])
        self.state([], [])
        self.assertEqual(t.rowCount(), 0)

    def test_panel_hides_when_tools_removed(self):
        self.state([], [res("a", True)])
        self.assertFalse(self.tab.tool_table.isHidden())
        self.ctx.set_tools([])
        self.assertTrue(self.tab.tool_table.isHidden())

    def test_overlay_is_drawn_in_verdict_colors(self):
        green = {"type": "rect", "pts": [10, 10, 60, 40], "ok": True, "label": ""}
        red = {"type": "line", "pts": [100, 60, 180, 60], "ok": False, "label": ""}
        pixmap = self.tab._render_frame(self.image, [], [res("a", True, overlay=[green]), res("b", False, overlay=[red])])
        img = pixmap.toImage()
        self.assertEqual(img.pixelColor(10, 25).name(), OVERLAY_OK.name())
        self.assertEqual(img.pixelColor(140, 60).name(), OVERLAY_NG.name())
        self.assertEqual(img.pixelColor(150, 20).name(), "#202020")

    def test_region_is_see_through_yellow(self):
        from app.widgets.tool_overlay import draw_region
        img = QImage(60, 60, QImage.Format_RGB32)
        img.fill(QColor("#202020"))
        painter = QPainter(img)
        mask = np.full((40, 40), 255, np.uint8)  # drawn at frame (10, 10)-(49, 49)
        mask[15:25, 15:25] = 0  # hole
        draw_region(painter, 10, 10, mask)
        painter.end()
        self.assertEqual(img.pixelColor(10, 30).name(), "#b07800")  # darker solid edge
        inside = img.pixelColor(18, 18)
        self.assertGreater(inside.red(), 0x60)  # tinted yellow ...
        self.assertLess(inside.red(), 0xd0)     # ... but the image shows through
        self.assertLess(inside.blue(), inside.red())
        self.assertEqual(img.pixelColor(30, 30).name(), "#202020")  # hole stays clear

    def test_all_overlay_types_draw_without_error(self):
        items = [
            {"type": "rect", "pts": [1, 1, 20, 20], "ok": True, "label": "r"},
            {"type": "mask_box", "pts": [1, 1, 20, 20], "ok": False, "label": "m"},
            {"type": "line", "pts": [0, 0, 30, 30], "ok": True, "label": "l"},
            {"type": "circle", "pts": [50, 50, 10], "ok": True, "label": "c"},
            {"type": "poly", "pts": [[0, 0], [30, 0], [30, 30]], "ok": False, "label": "p"},
            {"type": "text", "pts": [5, 90], "ok": True, "label": "t"},
            {"type": "region", "pts": [2, 2], "mask": np.full((20, 20), 255, np.uint8), "ok": True, "label": "a"},
            {"type": "region", "pts": [2, 2], "mask": np.zeros((0, 0), np.uint8), "ok": True, "label": ""},
            {"type": "region", "pts": [2, 2], "ok": True, "label": "no mask"},
            {"type": "bogus", "pts": [], "ok": True, "label": "x"},
        ]
        self.tab._render_frame(self.image, [], [res("a", True, overlay=items)])

    def test_cycle_counted_once_per_latched_state(self):
        before = self.ctx.settings["ng_count"]
        self.state([], [res("a", False)])
        self.assertEqual(self.ctx.settings["ng_count"], before + 1)
        self.tab.on_frame(self.image, [], [res("a", False)])  # same state, within hold time
        self.assertEqual(self.ctx.settings["ng_count"], before + 1)
        monitor_mod.save_ng_image.assert_called()  # NG frame saved (patched)

    def test_value_text(self):
        self.assertEqual(tool_value_text(res("w", True, "width", 4.0), True), "4.00 mm")
        self.assertEqual(tool_value_text(res("w", True, "width", 40.0), False), "40.00")
        self.assertEqual(tool_value_text(res("c", True, "area", 4.0), True), "4.00")
        self.assertEqual(tool_value_text(res("o", True, "ocr", 90.0, text="LOT 1\nx"), True), "LOT 1")


if __name__ == "__main__":
    unittest.main()
