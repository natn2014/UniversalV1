"""Tools tab, offscreen. Job configs go to a temp PART_CONFIGS_DIR; no camera,
tower light or kiosk is touched."""

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit  # noqa: E402

import app.context as context_mod  # noqa: E402
import app.jobs as jobs  # noqa: E402
import app.settings as settings_mod  # noqa: E402
import app.tabs.tools_tab as tools_tab_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.tabs.tools_tab import ToolsTab  # noqa: E402
from app.widgets.tool_params import CATEGORIES, KIND_USES, THUMBS_DIR, AddToolDialog  # noqa: E402

APP = QApplication.instance() or QApplication([])
RED = (0, 0, 220)


def frame():
    """Gray scene: a red patch, a white bar 40 px wide and a bright locator shape."""
    img = np.full((480, 640, 3), 90, np.uint8)
    img[100:200, 100:300] = RED
    img[300:380, 200:240] = 230
    cv2.rectangle(img, (400, 100), (440, 220), 255, -1)
    cv2.rectangle(img, (400, 190), (520, 220), 255, -1)
    cv2.circle(img, (470, 130), 12, 20, -1)
    return img


class ToolsTabTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", root / "app_settings.json"),
            mock.patch.object(jobs, "PART_CONFIGS_DIR", root),
            mock.patch.object(context_mod, "TOOLS_DIR", root / "tools"),
        ]
        for p in self.patches:
            p.start()
        jobs.save_part_config("JOB1", {"part_no": "JOB1", "roi": [0.1, 0.1, 0.5, 0.5]})
        self.ctx = AppContext()
        self.ctx.current_part_no = "JOB1"
        self.tab = ToolsTab(self.ctx)
        self.tab.resize(1024, 700)
        self.tab.set_frame(frame())

    def tearDown(self):
        self.tab.deleteLater()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def saved_tools(self):
        return jobs.load_part_config("JOB1").get("tools", [])


class AddTeachDelete(ToolsTabTest):
    def test_add_color_area_then_delete(self):
        tool = self.tab.add_tool("color_area", (0.18, 0.23, 0.44, 0.4))
        self.assertEqual(len(self.saved_tools()), 1)
        self.assertEqual(self.saved_tools()[0]["kind"], "color_area")
        self.assertEqual(self.tab.table.rowCount(), 1)
        self.assertIs(self.tab.selected_tool(), tool)
        self.assertGreater(self.tab._results[tool.name].value, 95)  # taught from its ROI: all red
        self.assertEqual(self.tab.table.item(0, 3).text(), "OK")
        self.assertEqual(jobs.load_part_config("JOB1")["roi"], [0.1, 0.1, 0.5, 0.5])  # rest untouched

        self.tab.delete_selected()
        self.assertEqual(self.saved_tools(), [])
        self.assertEqual(self.tab.table.rowCount(), 0)
        self.assertIsNone(self.tab.selected_tool())

    def test_draw_flow_through_the_view(self):
        self.tab.begin_add("width")
        self.assertEqual(self.tab._mode, "add")
        self.tab.view.roi_drawn.emit((0.25, 0.6, 0.5, 0.85))
        self.assertIsNone(self.tab._mode)
        self.assertEqual([t["kind"] for t in self.saved_tools()], ["width"])
        res = self.tab._results[self.tab.tools[0].name]
        self.assertAlmostEqual(res.value, 40, delta=1)

    def test_snapshot_needed_before_drawing(self):
        self.tab.set_frame(None)
        self.tab.begin_add("width")
        self.assertIsNone(self.tab._mode)

    def test_position_added_is_taught_and_template_saved(self):
        self.tab.add_tool("position", (0.55, 0.1, 0.9, 0.6))
        entry = self.saved_tools()[0]
        self.assertTrue(entry["template_file"])
        self.assertTrue((Path(self.tmp.name) / "tools" / entry["template_file"]).exists())
        self.assertTrue(self.tab._results[entry["name"]].ok)

    def test_teach_position_with_template_box(self):
        tool = self.tab.add_tool("position", (0.55, 0.1, 0.9, 0.6))
        self.tab._on_teach_clicked()
        self.assertEqual(self.tab._mode, "teach_box")
        self.tab.view.roi_drawn.emit((0.62, 0.2, 0.8, 0.45))
        self.assertEqual(tool.template_roi, [0.62, 0.2, 0.8, 0.45])

    def test_pick_color_teaches_from_a_point(self):
        tool = self.tab.add_tool("color_area", (0.1, 0.1, 0.9, 0.9))
        self.assertEqual(self.tab.pick_color_button.text(), "Pick Color")
        self.tab.pick_color_button.click()
        self.assertEqual(self.tab._mode, "pick")
        self.assertTrue(self.tab.pick_card.isVisibleTo(self.tab))
        self.tab.view.point_picked.emit((0.31, 0.31))  # inside the red patch
        self.assertEqual(self.tab._mode, "pick")  # stays open for more taps
        self.assertTrue(tool.h_lo > tool.h_hi or tool.h_lo <= 5)  # red hue range
        self.assertEqual(self.saved_tools()[0]["h_lo"], tool.h_lo)  # saved to the job

    def test_more_taps_add_and_buttons_adjust(self):
        tool = self.tab.add_tool("color_area", (0.1, 0.1, 0.9, 0.9))
        before = dict(tool.hsv_range())
        self.tab.pick_color_button.click()
        self.tab.view.point_picked.emit((0.31, 0.31))  # red
        red_only = tool.run(frame()).value
        self.tab.view.point_picked.emit((0.05, 0.9))  # gray background: added
        self.assertEqual(self.tab._pick_taps, 2)
        self.assertGreater(tool.run(frame()).value, red_only + 30)
        s_lo = tool.s_lo
        self.tab.wider_button.click()
        self.assertLessEqual(tool.s_lo, s_lo)
        v_lo = tool.v_lo
        self.tab.narrow_button.click()
        self.assertGreaterEqual(tool.v_lo, v_lo)
        self.tab.clear_pick_button.click()
        self.assertEqual(tool.hsv_range(), before)
        self.assertEqual(self.tab._pick_taps, 0)
        self.assertFalse(self.tab.wider_button.isEnabled())
        self.tab.done_pick_button.click()
        self.assertIsNone(self.tab._mode)
        self.assertFalse(self.tab.pick_card.isVisibleTo(self.tab))

    def test_selecting_another_tool_ends_the_pick(self):
        self.tab.add_tool("color_area", (0.1, 0.1, 0.9, 0.9))
        self.tab.add_tool("color_prohibit", (0.1, 0.1, 0.9, 0.9))
        self.tab.pick_color_button.click()
        self.tab.view.point_picked.emit((0.31, 0.31))
        self.assertEqual(self.tab._mode, "pick")
        self.tab.table.selectRow(0)
        self.assertIsNone(self.tab._mode)

    def test_pick_gray_sets_threshold_for_area(self):
        tool = self.tab.add_tool("area", (0.25, 0.55, 0.45, 0.85))  # holds the white bar
        self.assertEqual(self.tab.pick_color_button.text(), "Pick Gray")
        self.tab.pick_color_button.click()
        self.assertEqual(self.tab.pick_header.text(), "Brightness Extraction")
        self.tab.view.point_picked.emit((220 / 640, 340 / 480))  # on the bar (230) over 90
        self.assertEqual((tool.method, tool.polarity, tool.threshold), ("fixed", "bright", 160))
        bar = 40 * 80 * 100.0 / tool.crop(frame())[0].shape[0] / tool.crop(frame())[0].shape[1]
        self.assertAlmostEqual(tool.run(frame()).value, bar, delta=0.5)
        self.tab.wider_button.click()
        self.assertEqual(tool.threshold, 155)

    def test_pattern_search_asks_for_the_pattern_box_after_its_area(self):
        self.tab.begin_add("pattern_search")
        self.tab.view.roi_drawn.emit((0.1, 0.1, 0.9, 0.9))  # search area
        self.assertEqual(self.tab._mode, "teach_box")
        self.tab.view.roi_drawn.emit((0.15, 0.2, 0.48, 0.43))  # around the red patch
        tool = self.tab.tools[0]
        self.assertEqual(tool.kind, "pattern_search")
        self.assertEqual(tool.template_roi, [0.15, 0.2, 0.48, 0.43])
        self.assertEqual(tool.template.shape, (110, 211))

    def test_pick_button_follows_the_tool(self):
        self.tab.add_tool("brightness", (0.1, 0.1, 0.3, 0.3))
        self.assertFalse(self.tab.pick_color_button.isVisibleTo(self.tab))
        blob = self.tab.add_tool("blob_count", (0.1, 0.1, 0.3, 0.3))
        self.assertEqual(self.tab.pick_color_button.text(), "Pick Gray")
        blob.source = "color"
        self.tab._on_selected(self.tab.selected_index())
        self.assertEqual(self.tab.pick_color_button.text(), "Pick Color")

    def test_delete_is_a_touch_confirm(self):
        self.tab.add_tool("area", (0.1, 0.1, 0.3, 0.3))
        keep = self.tab.add_tool("width", (0.5, 0.5, 0.7, 0.7))
        self.tab.table.selectRow(0)
        with mock.patch.object(tools_tab_mod.DeleteToolDialog, "exec", return_value=QDialog.Rejected):
            self.tab._on_delete_clicked()
        self.assertEqual(len(self.saved_tools()), 2)
        with mock.patch.object(tools_tab_mod.DeleteToolDialog, "exec", return_value=QDialog.Accepted):
            self.tab._on_delete_clicked()
        self.assertEqual([t["name"] for t in self.saved_tools()], [keep.name])

    def test_delete_dialog_needs_no_typing_and_is_not_default(self):
        tool = self.tab.add_tool("pitch", (0.1, 0.1, 0.3, 0.3))
        dialog = tools_tab_mod.DeleteToolDialog(tool)
        self.assertEqual(dialog.findChildren(QLineEdit), [])
        self.assertFalse(dialog.delete_button.isDefault() or dialog.delete_button.autoDefault())
        self.assertGreaterEqual(dialog.delete_button.height(), 64)
        dialog.delete_button.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)

    def test_set_roi_for_selected_tool(self):
        tool = self.tab.add_tool("area", (0.1, 0.1, 0.3, 0.3))
        self.tab.roi_button.click()
        self.assertEqual(self.tab._mode, "set_roi")
        self.assertTrue(self.tab.roi_button.isChecked())
        self.tab.view.roi_drawn.emit((0.4, 0.4, 0.6, 0.6))
        self.assertEqual(tool.roi, (0.4, 0.4, 0.6, 0.6))
        self.assertEqual(self.saved_tools()[0]["roi"], [0.4, 0.4, 0.6, 0.6])


class LimitsParamsCalibration(ToolsTabTest):
    def test_limits_from_live_value(self):
        tool = self.tab.add_tool("brightness", (0.1, 0.2, 0.4, 0.4))
        self.tab._on_percent_clicked()
        live = self.tab._results[tool.name].value
        self.assertAlmostEqual(tool.low, live * 0.9, places=4)
        self.assertAlmostEqual(tool.high, live * 1.1, places=4)
        self.assertAlmostEqual(self.tab.low_spin.value(), live * 0.9, places=1)
        self.assertEqual(self.saved_tools()[0]["high"], tool.high)

    def test_limit_spin_boxes_edit_the_tool(self):
        tool = self.tab.add_tool("brightness", (0.1, 0.2, 0.4, 0.4))
        self.tab.high_check.setChecked(True)
        self.tab.high_spin.setValue(50.0)
        self.assertEqual(tool.high, 50.0)
        self.assertFalse(self.tab._results[tool.name].ok)  # brightness is about 90
        self.assertEqual(self.tab.table.item(0, 3).text(), "NG")
        self.tab.high_check.setChecked(False)
        self.assertIsNone(tool.high)
        self.tab.save()
        self.assertIsNone(self.saved_tools()[0]["high"])

    def test_parameter_form(self):
        tool = self.tab.add_tool("area", (0.1, 0.1, 0.4, 0.3))
        widgets = self.tab.form.widgets
        self.assertEqual({type(w).__name__ for w in (widgets["method"], widgets["threshold"])}, {"QComboBox", "QSpinBox"})
        widgets["threshold"].setValue(40)
        widgets["polarity"].setCurrentText("dark")
        self.assertEqual((tool.threshold, tool.polarity), (40, "dark"))
        self.tab.save()
        self.assertEqual(self.saved_tools()[0]["threshold"], 40)
        self.assertNotIn("pre", widgets)

    def test_form_types_for_each_param_kind(self):
        self.tab.add_tool("position", (0.55, 0.1, 0.9, 0.6))
        w = self.tab.form.widgets
        self.assertEqual(type(w["pyramid"]).__name__, "QCheckBox")
        self.assertEqual(type(w["min_score"]).__name__, "QDoubleSpinBox")
        self.assertEqual(type(w["mode"]).__name__, "QComboBox")
        self.assertNotIn("taught_center", w)

    def test_enabled_toggle_removes_the_result(self):
        tool = self.tab.add_tool("area", (0.1, 0.1, 0.4, 0.3))
        self.tab.enabled_check.setChecked(False)
        self.assertFalse(tool.enabled)
        self.assertEqual(self.tab.table.item(0, 3).text(), "—")

    def test_calibrate_mm(self):
        mm = self.tab.calibrate((0.25, 0.6, 0.5, 0.85), 5.0)  # the 40 px bar
        self.assertAlmostEqual(mm, 5.0 / 40, delta=0.003)
        self.assertEqual(jobs.load_part_config("JOB1")["mm_per_px"], mm)
        tool = self.tab.add_tool("width", (0.25, 0.6, 0.5, 0.85))
        self.assertAlmostEqual(self.tab._results[tool.name].value, 5.0, delta=0.15)
        self.assertIn("mm", self.tab.table.item(0, 2).text())

    def test_calibrate_failure(self):
        self.assertIsNone(self.tab.calibrate((0.7, 0.7, 0.9, 0.9), 5.0))
        self.assertIsNone(self.ctx.mm_per_px)

    def test_gate_toggle_saved(self):
        self.tab.add_tool("area", (0.1, 0.1, 0.4, 0.3))
        self.assertEqual(self.tab.gate_button.text(), "Gate: AND")
        self.tab.gate_button.click()
        self.assertEqual(self.tab.gate_button.text(), "Gate: OR")
        self.assertEqual(jobs.load_part_config("JOB1")["tools_gate"], "OR")

    def test_total_pill_follows_results(self):
        self.assertEqual(self.tab.total_pill.text(), "STANDBY")
        self.tab.add_tool("color_area", (0.18, 0.23, 0.44, 0.4))
        self.assertEqual(self.tab.total_pill.text(), "OK")
        self.tab.add_tool("color_prohibit", (0.18, 0.23, 0.44, 0.4))  # red is red: forbidden color present
        self.assertEqual(self.tab.total_pill.text(), "NG")

    def test_job_load_refreshes_the_list(self):
        self.tab.add_tool("area", (0.1, 0.1, 0.4, 0.3))
        config = jobs.load_part_config("JOB1")
        self.ctx.load_tools({"part_no": "OTHER"})
        self.assertEqual(self.tab.table.rowCount(), 0)
        self.ctx.load_tools(config)
        self.assertEqual(self.tab.table.rowCount(), 1)


class UiRules(ToolsTabTest):
    def test_touch_targets_are_at_least_48px(self):
        self.tab.add_tool("area", (0.1, 0.1, 0.4, 0.3))
        for b in (self.tab.teach_button, self.tab.roi_button, self.tab.delete_button, self.tab.gate_button,
                  self.tab.calibrate_button, self.tab.percent_button, self.tab.low_spin, self.tab.high_spin):
            self.assertGreaterEqual(b.minimumHeight(), 48, b)
        for w in self.tab.form.widgets.values():
            self.assertGreaterEqual(w.minimumHeight(), 48)

    def test_add_dialog_has_a_tile_per_kind_and_returns_the_pick(self):
        dialog = AddToolDialog()
        kinds = [k for _, ks in CATEGORIES for k in ks]
        self.assertEqual(len(kinds), 17)
        self.assertEqual(set(dialog.tiles), set(kinds))
        self.assertTrue(all(t.minimumHeight() >= 72 for t in dialog.tiles.values()))
        dialog.tiles["pitch"].click()
        self.assertEqual(dialog.selected_kind, "pitch")
        self.assertEqual(dialog.result(), QDialog.Accepted)

    def test_every_tile_shows_an_example_and_its_use(self):
        dialog = AddToolDialog()
        for kind, tile in dialog.tiles.items():
            self.assertTrue((THUMBS_DIR / f"{kind}.png").is_file(), kind)
            self.assertFalse(tile.picture.pixmap().isNull(), kind)
            self.assertTrue(KIND_USES.get(kind), kind)

    def test_every_registered_kind_is_offered(self):
        from app.vision.base import TOOL_KINDS
        offered = {k for _, ks in CATEGORIES for k in ks}
        self.assertEqual(offered, {k for k in TOOL_KINDS if not k.startswith("_")})

    def test_tab_sits_between_settings_and_job_change(self):
        source = (Path(context_mod.__file__).parent / "main_window.py").read_text()
        titles = re.findall(r'addTab\(self\.\w+, "([^"]+)"\)', source)
        self.assertEqual(titles[1:4], ["Settings", "Tools", "Job Change"])

    def test_screenshot_renders(self):
        self.tab.add_tool("width", (0.25, 0.6, 0.5, 0.85))
        self.tab.add_tool("color_area", (0.18, 0.23, 0.44, 0.4))
        pixmap = self.tab.grab()
        self.assertFalse(pixmap.isNull())
        out = os.environ.get("TOOLS_TAB_SHOT")
        if out:
            pixmap.save(out)


def qimage(bgr):
    from PySide6.QtGui import QImage
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class LiveStream(ToolsTabTest):
    """Live until Snapshot; the same button resumes the stream."""

    def setUp(self):
        super().setUp()
        self.tab.set_frame(None)
        self.tab.show()

    def send(self, results):
        self.tab._on_worker_frame(qimage(frame()), [], results)

    def test_live_frame_shows_worker_results_without_rerunning(self):
        from app.vision.base import ToolResult
        tool = self.tab.add_tool("brightness", (0.1, 0.1, 0.4, 0.4))
        with mock.patch.object(tools_tab_mod, "run_tools", side_effect=AssertionError("ran twice")):
            self.send([ToolResult(tool.name, "brightness", 123.0, True)])
        self.assertFalse(self.tab.frozen)
        self.assertEqual(self.tab.table.item(0, 2).text(), "123.00")
        self.assertFalse(self.tab.view.pixmap().isNull())
        self.assertEqual(self.tab.snapshot_button.text(), "Snapshot")

    def test_pending_slow_result_shows_waiting(self):
        from app.vision.base import ToolResult
        tool = self.tab.add_tool("ocr", (0.1, 0.1, 0.4, 0.4))
        self.send([ToolResult(tool.name, "ocr", 0.0, False, extra={"pending": True})])
        self.assertEqual(self.tab.table.item(0, 3).text(), "…")

    def test_snapshot_toggles_freeze_and_resume(self):
        self.send([])
        self.tab.toggle_snapshot()
        self.assertTrue(self.tab.frozen)
        self.assertEqual(self.tab.snapshot_button.text(), "Resume Live")
        self.tab._on_worker_frame(qimage(np.zeros((480, 640, 3), np.uint8)), [], [])
        self.assertEqual(int(self.tab._frame.max()), 255)  # a new live frame does not replace the frozen one
        self.tab.toggle_snapshot()
        self.assertFalse(self.tab.frozen)
        self.assertEqual(self.tab.snapshot_button.text(), "Snapshot")

    def test_drawing_while_live_freezes_first(self):
        self.send([])
        self.tab.begin_add("width")
        self.assertTrue(self.tab.frozen)
        self.assertEqual(self.tab._mode, "add")
        self.tab.view.roi_drawn.emit((0.25, 0.6, 0.5, 0.85))
        self.assertAlmostEqual(self.tab._results[self.tab.tools[0].name].value, 40, delta=1)

    def test_resume_cancels_a_drawing_mode(self):
        self.send([])
        self.tab.begin_add("width")
        self.tab.resume_live()
        self.assertIsNone(self.tab._mode)
        self.assertEqual(self.saved_tools(), [])


if __name__ == "__main__":
    unittest.main()
