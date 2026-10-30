"""Standard-mark dataset generator and the job's Required Marks (judgement,
Settings card, Monitor "Missing" label), offscreen with settings in a temp dir."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import app.settings as settings_mod  # noqa: E402
import app.tabs.monitor_tab as monitor_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.judgement import DetectionFilters  # noqa: E402
from app.marking.classes import STANDARD_CLASSES, read_classes_file  # noqa: E402
from app.tabs.monitor_tab import MonitorTab  # noqa: E402
from app.theme import NG_RED, OK_GREEN  # noqa: E402
from app.tabs.settings_tab import SettingsTab  # noqa: E402
from app.yolo_labels import read_yolo_label_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from make_synthetic_mark import make_dataset  # noqa: E402

APP = QApplication.instance() or QApplication([])


def det(cls, conf=0.9):
    return {"class_name": cls, "conf": conf, "x1": 10, "y1": 10, "x2": 40, "y2": 40, "label": cls}


class Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "a"
        cls.truth = make_dataset(cls.root, 20, seed=1, size=(640, 360))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_train_tab_layout(self):
        self.assertEqual(read_classes_file(self.root), STANDARD_CLASSES)
        self.assertEqual(len(list((self.root / "images").glob("*.jpg"))), 20)
        self.assertEqual(len(list((self.root / "labels").glob("*.txt"))), 20)

    def test_labels_are_valid_and_match_truth(self):
        for entry in self.truth:
            stem = Path(entry["image"]).stem
            boxes = read_yolo_label_file(self.root / "labels" / f"{stem}.txt")
            for b in boxes:
                self.assertLess(b["cls_id"], len(STANDARD_CLASSES))
                for v in (b["cx"], b["cy"], b["w"], b["h"]):
                    self.assertTrue(0.0 <= v <= 1.0, (stem, b))
                self.assertGreater(b["w"] * b["h"], 0)
            self.assertEqual(sorted({STANDARD_CLASSES[b["cls_id"]] for b in boxes}), entry["present"])
            self.assertFalse(set(entry["present"]) & set(entry["absent"]))

    def test_same_seed_same_labels(self):
        other = Path(self.tmp.name) / "b"
        make_dataset(other, 5, seed=1, size=(640, 360))
        for i in range(1, 6):
            name = f"labels/mark_{i:04d}.txt"
            self.assertEqual((other / name).read_text(), (self.root / name).read_text())

    def test_every_class_is_drawn(self):
        root = Path(self.tmp.name) / "all"
        truth = make_dataset(root, 60, seed=2, size=(640, 360))
        seen = {c for t in truth for c in t["present"]}
        self.assertEqual(seen, set(STANDARD_CLASSES))

    def test_agc_layout_order(self):
        import numpy as np
        import make_synthetic_mark as g
        for seed in range(20):
            rng = np.random.default_rng(seed)
            _, boxes = g.agc_layout(g.Pen(rng), rng)
            top = {}
            for k, (x1, y1, x2, y2) in boxes:
                top.setdefault(k, y1)
            if "logo_agc" in top and "e_mark" in top:
                self.assertLess(top["logo_agc"], top["e_mark"])
            if "e_mark" in top and "date_code" in top:
                self.assertLess(top["e_mark"], top["date_code"])

    def test_background_images_have_an_empty_label_file(self):
        with mock.patch("make_synthetic_mark.BACKGROUND_SHARE", 1.0):
            root = Path(self.tmp.name) / "bg"
            truth = make_dataset(root, 2, seed=3, size=(320, 180))
        self.assertEqual([t["present"] for t in truth], [[], []])
        self.assertEqual((root / "labels" / "mark_0001.txt").read_text(), "")


class Prelabel(unittest.TestCase):
    def test_writes_model_boxes_and_classes(self):
        from types import SimpleNamespace
        from prelabel_marks import prelabel
        box = SimpleNamespace(cls=1, xyxy=[SimpleNamespace(tolist=lambda: [10.0, 20.0, 30.0, 60.0])])
        model = SimpleNamespace(names={0: "e_mark", 1: "dot"},
                                predict=lambda *a, **k: [SimpleNamespace(orig_shape=(100, 200), boxes=[box])])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            for name in ("a.png", "b.png", "notes.txt"):
                (root / "images" / name).write_bytes(b"")
            (root / "labels").mkdir()
            (root / "labels" / "b.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            self.assertEqual(prelabel(model, root), {"a": 1})
            self.assertEqual(read_classes_file(root), ["e_mark", "dot"])
            self.assertEqual(read_yolo_label_file(root / "labels" / "a.txt"),
                             [{"cls_id": 1, "cx": 0.1, "cy": 0.4, "w": 0.1, "h": 0.4}])
            self.assertEqual((root / "labels" / "b.txt").read_text(), "0 0.5 0.5 0.1 0.1\n")  # kept


class Judge(unittest.TestCase):
    def test_required_marks(self):
        f = DetectionFilters(judge_mode="required", required_classes={"e_mark", "tis_mark"})
        self.assertEqual(f.judge([det("e_mark"), det("tis_mark"), det("dot")]), "GREEN")
        self.assertEqual(f.judge([det("e_mark"), det("dot")]), "RED")
        self.assertEqual(f.missing([det("e_mark"), det("dot")]), ["tis_mark"])
        self.assertEqual(f.judge([]), "YELLOW")
        self.assertEqual(f.missing([]), [])

    def test_nothing_required_keeps_target_logic(self):
        f = DetectionFilters(target_class_1="e_mark")
        self.assertEqual(f.judge([det("e_mark")]), "GREEN")
        self.assertEqual(f.judge([det("dot")]), "RED")
        self.assertEqual(f.missing([det("dot")]), [])


class JudgeMode(unittest.TestCase):
    def filters(self, mode="required"):
        return DetectionFilters(judge_mode=mode, confidence_threshold=0.5, selected_classes={"e_mark", "tis_mark", "dot"},
                                required_classes={"e_mark", "tis_mark"}, mark_conf={"e_mark": 0.25, "tis_mark": 0.6})

    def test_detection_mode_ignores_required_marks(self):
        f = self.filters("detection")
        f.target_class_1 = "dot"
        self.assertEqual(f.judge(f.apply([det("dot")])), "GREEN")
        self.assertEqual(f.missing(f.apply([det("dot")])), [])
        self.assertEqual(f.mark_results([det("dot")]), [])
        self.assertEqual(f.apply([det("e_mark", 0.3)]), [])  # the slider's 0.5 for every class

    def test_each_mark_has_its_own_confidence(self):
        f = self.filters()
        self.assertEqual(f.judge(f.apply([det("e_mark", 0.3), det("tis_mark", 0.7)])), "GREEN")
        self.assertEqual(f.judge(f.apply([det("e_mark", 0.3), det("tis_mark", 0.55)])), "RED")
        self.assertEqual(f.missing(f.apply([det("e_mark", 0.3), det("tis_mark", 0.55)])), ["tis_mark"])
        self.assertEqual(f.apply([det("dot", 0.45)]), [])  # not required: the slider's 0.5

    def test_mark_results_rows(self):
        f = self.filters()
        self.assertEqual(f.mark_results([det("e_mark", 0.3), det("e_mark", 0.8), det("tis_mark", 0.55)]), [
            {"class_name": "e_mark", "conf": 0.8, "min_conf": 0.25, "ok": True},
            {"class_name": "tis_mark", "conf": 0.55, "min_conf": 0.6, "ok": False}])
        rows = f.mark_results([det("e_mark", 0.8)])
        self.assertEqual([(r["conf"], r["ok"]) for r in rows], [(0.8, True), (None, False)])
        self.assertEqual([r["ok"] for r in f.mark_results([det("dot", 0.1)])], [None, None])  # no part


class TabsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", Path(self.tmp.name) / "app_settings.json"),
            mock.patch.object(monitor_mod, "save_ng_image", mock.Mock()),
        ]
        for p in self.patches:
            p.start()
        self.ctx = AppContext()
        self.settings = SettingsTab(self.ctx)
        self.ctx.model_classes = ["dot", "e_mark", "tis_mark"]
        self.ctx.model_classes_changed.emit(self.ctx.model_classes)

    def tearDown(self):
        self.settings.deleteLater()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def ticked(self):
        return {c.text() for c in self.settings._required_checkboxes() if c.isChecked()}


class SettingsCard(TabsTest):
    def test_checkboxes_per_class_none_ticked(self):
        self.assertEqual([c.text() for c in self.settings._required_checkboxes()], ["dot", "e_mark", "tis_mark"])
        self.assertEqual(self.ticked(), set())
        self.assertEqual(self.ctx.filters.required_classes, set())

    def test_teach_from_current_part(self):
        self.ctx.last_detections = [det("e_mark"), det("tis_mark"), det("e_mark")]
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes) as ask:
            self.settings.teach_required_marks()
        self.assertIn("e_mark, tis_mark", ask.call_args[0][2])
        self.assertEqual(self.ticked(), {"e_mark", "tis_mark"})
        self.assertEqual(self.ctx.filters.required_classes, {"e_mark", "tis_mark"})

    def test_teach_without_a_frame_changes_nothing(self):
        with mock.patch.object(QMessageBox, "information") as info:
            self.settings.teach_required_marks()
        info.assert_called_once()
        self.assertEqual(self.ticked(), set())

    def test_saved_with_the_job(self):
        self.settings.set_required_marks(["dot", "tis_mark"])
        config = self.settings.collect_config()
        self.assertEqual(config["required_classes"], ["dot", "tis_mark"])
        self.settings.set_required_marks([])
        self.settings.apply_config(config)
        self.assertEqual(self.ctx.filters.required_classes, {"dot", "tis_mark"})
        self.settings.apply_config({k: v for k, v in config.items() if k != "required_classes"})
        self.assertEqual(self.ctx.filters.required_classes, set())  # older job: nothing required


class Switch(TabsTest):
    def test_switch_shows_the_controls_of_its_mode(self):
        self.assertEqual(self.ctx.filters.judge_mode, "detection")
        self.assertFalse(self.settings.target_box.isHidden())
        self.assertTrue(self.settings.required_card.isHidden())
        self.settings.judge_mode_buttons["required"].click()
        self.assertEqual(self.ctx.filters.judge_mode, "required")
        self.assertTrue(self.settings.target_box.isHidden())
        self.assertFalse(self.settings.required_card.isHidden())

    def test_min_conf_per_mark(self):
        spin = self.settings._mark_conf_spins["tis_mark"]
        self.assertFalse(spin.isEnabled())
        self.settings.set_required_marks(["tis_mark"])
        self.assertTrue(spin.isEnabled())
        spin.setValue(65)
        self.assertEqual(self.ctx.filters.mark_conf["tis_mark"], 0.65)

    def test_saved_with_the_job(self):
        self.settings.set_judge_mode("required")
        self.settings.set_required_marks(["e_mark", "tis_mark"])
        self.settings.set_mark_conf({"e_mark": 0.3, "tis_mark": 0.7})
        config = self.settings.collect_config()
        self.assertEqual(config["judge_mode"], "required")
        self.assertEqual(config["mark_conf"], {"e_mark": 0.3, "tis_mark": 0.7})
        self.settings.set_judge_mode("detection")
        self.settings.set_mark_conf({"e_mark": 0.9})
        self.settings.apply_config(config)
        self.assertEqual(self.ctx.filters.judge_mode, "required")
        self.assertEqual(self.ctx.filters.mark_conf["e_mark"], 0.3)

    def test_older_job(self):
        old = {"confidence_threshold": 0.45, "required_classes": ["e_mark"]}
        self.settings.apply_config(old)
        self.assertEqual(self.ctx.filters.judge_mode, "required")  # ticked marks decided before
        self.assertEqual(self.ctx.filters.mark_conf["e_mark"], 0.45)  # with the slider's confidence
        self.settings.apply_config({"confidence_threshold": 0.45})
        self.assertEqual(self.ctx.filters.judge_mode, "detection")


class MonitorMarks(TabsTest):
    def test_mark_table_like_the_tool_table(self):
        tab = MonitorTab(self.ctx)
        image = QImage(200, 100, QImage.Format_RGB888)
        image.fill(QColor("#202020"))
        self.ctx.filters.selected_classes = {"dot", "e_mark", "tis_mark"}
        self.assertFalse(tab.table.isHidden())
        self.assertTrue(tab.mark_table.isHidden())
        self.settings.set_judge_mode("required")
        self.settings.set_required_marks(["e_mark", "tis_mark"])
        self.settings.set_mark_conf({"e_mark": 0.5, "tis_mark": 0.8})
        tab._reset_judgement()
        tab.on_frame(image, [det("e_mark", 0.9), det("tis_mark", 0.6)], [])
        self.assertTrue(tab.table.isHidden())
        self.assertFalse(tab.mark_table.isHidden())
        cells = [[tab.mark_table.item(r, c).text() for c in range(4)] for r in range(tab.mark_table.rowCount())]
        self.assertEqual(cells, [["e_mark", "0.90", "0.50", "OK"], ["tis_mark", "0.60", "0.80", "NG"]])
        self.assertEqual(tab.mark_table.item(1, 3).background().color().name(), QColor(NG_RED).name())
        self.assertEqual(tab.mark_table.item(0, 3).background().color().name(), QColor(OK_GREEN).name())
        self.assertEqual(tab._latch.state, "RED")
        tab.deleteLater()


class MonitorMissing(TabsTest):
    def test_missing_marks_shown_and_ng(self):
        tab = MonitorTab(self.ctx)
        image = QImage(200, 100, QImage.Format_RGB888)
        image.fill(QColor("#202020"))
        self.ctx.filters.selected_classes = {"dot", "e_mark", "tis_mark"}
        self.settings.set_judge_mode("required")
        self.settings.set_required_marks(["e_mark", "tis_mark"])
        tab._reset_judgement()
        tab.on_frame(image, [det("e_mark")], [])
        self.assertEqual(tab._latch.state, "RED")
        self.assertEqual(tab.missing_label.text(), "Missing: tis_mark")
        self.assertEqual([d["class_name"] for d in self.ctx.last_detections], ["e_mark"])
        tab._reset_judgement()
        tab.on_frame(image, [det("e_mark"), det("tis_mark")], [])
        self.assertEqual(tab._latch.state, "GREEN")
        self.assertEqual(tab.missing_label.text(), "")
        tab.deleteLater()


if __name__ == "__main__":
    unittest.main()
