"""Pose datasets: keypoints.txt, pose label files, data.yaml, and the Train
tab's pose flow (draw a box, tap each keypoint, drag one, train-ready count).
Offscreen; settings in a temp dir."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import app.paths as paths_mod  # noqa: E402
import app.settings as settings_mod  # noqa: E402
from app.augmentation import build_hyp, describe  # noqa: E402
from app.pose_labels import (  # noqa: E402
    HIDDEN,
    NOT_LABELLED,
    VISIBLE,
    dataset_yaml,
    is_pose_dataset,
    kpts_to_normalized,
    kpts_to_pixel,
    label_is_complete,
    parse_keypoint_names,
    read_keypoints_file,
    read_pose_label_file,
    write_keypoints_file,
    write_pose_label_file,
)
from app.yolo_labels import read_yolo_label_file  # noqa: E402

APP = QApplication.instance() or QApplication([])


class KeypointNames(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_keypoint_names(" head, tail ,"), (["head", "tail"], None))
        self.assertEqual(parse_keypoint_names("head\ntail\nnotch")[0], ["head", "tail", "notch"])
        for bad in ("", " , ", "head, Head", "left pin, tail"):
            names, problem = parse_keypoint_names(bad)
            self.assertIsNone(names, bad)
            self.assertTrue(problem, bad)

    def test_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(read_keypoints_file(root))
            self.assertFalse(is_pose_dataset(root))
            write_keypoints_file(root, ["head", "tail"])
            self.assertTrue(is_pose_dataset(root))
            self.assertEqual(read_keypoints_file(root), ["head", "tail"])


class PoseLabelFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.label = Path(self.tmp.name) / "a.txt"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        boxes = [{"cls_id": 1, "cx": 0.5, "cy": 0.4, "w": 0.2, "h": 0.1,
                  "kpts": [(0.45, 0.4, VISIBLE), (0.55, 0.41, HIDDEN)]}]
        write_pose_label_file(self.label, boxes, 2)
        fields = self.label.read_text().split()
        self.assertEqual(len(fields), 5 + 3 * 2)
        back = read_pose_label_file(self.label, 2)
        self.assertEqual(back[0]["cls_id"], 1)
        for (x, y, v), (ex, ey, ev) in zip(back[0]["kpts"], boxes[0]["kpts"]):
            self.assertAlmostEqual(x, ex, places=5)
            self.assertAlmostEqual(y, ey, places=5)
            self.assertEqual(v, ev)
        # The detect reader skips pose lines rather than misreading them.
        self.assertEqual(read_yolo_label_file(self.label), [])

    def test_unplaced_points_written_as_zero(self):
        write_pose_label_file(self.label, [{"cls_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1,
                                            "kpts": [(0.3, 0.3, VISIBLE)]}], 2)
        self.assertTrue(self.label.read_text().strip().endswith("0.000000 0.000000 0"))
        self.assertFalse(label_is_complete(read_pose_label_file(self.label, 2)))

    def test_detect_line_reads_as_unplaced(self):
        self.label.write_text("0 0.5 0.5 0.2 0.2\n")
        boxes = read_pose_label_file(self.label, 2)
        self.assertEqual(boxes[0]["kpts"], [(0.0, 0.0, NOT_LABELLED)] * 2)
        self.assertFalse(label_is_complete(boxes))

    def test_wrong_keypoint_count_skipped(self):
        self.label.write_text("0 0.5 0.5 0.2 0.2 0.1 0.1 2\n")  # 1 point, dataset has 2
        self.assertEqual(read_pose_label_file(self.label, 2), [])

    def test_pixel_conversion(self):
        px = kpts_to_pixel([(0.5, 0.25, VISIBLE), (0, 0, NOT_LABELLED)], 200, 100)
        self.assertEqual(px, [(100.0, 25.0, VISIBLE), (0.0, 0.0, NOT_LABELLED)])
        self.assertEqual(kpts_to_normalized(px, 200, 100), [(0.5, 0.25, VISIBLE), (0.0, 0.0, NOT_LABELLED)])
        self.assertEqual(kpts_to_normalized([(250.0, -5.0, VISIBLE)], 200, 100), [(1.0, 0.0, VISIBLE)])

    def test_complete(self):
        self.assertFalse(label_is_complete([]))
        self.assertTrue(label_is_complete([{"kpts": [(0.1, 0.1, VISIBLE), (0.2, 0.2, HIDDEN)]}]))


class DatasetYaml(unittest.TestCase):
    def test_detect_has_no_keypoints(self):
        data = dataset_yaml(Path("/tmp/ds"), "labelled.txt", ["a", "b"])
        self.assertEqual(data["nc"], 2)
        self.assertNotIn("kpt_shape", data)

    def test_pose(self):
        data = dataset_yaml(Path("/tmp/ds"), "labelled.txt", ["part"], ["head", "tail", "notch"])
        self.assertEqual(data["kpt_shape"], [3, 3])
        self.assertEqual(data["flip_idx"], [0, 1, 2])
        # Left to Ultralytics' default: a tight value stopped the points being learned.
        self.assertNotIn("kpt_oks_sigmas", data)
        self.assertEqual(data["kpt_names"], {0: ["head", "tail", "notch"]})


class RotationAugment(unittest.TestCase):
    def test_any_rotation(self):
        self.assertEqual(build_hyp("light", rotate_any=True)["degrees"], 180.0)
        self.assertEqual(build_hyp("light")["degrees"], build_hyp("light", rotate_any=False)["degrees"])
        self.assertTrue(describe("light", False, False, True).endswith(", any rotation"))


class ExifRotation(unittest.TestCase):
    def test_train_tab_shows_phone_photos_upright_like_opencv(self):
        # A 40x20 picture stored sideways with EXIF "rotate 90° CW" (6):
        # OpenCV / Ultralytics read it 20 wide x 40 tall; so must the Train tab.
        from PIL import Image
        from app.tabs.train_tab import load_upright
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phone.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (40, 20), (90, 90, 90)).save(path, exif=exif)
            h, w = cv2.imread(str(path)).shape[:2]
            pixmap = load_upright(path)
            self.assertEqual((pixmap.width(), pixmap.height()), (w, h))
            self.assertEqual((w, h), (20, 40))


class TrainTabPoseFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", root / "app_settings.json"),
            mock.patch.object(paths_mod, "MARK_MODEL_PATH", root / "missing.pt"),
        ]
        for p in self.patches:
            p.start()
        from app.context import AppContext
        from app.tabs.train_tab import TrainTab
        self.ctx = AppContext()
        self.tab = TrainTab(self.ctx)
        self.dataset = root / "ds"
        self.images = self.dataset / "images"
        self.images.mkdir(parents=True)
        for name in ("a.jpg", "b.jpg"):
            cv2.imwrite(str(self.images / name), np.full((100, 200, 3), 128, np.uint8))

    def tearDown(self):
        self.tab.deleteLater()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def make_pose(self, names="head, tail"):
        self.tab.scan_train_folder(self.images)
        with mock.patch("app.tabs.train_tab.get_text", return_value=(names, True)):
            self.tab._on_make_pose_dataset()

    def test_detect_mode_by_default(self):
        self.tab.scan_train_folder(self.images)
        self.assertIsNone(self.tab._pose_names)
        self.assertFalse(self.tab.make_pose_button.isHidden())
        self.assertTrue(self.tab.kpt_box.isHidden())

    def test_make_pose_dataset(self):
        self.make_pose()
        self.assertEqual(read_keypoints_file(self.dataset), ["head", "tail"])
        self.assertEqual((self.dataset / "classes.txt").read_text().split(), ["part"])
        self.assertEqual(self.tab._pose_names, ["head", "tail"])
        self.assertTrue(self.tab.make_pose_button.isHidden())
        self.assertFalse(self.tab.kpt_box.isHidden())

    def test_draw_box_then_tap_points(self):
        self.make_pose()
        self.tab.train_class_combo.setCurrentText("part")
        self.tab._on_train_box_drawn(40, 20, 160, 80)
        self.assertEqual(self.tab._next_kpt, 0)
        self.assertTrue(self.tab.train_image_widget.point_mode())
        label = self.dataset / "labels" / "a.txt"
        self.assertFalse(label_is_complete(read_pose_label_file(label, 2)))

        self.tab._on_point_placed(150.0, 50.0)  # head
        self.assertEqual(self.tab._next_kpt, 1)
        self.tab.kpt_hidden_button.setChecked(True)
        self.tab._on_point_placed(50.0, 50.0)  # tail, covered
        self.assertEqual(self.tab._next_kpt, -1)
        self.assertFalse(self.tab.train_image_widget.point_mode())

        boxes = read_pose_label_file(label, 2)
        self.assertTrue(label_is_complete(boxes))
        (hx, hy, hv), (tx, ty, tv) = boxes[0]["kpts"]
        self.assertAlmostEqual(hx, 0.75, places=4)
        self.assertAlmostEqual(hy, 0.5, places=4)
        self.assertEqual((hv, tv), (VISIBLE, HIDDEN))
        self.assertAlmostEqual(tx, 0.25, places=4)

        # Dragging a dot saves its new place.
        self.tab._train_boxes[0]["kpts"][0] = (160.0, 40.0, VISIBLE)
        self.tab._on_point_moved(0, 160.0, 40.0)
        self.assertAlmostEqual(read_pose_label_file(label, 2)[0]["kpts"][0][0], 0.8, places=4)

        # Reload: points come back in pixels.
        self.tab._load_train_index(0)
        self.assertAlmostEqual(self.tab._train_boxes[0]["kpts"][0][0], 160.0, places=1)

        # Only the finished image goes into training.
        self.assertEqual(self.tab._pose_label_counts(), (1, 0))
        data_yaml = self.tab._write_dataset_files()
        self.assertIn("kpt_shape", data_yaml.read_text())
        self.assertEqual(len((self.dataset / "labelled.txt").read_text().splitlines()), 1)

    def test_unfinished_image_left_out(self):
        self.make_pose()
        self.tab.train_class_combo.setCurrentText("part")
        self.tab._on_train_box_drawn(40, 20, 160, 80)
        self.tab._on_point_placed(150.0, 50.0)  # tail still missing
        self.assertEqual(self.tab._pose_label_counts(), (0, 1))
        self.assertIsNone(self.tab._write_dataset_files())

    def test_replace_points(self):
        self.make_pose()
        self.tab.train_class_combo.setCurrentText("part")
        self.tab._on_train_box_drawn(40, 20, 160, 80)
        self.tab._on_point_placed(150.0, 50.0)
        self.tab._on_point_placed(50.0, 50.0)
        self.tab._on_replace_points()
        self.assertEqual(self.tab._next_kpt, 0)
        self.assertEqual(self.tab._train_boxes[0]["kpts"], [(0.0, 0.0, NOT_LABELLED)] * 2)

    def test_names_count_locked_once_points_placed(self):
        self.make_pose()
        self.tab.train_class_combo.setCurrentText("part")
        self.tab._on_train_box_drawn(40, 20, 160, 80)
        self.tab._on_point_placed(150.0, 50.0)
        with mock.patch("app.tabs.train_tab.get_text", return_value=("head, tail, notch", True)):
            self.tab._on_edit_keypoint_names()
        self.assertEqual(read_keypoints_file(self.dataset), ["head", "tail"])
        with mock.patch("app.tabs.train_tab.get_text", return_value=("front, back", True)):
            self.tab._on_edit_keypoint_names()
        self.assertEqual(read_keypoints_file(self.dataset), ["front", "back"])

    def test_training_config_is_pose(self):
        self.make_pose()
        self.tab.train_class_combo.setCurrentText("part")
        self.tab._on_train_box_drawn(40, 20, 160, 80)
        self.tab._on_point_placed(150.0, 50.0)
        self.tab._on_point_placed(50.0, 50.0)
        with mock.patch("app.tabs.train_tab.TrainingWorker") as worker:
            self.tab._start_training()
        config = worker.call_args[0][0]
        self.assertEqual(config["task"], "pose")
        self.assertEqual(config["kpt_count"], 2)
        self.assertEqual(config["name"], "pose")
        self.ctx.training_active = False


if __name__ == "__main__":
    unittest.main()
