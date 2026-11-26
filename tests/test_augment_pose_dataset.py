"""tools/augment_pose_dataset.py: auto head-up labels and turned copies whose
box and keypoints follow the part."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import augment_pose_dataset as aug  # noqa: E402
from app.pose_labels import read_keypoints_file, read_pose_label_file, write_keypoints_file  # noqa: E402
from app.pose_labels import write_pose_label_file  # noqa: E402


def photo(path: Path) -> None:
    """A bright upright part (taller than wide) centred on a dark table."""
    img = np.full((800, 600, 3), 40, np.uint8)
    cv2.rectangle(img, (250, 300), (350, 500), (200, 200, 200), -1)
    cv2.imwrite(str(path), img)


def angle_of(label: dict, w: int = 600, h: int = 800) -> float:
    """Tail -> head direction in pixels (normalised x and y have different scales)."""
    (hx, hy, _), (tx, ty, _) = label["kpts"]
    return math.degrees(math.atan2(-(hy - ty) * h, (hx - tx) * w))


class AugmentPoseDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "photos").mkdir()
        photo(self.root / "photos" / "a.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def test_auto_head_up_and_turned_copies(self):
        out = self.root / "ds"
        self.assertEqual(aug.main([str(self.root / "photos"), str(out), "--angles", "3,5", "--auto-head-up"]), 0)
        self.assertEqual(read_keypoints_file(out), ["head", "tail"])
        stems = sorted(p.stem for p in (out / "images").iterdir())
        self.assertEqual(stems, ["a", "a_r+3", "a_r+5", "a_r-3", "a_r-5"])

        base = read_pose_label_file(out / "labels" / "a.txt", 2)[0]
        (hx, hy, _), (tx, ty, _) = base["kpts"]
        self.assertAlmostEqual(hx * 600, 300, delta=4)  # middle of the top edge
        self.assertAlmostEqual(hy * 800, 300, delta=4)
        self.assertAlmostEqual(ty * 800, 500, delta=4)  # middle of the bottom edge
        self.assertAlmostEqual(angle_of(base), 90.0, delta=0.5)

        for angle in (3, 5, -3, -5):
            label = read_pose_label_file(out / "labels" / f"a_r{angle:+d}.txt", 2)[0]
            self.assertAlmostEqual(angle_of(label), 90.0 + angle, delta=0.5)  # turned with the image
            self.assertAlmostEqual(label["cx"], 0.5, delta=0.01)  # turned about the centre
            self.assertGreater(label["w"], base["w"])  # the box grows to hold the turned part

    def test_existing_label_kept_and_turned(self):
        ds = self.root / "src"
        (ds / "images").mkdir(parents=True)
        (ds / "labels").mkdir()
        photo(ds / "images" / "a.jpg")
        write_keypoints_file(ds, ["front", "back"])
        # Labelled by hand pointing right (front at the right end).
        write_pose_label_file(ds / "labels" / "a.txt", [{
            "cls_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3,
            "kpts": [(0.6, 0.5, 2), (0.4, 0.5, 2)]}], 2)
        out = self.root / "ds"
        aug.main([str(ds), str(out), "--angles", "4"])
        self.assertEqual(read_keypoints_file(out), ["front", "back"])
        self.assertAlmostEqual(angle_of(read_pose_label_file(out / "labels" / "a.txt", 2)[0]), 0.0, delta=0.5)
        self.assertAlmostEqual(angle_of(read_pose_label_file(out / "labels" / "a_r+4.txt", 2)[0]), 4.0, delta=0.5)

    def test_unlabelled_photo_skipped_without_auto(self):
        out = self.root / "ds"
        self.assertEqual(aug.main([str(self.root / "photos"), str(out)]), 1)
        self.assertEqual(list((out / "images").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
