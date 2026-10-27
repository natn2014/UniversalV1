import tempfile
import unittest
from pathlib import Path

from app.marking.classes import read_classes_file, validate_class_name, write_classes_file
from app.yolo_labels import read_yolo_label_file, write_yolo_label_file


class ValidateClassName(unittest.TestCase):
    def test_rejects(self):
        existing = ["Lead", "NoTape"]
        for name in ("", "   ", "a\nb", "lead", " NoTape "):
            self.assertIsNotNone(validate_class_name(name, existing), name)

    def test_accepts_new_name(self):
        self.assertIsNone(validate_class_name(" Scratch ", ["Lead", "NoTape"]))

    def test_rename_to_other_case_of_itself(self):
        # Callers pass the list without the class being renamed.
        self.assertIsNone(validate_class_name("NOTAPE", ["Lead"]))


class AddRenameRoundTrip(unittest.TestCase):
    def test_ids_stay_put(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label = root / "labels" / "a.txt"
            label.parent.mkdir()
            write_yolo_label_file(label, [{"cls_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}])
            before = label.read_text()

            write_classes_file(root, ["Lead", "NoTape"])
            write_classes_file(root, read_classes_file(root) + ["Scratch"])  # add
            names = read_classes_file(root)
            names[1] = "MissingTape"  # rename keeps the position
            write_classes_file(root, names)

            self.assertEqual(read_classes_file(root), ["Lead", "MissingTape", "Scratch"])
            self.assertEqual(label.read_text(), before)
            self.assertEqual(read_yolo_label_file(label)[0]["cls_id"], 1)


if __name__ == "__main__":
    unittest.main()
