"""Creates an empty marking dataset folder for the Train tab.

    python3 tools/init_marking_dataset.py element datasets/marking_elements
    python3 tools/init_marking_dataset.py char    datasets/marking_chars
    python3 tools/init_marking_dataset.py standard datasets/standard_marks

Copy or Capture images into <dir>/images, then open <dir>/images in the Train
tab: it labels and trains with <dir>/classes.txt. Edit the logo_<brand> lines
in classes.txt to match the brands on your line before labelling.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.marking.classes import CHAR_CLASSES, ELEMENT_CLASSES, STANDARD_CLASSES, read_classes_file, write_classes_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["element", "char", "standard"])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    root = args.folder
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    if read_classes_file(root):
        print(f"{root / 'classes.txt'} already exists; left unchanged.")
    else:
        classes = {"element": ELEMENT_CLASSES, "char": CHAR_CLASSES, "standard": STANDARD_CLASSES}[args.kind]
        write_classes_file(root, classes)
        print(f"Created {root} with {args.kind} classes.")


if __name__ == "__main__":
    main()
