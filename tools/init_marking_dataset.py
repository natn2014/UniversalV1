"""Creates an empty marking dataset folder for the Train tab.

    python3 tools/init_marking_dataset.py element datasets/marking_elements
    python3 tools/init_marking_dataset.py char    datasets/marking_chars

Copy or Capture images into <dir>/images, then open <dir>/images in the Train
tab: it labels and trains with <dir>/classes.txt. Edit the logo_<brand> lines
in classes.txt to match the brands on your line before labelling.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.marking.classes import CHAR_CLASSES, ELEMENT_CLASSES, read_classes_file, write_classes_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["element", "char"])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    root = args.folder
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    if read_classes_file(root):
        print(f"{root / 'classes.txt'} already exists; left unchanged.")
    else:
        write_classes_file(root, ELEMENT_CLASSES if args.kind == "element" else CHAR_CLASSES)
        print(f"Created {root} with {args.kind} classes.")


if __name__ == "__main__":
    main()
