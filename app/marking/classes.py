"""Class lists for the two marking models and the dataset classes.txt file.

A dataset folder (<root>/images, <root>/labels) may hold <root>/classes.txt,
one class per line; the Train tab then labels and trains with those classes
instead of the loaded model's, so a new model can start from yolo26n.pt.
"""

import string
from pathlib import Path
from typing import List, Optional

# Element model: one logo_<brand> class per manufacturer (the brand after the
# prefix is what standards.parse() receives as the manufacturer), plus the
# graphical marks and the regions the char model reads.
LOGO_PREFIX = "logo_"
ELEMENT_CLASSES = [
    "logo_pilkington", "logo_sekurit", "logo_fuyao", "logo_xyg",
    "e_mark",      # circle with E<country>; its text is read by the char model
    "ccc_mark",    # CCC logo
    "text_line",   # one printed line: 43R-..., DOT..., E000199, LAMINATED
    "date_code",   # digit + dots, or the XYG digit pyramid
    "class_mark",  # glass class mark (//, II, V-VI ...)
]
# Elements whose crops go to the char model.
TEXT_ELEMENTS = ("e_mark", "text_line", "date_code", "class_mark")

# Char model: digits, capitals and the symbols used in markings (named, since
# "." "-" "/" make poor class names; reader.CHAR_ALIASES maps them back).
CHAR_CLASSES = list(string.digits) + list(string.ascii_uppercase) + ["dash", "dot", "slash"]

CLASSES_FILE = "classes.txt"


def manufacturer_from_class(class_name: str) -> Optional[str]:
    if class_name.startswith(LOGO_PREFIX):
        return class_name[len(LOGO_PREFIX):].upper()
    return None


def read_classes_file(dataset_root: Path) -> Optional[List[str]]:
    path = dataset_root / CLASSES_FILE
    if not path.exists():
        return None
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return names or None


def write_classes_file(dataset_root: Path, names: List[str]) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / CLASSES_FILE).write_text("\n".join(names) + "\n")
