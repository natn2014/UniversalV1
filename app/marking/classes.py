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
# Standard-mark model: one class per approval / standard mark, so the
# detector alone tells which standards a windshield carries (and a job can
# require some of them). The order is the class id: append, never reorder.
# tools/make_synthetic_mark.py --dataset draws training images for it.
STANDARD_CLASSES = [
    "e_mark",       # ECE R43: circle with E<n> and the 43R-00xxxx approval number
    "dot",          # US DOT / FMVSS 205: DOT-### M###
    "as_class",     # ANSI Z26.1 item: AS1 / AS2 / AS3
    "ccc_mark",     # China CCC oval with its E000xxx factory code
    "tis_mark",     # Thai TIS emblem with TIS ####-####
    "sni_mark",     # Indonesia SNI with the maker code
    "inmetro",      # Brazil INMETRO
    "asnzs",        # Australia / New Zealand AS/NZS 2080
    "bis_isi",      # India ISI mark with IS 2553 and CM/L licence
    "laminated",    # glass type text
    "glass_class",  # light-transmission class: //, II, V, V-VI
    "date_code",    # digit + dots
    "logo_acme", "logo_nova", "logo_orion",  # made-up makers; rename to the line's brands
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


def validate_class_name(name: str, existing: List[str]) -> Optional[str]:
    """Why `name` can't be a new class name (None = fine). Compare against
    `existing` without the class being renamed."""
    if not name.strip():
        return "Class name can't be empty."
    if "\n" in name or "\r" in name:
        return "Class name must be one line."
    if name.strip().lower() in (n.lower() for n in existing):
        return f'There is already a class "{name.strip()}".'
    return None


def write_classes_file(dataset_root: Path, names: List[str]) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / CLASSES_FILE).write_text("\n".join(names) + "\n")
