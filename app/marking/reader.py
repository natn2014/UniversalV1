"""Turns the char model's detections on a text crop into text lines.

Detections use the VideoWorker dict format (x1, y1, x2, y2, class_name,
conf). Class names are the characters themselves ("A", "7", ...) or one of
CHAR_ALIASES for symbols that can't be file-safe class names.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Tuple

CHAR_ALIASES = {"dot": ".", "dash": "-", "hyphen": "-", "slash": "/"}


@dataclass
class ReadChar:
    text: str
    conf: float
    box: Tuple[int, int, int, int]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


@dataclass
class ReadLine:
    chars: List[ReadChar] = field(default_factory=list)
    space_factor: float = 0.6

    @property
    def text(self) -> str:
        """Characters left to right, with a space where the gap is wider than
        space_factor x the line's median character height."""
        if not self.chars:
            return ""
        heights = [c.height for c in self.chars if c.text != "."] or [c.height for c in self.chars]
        gap_limit = self.space_factor * statistics.median(heights)
        out = [self.chars[0].text]
        for prev, cur in zip(self.chars, self.chars[1:]):
            if cur.box[0] - prev.box[2] > gap_limit:
                out.append(" ")
            out.append(cur.text)
        return "".join(out)

    @property
    def min_conf(self) -> float:
        return min((c.conf for c in self.chars), default=0.0)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return (min(c.box[0] for c in self.chars), min(c.box[1] for c in self.chars),
                max(c.box[2] for c in self.chars), max(c.box[3] for c in self.chars))


def _to_char(det: dict) -> ReadChar:
    name = str(det.get("class_name", "?"))
    return ReadChar(CHAR_ALIASES.get(name.lower(), name),
                    float(det.get("conf", 0.0)),
                    (int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])))


def _overlaps_row(char: ReadChar, row: List[ReadChar], min_overlap: float) -> bool:
    top = min(c.box[1] for c in row)
    bottom = max(c.box[3] for c in row)
    overlap = min(bottom, char.box[3]) - max(top, char.box[1])
    return overlap >= min_overlap * max(1, min(char.height, bottom - top))


def assemble_lines(detections: List[dict], space_factor: float = 0.6,
                   min_overlap: float = 0.5) -> List[ReadLine]:
    """Groups characters into rows (top to bottom) by vertical overlap and
    sorts each row left to right."""
    chars = sorted((_to_char(d) for d in detections), key=lambda c: (c.box[1] + c.box[3]) / 2)
    rows: List[List[ReadChar]] = []
    # Tallest first so small dots join the row of the digit they sit beside.
    for char in sorted(chars, key=lambda c: -c.height):
        for row in rows:
            if _overlaps_row(char, row, min_overlap):
                row.append(char)
                break
        else:
            rows.append([char])
    rows.sort(key=lambda row: min(c.box[1] for c in row))
    return [ReadLine(sorted(row, key=lambda c: c.box[0]), space_factor) for row in rows]
