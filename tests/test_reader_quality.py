import unittest

import cv2
import numpy as np

from app.marking import quality as q
from app.marking.reader import assemble_lines


def det(ch, x, y, w=10, h=20, conf=0.95):
    return {"class_name": ch, "conf": conf, "x1": x, "y1": y, "x2": x + w, "y2": y + h}


def row(text, x0=0, y=0, conf=0.95, dot_names=("dot",)):
    """Detections for `text`, 12 px pitch, a 16 px extra gap per space."""
    out, x = [], x0
    for ch in text:
        if ch == " ":
            x += 16
            continue
        if ch == ".":
            out.append(det("dot", x, y + 15, 4, 4, conf))
            x += 8
        else:
            out.append(det({"-": "dash"}.get(ch, ch), x, y, conf=conf))
            x += 12
    return out


class Reader(unittest.TestCase):
    def test_single_line_with_spaces(self):
        lines = assemble_lines(row("43R - 001582")[::-1])
        self.assertEqual([l.text for l in lines], ["43R - 001582"])

    def test_rows_top_to_bottom(self):
        dets = row("E 000199", y=40) + row("43R-001582", y=0) + row("3...", y=80)
        self.assertEqual([l.text for l in assemble_lines(dets)], ["43R-001582", "E 000199", "3..."])

    def test_dots_before_digit(self):
        self.assertEqual(assemble_lines(row("...2"))[0].text, "...2")


class TextDefects(unittest.TestCase):
    def line(self, text, conf=0.95):
        return assemble_lines(row(text, conf=conf))[0]

    def test_diff_ops(self):
        self.assertEqual(q.diff_text("43R001582", "43R001S82"), [("replace", 6, "5", "S")])
        self.assertEqual(q.diff_text("43R001582", "43R00158"), [("missing", 8, "2", "")])
        self.assertEqual(q.diff_text("AS2", "AS2X"), [("extra", 3, "", "X")])

    def test_ok_line_ignores_spacing(self):
        self.assertEqual(q.check_line("ece_approval", "43R-001582", self.line("43R - 001582")), [])

    def test_misprint(self):
        issues = q.check_line("ece_approval", "43R-001582", self.line("43R-00182"))
        self.assertEqual([i.code for i in issues], ["MISPRINT"])
        self.assertIn("'5' missing", issues[0].message)

    def test_faint(self):
        dets = row("DOT682")
        dets[3]["conf"] = 0.3
        issues = q.check_line("dot", "DOT682", assemble_lines(dets)[0])
        self.assertEqual([i.code for i in issues], ["FAINT_CHAR"])

    def test_line_missing(self):
        self.assertEqual([i.code for i in q.check_line("ccc", "E000199", None)], ["LINE_MISSING"])

    def test_missing_elements(self):
        issues = q.check_missing(["logo", "e_mark", "ccc_mark"], ["logo", "e_mark"])
        self.assertEqual([i.element for i in issues], ["ccc_mark"])


def label(text="43R-001582", shift=(0, 0)):
    img = np.full((60, 260), 200, np.uint8)
    cv2.putText(img, text, (10 + shift[0], 42 + shift[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 30, 2)
    return img


class PrintDefects(unittest.TestCase):
    def test_same_print_shifted_and_dimmer(self):
        crop = (label(shift=(3, 2)) * 0.8 + 20).astype(np.uint8)
        self.assertEqual(q.check_print("ece_approval", label(), crop), [])

    def test_smear(self):
        crop = label()
        cv2.circle(crop, (130, 30), 14, 30, -1)
        self.assertIn("PRINT_FLAW", [i.code for i in q.check_print("ece_approval", label(), crop)])

    def test_broken_characters(self):
        crop = label()
        crop[:, 150:200] = 200  # two characters not printed
        self.assertIn("PRINT_FLAW", [i.code for i in q.check_print("ece_approval", label(), crop)])

    def test_positions(self):
        anchor = (0, 0, 40, 40)
        master = {"ece_approval": q.relative_offset((60, 0, 200, 40), anchor)}
        ok = {"e_mark": anchor, "ece_approval": (65, 2, 205, 42)}
        moved = {"e_mark": anchor, "ece_approval": (60, 40, 200, 80)}
        self.assertEqual(q.check_positions(ok, "e_mark", master), [])
        self.assertEqual([i.code for i in q.check_positions(moved, "e_mark", master)], ["POSITION"])


if __name__ == "__main__":
    unittest.main()
