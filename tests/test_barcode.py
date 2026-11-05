"""Code 39 scans back to the saved job name (app/barcode.py)."""

import unittest

from app.barcode import decode_full_ascii, encode_full_ascii, match_part_no, scan_readings

JOBS = ["1500D/H27/I-MM-3EFHMX-A6/D12", "2500D/H23/I-MM-3EFHMX/D045",
        "15K45/H40/I-MM-4FYMX-C8/HLF/I/D15", "8850304144380", "MarkTest"]


class Code39ToJob(unittest.TestCase):
    def test_literal_slashes_are_not_escapes(self):
        """'/H', '/I', '/D' in a part number used to decode to '(' ')' and a cut."""
        for part in JOBS:
            self.assertEqual(match_part_no(part, JOBS), part)

    def test_saved_name_encoded_back_matches(self):
        for part in JOBS:
            self.assertEqual(match_part_no(encode_full_ascii(part), JOBS), part)
            self.assertEqual(decode_full_ascii(encode_full_ascii(part)), part)

    def test_case_and_file_name(self):
        self.assertEqual(match_part_no("MARKTEST", JOBS), "MarkTest")
        self.assertEqual(match_part_no("1500D_H27_I-MM-3EFHMX-A6_D12", JOBS), JOBS[0])

    def test_check_character_and_label_format(self):
        body = "8850304144380"
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"
        check = chars[sum(chars.index(c) for c in body) % 43]
        self.assertEqual(match_part_no(body + check, JOBS), body)
        self.assertEqual(match_part_no("X$MarkTest$Y", JOBS), "MarkTest")

    def test_unknown_scan_stays_literal(self):
        self.assertEqual(match_part_no("NEW/PART-1", JOBS), "NEW/PART-1")
        self.assertEqual(match_part_no("  ", JOBS), "")
        self.assertIn("new/Part", scan_readings("+N+E+W/O+P+A+R+T".replace("+P", "P")))


if __name__ == "__main__":
    unittest.main()


class PartLabel(unittest.TestCase):
    def test_full_ascii_shown(self):
        from app.barcode import part_label
        self.assertEqual(part_label("1500D/H27/I-MM-3EFHMX-A6/D12"), "1500D(27)-MM-3EFHMX-A6$12")
        self.assertEqual(part_label("MarkTest"), "MarkTest")
        self.assertIsNone(part_label(None))
