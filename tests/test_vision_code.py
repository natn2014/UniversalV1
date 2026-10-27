import unittest

import cv2
import numpy as np

from app.vision.code import CodeTool


def qr_frame(text="PART-123", module=8, size=(240, 320)):
    qr = cv2.QRCodeEncoder.create().encode(text)
    qr = cv2.resize(qr, None, fx=module, fy=module, interpolation=cv2.INTER_NEAREST)
    frame = np.full((*size, 3), 255, np.uint8)
    h, w = qr.shape[:2]
    y, x = (size[0] - h) // 2, (size[1] - w) // 2
    frame[y:y + h, x:x + w] = qr[..., None] if qr.ndim == 2 else qr
    return frame


class Code(unittest.TestCase):
    def test_reads_qr(self):
        res = CodeTool().run(qr_frame())
        self.assertEqual((res.value, res.text, res.ok), (1.0, "PART-123", True))
        self.assertEqual(res.overlay[1]["type"], "poly")

    def test_expected_and_pattern(self):
        self.assertTrue(CodeTool(expected="PART-123").run(qr_frame()).ok)
        self.assertFalse(CodeTool(expected="PART-999").run(qr_frame()).ok)
        self.assertTrue(CodeTool(pattern=r"PART-\d{3}").run(qr_frame()).ok)
        res = CodeTool(pattern=r"LOT-\d+").run(qr_frame())
        self.assertFalse(res.ok)
        self.assertEqual(res.value, 1)  # it was read, just not the right one

    def test_job_part_number(self):
        tool = CodeTool(use_job_part_no=True)
        tool.part_no = "PART-123"
        self.assertTrue(tool.run(qr_frame()).ok)
        tool.part_no = "OTHER"
        self.assertFalse(tool.run(qr_frame()).ok)
        tool.part_no = None
        self.assertEqual(tool.run(qr_frame()).extra["reason"], "no job part number")

    def test_blank_roi(self):
        res = CodeTool().run(np.full((100, 100, 3), 255, np.uint8))
        self.assertEqual(res.value, 0)
        self.assertFalse(res.ok)

    def test_small_code_found_at_2x(self):
        self.assertEqual(CodeTool().run(qr_frame(module=2)).text, "PART-123")

    def test_roi_and_symbology_filter(self):
        self.assertTrue(CodeTool(symbology="qrcode", roi=(0.1, 0.1, 0.9, 0.9)).run(qr_frame()).ok)
        self.assertFalse(CodeTool(symbology="code39").run(qr_frame()).ok)

    def test_dollar_label_is_cleaned(self):
        self.assertEqual(CodeTool().run(qr_frame("PRE$ABC-1$SUF")).text, "ABC-1")


if __name__ == "__main__":
    unittest.main()
