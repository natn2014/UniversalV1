import unittest

from app.vision.base import ToolResult
from app.vision.total import total_state


def res(ok, kind="area"):
    return ToolResult("t", kind, 0.0, ok)


class Total(unittest.TestCase):
    def test_and_gate(self):
        self.assertEqual(total_state([res(True), res(True)], None, "AND"), "GREEN")
        self.assertEqual(total_state([res(True), res(False)], None, "AND"), "RED")

    def test_or_gate(self):
        self.assertEqual(total_state([res(False), res(True)], None, "OR"), "GREEN")
        self.assertEqual(total_state([res(False), res(False)], None, "OR"), "RED")

    def test_yolo_is_one_more_input(self):
        self.assertEqual(total_state([res(True)], "GREEN"), "GREEN")
        self.assertEqual(total_state([res(True)], "RED"), "RED")
        self.assertEqual(total_state([res(False)], "GREEN", "OR"), "GREEN")
        self.assertEqual(total_state([res(False)], "RED", "OR"), "RED")
        self.assertEqual(total_state([], "GREEN"), "GREEN")
        self.assertEqual(total_state([], "RED"), "RED")

    def test_yolo_yellow_means_no_part(self):
        self.assertEqual(total_state([res(True)], "YELLOW"), "YELLOW")
        self.assertEqual(total_state([res(True)], "YELLOW", "OR"), "YELLOW")

    def test_position_not_found(self):
        self.assertEqual(total_state([res(True)], "GREEN", position_found=False), "YELLOW")

    def test_position_result_is_not_a_gate_input(self):
        self.assertEqual(total_state([res(False, "position"), res(True)], None), "GREEN")
        self.assertEqual(total_state([res(True, "position")], None), "YELLOW")

    def test_nothing_to_judge(self):
        self.assertEqual(total_state([], None), "YELLOW")
        self.assertEqual(total_state([], None, "OR"), "YELLOW")

    def test_pending_slow_tool_is_standby(self):
        pending = ToolResult("ocr", "ocr", 0.0, False, extra={"pending": True})
        for gate in ("AND", "OR"):
            self.assertEqual(total_state([res(True), pending], None, gate), "YELLOW")
            self.assertEqual(total_state([res(True), pending], "GREEN", gate), "YELLOW")


if __name__ == "__main__":
    unittest.main()
