import ast
import json
import unittest
from pathlib import Path

import numpy as np

from app.vision import base
from app.vision.base import Offset, Tool, ToolResult, create, register, rotate_points, shift_overlay


@register("_dummy")
class DummyTool(Tool):
    PARAMS = {"gain": 2.0, "tags": []}

    def process(self, patch, origin):
        return self.result(float(patch.mean()) * self.gain, [base.overlay_item("line", [0, 0, 5, 5])])


class Judge(unittest.TestCase):
    def test_limits(self):
        t = DummyTool(low=10, high=20)
        self.assertTrue(t.judge(10) and t.judge(20) and t.judge(15))
        self.assertFalse(t.judge(9.9) or t.judge(20.1))
        self.assertTrue(DummyTool(low=10, high=None).judge(1e9))
        self.assertFalse(DummyTool(low=10, high=None).judge(5))
        self.assertTrue(DummyTool(low=None, high=20).judge(-1e9))
        self.assertTrue(DummyTool(low=None, high=None).judge(123))


class Crop(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 200, 3), np.uint8)

    def test_plain_crop(self):
        t = DummyTool(roi=(0.25, 0.25, 0.5, 0.75))
        patch, origin = t.crop(self.frame, Offset())
        self.assertEqual(origin, (50, 25))
        self.assertEqual(patch.shape[:2], (50, 50))

    def test_shift_clamps_at_frame_edge(self):
        t = DummyTool(roi=(0.5, 0.5, 0.75, 1.0))
        patch, origin = t.crop(self.frame, Offset(dx=500, dy=-500))
        self.assertEqual(origin, (150, 0))
        self.assertEqual(patch.shape[:2], (50, 50))

    def test_no_follow_ignores_offset(self):
        t = DummyTool(roi=(0.25, 0.25, 0.5, 0.75), follow_position=False)
        self.assertEqual(t.crop(self.frame, Offset(dx=10, dy=10))[1], (50, 25))

    def test_not_found_offset_ignored(self):
        t = DummyTool(roi=(0.25, 0.25, 0.5, 0.75))
        self.assertEqual(t.crop(self.frame, Offset(dx=10, found=False))[1], (50, 25))


class Run(unittest.TestCase):
    def test_result_and_overlay_in_frame_pixels(self):
        frame = np.full((100, 200, 3), 10, np.uint8)
        t = DummyTool(name="d", roi=(0.5, 0.5, 1.0, 1.0), high=100)
        res = t.run(frame, Offset())
        self.assertIsInstance(res, ToolResult)
        self.assertEqual((res.name, res.kind, res.value, res.ok), ("d", "_dummy", 20.0, True))
        self.assertEqual(res.overlay[0]["pts"], [100, 50, 200, 100])
        self.assertEqual(res.overlay[1]["pts"], [100, 50, 105, 55])

    def test_empty_frame_does_not_raise(self):
        res = DummyTool().run(np.zeros((0, 0, 3), np.uint8))
        self.assertFalse(res.ok)
        self.assertTrue(res.text)


class Helpers(unittest.TestCase):
    def test_rotate_points(self):
        out = rotate_points([(10, 0)], (0, 0), 90)
        self.assertAlmostEqual(out[0][0], 0, places=6)
        self.assertAlmostEqual(out[0][1], -10, places=6)

    def test_shift_overlay_each_type(self):
        items = [
            {"type": "circle", "pts": [5, 6, 3], "ok": True, "label": ""},
            {"type": "poly", "pts": [[0, 0], [1, 1]], "ok": True, "label": ""},
            {"type": "text", "pts": [1, 2], "ok": True, "label": "x"},
        ]
        out = shift_overlay(items, 10, 20)
        self.assertEqual(out[0]["pts"], [15, 26, 3])
        self.assertEqual(out[1]["pts"], [[10, 20], [11, 21]])
        self.assertEqual(out[2]["pts"], [11, 22])
        self.assertEqual(items[0]["pts"], [5, 6, 3])


class Registry(unittest.TestCase):
    def test_round_trip(self):
        t = DummyTool(name="n", roi=(0.1, 0.2, 0.3, 0.4), low=1, high=None, enabled=False, gain=3.5)
        d = json.loads(json.dumps(t.to_dict()))
        again = Tool.from_dict(d)
        self.assertIsInstance(again, DummyTool)
        self.assertEqual(again.to_dict(), t.to_dict())
        self.assertIsNone(again.high)
        self.assertEqual(again.gain, 3.5)

    def test_unknown_kind(self):
        with self.assertRaises(KeyError):
            create("no_such_tool")
        with self.assertRaises(KeyError):
            Tool.from_dict({"kind": "no_such_tool"})

    def test_default_limits_per_class(self):
        class Limited(DummyTool):
            DEFAULT_HIGH = 5.0
        self.assertEqual(Limited().high, 5.0)
        self.assertIsNone(Limited(high=None).high)

    def test_mutable_param_default_not_shared(self):
        a, b = DummyTool(), DummyTool()
        a.tags.append(1)
        self.assertEqual(b.tags, [])


class NoQt(unittest.TestCase):
    def test_vision_package_does_not_import_qt(self):
        for path in Path(base.__file__).parent.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    self.assertFalse(n.startswith(("PySide6", "PyQt")), f"{path.name} imports {n}")


if __name__ == "__main__":
    unittest.main()
