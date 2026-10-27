import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from PySide6.QtCore import QObject

from app import context as context_mod
from app import jobs
from app.context import AppContext
from app.vision.base import Tool, register
from app.vision.position import PositionTool
from app.vision.store import tools_from_config, tools_to_config


@register("_store_dummy")
class StoreDummy(Tool):
    PARAMS = {"gain": 1.5}


def taught_position():
    frame = np.full((240, 320, 3), 20, np.uint8)
    cv2.rectangle(frame, (120, 90), (200, 150), 255, -1)
    t = PositionTool(name="pos 1", roi=(0.2, 0.2, 0.8, 0.8))
    t.teach(frame, template_roi=(0.35, 0.35, 0.65, 0.65))
    return t


class RoundTrip(unittest.TestCase):
    def test_two_tools_with_a_template_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_dir = Path(tmp) / "tools"
            tools = [taught_position(), StoreDummy(name="d", gain=2.5, low=1, high=None)]
            entries = tools_to_config(tools, "PART-1", tools_dir)
            entries = json.loads(json.dumps(entries))  # must be JSON-safe
            self.assertTrue((tools_dir / entries[0]["template_file"]).exists())
            self.assertNotIn("template_file", entries[1])

            loaded = tools_from_config(entries, "PART-1", tools_dir)
            self.assertEqual([t.to_dict() for t in loaded], entries)
            self.assertTrue(np.array_equal(loaded[0].template, tools[0].template))
            self.assertIsNone(loaded[1].template)

    def test_stale_template_files_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_dir = Path(tmp)
            tools_to_config([taught_position()], "P", tools_dir)
            self.assertEqual(len(list((tools_dir / "P").glob("*.png"))), 1)
            tools_to_config([], "P", tools_dir)
            self.assertEqual(list((tools_dir / "P").glob("*.png")), [])

    def test_same_name_tools_do_not_share_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = tools_to_config([taught_position(), taught_position()], "P", Path(tmp))
            self.assertNotEqual(entries[0]["template_file"], entries[1]["template_file"])

    def test_broken_entries_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = StoreDummy(name="ok").to_dict()
            loaded = tools_from_config([{"kind": "nope"}, {"name": "no kind"}, good, "junk"], "P", Path(tmp))
            self.assertEqual([t.name for t in loaded], ["ok"])

    def test_missing_template_file_keeps_tool_untaught(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = taught_position().to_dict()
            entry["template_file"] = "P/gone.png"
            loaded = tools_from_config([entry], "P", Path(tmp))
            self.assertEqual(len(loaded), 1)
            self.assertIsNone(loaded[0].template)

    def test_no_tools_key_loads_empty(self):
        self.assertEqual(tools_from_config(None, "P", Path("/nonexistent")), [])
        self.assertEqual(tools_from_config([], "P", Path("/nonexistent")), [])


class ContextSave(unittest.TestCase):
    def make_ctx(self):
        ctx = AppContext.__new__(AppContext)  # no camera / tower / settings
        QObject.__init__(ctx)
        ctx.tools, ctx.tools_gate, ctx.current_part_no = [], "AND", "JOB1"
        return ctx

    def test_job_json_and_png_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(jobs, "PART_CONFIGS_DIR", root), \
                    mock.patch.object(context_mod, "TOOLS_DIR", root / "tools"):
                jobs.save_part_config("JOB1", {"part_no": "JOB1", "roi": [0.1, 0.1, 0.5, 0.5]})
                ctx = self.make_ctx()
                ctx.tools_gate = "OR"
                ctx.set_tools([taught_position(), StoreDummy(name="d")], save_to_job=True)

                config = jobs.load_part_config("JOB1")
                self.assertEqual(len(config["tools"]), 2)
                self.assertEqual(config["tools_gate"], "OR")
                self.assertEqual(config["roi"], [0.1, 0.1, 0.5, 0.5])  # untouched keys stay

                other = self.make_ctx()
                other.load_tools(config)
                self.assertEqual([t.to_dict() for t in other.tools], config["tools"])
                self.assertEqual(other.tools_gate, "OR")

    def test_old_job_without_tools(self):
        ctx = self.make_ctx()
        ctx.tools = [StoreDummy()]
        ctx.load_tools({"part_no": "OLD"})
        self.assertEqual(ctx.tools, [])
        self.assertEqual(ctx.tools_gate, "AND")

    def test_not_saved_unless_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(jobs, "PART_CONFIGS_DIR", root):
                jobs.save_part_config("JOB1", {"part_no": "JOB1"})
                self.make_ctx().set_tools([StoreDummy()])
                self.assertNotIn("tools", jobs.load_part_config("JOB1"))


if __name__ == "__main__":
    unittest.main()
