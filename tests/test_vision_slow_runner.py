import threading
import time
import unittest

import numpy as np

from app.camera import VideoWorker
from app.vision.base import Offset, Tool, register
from app.vision.brightness import BrightnessAverageTool
from app.vision.runner import run_tools_with_offset
from app.vision.slow_runner import SlowToolRunner
from tests.test_vision_runner import H, W, job_tools, scene

RUN_S = 0.15


@register("_sleepy")
class Sleepy(Tool):
    """Stands in for OCR: slow, and counts how often it really runs."""

    SLOW = True
    runs = 0
    lock = threading.Lock()

    def process(self, patch, origin):
        time.sleep(RUN_S)
        with Sleepy.lock:
            Sleepy.runs += 1
        return self.result(float(patch.mean()), text="read")


def wait_idle(runner: SlowToolRunner, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while not runner.due(time.monotonic(), 1000.0) and time.monotonic() < end:
        time.sleep(0.01)


class SplitRunner(unittest.TestCase):
    def test_fast_and_slow_selection(self):
        fast, slow = BrightnessAverageTool(name="b"), Sleepy(name="s")
        frame = scene()
        self.assertEqual([r.name for r in run_tools_with_offset(frame, [fast, slow], slow=False)[0]], ["b"])
        self.assertEqual([r.name for r in run_tools_with_offset(frame, [fast, slow], slow=True)[0]], ["s"])
        self.assertEqual(len(run_tools_with_offset(frame, [fast, slow])[0]), 2)

    def test_given_offset_skips_position(self):
        pos, _ = job_tools()
        results, offset = run_tools_with_offset(scene(), [pos, Sleepy(name="s")], slow=True,
                                                offset=Offset(dx=5))
        self.assertEqual([r.kind for r in results], ["_sleepy"])
        self.assertEqual(offset.dx, 5)


class Runner(unittest.TestCase):
    def setUp(self):
        self.runner = SlowToolRunner()
        self.tool = Sleepy(name="ocr")
        Sleepy.runs = 0

    def tearDown(self):
        self.runner.stop()

    def submit(self, epoch=1, now=None):
        self.runner.submit(scene(), Offset(), [self.tool], epoch, time.monotonic() if now is None else now)

    def test_mailbox_keeps_only_the_newest_frame(self):
        for _ in range(10):
            self.submit()
        time.sleep(RUN_S * 3)
        self.assertLessEqual(Sleepy.runs, 2)

    def test_due_respects_rate_and_busy(self):
        self.assertTrue(self.runner.due(100.0, 2.0))
        self.submit(now=100.0)
        self.assertFalse(self.runner.due(100.0, 2.0))  # busy
        wait_idle(self.runner)
        self.assertFalse(self.runner.due(100.3, 2.0))  # within 0.5 s of the submit
        self.assertTrue(self.runner.due(100.5, 2.0))

    def test_result_counts_only_for_its_epoch_and_age(self):
        now = time.monotonic()
        self.assertTrue(self.runner.results_for([self.tool], 1, now, 1.0)[0].extra["pending"])
        self.submit(epoch=1, now=now)
        wait_idle(self.runner)
        res = self.runner.results_for([self.tool], 1, now + 0.2, 1.0)[0]
        self.assertNotIn("pending", res.extra)
        self.assertEqual(res.text, "read")
        self.assertAlmostEqual(res.extra["age_ms"], 200, delta=1)
        self.assertTrue(self.runner.results_for([self.tool], 2, now + 0.2, 1.0)[0].extra["pending"])
        self.assertTrue(self.runner.results_for([self.tool], 1, now + 1.5, 1.0)[0].extra["pending"])

    def test_clear_drops_cached_and_in_flight_results(self):
        now = time.monotonic()
        self.submit(now=now)
        self.runner.clear()  # while the run is in flight
        wait_idle(self.runner)
        self.assertTrue(self.runner.results_for([self.tool], 1, now, 10.0)[0].extra["pending"])

    def test_stop_joins_and_submit_restarts(self):
        self.submit()
        start = time.monotonic()
        self.runner.stop()
        self.assertLess(time.monotonic() - start, 1.0)
        self.submit()
        wait_idle(self.runner)
        self.assertGreaterEqual(Sleepy.runs, 1)


class WorkerVision(unittest.TestCase):
    """VideoWorker._run_vision: slow tools join in from their own thread, and
    a new part never shows the previous part's slow result."""

    def setUp(self):
        self.worker = VideoWorker()
        self.worker.set_slow_hz(20.0)
        pos, _ = job_tools()
        self.slow = Sleepy(name="ocr", roi=(0.4, 0.62, 0.7, 0.88))
        self.worker.set_tools([pos, self.slow])

    def tearDown(self):
        self.worker._slow.stop()

    def run_until_read(self, frame, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            results = self.worker._run_vision(frame, self.worker._tools)
            slow = [r for r in results if r.name == "ocr"]
            if slow and "pending" not in slow[0].extra:
                return results
            time.sleep(0.02)
        self.fail("slow result never arrived")

    def test_pending_then_result(self):
        first = self.worker._run_vision(scene(), self.worker._tools)
        self.assertEqual([r.kind for r in first], ["position", "_sleepy"])
        self.assertTrue(first[1].extra["pending"])
        results = self.run_until_read(scene())
        self.assertTrue(results[1].extra["slow"])

    def test_new_part_starts_pending(self):
        self.run_until_read(scene())
        lost = self.worker._run_vision(np.full((H, W, 3), 20, np.uint8), self.worker._tools)
        self.assertEqual([r.kind for r in lost], ["position"])
        back = self.worker._run_vision(scene(), self.worker._tools)
        self.assertTrue(back[1].extra["pending"])

    def test_fast_loop_is_not_blocked(self):
        start = time.perf_counter()
        for _ in range(10):
            self.worker._run_vision(scene(), self.worker._tools)
        self.assertLess(time.perf_counter() - start, RUN_S)


if __name__ == "__main__":
    unittest.main()
