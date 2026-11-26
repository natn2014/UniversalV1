"""Training picks a batch that fits in free memory (the OOM kill of
2026-11-25), and refuses to start when even the smallest does not fit."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app.training as training  # noqa: E402
from app.training import TrainingWorker, batch_for_memory, free_memory_mb  # noqa: E402


def meminfo(path: Path, available_mb: int, swap_free_mb: int) -> str:
    path.write_text(f"MemTotal:        7660000 kB\nMemFree:          300000 kB\n"
                    f"MemAvailable:    {available_mb * 1024} kB\nSwapTotal:       2097148 kB\n"
                    f"SwapFree:        {swap_free_mb * 1024} kB\n")
    return str(path)


class FreeMemory(unittest.TestCase):
    def test_reads_available_plus_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(free_memory_mb(meminfo(Path(tmp) / "m", 2768, 984)), 3752)

    def test_missing_file(self):
        self.assertIsNone(free_memory_mb("/nonexistent/meminfo"))


class BatchChoice(unittest.TestCase):
    def test_night_of_the_kill_gets_batch_4(self):
        # VS Code + Firefox open: ~3.75 GB free. Batch 8 was killed; 4 fits.
        self.assertEqual(batch_for_memory(8, 640, 3752), 4)

    def test_plenty_of_memory_keeps_the_request(self):
        self.assertEqual(batch_for_memory(8, 640, 6000), 8)
        self.assertEqual(batch_for_memory(4, 640, 6000), 4)  # never above the request

    def test_low_memory(self):
        self.assertEqual(batch_for_memory(8, 640, 2700), 2)
        self.assertIsNone(batch_for_memory(8, 640, 2000))

    def test_smaller_images_need_less(self):
        self.assertEqual(batch_for_memory(8, 320, 2700), 8)

    def test_unknown_memory_keeps_the_request(self):
        self.assertEqual(batch_for_memory(8, 640, None), 8)


class WorkerChecksFirst(unittest.TestCase):
    def config(self):
        return {"model_path": "x.pt", "data_yaml": "d.yaml", "epochs": 1, "imgsz": 640, "batch": 8,
                "device": "cpu", "project": "p", "name": "n", "patience": 1, "optimizer": "auto", "lr0": 0.01}

    def test_refuses_without_loading_a_model(self):
        worker = TrainingWorker(self.config())
        errors = []
        worker.training_error.connect(errors.append)
        with mock.patch.object(training, "free_memory_mb", return_value=1500), \
                mock.patch.object(training, "YOLO") as yolo:
            worker.run()
        yolo.assert_not_called()
        self.assertIn("Not enough free memory", errors[0])

    def test_lowers_the_batch_and_says_so(self):
        worker = TrainingWorker(self.config())
        status = []
        worker.status_update.connect(status.append)
        with mock.patch.object(training, "free_memory_mb", return_value=3752), \
                mock.patch.object(training, "YOLO") as yolo:
            worker.run()
        self.assertEqual(yolo.return_value.train.call_args.kwargs["batch"], 4)
        self.assertTrue(any("batch 4 instead of 8" in s for s in status))


class ReleasesMemory(unittest.TestCase):
    def test_after_success_and_after_failure(self):
        cfg = WorkerChecksFirst().config()
        for train_effect in (None, RuntimeError("boom")):
            worker = TrainingWorker(dict(cfg))
            with mock.patch.object(training, "free_memory_mb", return_value=6000), \
                    mock.patch.object(training, "YOLO") as yolo, \
                    mock.patch.object(training, "release_training_memory") as release:
                yolo.return_value.train.side_effect = train_effect
                worker.run()
            release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
