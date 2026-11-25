import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import autocapture
from app.autocapture import CLEAR_FRAMES, AutoCaptureGate, Master, count_images, hamming, image_hash, load_masters


def pattern(seed: int, size=(120, 160)) -> np.ndarray:
    """A random blocky RGB image; different seeds give unrelated layouts."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (8, 10, 3), dtype=np.uint8)
    return np.kron(blocks, np.ones((size[0] // 8, size[1] // 10, 1), dtype=np.uint8))


def master(name: str, image: np.ndarray) -> Master:
    return Master(f"part/{name}.jpg", image_hash(image))


class Hashing(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(hamming(image_hash(pattern(1)), image_hash(pattern(1))), 0)

    def test_noise_and_brightness_stay_close(self):
        base = pattern(1)
        rng = np.random.default_rng(7)
        noisy = np.clip(base.astype(int) + rng.integers(-8, 9, base.shape) + 20, 0, 255).astype(np.uint8)
        self.assertLessEqual(hamming(image_hash(base), image_hash(noisy)), 10)

    def test_size_independent(self):
        self.assertLessEqual(hamming(image_hash(pattern(1)), image_hash(pattern(1, (480, 640)))), 4)

    def test_different_images_far_apart(self):
        self.assertGreater(hamming(image_hash(pattern(1)), image_hash(pattern(2))), 16)

    def test_grayscale_input(self):
        gray = pattern(3)[:, :, 0]
        self.assertIsInstance(image_hash(gray), int)


class Gate(unittest.TestCase):
    def setUp(self):
        self.masters = [master("a", pattern(1)), master("b", pattern(2))]

    def gate(self, **kw):
        args = dict(threshold=10, interval_s=1.0, max_images=3)
        args.update(kw)
        return AutoCaptureGate(**args)

    def test_picks_nearest_master(self):
        found = self.gate().check(image_hash(pattern(2)), self.masters, now=0)
        self.assertIs(found, self.masters[1])

    def test_rejects_above_threshold(self):
        self.assertIsNone(self.gate().check(image_hash(pattern(9)), self.masters, now=0))

    def test_no_masters(self):
        self.assertIsNone(self.gate().check(image_hash(pattern(1)), [], now=0))

    def test_interval(self):
        gate = self.gate()
        gate.record_saved(image_hash(pattern(1)), now=0)
        b = image_hash(pattern(2))
        self.assertIsNone(gate.check(b, self.masters, now=0.5))
        self.assertIsNotNone(gate.check(b, self.masters, now=1.0))

    def test_same_view_not_saved_twice(self):
        gate = self.gate(interval_s=0)
        a = image_hash(pattern(1))
        gate.record_saved(a, now=0)
        self.assertIsNone(gate.check(a, self.masters, now=5))
        self.assertIsNotNone(gate.check(image_hash(pattern(2)), self.masters, now=5))

    def test_stops_at_max(self):
        gate = self.gate(max_images=2, count=1, interval_s=0)
        self.assertFalse(gate.full)
        gate.record_saved(image_hash(pattern(1)), now=0)
        self.assertTrue(gate.full)
        self.assertIsNone(gate.check(image_hash(pattern(2)), self.masters, now=1))


class WaitForPartToLeave(unittest.TestCase):
    def setUp(self):
        self.a = pattern(1)
        self.masters = [master("a", self.a), master("b", pattern(2))]
        self.gate = AutoCaptureGate(threshold=10, interval_s=0, max_images=100, require_clear=True)
        self.gate.record_saved(image_hash(self.a), now=0, master=self.masters[0])
        # Same part, still in view, but 6 bits off: past DEDUP_BITS, within threshold.
        self.moved = image_hash(self.a) ^ 0b111111
        self.empty = image_hash(pattern(9))

    def test_part_still_in_view_not_saved_again(self):
        self.assertTrue(self.gate.waiting)
        distance = hamming(self.moved, self.masters[0].hash)
        self.assertTrue(autocapture.DEDUP_BITS <= hamming(self.moved, image_hash(self.a)) and distance <= 10)
        for t in range(1, 20):
            self.assertIsNone(self.gate.check(self.moved, self.masters, now=t))

    def test_saves_again_after_part_leaves(self):
        for t in range(CLEAR_FRAMES):
            self.assertIsNone(self.gate.check(self.empty, self.masters, now=t))
        self.assertFalse(self.gate.waiting)
        self.assertIsNotNone(self.gate.check(self.moved, self.masters, now=10))

    def test_same_position_after_leaving_is_saved(self):
        for t in range(CLEAR_FRAMES):
            self.gate.check(self.empty, self.masters, now=t)
        self.assertIsNotNone(self.gate.check(image_hash(self.a), self.masters, now=10))

    def test_swapped_for_other_part_without_gap(self):
        b = self.masters[1].hash
        for t in range(CLEAR_FRAMES):
            self.assertIsNone(self.gate.check(b, self.masters, now=t))
        self.assertIs(self.gate.check(b, self.masters, now=10), self.masters[1])

    def test_brief_gap_does_not_rearm(self):
        for t in range(CLEAR_FRAMES - 1):
            self.gate.check(self.empty, self.masters, now=t)
        self.gate.check(self.moved, self.masters, now=5)  # back in view: count restarts
        for t in range(CLEAR_FRAMES - 1):
            self.gate.check(self.empty, self.masters, now=6 + t)
        self.assertTrue(self.gate.waiting)

    def test_near_threshold_is_not_clear(self):
        near = master("a", self.a).hash ^ 0b111111111111  # 12 bits off: > threshold, < threshold + margin
        for t in range(10):
            self.gate.check(near, self.masters, now=t)
        self.assertTrue(self.gate.waiting)

    def test_off_saves_part_again(self):
        gate = AutoCaptureGate(threshold=10, interval_s=0, max_images=100, require_clear=False)
        gate.record_saved(image_hash(self.a), now=0)
        self.assertFalse(gate.waiting)
        self.assertIsNotNone(gate.check(self.moved, self.masters, now=1))


class LoadMasters(unittest.TestCase):
    def test_drops_empty_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = autocapture.MASTERS_DIR
            autocapture.MASTERS_DIR = Path(tmp)
            try:
                (Path(tmp) / "p").mkdir()
                for name, seed in (("x.jpg", 1), ("y.jpg", 2), ("z.jpg", 3), ("w.jpg", 4)):
                    cv2.imwrite(str(Path(tmp) / "p" / name), pattern(seed))
                entries = [None, {"class_name": "A b", "image": "p/x.jpg"}, {"image": "p/gone.jpg"},
                           {"class_name": "B", "image": "p/y.jpg"}, {"class_name": "C", "image": "p/z.jpg"},
                           {"class_name": "D", "image": "p/w.jpg"}]
                masters = load_masters(entries)
            finally:
                autocapture.MASTERS_DIR = old
        # Old configs carried a class_name; it is ignored, not required.
        self.assertEqual([m.image for m in masters], ["p/x.jpg", "p/y.jpg", "p/z.jpg", "p/w.jpg"])
        self.assertEqual(masters[0].to_entry(), {"image": "p/x.jpg"})
        self.assertEqual(load_masters(None), [])


class Counting(unittest.TestCase):
    def test_counts_only_job_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("images/1_m1.jpg", "images/2_m2.png", "images/3_m1.jpg",
                        "labels/1_m1.txt", "stray.jpg", "images/notes.txt",
                        "master_1/images/old.jpg"):  # old per-master layout: not counted
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            self.assertEqual(count_images(root), 3)
            self.assertEqual(count_images(root / "missing"), 0)


if __name__ == "__main__":
    unittest.main()
