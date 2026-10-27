import unittest

import numpy as np

from app.vision.edges import find_edges, gradient, profile


def step(n=100, at=40, lo=0.0, hi=100.0):
    p = np.full(n, lo)
    p[at:] = hi
    return p


class Profile(unittest.TestCase):
    def test_axes(self):
        patch = np.zeros((10, 20), np.uint8)
        patch[:, 5:] = 100
        self.assertEqual(profile(patch, "x").shape, (20,))
        self.assertEqual(profile(patch, "y").shape, (10,))
        self.assertEqual(profile(patch, "x")[7], 100)

    def test_bgr_input(self):
        self.assertEqual(profile(np.zeros((6, 8, 3), np.uint8)).shape, (8,))

    def test_gradient_length(self):
        self.assertEqual(len(gradient(step())), 100)


class FindEdges(unittest.TestCase):
    def test_sharp_step_position(self):
        edges = find_edges(step(at=40))
        self.assertEqual(len(edges), 1)
        self.assertAlmostEqual(edges[0][0], 40.0, delta=0.2)
        self.assertGreater(edges[0][1], 0)

    def test_blurred_step_at_40_5(self):
        x = np.arange(100)
        p = 100 / (1 + np.exp(-(x + 0.5 - 40.5) / 2.0))  # centre at coordinate 40.5
        edges = find_edges(p, threshold=5)
        self.assertEqual(len(edges), 1)
        self.assertAlmostEqual(edges[0][0], 40.5, delta=0.2)

    def test_polarity_filter(self):
        p = np.zeros(100)
        p[30:60] = 100  # rising at 30, falling at 60
        self.assertEqual(len(find_edges(p)), 2)
        rising = find_edges(p, polarity="rising")
        falling = find_edges(p, polarity="falling")
        self.assertEqual((len(rising), len(falling)), (1, 1))
        self.assertAlmostEqual(rising[0][0], 30, delta=0.3)
        self.assertAlmostEqual(falling[0][0], 60, delta=0.3)
        self.assertLess(falling[0][1], 0)

    def test_min_distance(self):
        def two_steps(gap):
            p = np.zeros(100)
            p[40:] = 50
            p[40 + gap:] = 100
            return p
        self.assertEqual(len(find_edges(two_steps(2), threshold=10, min_distance=3, smooth=0)), 1)
        self.assertEqual(len(find_edges(two_steps(4), threshold=10, min_distance=3, smooth=0)), 2)
        self.assertEqual(len(find_edges(two_steps(4), threshold=10, min_distance=6, smooth=0)), 1)

    def test_threshold_and_flat(self):
        self.assertEqual(find_edges(np.full(50, 80.0)), [])
        self.assertEqual(find_edges(step(hi=10), threshold=20), [])

    def test_short_profile(self):
        self.assertEqual(find_edges(np.array([1.0, 2.0])), [])
        self.assertEqual(find_edges(np.array([])), [])


if __name__ == "__main__":
    unittest.main()
