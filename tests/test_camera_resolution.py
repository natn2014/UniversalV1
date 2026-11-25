import unittest

from app.camera import DEFAULT_RESOLUTION, RESOLUTION_CHOICES, parse_resolution


class ParseResolution(unittest.TestCase):
    def test_choices_parse(self):
        for _, value in RESOLUTION_CHOICES:
            w, h = parse_resolution(value)
            self.assertGreater(w, h)

    def test_default_is_full_hd(self):
        self.assertEqual(parse_resolution(DEFAULT_RESOLUTION), (1920, 1080))

    def test_bad_values_fall_back_to_default(self):
        for bad in ("", "1920", "axb", "0x0", None):
            self.assertEqual(parse_resolution(bad), (1920, 1080), bad)


if __name__ == "__main__":
    unittest.main()
