import unittest

from app.augmentation import DEFAULT_PRESET, PRESETS, build_hyp, describe


class Presets(unittest.TestCase):
    def test_flips_always_explicit(self):
        for preset in PRESETS:
            hyp = build_hyp(preset)
            self.assertEqual((hyp["fliplr"], hyp["flipud"]), (0.0, 0.0), preset)
            self.assertEqual(build_hyp(preset, mirror=True)["fliplr"], 0.5)
            self.assertEqual(build_hyp(preset, flip_ud=True)["flipud"], 0.5)

    def test_off_changes_nothing(self):
        self.assertTrue(all(v == 0.0 for v in build_hyp("off").values()))

    def test_presets_cover_the_same_keys(self):
        # Switching presets must never leave an Ultralytics default (e.g.
        # mosaic=1.0) active because a key was left out.
        keys = {frozenset(p) for p in PRESETS.values()}
        self.assertEqual(len(keys), 1)
        self.assertIn("mosaic", next(iter(keys)))

    def test_strength_increases(self):
        for key in ("hsv_v", "degrees", "scale", "mosaic"):
            self.assertLessEqual(PRESETS["off"][key], PRESETS["light"][key])
            self.assertLess(PRESETS["light"][key], PRESETS["strong"][key])

    def test_unknown_preset_uses_default(self):
        self.assertEqual(build_hyp("bogus"), build_hyp(DEFAULT_PRESET))

    def test_valid_ultralytics_settings(self):
        from ultralytics.cfg import get_cfg
        for preset in PRESETS:
            cfg = get_cfg(overrides=build_hyp(preset, mirror=True, flip_ud=True))
            self.assertEqual(cfg.fliplr, 0.5)

    def test_describe(self):
        self.assertEqual(describe("light", False, False), "Augment: Light, no mirror")
        self.assertEqual(describe("strong", True, True), "Augment: Strong, mirror+upside-down")


if __name__ == "__main__":
    unittest.main()
