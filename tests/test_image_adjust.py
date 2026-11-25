"""Camera image adjustments: each step on a synthetic frame, the settings
round trip, the worker hook and the Settings > Camera > Image card."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import app.settings as settings_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.image_adjust import ImageAdjust  # noqa: E402
from app.tabs.settings_tab import SettingsTab  # noqa: E402

APP = QApplication.instance() or QApplication([])


def frame():
    """4x6 BGR ramp with a coloured corner, so flips and colour are visible."""
    f = np.zeros((4, 6, 3), np.uint8)
    f[..., :] = np.arange(6, dtype=np.uint8)[None, :, None] * 40
    f[0, 0] = (0, 0, 200)
    return f


class Apply(unittest.TestCase):
    def test_neutral_is_identity(self):
        self.assertTrue(ImageAdjust().is_neutral)
        np.testing.assert_array_equal(ImageAdjust().apply(frame()), frame())

    def test_mirror_and_flip(self):
        f = frame()
        np.testing.assert_array_equal(ImageAdjust(mirror=True).apply(f), f[:, ::-1])
        np.testing.assert_array_equal(ImageAdjust(flip=True).apply(f), f[::-1])
        np.testing.assert_array_equal(ImageAdjust(mirror=True, flip=True).apply(f), f[::-1, ::-1])
        np.testing.assert_array_equal(f, frame())  # input untouched

    def test_brightness_contrast_gamma_are_monotone(self):
        f = frame()
        self.assertGreater(ImageAdjust(brightness=30).apply(f).mean(), f.mean())
        self.assertLess(ImageAdjust(brightness=-30).apply(f).mean(), f.mean())
        self.assertGreater(ImageAdjust(contrast=150).apply(f).std(), f.std())
        self.assertGreater(ImageAdjust(gamma=1.5).apply(f).mean(), f.mean())
        self.assertLess(ImageAdjust(gamma=0.7).apply(f).mean(), f.mean())

    def test_saturation(self):
        gray = ImageAdjust(saturation=0).apply(frame())
        b, g, r = gray[0, 0]
        self.assertLessEqual(max(b, g, r) - min(b, g, r), 2)
        self.assertEqual(ImageAdjust(clahe=True).apply(frame()).shape, frame().shape)


class SettingsAndWorker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = Path(self.tmp.name) / "app_settings.json"
        self.patch = mock.patch.object(settings_mod, "APP_SETTINGS_FILE", self.file)
        self.patch.start()
        self.ctx = AppContext()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_settings_round_trip(self):
        adjust = ImageAdjust(mirror=True, brightness=20, gamma=1.25)
        self.ctx.settings.set(**adjust.to_settings())
        self.assertEqual(ImageAdjust.from_settings(self.ctx.settings), adjust)
        self.assertEqual(self.ctx.worker._adjust, adjust)  # live, before Save
        self.ctx.settings.commit()
        self.assertTrue(json.loads(self.file.read_text())["camera_mirror"])

    def test_image_card(self):
        tab = SettingsTab(self.ctx)
        card = tab.image_card
        card.mirror_button.click()
        card.sliders["gamma"].setValue(150)
        self.assertEqual(self.ctx.worker._adjust, ImageAdjust(mirror=True, gamma=1.5))
        self.assertEqual(sorted(tab.unsaved_changes()), ["camera_gamma", "camera_mirror"])
        tab.revert()
        self.assertFalse(card.mirror_button.isChecked())
        self.assertEqual(card.sliders["gamma"].value(), 100)
        self.assertTrue(self.ctx.worker._adjust.is_neutral)
        card.sliders["brightness"].setValue(40)
        card.reset_button.click()
        self.assertEqual(card.sliders["brightness"].value(), 0)
        tab.deleteLater()


if __name__ == "__main__":
    unittest.main()
