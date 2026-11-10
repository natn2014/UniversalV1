"""Gauge Reader: the angle maths and its guards, limits, the steadying hold, the
capture rule, the job round trip, the NG log tag, and (when the model and
Ultralytics are here) the real model on real frames."""

import json
import math
import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import app.gauge_log as gauge_log_mod  # noqa: E402
import app.jobs as jobs_mod  # noqa: E402
import app.ng_log as ng_log_mod  # noqa: E402
import app.paths as paths_mod  # noqa: E402
import app.settings as settings_mod  # noqa: E402
import app.tabs.settings_tab as settings_tab_mod  # noqa: E402
from app.gauge import (  # noqa: E402
    CLASS_CENTER,
    CLASS_END,
    CLASS_START,
    CLASS_TIP,
    GAUGE_INFERENCE,
    NO_READ,
    OK,
    OVER,
    UNDER,
    GaugeCaptureGate,
    GaugeHold,
    GaugeReading,
    GaugeSpec,
    PointSmoother,
    gauge_state,
    labelled_points,
    needle_ratio,
    out_of_spec_text,
    read_gauge,
)
from app.judgement import DetectionFilters  # noqa: E402
from app.optional_deps import YOLO  # noqa: E402

APP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parent.parent
SCALE = GaugeSpec(value_at_start=0.0, value_at_end=100.0, unit="%", spec_min=20.0, spec_max=80.0)
CENTER = (500.0, 500.0)
RADIUS = 200.0
NEEDLE = 130.0  # short enough to stay between Start and End all round the dial
# A 270 degree dial, Start lower left and End lower right; screen angles (y down).
START_ANGLE, SWEEP = 135.0, 270.0


def on_dial(angle: float, radius: float = RADIUS) -> tuple:
    a = math.radians(angle)
    return CENTER[0] + radius * math.cos(a), CENTER[1] + radius * math.sin(a)


def det(cls_id: int, point: tuple, conf: float = 0.9) -> dict:
    x, y = point
    return {"x1": x - 5, "y1": y - 5, "x2": x + 5, "y2": y + 5, "cls_id": cls_id, "conf": conf, "class_name": str(cls_id)}


def dial(percent: float = None, tip_angle: float = None) -> list:
    if tip_angle is None:
        tip_angle = START_ANGLE + SWEEP * percent / 100
    return [det(CLASS_CENTER, CENTER), det(CLASS_START, on_dial(START_ANGLE)),
            det(CLASS_END, on_dial(START_ANGLE + SWEEP)), det(CLASS_TIP, on_dial(tip_angle, NEEDLE))]


def reading(side: str, percent: float = 50.0) -> GaugeReading:
    return GaugeReading(side, percent, percent, {})


class Geometry(unittest.TestCase):
    def test_needle_positions(self):
        for percent in (0, 25, 50, 75, 100):
            self.assertAlmostEqual(read_gauge(dial(percent), SCALE).value, percent, places=2)

    def test_matches_the_original_mpa_conversion(self):
        spec = GaugeSpec()  # 0.1 at Start, 0 at End
        for percent in (0, 37, 50, 100):
            original = max(0, min((1 - percent / 100) / 10, 1))
            self.assertAlmostEqual(read_gauge(dial(percent), spec).value, original, places=4)

    def test_measured_from_end_0_to_start_full_scale(self):
        spec = GaugeSpec()
        for percent_from_start, value, percent in ((100, 0.0, 0.0), (50, 0.05, 50.0), (0, 0.1, 100.0)):
            result = read_gauge(dial(percent_from_start), spec)
            self.assertAlmostEqual(result.value, value, places=4)
            self.assertAlmostEqual(result.percent, percent, places=2)

    def test_needle_just_past_start_pins_to_full_scale(self):
        self.assertEqual(read_gauge(dial(tip_angle=START_ANGLE - 5), SCALE).percent, 100.0)

    def test_needle_just_past_end_is_low_not_full_scale(self):
        self.assertEqual(read_gauge(dial(tip_angle=START_ANGLE + SWEEP + 5), SCALE).percent, 0.0)

    def test_dead_zone_splits_at_its_middle(self):
        middle = START_ANGLE + SWEEP + 45
        self.assertEqual(read_gauge(dial(tip_angle=middle - 1), SCALE).percent, 0.0)
        self.assertEqual(read_gauge(dial(tip_angle=middle + 1), SCALE).percent, 100.0)

    def test_every_point_is_required_center_too(self):
        for missing in (CLASS_CENTER, CLASS_START, CLASS_END, CLASS_TIP):
            dets = [d for d in dial(50) if d["cls_id"] != missing]
            self.assertEqual(read_gauge(dets, SCALE).side, NO_READ, missing)

    def test_most_confident_point_of_a_class_wins(self):
        dets = dial(50) + [det(CLASS_TIP, on_dial(START_ANGLE), conf=0.2)]
        self.assertAlmostEqual(read_gauge(dets, SCALE).value, 50.0, places=2)

    def test_start_end_swapped_by_the_model_is_corrected(self):
        dets = dial(25)
        dets[1]["cls_id"], dets[2]["cls_id"] = CLASS_END, CLASS_START
        self.assertAlmostEqual(read_gauge(dets, SCALE).value, 25.0, places=2)

    def test_span_guard_scales_with_the_image(self):
        # Start and End on nearly the same mark: not a gauge, at any size.
        dets = dial(50)
        dets[2] = det(CLASS_END, on_dial(START_ANGLE + 2))
        self.assertEqual(read_gauge(dets, SCALE).side, NO_READ)
        # A small dial (the old fixed 50 px limit would have refused it) reads.
        small = [det(c["cls_id"], (500 + (c["x1"] + 5 - 500) * 0.15, 500 + (c["y1"] + 5 - 500) * 0.15))
                 for c in dial(50)]
        self.assertAlmostEqual(read_gauge(small, SCALE).value, 50.0, places=1)

    def test_tip_on_the_center_is_no_reading(self):
        dets = dial(50)
        dets[3] = det(CLASS_TIP, CENTER)
        self.assertEqual(read_gauge(dets, SCALE).side, NO_READ)

    def test_no_sweep(self):
        self.assertIsNone(needle_ratio((0, 0), (1, 1), (2, 2), (0, 5)))


class Position(unittest.TestCase):
    def test_labels_swapped_by_the_model_still_read(self):
        dets = dial(25)
        dets[1]["cls_id"], dets[2]["cls_id"] = CLASS_END, CLASS_START
        self.assertAlmostEqual(read_gauge(dets, SCALE).value, 25.0, places=2)

    def test_both_ends_labelled_the_same_still_read(self):
        dets = dial(25)
        dets[2]["cls_id"] = CLASS_START
        self.assertAlmostEqual(read_gauge(dets, SCALE).value, 25.0, places=2)

    def test_center_left_of_start_is_refused(self):
        dets = dial(50)
        dets[0] = det(CLASS_CENTER, (250.0, 500.0))
        result = read_gauge(dets, SCALE)
        self.assertEqual(result.side, NO_READ)
        self.assertTrue(result.problem)

    def test_tip_off_the_dial_is_refused(self):
        dets = dial(50)
        dets[3] = det(CLASS_TIP, (900.0, 450.0))  # twice the radius out
        self.assertTrue(read_gauge(dets, SCALE).problem)

    def test_long_pointer_past_start_reads(self):
        # The station's vacuum gauge: near -0.09 its pointer reaches left of the Start tick.
        tip_angle = START_ANGLE + SWEEP * 0.2
        dets = dial(tip_angle=tip_angle)
        dets[3] = det(CLASS_TIP, on_dial(tip_angle, RADIUS * 0.97))
        self.assertLess(dets[3]["x1"] + 5, on_dial(START_ANGLE)[0])
        self.assertAlmostEqual(read_gauge(dets, SCALE).value, 20.0, places=1)

    def test_a_surer_box_in_the_wrong_place_loses(self):
        dets = dial(50, )
        dets[3]["conf"] = 0.3
        dets.append(det(CLASS_TIP, (900.0, 450.0), conf=0.95))
        self.assertAlmostEqual(read_gauge(dets, SCALE).value, 50.0, places=2)

    def test_most_confident_valid_set_wins(self):
        dets = dial(50)
        dets.append(det(CLASS_TIP, on_dial(START_ANGLE + SWEEP * 0.75, NEEDLE), conf=0.5))
        self.assertAlmostEqual(read_gauge(dets, SCALE).value, 50.0, places=2)

    def test_tip_past_end_reads(self):
        dets = dial(50)
        end_x = on_dial(START_ANGLE + SWEEP)[0]
        dets[3] = det(CLASS_TIP, (end_x + 20, 420.0))
        self.assertTrue(read_gauge(dets, SCALE).read)

    def test_position_replaces_the_class(self):
        dets = dial(25)
        dets[1]["cls_id"], dets[2]["cls_id"] = CLASS_END, CLASS_START  # "End" on the left
        points = read_gauge(dets, SCALE).points
        self.assertLess(points["start"][0], CENTER[0])
        self.assertGreater(points["end"][0], CENTER[0])

    def test_both_ends_on_one_side_of_center_are_refused(self):
        for start_point, end_point in (((300.0, 700.0), (400.0, 700.0)),  # both left
                                       ((600.0, 700.0), (700.0, 700.0)),  # both right
                                       ((480.0, 700.0), (520.0, 700.0))):  # only just either side
            dets = dial(50)
            dets[1], dets[2] = det(CLASS_START, start_point), det(CLASS_END, end_point)
            result = read_gauge(dets, SCALE)
            self.assertEqual(result.side, NO_READ, start_point)
            self.assertTrue(result.problem, start_point)

    def test_one_mark_seen_as_both_leaves_no_second_end(self):
        dets = dial(50)
        dets[2] = det(CLASS_END, on_dial(START_ANGLE))
        self.assertEqual(read_gauge(dets, SCALE).side, NO_READ)

    def test_repeats_of_one_mark_dont_crowd_out_the_other_end(self):
        x, y = on_dial(START_ANGLE)
        dets = dial(50)
        dets[2]["conf"] = 0.2
        dets += [det(CLASS_END, (x + i, y), conf=0.95) for i in range(1, 5)]
        self.assertAlmostEqual(read_gauge(dets, SCALE).value, 50.0, delta=0.2)  # Start is up to 1 px off

    def test_labels_for_the_view_go_by_side_of_center(self):
        dets = dial(50)
        dets[1]["cls_id"], dets[2]["cls_id"] = CLASS_END, CLASS_START
        names = dict(labelled_points(dets))
        self.assertLess(names["Start"][0], CENTER[0])
        self.assertGreater(names["End"][0], CENTER[0])

    def test_one_mark_seen_as_both_is_labelled_once(self):
        point = on_dial(START_ANGLE)
        dets = [det(CLASS_CENTER, CENTER), det(CLASS_START, point), det(CLASS_END, point, conf=0.5)]
        self.assertEqual([name for name, _ in labelled_points(dets)], ["Center", "Start"])

    def test_view_shows_one_of_each_point(self):
        x, y = CENTER
        dets = dial(50) + [det(CLASS_CENTER, (x + 60, y + 40), conf=0.5), det(CLASS_CENTER, (x - 70, y), conf=0.4),
                           det(CLASS_TIP, (x + 20, y + 150), conf=0.3),
                           det(CLASS_START, (x - 120, y + 190), conf=0.6), det(CLASS_END, (x + 150, y + 100), conf=0.5)]
        named = labelled_points(dets)
        self.assertEqual(sorted(name for name, _ in named), ["Center", "End", "Start", "Tip"])
        points = dict(named)
        self.assertEqual(points["Center"], CENTER)  # the surest one
        self.assertEqual(points["Start"], on_dial(START_ANGLE))
        self.assertEqual(points["End"], on_dial(START_ANGLE + SWEEP))

    def test_without_a_center_the_side_is_unknown(self):
        dets = [det(CLASS_END, on_dial(START_ANGLE))]
        self.assertEqual([name for name, _ in labelled_points(dets)], ["Start/End"])

    def test_missing_point_is_not_a_position_problem(self):
        self.assertFalse(read_gauge(dial(50)[:3], SCALE).problem)


class Smoothing(unittest.TestCase):
    W = 1920.0

    def test_small_shifts_are_steadied(self):
        s = PointSmoother()
        s.update([("Center", (100.0, 100.0))], 0.0, 1.0, self.W)
        out = dict(s.update([("Center", (110.0, 100.0))], 0.1, 1.0, self.W))
        self.assertEqual(out["Center"], (105.0, 100.0))  # half way

    def test_a_jump_is_taken_at_once(self):
        s = PointSmoother()
        s.update([("Center", (100.0, 100.0))], 0.0, 1.0, self.W)
        out = dict(s.update([("Center", (400.0, 100.0))], 0.1, 1.0, self.W))  # the next gauge
        self.assertEqual(out["Center"], (400.0, 100.0))

    def test_a_missed_point_stays_for_the_hold_time(self):
        s = PointSmoother()
        s.update([("Center", (100.0, 100.0)), ("Tip", (50.0, 60.0))], 0.0, 1.0, self.W)
        self.assertIn("Tip", dict(s.update([("Center", (100.0, 100.0))], 0.5, 1.0, self.W)))
        self.assertNotIn("Tip", dict(s.update([("Center", (100.0, 100.0))], 1.6, 1.0, self.W)))

    def test_a_name_twice_is_drawn_as_it_is(self):
        s = PointSmoother()
        named = [("Start/End", (10.0, 10.0)), ("Start/End", (90.0, 10.0))]
        self.assertEqual(s.update(named, 0.0, 1.0, self.W), named)


class Limits(unittest.TestCase):
    def test_sides_and_states(self):
        for percent, side, state in ((50, OK, "GREEN"), (85, OVER, "RED"), (10, UNDER, "RED")):
            result = read_gauge(dial(percent), SCALE)
            self.assertEqual(result.side, side)
            self.assertEqual(gauge_state(result), state)
        self.assertEqual(gauge_state(GaugeReading()), "YELLOW")

    def test_limits_are_inclusive(self):
        self.assertEqual(read_gauge(dial(20), SCALE).side, OK)
        self.assertEqual(read_gauge(dial(80), SCALE).side, OK)

    def test_reversed_scale_judges_the_value_not_the_needle(self):
        # Vacuum dial: the value falls as the needle climbs.
        spec = GaugeSpec(value_at_start=0.1, value_at_end=0.0, unit="MPa", spec_min=0.04, spec_max=0.07)
        self.assertEqual(read_gauge(dial(10), spec).side, OVER)  # 0.09 MPa
        self.assertEqual(read_gauge(dial(90), spec).side, UNDER)  # 0.01 MPa
        self.assertEqual(read_gauge(dial(50), spec).side, OK)

    def test_invalid_spec_gives_no_reading(self):
        for spec in (GaugeSpec(spec_min=5, spec_max=5), GaugeSpec(spec_min=6, spec_max=5),
                     GaugeSpec(value_at_start=1, value_at_end=1)):
            self.assertFalse(spec.valid)
            self.assertEqual(read_gauge(dial(50), spec).side, NO_READ)

    def test_messages(self):
        self.assertEqual(out_of_spec_text(read_gauge(dial(85), SCALE), SCALE), "OVER: 85.0000 % > MAX 80.0000")
        self.assertEqual(out_of_spec_text(read_gauge(dial(10), SCALE), SCALE), "UNDER: 10.0000 % < MIN 20.0000")
        self.assertEqual(out_of_spec_text(read_gauge(dial(50), SCALE), SCALE), "")

    def test_filters_judge_and_keep_every_point(self):
        filters = DetectionFilters(judge_mode="gauge", gauge=SCALE, confidence_threshold=0.95, max_detections=1)
        filtered = filters.apply(dial(50))  # no selected classes, conf too low, capped: all still count
        self.assertEqual(len(filtered), 4)
        self.assertEqual(filters.judge(filtered), "GREEN")
        self.assertEqual(filters.judge([]), "YELLOW")


class Hold(unittest.TestCase):
    def feed(self, hold, percents, start=0.0):
        out = None
        for i, p in enumerate(percents):
            out = hold.update(reading(OK, p), start + i * 0.05, 1.0)
        return out

    def test_waits_for_a_few_readings(self):
        hold = GaugeHold()
        self.assertFalse(hold.update(reading(OK), 0.0, 1.0).read)
        self.assertFalse(hold.update(reading(OK), 0.1, 1.0).read)
        self.assertTrue(hold.update(reading(OK), 0.2, 1.0).read)

    def test_one_bad_frame_cannot_flip_it(self):
        hold = GaugeHold()
        self.feed(hold, [50, 50, 50, 50])
        result = hold.update(reading(OVER, 99), 0.3, 1.0)
        self.assertEqual(result.side, OK)
        self.assertEqual(result.percent, 50)

    def test_a_real_change_comes_through(self):
        hold = GaugeHold()
        self.feed(hold, [50, 50, 50])
        for i in range(3):
            result = hold.update(reading(OVER, 90), 0.2 + i * 0.05, 1.0)
        self.assertEqual(result.side, OVER)

    def test_points_lost_briefly_keeps_the_reading(self):
        hold = GaugeHold()
        self.feed(hold, [50, 50, 50])
        self.assertTrue(hold.update(GaugeReading(), 0.5, 1.0).read)

    def test_lost_for_longer_starts_over(self):
        hold = GaugeHold()
        self.feed(hold, [50, 50, 50])
        self.assertFalse(hold.update(GaugeReading(), 2.0, 1.0).read)
        self.assertFalse(hold.update(reading(OK), 2.1, 1.0).read)  # history forgotten

    def test_hold_off(self):
        hold = GaugeHold()
        self.feed(hold, [50, 50, 50])
        self.assertFalse(hold.update(GaugeReading(), 0.2, 0.0).read)


class CaptureGate(unittest.TestCase):
    def test_counted_ng_is_the_normal_save(self):
        gate = GaugeCaptureGate()
        self.assertFalse(gate.update(OVER, True, True))

    def test_staying_out_of_spec_saves_nothing_more(self):
        gate = GaugeCaptureGate()
        gate.update(OVER, True, True)
        for _ in range(100):
            self.assertFalse(gate.update(OVER, True, False))

    def test_swing_saves_once_each_way(self):
        gate = GaugeCaptureGate()
        gate.update(OVER, True, True)
        self.assertTrue(gate.update(UNDER, True, False))
        self.assertFalse(gate.update(UNDER, True, False))
        self.assertTrue(gate.update(OVER, True, False))

    def test_lost_reading_does_not_end_an_excursion(self):
        gate = GaugeCaptureGate()
        gate.update(OVER, True, True)
        self.assertFalse(gate.update(NO_READ, True, False))
        self.assertFalse(gate.update(OVER, True, False))

    def test_back_in_spec_ends_it(self):
        gate = GaugeCaptureGate()
        gate.update(OVER, True, True)
        gate.update(OK, False, False)
        self.assertFalse(gate.update(OVER, False, False))  # waits for the judgement
        self.assertFalse(gate.update(OVER, True, True))

    def test_out_of_spec_under_an_ng_already_showing(self):
        gate = GaugeCaptureGate()  # e.g. a tool's NG was counted first
        self.assertTrue(gate.update(OVER, True, False))
        self.assertFalse(gate.update(OVER, True, False))


class Spec(unittest.TestCase):
    def test_round_trip(self):
        spec = GaugeSpec(1.0, 2.0, "bar", 1.2, 1.8)
        self.assertEqual(GaugeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))), spec)

    def test_older_job_and_damaged_values(self):
        self.assertEqual(GaugeSpec.from_dict(None), GaugeSpec())
        self.assertEqual(GaugeSpec.from_dict({"spec_min": "x", "other": 1}), GaugeSpec())

    def test_inspection_type(self):
        self.assertEqual(jobs_mod.inspection_types({"model_path": "a.pt", "judge_mode": "gauge"}), ["gauge"])


class NgLogTag(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(ng_log_mod, "NG_LOGS_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        mock.patch.object(ng_log_mod, "_total_bytes", None).start()
        self.addCleanup(mock.patch.stopall)
        image = QImage(64, 48, QImage.Format_RGB888)
        image.fill(0)
        self.pixmap = QPixmap.fromImage(image)

    def test_tag_in_name_and_listing_and_full_image(self):
        ng_log_mod.save_ng_image(self.pixmap, "P1", tag="OVER")
        name, thumbs, count = ng_log_mod.collect_ng_log()[0]
        self.assertEqual((name, count), ("P1", 1))
        self.assertTrue(thumbs[0].name.endswith("_OVER_thumb.jpg"))
        self.assertTrue(ng_log_mod.full_image_for_thumb(thumbs[0]).exists())
        self.assertTrue(ng_log_mod.ng_stamp_text(thumbs[0]).endswith("  OVER"))

    def test_untagged_stays_as_it_was(self):
        ng_log_mod.save_ng_image(self.pixmap, "P1")
        thumb = ng_log_mod.collect_ng_log()[0][1][0]
        self.assertRegex(ng_log_mod.ng_stamp_text(thumb), r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

    def test_tagged_images_are_pruned_like_the_rest(self):
        ng_log_mod.save_ng_image(self.pixmap, "P1", tag="UNDER")
        self.assertEqual(ng_log_mod.prune_ng_log(0)[0], 1)
        self.assertEqual(ng_log_mod.collect_ng_log(), [])


class App(unittest.TestCase):
    """The whole window: job round trip, verdict, counter and tagged image."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", root / "app_settings.json"),
            mock.patch.object(paths_mod, "MARK_MODEL_PATH", root / "missing.pt"),
            mock.patch.object(paths_mod, "GAUGE_MODEL_PATH", root / "missing_gauge.pt"),
            mock.patch.object(jobs_mod, "PART_CONFIGS_DIR", root / "jobs"),
            mock.patch.object(jobs_mod, "LAST_PART_FILE", root / "jobs" / ".last_part"),
            mock.patch.object(ng_log_mod, "NG_LOGS_DIR", root / "ng"),
            mock.patch.object(paths_mod, "GAUGE_LOGS_DIR", root / "gauge"),
            mock.patch.object(settings_tab_mod, "find_cameras", return_value=[]),
            mock.patch("app.tower.TowerController.connect_device"),
            mock.patch("app.main_window.MainWindow.confirm_unsaved", return_value=True),
            mock.patch("app.tabs.monitor_tab.MonitorTab.ensure_live", return_value=False),
        ]
        for p in cls.patches:
            p.start()
        from app.main_window import MainWindow
        cls.win = MainWindow()
        cls.settings = cls.win.settings_tab
        cls.monitor = cls.win.monitor_tab
        cls.ctx = cls.win.ctx

    @classmethod
    def tearDownClass(cls):
        cls.ctx.relay.shutdown()
        cls.win.deleteLater()
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def setUp(self):
        self.settings.set_gauge(SCALE.to_dict())
        self.settings.set_judge_mode("gauge")
        self.ctx.settings.set(state_hold_ms=0, mark_hold_ms=0, ok_count=0, ng_count=0)
        self.monitor._reset_judgement()
        self.monitor._preview = False
        self.ctx.gauge_log.set_part(None)
        self.image = QImage(1000, 1000, QImage.Format_RGB888)
        self.image.fill(0)

    def frames(self, percent, count=5):
        for _ in range(count):
            self.monitor.on_frame(self.image, dial(percent), [])

    def ng_images(self):
        groups = ng_log_mod.collect_ng_log()
        return [t.name for t in groups[0][1]] if groups else []

    def clear_ng(self):
        ng_log_mod.prune_ng_log(0)

    def test_inference_args_follow_the_mode(self):
        self.assertEqual(self.ctx.worker._inference_args, GAUGE_INFERENCE)
        self.settings.set_judge_mode("detection")
        self.assertEqual(self.ctx.worker._inference_args, {})

    def test_job_round_trip_and_old_job(self):
        spec = {"value_at_start": 0.1, "value_at_end": 0.0, "unit": "MPa", "spec_min": 0.04, "spec_max": 0.07}
        self.settings.apply_config({"judge_mode": "gauge", "gauge": spec})
        self.assertEqual(self.settings.job_settings()["gauge"], spec)
        self.assertTrue(self.ctx.filters.by_gauge)
        self.settings.apply_config({"judge_mode": "detection"})
        self.assertEqual(self.ctx.filters.gauge, GaugeSpec())
        self.assertFalse(self.ctx.filters.by_gauge)

    def test_save_refused_when_limits_are_wrong(self):
        self.ctx.set_current_part("G1")
        jobs_mod.save_part_config("G1", self.settings.collect_config())
        before = jobs_mod.load_part_config("G1")["gauge"]
        self.settings.gauge_min_spin.setValue(90)  # above MAX
        self.settings.save()
        self.assertEqual(jobs_mod.load_part_config("G1")["gauge"], before)
        self.settings.gauge_min_spin.setValue(30)
        self.settings.save()
        self.assertEqual(jobs_mod.load_part_config("G1")["gauge"]["spec_min"], 30)
        self.ctx.set_current_part(None)

    def test_in_spec_is_ok_with_no_image(self):
        self.clear_ng()
        self.frames(50)
        self.assertEqual(self.ctx.settings["ok_count"], 1)
        self.assertEqual(self.ctx.settings["ng_count"], 0)
        self.assertEqual(self.ng_images(), [])
        self.assertEqual(self.monitor.gauge_table.item(0, 3).text(), "OK")

    def test_over_is_ng_with_one_tagged_image(self):
        self.clear_ng()
        self.frames(85, 10)
        self.assertEqual(self.ctx.settings["ng_count"], 1)
        images = self.ng_images()
        self.assertEqual(len(images), 1)
        self.assertIn("_OVER", images[0])
        self.assertEqual(self.monitor.gauge_table.item(0, 3).text(), "OVER")

    def test_under_then_swing_saves_a_second_tagged_image(self):
        self.clear_ng()
        self.frames(10, 5)
        self.assertIn("_UNDER", self.ng_images()[0])
        self.frames(90, 5)
        images = self.ng_images()
        self.assertEqual(len(images), 2)
        self.assertEqual(sum("_OVER" in i for i in images), 1)
        self.assertEqual(self.ctx.settings["ng_count"], 1)

    def test_readings_are_recorded_but_not_in_preview(self):
        path = gauge_log_mod.daily_log_path(None, datetime.now())
        def recorded():
            return sum(len(v) for v in gauge_log_mod.read_day(path).values())
        before = recorded()
        self.frames(50, 5)
        self.assertEqual(recorded(), before + 1)  # 1 a second, not 1 a frame
        self.ctx.gauge_log._last_at = float("-inf")
        self.monitor._reset_judgement()
        self.monitor._preview = True
        self.frames(50, 5)
        self.assertEqual(recorded(), before + 1)

    def paint_overlay(self, detections):
        from PySide6.QtGui import QPainter
        canvas = QImage(1000, 1000, QImage.Format_RGB32)
        canvas.fill(0)
        painter = QPainter(canvas)
        self.monitor._paint_overlay(painter, canvas, detections, [])
        painter.end()
        return lambda x, y: canvas.pixelColor(x, y).name()

    def test_overlay_is_points_and_a_center_to_tip_line(self):
        from app.tabs.monitor_tab import GAUGE_POINT_COLOR
        from app.theme import OK_GREEN
        self.frames(50, 5)  # needle straight up
        pixel = self.paint_overlay(dial(50))
        self.assertEqual(pixel(500, 500), GAUGE_POINT_COLOR)  # Center
        x, y = on_dial(START_ANGLE)
        self.assertEqual(pixel(round(x), round(y)), GAUGE_POINT_COLOR)  # Start
        self.assertEqual(pixel(500, 370), OK_GREEN.lower())  # Tip
        self.assertEqual(pixel(500, 440), OK_GREEN.lower())  # the line between them
        self.assertEqual(pixel(430, 570), "#000000")  # no line from Center to Start
        self.assertEqual(pixel(495 - 12, 495 - 12), "#000000")  # no box

    def test_without_a_reading_the_detections_still_show_as_points(self):
        from app.tabs.monitor_tab import GAUGE_POINT_COLOR
        self.monitor._reset_judgement()
        self.monitor.on_frame(self.image, dial(50), [])  # one frame: not read yet
        pixel = self.paint_overlay(dial(50))
        self.assertEqual(pixel(500, 370), GAUGE_POINT_COLOR)
        self.assertEqual(pixel(500, 440), "#000000")  # no needle until it is read

    def test_one_mark_seen_as_both_shows_one_label(self):
        self.monitor._reset_judgement()
        point = on_dial(START_ANGLE)
        dets = [det(CLASS_CENTER, CENTER), det(CLASS_START, point), det(CLASS_END, point, conf=0.5)]
        self.monitor.on_frame(self.image, dets, [])
        with mock.patch("app.tabs.monitor_tab.draw_label") as label:
            self.paint_overlay(dets)
        self.assertEqual([call.args[3] for call in label.call_args_list], ["Center", "Start"])

    def test_fourth_day_is_refused_with_a_message(self):
        tab = self.win.gauge_data_tab
        days = [date(2026, 11, d) for d in (9, 8, 7, 6)]
        tab._days = days
        tab._selection.slots = days[:3]
        tab.day_list.clear()
        tab.day_list.addItems([d.isoformat() for d in days])
        self.addCleanup(tab.day_list.clear)
        self.addCleanup(tab._days.clear)
        messages = []
        self.ctx.status.connect(messages.append)
        self.addCleanup(self.ctx.status.disconnect, messages.append)
        tab._on_day_clicked(tab.day_list.item(3))
        self.assertEqual(messages, ["Up to 3 days at once: tap one to remove it first."])
        self.assertEqual(tab._selection.slots, days[:3])
        tab._selection.reset(date.today())

    def test_part_and_model_smaller_on_a_small_screen(self):
        # The style is only applied to a shown window; which one is chosen is what matters.
        for (width, height), name in (((1024, 768), "monitorPartModelCompact"), ((1920, 1080), "monitorPartModel")):
            self.win.resize(width, height)
            self.monitor._fit_part_model()
            self.assertEqual(self.monitor.part_label.objectName(), name)
            self.assertEqual(self.monitor.model_label.objectName(), name)
        self.win.resize(1024, 768)

    def test_gauge_data_tab_only_for_gauge_jobs(self):
        index = self.win.tabs.indexOf(self.win.gauge_data_tab)
        self.assertTrue(self.win.tabs.isTabVisible(index))
        self.settings.set_judge_mode("required")
        self.assertFalse(self.win.tabs.isTabVisible(index))
        self.assertFalse(self.ctx.worker._qr_scan)
        self.settings.set_judge_mode("gauge")
        self.assertTrue(self.ctx.worker._qr_scan)

    def test_all_tabs_fit_a_1024_panel_in_both_languages(self):
        self.win.resize(1024, 768)
        bar = self.win.tabs.tabBar()
        room = 1024 - self.win.tabs.cornerWidget(Qt.TopRightCorner).sizeHint().width() - 24
        for language in ("en", "th"):
            self.win.set_language(language)
            APP.processEvents()
            self.assertLessEqual(bar.sizeHint().width(), room, language)
        self.win.set_language("en")

    def test_switching_language_keeps_the_reading_and_saves_nothing_extra(self):
        self.clear_ng()
        self.frames(85, 5)
        self.win.set_language("th")
        self.monitor._show_judge_mode()
        self.assertTrue(self.monitor._gauge_reading.read)
        self.frames(85, 5)
        self.win.set_language("en")
        self.assertEqual(len(self.ng_images()), 1)

    def test_out_of_position_shows_a_note_not_ng(self):
        dets = dial(50)
        dets[0] = det(CLASS_CENTER, (250.0, 500.0))
        for _ in range(5):
            self.monitor.on_frame(self.image, dets, [])
        self.assertEqual(self.ctx.settings["ng_count"], 0)
        self.assertIn("position", self.monitor.layout_label.text())

    def test_no_reading_is_standby_with_nothing_saved(self):
        self.clear_ng()
        for _ in range(5):
            self.monitor.on_frame(self.image, [], [])
        self.assertEqual(self.ctx.settings["ng_count"], 0)
        self.assertEqual(self.ctx.settings["ok_count"], 0)
        self.assertEqual(self.ng_images(), [])

    def test_preview_saves_nothing(self):
        self.clear_ng()
        self.monitor._preview = True
        self.frames(85, 5)
        self.assertEqual(self.ctx.settings["ng_count"], 0)
        self.assertEqual(self.ng_images(), [])


class DataRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(paths_mod, "GAUGE_LOGS_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = gauge_log_mod.GaugeDataLog()
        self.log.set_part("P1")

    def today_values(self, box):
        path = gauge_log_mod.daily_log_path(self.log.part, datetime.now())
        return gauge_log_mod.read_day(path).get(box, [])

    def test_box_label_matches_the_original(self):
        self.assertEqual(gauge_log_mod.next_box_label("Box No.38"), "Box No.39")
        self.assertEqual(gauge_log_mod.next_box_label("Box No.43"), "Box No.1")
        self.assertEqual(gauge_log_mod.next_box_label("box no.5"), "Box No.6")  # always the chart's spelling
        self.assertEqual(gauge_log_mod.next_box_label("OTHER"), "OTHER")
        self.assertEqual(len(gauge_log_mod.box_labels()), 43)

    def test_csv_same_columns_as_the_original(self):
        when = datetime(2026, 11, 9, 12, 0, 1, 5)
        self.log.record(0.0777, 0.0, when)
        path = Path(self.tmp.name) / "P1" / "2026_11_09_log.csv"
        self.assertEqual(path.read_text().splitlines(),
                         ["Timestamp,Converted Value,QR Code Data", "2026-11-09 12:00:01.000005,0.0777,N/A"])

    def test_one_reading_a_second(self):
        self.assertTrue(self.log.record(1.0, 10.0))
        self.assertFalse(self.log.record(2.0, 10.5))
        self.assertTrue(self.log.record(3.0, 11.0))
        self.assertEqual(self.today_values("N/A"), [1.0, 3.0])

    def test_qr_ignored_until_the_gauge_is_read(self):
        self.log.set_qr("Box No.7")
        self.assertEqual(self.log.qr, "N/A")
        self.log.record(1.0, 0.0)
        self.log.set_qr("Box No.7")
        self.assertEqual(self.log.qr, "Box No.8")
        self.log.record(1.0, 2.0)
        self.assertEqual(self.today_values("Box No.8"), [1.0])

    def test_every_reading_of_the_day_counts(self):
        for i in range(1005):
            self.log.record(float(i), float(i) * 2)
        cache = gauge_log_mod.DayStatsCache()
        stats = cache.get("P1", datetime.now().date())
        self.assertEqual(stats["N/A"]["count"], 1005)
        self.assertEqual(stats["N/A"]["med"], 502.0)

    def test_damaged_line_is_skipped(self):
        self.log.record(1.5, 0.0)
        path = gauge_log_mod.daily_log_path("P1", datetime.now())
        with open(path, "ab") as f:
            f.write(b"2026-01-01 00:00:00,not-a-num\n\xff\xfe,1\n2026-01-01 00:00:0")  # power lost mid-line
        self.assertEqual(gauge_log_mod.read_day(path), {"N/A": [1.5]})

    def test_jobs_are_separate(self):
        self.log.record(1.0, 0.0)
        self.log.set_part("P2")
        self.assertEqual(self.today_values("N/A"), [])

    def test_days_newest_first_and_other_files_ignored(self):
        self.log.record(1.0, 0.0, datetime(2026, 11, 7, 8))
        self.log.record(1.0, 2.0, datetime(2026, 11, 9, 8))
        self.log.record(1.0, 4.0, datetime(2026, 11, 8, 8))
        (Path(self.tmp.name) / "P1" / "notes_log.csv").write_text("x")
        self.assertEqual(gauge_log_mod.available_days("P1"), [date(2026, 11, 9), date(2026, 11, 8), date(2026, 11, 7)])
        self.assertEqual(gauge_log_mod.available_days("no such job"), [])

    def test_cache_reads_a_day_again_only_when_it_changed(self):
        day = datetime(2026, 11, 9, 8)
        self.log.record(1.0, 0.0, day)
        cache = gauge_log_mod.DayStatsCache()
        with mock.patch.object(gauge_log_mod, "read_day", wraps=gauge_log_mod.read_day) as reads:
            cache.get("P1", day.date())
            cache.get("P1", day.date())
            self.assertEqual(reads.call_count, 1)
            self.log.record(3.0, 2.0, day)
            self.assertEqual(cache.get("P1", day.date())["N/A"]["count"], 2)
            self.assertEqual(reads.call_count, 2)
        self.assertEqual(cache.get("P1", date(2026, 1, 1)), {})

    def test_summary_lowest_first(self):
        stats = gauge_log_mod.day_stats({"Box No.1": [3.0], "Box No.2": [1.0, 2.0]})
        self.assertEqual(gauge_log_mod.box_summary(stats), [("Box No.2", 1.5), ("Box No.1", 3.0)])

    def make_days(self, *days):
        for number, day in enumerate(days):
            self.log.record(1.0, number * 2.0, datetime(2026, 11, day, 8))

    def test_usb_drives_only_writable_media_mounts(self):
        root = Path(self.tmp.name)
        (root / "MY USB").mkdir()
        (root / "RO").mkdir()
        mounts = root / "mounts"
        mounts.write_text(f"/dev/sda1 {root}/MY\\040USB vfat rw,nosuid 0 0\n"
                          f"/dev/sdb1 {root}/RO vfat ro,nosuid 0 0\n"
                          f"/dev/nvme0 / ext4 rw 0 0\n"
                          f"/dev/sdc1 {root}/gone vfat rw 0 0\n")
        with mock.patch.object(gauge_log_mod, "_MEDIA_ROOTS", (str(root),)):
            self.assertEqual(gauge_log_mod.usb_drives(mounts), [root / "MY USB"])
        self.assertEqual(gauge_log_mod.usb_drives(root / "no such file"), [])

    def test_export_copies_only_the_days_to_a_new_folder(self):
        self.make_days(7, 8, 9)
        drive = Path(tempfile.mkdtemp(dir=self.tmp.name))
        days = [date(2026, 11, 7), date(2026, 11, 8)]
        first = gauge_log_mod.export_days("P1", days, drive, datetime(2026, 11, 9, 10, 0, 0))
        self.assertEqual(first, drive / "gauge_export" / "P1_20261109_100000")
        self.assertEqual(sorted(f.name for f in first.iterdir()), ["2026_11_07_log.csv", "2026_11_08_log.csv"])
        self.assertEqual(gauge_log_mod.read_day(first / "2026_11_07_log.csv"), {"N/A": [1.0]})
        again = gauge_log_mod.export_days("P1", days, drive, datetime(2026, 11, 9, 10, 0, 1))
        self.assertNotEqual(first, again)
        self.assertTrue(first.exists())

    def test_failed_export_leaves_nothing_half_done(self):
        self.make_days(7)
        drive = Path(tempfile.mkdtemp(dir=self.tmp.name))
        with self.assertRaises(OSError):
            gauge_log_mod.export_days("P1", [date(2026, 11, 7), date(2026, 11, 8)], drive)  # no file for the 8th
        self.assertEqual(list((drive / "gauge_export").iterdir()), [])

    def export_dialog(self, drives):
        from app.widgets.gauge_export_dialog import GaugeExportDialog
        patcher = mock.patch("app.widgets.gauge_export_dialog.usb_drives", return_value=drives)
        patcher.start()
        self.addCleanup(patcher.stop)
        dialog = GaugeExportDialog("P1", [date(2026, 11, d) for d in (9, 8, 7)])
        self.addCleanup(dialog._timer.stop)
        return dialog

    def test_dialog_range_and_drive_decide_export(self):
        self.make_days(7, 8, 9)
        drive = Path(tempfile.mkdtemp(dir=self.tmp.name))
        dialog = self.export_dialog([drive])
        self.assertEqual(len(dialog._chosen_days()), 3)  # the whole record by default
        self.assertTrue(dialog.export_button.isEnabled())
        dialog.from_combo.setCurrentIndex(0)  # the 9th
        dialog.to_combo.setCurrentIndex(1)  # the 8th
        self.assertEqual(dialog._chosen_days(), [])
        self.assertFalse(dialog.export_button.isEnabled())
        self.assertEqual(dialog.status_label.text(), "From date is after To date.")

    def test_dialog_without_a_drive_cannot_export(self):
        self.make_days(7, 8, 9)
        dialog = self.export_dialog([])
        self.assertFalse(dialog.export_button.isEnabled())
        self.assertEqual(dialog.status_label.text(), "Plug in a USB drive.")

    def test_dialog_exports_the_range_and_stays_open_while_copying(self):
        self.make_days(7, 8, 9)
        drive = Path(tempfile.mkdtemp(dir=self.tmp.name))
        dialog = self.export_dialog([drive])
        dialog.from_combo.setCurrentIndex(1)  # the 8th
        dialog.to_combo.setCurrentIndex(0)  # the 9th
        dialog._export()
        dialog.reject()  # ignored while copying
        self.assertFalse(dialog.close_button.isEnabled())
        while dialog.busy or not dialog.close_button.isEnabled():
            APP.processEvents()
        folders = list((drive / "gauge_export").iterdir())
        self.assertEqual(sorted(f.name for f in folders[0].iterdir()), ["2026_11_08_log.csv", "2026_11_09_log.csv"])
        self.assertIn("Saved 2 days", dialog.status_label.text())
        self.assertTrue(dialog.export_button.isEnabled())

    def test_compare_table_is_in_box_order_with_a_gap_for_a_missing_box(self):
        from app.tabs.gauge_data_tab import summary_table
        first = gauge_log_mod.day_stats({"Box No.10": [1.0], "Box No.2": [9.0], "N/A": [5.0]})
        second = gauge_log_mod.day_stats({"Box No.2": [3.0]})
        headers, rows = summary_table([(0, date(2026, 11, 8), first), (1, date(2026, 11, 9), second)])
        self.assertEqual(headers, ["#", "11-08", "11-09"])
        self.assertEqual(rows, [("2", [9.0, 3.0]), ("10", [1.0, None]), ("N/A", [5.0, None])])

    def test_day_keeps_its_colour_and_a_fourth_is_refused(self):
        from app.tabs.gauge_data_tab import DaySelection
        one, two, three, four = (date(2026, 11, d) for d in (6, 7, 8, 9))
        chosen = DaySelection(one)
        self.assertFalse(chosen.toggle(one))  # the last one stays
        self.assertTrue(chosen.toggle(two))
        self.assertTrue(chosen.toggle(three))
        self.assertTrue(chosen.full)
        self.assertFalse(chosen.toggle(four))
        self.assertTrue(chosen.toggle(two))
        self.assertEqual(chosen.chosen(), [(0, one), (2, three)])  # three's colour is unchanged
        self.assertTrue(chosen.toggle(four))
        self.assertEqual(chosen.chosen(), [(0, one), (1, four), (2, three)])

    def test_today_stays_today_after_midnight(self):
        from app.tabs.gauge_data_tab import DaySelection
        old, new = date(2026, 11, 8), date(2026, 11, 9)
        chosen = DaySelection(old)
        chosen.follow(old, new)
        self.assertEqual(chosen.chosen(), [(0, new)])
        chosen.toggle(old)
        chosen.follow(old, new)  # both selected, e.g. the clock was set at boot
        self.assertEqual(chosen.chosen(), [(0, new)])

    def test_real_qr_label(self):
        import cv2
        frame = cv2.imread(str(ROOT / "tests" / "data" / "no_gauge_frame.png"))
        self.assertEqual(gauge_log_mod.read_qr(frame), "Box No.38")

    def test_chart_renders_with_data_and_empty(self):
        from app.tabs.gauge_data_tab import render_box_plot
        stats = gauge_log_mod.day_stats({"Box No.3": [0.05 + i / 1000 for i in range(20)], "N/A": [0.07]})
        today = date(2026, 11, 9)
        cases = ([(0, today, {})], [(0, today, stats)], [(0, today, stats), (1, date(2026, 11, 8), {}), (2, date(2026, 11, 7), stats)])
        for days in cases:
            image = render_box_plot(days, 700, 400, "MPa")
            self.assertEqual((image.width(), image.height()), (700, 400))


@unittest.skipUnless(YOLO is not None and (ROOT / "weights" / "gauge_epoch-100.pt").exists(), "needs Ultralytics and the model")
class RealModel(unittest.TestCase):
    def test_model_loads_and_a_frame_without_a_gauge_is_no_reading(self):
        import cv2
        model = YOLO(str(ROOT / "weights" / "gauge_epoch-100.pt"))
        self.assertEqual(sorted(model.names.values()), ["Center", "End", "Start", "Tip"])
        frame = cv2.imread(str(ROOT / "tests" / "data" / "no_gauge_frame.png"))
        boxes = model(frame, verbose=False, **GAUGE_INFERENCE)[0].boxes
        dets = [{"x1": float(b.xyxy[0][0]), "y1": float(b.xyxy[0][1]), "x2": float(b.xyxy[0][2]),
                 "y2": float(b.xyxy[0][3]), "cls_id": int(b.cls[0]), "conf": float(b.conf[0])} for b in boxes]
        self.assertEqual(read_gauge(dets, GaugeSpec()).side, NO_READ)


if __name__ == "__main__":
    unittest.main()
