"""Gauge Reader: turns the gauge model's four points into a value and a verdict.

weights/gauge_epoch-100.pt is a detection model with one class per point of a
dial gauge (by class id, see CLASS_*). The value comes from the needle's angle
between the Start and End of the scale. No Qt here, so all of it is testable.

Two different pairs of numbers, easily confused:
  value_at_start / value_at_end  what the dial shows at each end of its scale
  spec_min / spec_max            the job's OK range; outside it the part is NG
"""

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from itertools import permutations, product
from typing import Deque, Dict, List, Optional, Tuple

# The model's own names have 1 = End and 2 = Start, and it mixes them up (see
# _candidates). Its class is never used for these two: left of Center is Start,
# right of Center is End.
CLASS_CENTER, CLASS_START, CLASS_END, CLASS_TIP = 0, 1, 2, 3

# What the gauge model was tuned with: its points are small and sit close
# together, so the usual defaults (conf 0.25, iou 0.7) lose some of them.
GAUGE_INFERENCE = {"conf": 0.10, "iou": 0.35, "imgsz": 640}

# Geometry that can't be a real gauge: no reading rather than a wrong one.
# Ratios to the dial's radius, so they hold at any camera resolution.
MIN_SIDE_RATIO = 0.15  # how far Start and End are to the left / right of Center
MIN_NEEDLE_RATIO = 0.2  # Center to Tip
ORDER_TOLERANCE_RATIO = 0.05  # how far the Tip may stray past Start / End
TOP_CANDIDATES = 3  # boxes tried per point; the model runs at a low confidence

VALUE_DECIMALS = 4  # the value is judged as shown, so 0.0700 is never "> 0.0700"

OK, OVER, UNDER, NO_READ = "ok", "over", "under", "no_read"
OUT_OF_SPEC = (OVER, UNDER)

Point = Tuple[float, float]


@dataclass
class GaugeSpec:
    """One job's gauge. The defaults are the vacuum gauge this model was
    trained on (0.1 MPa at Start, 0 at End) with the whole scale accepted."""

    value_at_start: float = 0.1
    value_at_end: float = 0.0
    unit: str = "MPa"
    spec_min: float = 0.0
    spec_max: float = 0.1

    @property
    def valid(self) -> bool:
        return self.spec_min < self.spec_max and self.value_at_start != self.value_at_end

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "GaugeSpec":
        """From a job's "gauge" key; older jobs have none, and a damaged value
        falls back to its default."""
        spec = cls()
        for key, value in (data or {}).items():
            if key == "unit":
                spec.unit = str(value)
            elif key in ("value_at_start", "value_at_end", "spec_min", "spec_max"):
                try:
                    setattr(spec, key, float(value))
                except (TypeError, ValueError):
                    pass
        return spec


@dataclass
class GaugeReading:
    side: str = NO_READ
    value: float = 0.0
    percent: float = 0.0  # needle position along the scale, End = 0, Start = 100
    points: Dict[str, Point] = field(default_factory=dict)  # center, start, end, tip
    problem: bool = False  # all points found, but not where a gauge has them

    @property
    def read(self) -> bool:
        return self.side != NO_READ


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(point: Point, center: Point) -> float:
    return math.degrees(math.atan2(point[1] - center[1], point[0] - center[0]))


def needle_ratio(center: Point, start: Point, end: Point, tip: Point) -> Optional[float]:
    """Needle position measured from End (0) to Start (1), over the top of the
    dial; None without a sweep. Past either end it pins to the nearer one: the
    gap past End and the gap past Start are the same arc, split at its middle.
    (Without this a needle just past End measures ~355 degrees and reads as
    full scale.)"""
    origin = _angle(end, center)
    sweep = (origin - _angle(start, center)) % 360
    if sweep == 0:
        return None
    needle = (origin - _angle(tip, center)) % 360
    if needle <= sweep:
        return needle / sweep
    return 1.0 if needle < sweep + (360 - sweep) / 2 else 0.0


def _distinct(found: List[tuple]) -> List[tuple]:
    """Most confident first, without repeats: a detection centred inside the
    box of a surer one is the same mark seen twice."""
    kept: List[tuple] = []
    for item in sorted(found, key=lambda c: c[0], reverse=True):
        x, y = item[1]
        if not any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for _, _, box in kept):
            kept.append(item)
    return kept


def _top(found: List[tuple], count: int) -> List[Tuple[float, Point]]:
    return [(conf, point) for conf, point, _ in _distinct(found)[:count]]


def _candidates(detections: List[dict]) -> tuple:
    """(centers, scale ends, tips), each [(conf, box centre)], most confident
    first. Start and End are one pool: the model was trained with left-right
    mirroring, so its Start / End labels can't be trusted, only where they are."""
    centers, ends, tips = [], [], []
    for det in detections:
        cls_id = det.get("cls_id")
        role = {CLASS_CENTER: centers, CLASS_START: ends, CLASS_END: ends, CLASS_TIP: tips}.get(cls_id)
        if role is not None:
            box = (det["x1"], det["y1"], det["x2"], det["y2"])
            role.append((det.get("conf", 0.0), ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box))
    return _top(centers, TOP_CANDIDATES), _top(ends, TOP_CANDIDATES + 1), _top(tips, TOP_CANDIDATES)


def labelled_points(detections: List[dict]) -> List[Tuple[str, Point]]:
    """What the model found, for the view while there is no reading: each mark
    once, and a scale end named by its side of Center, whatever its class."""
    centers, ends, tips = _candidates(detections)
    named = [("Center", point) for _, point in centers] + [("Tip", point) for _, point in tips]
    for _, point in ends:
        side = "Start/End" if not centers else "Start" if point[0] < centers[0][1][0] else "End"
        named.append((side, point))
    return named


def in_position(center: Point, start: Point, end: Point, tip: Point) -> bool:
    """Start on the left of Center and End on its right, the Tip between them;
    and nothing that can't be a dial (see the ratios above)."""
    radius = (_distance(center, start) + _distance(center, end)) / 2
    side = MIN_SIDE_RATIO * radius
    margin = ORDER_TOLERANCE_RATIO * radius
    return (radius > 0
            and start[0] <= center[0] - side
            and end[0] >= center[0] + side
            and start[0] - margin <= tip[0] <= end[0] + margin
            and _distance(center, tip) >= MIN_NEEDLE_RATIO * radius)


def read_gauge(detections: List[dict], spec: GaugeSpec) -> GaugeReading:
    """Takes the most confident set of points that is in position, so a wrong
    box the model is surer of can't replace a right one. in_position decides
    which scale end is Start and which is End. All four points are needed, the
    centre too: without it there is no angle to measure."""
    if not spec.valid:
        return GaugeReading()
    centers, ends, tips = _candidates(detections)
    if not centers or not tips or len(ends) < 2:
        return GaugeReading()
    best, best_score = None, -1.0
    for (c_conf, center), (t_conf, tip), ((a_conf, start), (b_conf, end)) in product(centers, tips, permutations(ends, 2)):
        score = c_conf + t_conf + a_conf + b_conf
        if score > best_score and in_position(center, start, end, tip):
            best, best_score = (center, start, end, tip), score
    if best is None:
        return GaugeReading(problem=True)
    center, start, end, tip = best
    ratio = needle_ratio(center, start, end, tip)
    if ratio is None:
        return GaugeReading(problem=True)
    value = round(spec.value_at_end + ratio * (spec.value_at_start - spec.value_at_end), VALUE_DECIMALS)
    side = OVER if value > spec.spec_max else UNDER if value < spec.spec_min else OK
    return GaugeReading(side, value, ratio * 100, {"center": center, "start": start, "end": end, "tip": tip})


def gauge_state(reading: GaugeReading) -> str:
    """STANDBY while there is no reading, OK in spec (limits included), else NG."""
    if not reading.read:
        return "YELLOW"
    return "GREEN" if reading.side == OK else "RED"


def format_value(value: float, spec: GaugeSpec) -> str:
    return f"{value:.{VALUE_DECIMALS}f} {spec.unit}".rstrip()


def out_of_spec_text(reading: GaugeReading, spec: GaugeSpec) -> str:
    """"OVER: 0.0812 MPa > MAX 0.0700"; "" while in spec or not read."""
    if reading.side == OVER:
        return f"OVER: {format_value(reading.value, spec)} > MAX {spec.spec_max:.{VALUE_DECIMALS}f}"
    if reading.side == UNDER:
        return f"UNDER: {format_value(reading.value, spec)} < MIN {spec.spec_min:.{VALUE_DECIMALS}f}"
    return ""


class GaugeHold:
    """Steadies the reading (time passed in, so it is testable).

    The median of the last few readings, so one misplaced point doesn't make
    an NG; nothing is reported until a few readings agree on where it is.
    When points drop out for a moment the last result stays for `hold_s`, and
    after that the history is forgotten."""

    WINDOW = 5
    MIN_READINGS = 3

    def __init__(self) -> None:
        self._readings: Deque[GaugeReading] = deque(maxlen=self.WINDOW)
        self._last: Optional[GaugeReading] = None
        self._last_at = 0.0

    def reset(self) -> None:
        self._readings.clear()
        self._last = None

    def update(self, reading: GaugeReading, now: float, hold_s: float) -> GaugeReading:
        if reading.read:
            self._readings.append(reading)
            self._last_at = now
            if len(self._readings) < self.MIN_READINGS:
                return GaugeReading()
            ordered = sorted(self._readings, key=lambda r: r.percent)
            self._last = ordered[len(ordered) // 2]
            return self._last
        if self._readings and now - self._last_at <= hold_s:
            return self._last or GaugeReading()
        self.reset()
        return GaugeReading()


class GaugeCaptureGate:
    """Says when out of spec needs an image the normal NG save doesn't cover.

    The NG log already saves one image each time an NG is counted. On top of
    that, while the judgement stays NG: a gauge that goes out of spec with no
    new NG counted (one was already showing), or that swings straight from
    OVER to UNDER or back. A gauge staying out of spec saves nothing more, so
    a long excursion can't fill the disk."""

    def __init__(self) -> None:
        self._captured: Optional[str] = None  # the side of this excursion's last image

    def reset(self) -> None:
        self._captured = None

    def update(self, side: str, state_red: bool, counted_red: bool) -> bool:
        """`state_red`: the judgement shown is NG; `counted_red`: an NG was
        counted this frame, so the normal save has this image. A lost reading
        doesn't end an excursion, only coming back in spec does."""
        if side == OK:
            self._captured = None
        if side not in OUT_OF_SPEC:
            return False
        if counted_red:
            self._captured = side
            return False
        if state_red and side != self._captured:
            self._captured = side
            return True
        return False
