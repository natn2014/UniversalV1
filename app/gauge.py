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
from typing import Deque, Dict, List, Optional, Tuple

# The model's own names have 1 = End and 2 = Start; which is which doesn't
# matter, as read_gauge() takes the left one of the two as Start.
CLASS_CENTER, CLASS_START, CLASS_END, CLASS_TIP = 0, 1, 2, 3

# What the gauge model was tuned with: its points are small and sit close
# together, so the usual defaults (conf 0.25, iou 0.7) lose some of them.
GAUGE_INFERENCE = {"conf": 0.10, "iou": 0.35, "imgsz": 640}

# Geometry that can't be a real gauge: no reading rather than a wrong one.
# Ratios to the dial's radius, so they hold at any camera resolution.
MIN_SPAN_RATIO = 0.25  # Start to End
MIN_NEEDLE_RATIO = 0.2  # Center to Tip

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
    percent: float = 0.0  # needle position along the scale, Start = 0, End = 100
    points: Dict[str, Point] = field(default_factory=dict)  # center, start, end, tip

    @property
    def read(self) -> bool:
        return self.side != NO_READ


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(point: Point, center: Point) -> float:
    return math.degrees(math.atan2(point[1] - center[1], point[0] - center[0]))


def needle_ratio(center: Point, start: Point, end: Point, tip: Point) -> Optional[float]:
    """Needle position from Start (0) to End (1); None without a sweep. Past
    either end it pins to the nearer one: the gap below Start and the gap
    above End are the same arc, split at its middle. (Without this a needle
    just below Start measures ~355 degrees and reads as full scale.)"""
    origin = _angle(start, center)
    sweep = (_angle(end, center) - origin) % 360
    if sweep == 0:
        return None
    needle = (_angle(tip, center) - origin) % 360
    if needle <= sweep:
        return needle / sweep
    return 1.0 if needle < sweep + (360 - sweep) / 2 else 0.0


def _best_points(detections: List[dict]) -> Dict[int, Point]:
    """The centre of the most confident box of each gauge class, by class id."""
    best: Dict[int, dict] = {}
    for det in detections:
        cls_id = det.get("cls_id")
        if cls_id in (CLASS_CENTER, CLASS_START, CLASS_END, CLASS_TIP):
            if cls_id not in best or det.get("conf", 0.0) > best[cls_id].get("conf", 0.0):
                best[cls_id] = det
    return {c: ((d["x1"] + d["x2"]) / 2, (d["y1"] + d["y2"]) / 2) for c, d in best.items()}


def read_gauge(detections: List[dict], spec: GaugeSpec) -> GaugeReading:
    """All four points are needed, the centre too: without it there is no
    angle to measure, and a wrong number could pass a bad part."""
    if not spec.valid:
        return GaugeReading()
    found = _best_points(detections)
    if len(found) < 4:
        return GaugeReading()
    center, start, end, tip = found[CLASS_CENTER], found[CLASS_START], found[CLASS_END], found[CLASS_TIP]
    if start[0] > end[0]:  # Start is the left end of the scale
        start, end = end, start
    radius = (_distance(center, start) + _distance(center, end)) / 2
    if (radius <= 0 or _distance(start, end) < MIN_SPAN_RATIO * radius
            or _distance(center, tip) < MIN_NEEDLE_RATIO * radius):
        return GaugeReading()
    ratio = needle_ratio(center, start, end, tip)
    if ratio is None:
        return GaugeReading()
    value = round(spec.value_at_start + ratio * (spec.value_at_end - spec.value_at_start), VALUE_DECIMALS)
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
