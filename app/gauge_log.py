"""Gauge data record: one reading a second to a daily CSV per job, tagged with
the box number from the QR label, and the recent rows for the Gauge Data chart.

gauge_logs/<part>/YYYY_MM_DD_log.csv, same columns as Gauge_reader-main:
Timestamp, Converted Value, QR Code Data.
"""

import csv
import re
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import cv2

from . import optional_deps, paths

BOX_COUNT = 43  # boxes on the line; Box No.43 wraps back to Box No.1
NO_QR = "N/A"
RECORD_INTERVAL_S = 1.0
QR_SCAN_INTERVAL_S = 2.0
BUFFER_ROWS = 1000  # what the chart and summary show
CSV_HEADER = ["Timestamp", "Converted Value", "QR Code Data"]
_BOX_QR_RE = re.compile(r"Box\s+No\.(\d+)", re.IGNORECASE)

Row = Tuple[str, float, str]


def box_labels() -> List[str]:
    return [f"Box No.{n}" for n in range(1, BOX_COUNT + 1)]


def next_box_label(raw: str) -> str:
    """The box a QR label belongs to: the number on it plus one, as the gauge
    being measured is one position on from the label in view (43 wraps to 1).
    Other QR text is kept as it is."""
    match = _BOX_QR_RE.fullmatch(raw.strip())
    if match is None:
        return raw.strip()
    number = int(match.group(1))
    return f"Box No.{1 if number >= BOX_COUNT else number + 1}"


def read_qr(frame_bgr) -> str:
    """The first QR code in the frame, "" if none. zbar first: on the station's
    own frames it reads labels that OpenCV's detector misses."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if optional_deps.zbar_decode is not None:
        for result in optional_deps.zbar_decode(gray):
            if result.type == "QRCODE":
                return result.data.decode("utf-8", "replace").strip()
        return ""
    return cv2.QRCodeDetector().detectAndDecode(gray)[0].strip()


def values_by_box(rows: List[Row]) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}
    for _, value, box in rows:
        grouped.setdefault(box, []).append(value)
    return grouped


def box_summary(rows: List[Row]) -> List[Tuple[str, float]]:
    """(box, average), lowest average first."""
    averages = [(box, sum(values) / len(values)) for box, values in values_by_box(rows).items()]
    return sorted(averages, key=lambda item: item[1])


def _part_dir(part: Optional[str]):
    return paths.GAUGE_LOGS_DIR / paths.sanitize_part_no(part or paths.NO_PART_NAME)


def daily_log_path(part: Optional[str], when: datetime):
    return _part_dir(part) / f"{when.strftime('%Y_%m_%d')}_log.csv"


def _read_tail(path) -> List[Row]:
    """The last rows of a day's file; a damaged line is skipped."""
    rows: Deque[Row] = deque(maxlen=BUFFER_ROWS)
    try:
        with open(path, newline="") as f:
            for record in csv.reader(f):
                try:
                    rows.append((record[0], float(record[1]), record[2]))
                except (IndexError, ValueError):
                    continue  # the header, or a line cut short by a power loss
    except OSError:
        return []
    return list(rows)


class GaugeDataLog:
    """Not thread-safe: used from the UI thread (the chart takes a snapshot)."""

    def __init__(self) -> None:
        self.rows: Deque[Row] = deque(maxlen=BUFFER_ROWS)
        self.qr = NO_QR
        self.part: Optional[str] = None
        self._gauge_seen = False
        self._last_at = float("-inf")

    def set_part(self, part: Optional[str]) -> None:
        """Another job: its own file, and today's rows so a restart keeps the chart."""
        self.part = part
        self.qr = NO_QR
        self._gauge_seen = False
        self._last_at = float("-inf")
        self.rows = deque(_read_tail(daily_log_path(part, datetime.now())), maxlen=BUFFER_ROWS)

    def set_qr(self, raw: str) -> None:
        """Ignored until the gauge has been read, like the original: a label
        in view before the gauge is must not name the first readings."""
        if self._gauge_seen and raw.strip():
            self.qr = next_box_label(raw)

    def record(self, value: float, now: float, when: Optional[datetime] = None) -> bool:
        """Appends one reading, at most once per RECORD_INTERVAL_S (`now` is
        monotonic). Raises OSError if the file can't be written."""
        self._gauge_seen = True
        if now - self._last_at < RECORD_INTERVAL_S:
            return False
        self._last_at = now
        when = when or datetime.now()
        path = daily_log_path(self.part, when)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        stamp = when.isoformat(sep=" ", timespec="microseconds")
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(CSV_HEADER)
            writer.writerow([stamp, value, self.qr])
        self.rows.append((stamp, value, self.qr))
        return True
