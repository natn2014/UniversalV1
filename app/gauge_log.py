"""Gauge data record: one reading a second to a daily CSV per job, tagged with
the box number from the QR label, and a per-box summary of a day for the
Gauge Data chart.

gauge_logs/<part>/YYYY_MM_DD_log.csv, same columns as Gauge_reader-main:
Timestamp, Converted Value, QR Code Data.
"""

import csv
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import optional_deps, paths

BOX_COUNT = 43  # boxes on the line; Box No.43 wraps back to Box No.1
NO_QR = "N/A"
RECORD_INTERVAL_S = 1.0
QR_SCAN_INTERVAL_S = 2.0
CSV_HEADER = ["Timestamp", "Converted Value", "QR Code Data"]
_DAY_FILE_FORMAT = "%Y_%m_%d_log.csv"
_BOX_QR_RE = re.compile(r"Box\s+No\.(\d+)", re.IGNORECASE)

DayStats = Dict[str, dict]  # box -> matplotlib boxplot stats, plus "count"


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


def _part_dir(part: Optional[str]) -> Path:
    return paths.GAUGE_LOGS_DIR / paths.sanitize_part_no(part or paths.NO_PART_NAME)


def daily_log_path(part: Optional[str], when: date) -> Path:
    return _part_dir(part) / when.strftime(_DAY_FILE_FORMAT)


def available_days(part: Optional[str]) -> List[date]:
    """The days this job has a log for, newest first."""
    days = []
    for path in _part_dir(part).glob("*_log.csv"):
        try:
            days.append(datetime.strptime(path.name, _DAY_FILE_FORMAT).date())
        except ValueError:
            continue
    return sorted(days, reverse=True)


def read_day(path: Path) -> Dict[str, List[float]]:
    """A day's values by box; damaged lines are skipped."""
    grouped: Dict[str, List[float]] = {}
    try:
        with open(path, newline="", errors="replace") as f:
            for record in csv.reader(f):
                try:
                    value, box = float(record[1]), record[2]
                except (IndexError, ValueError):
                    continue  # the header, or a line cut short by a power loss
                grouped.setdefault(box, []).append(value)
    except (OSError, csv.Error):
        pass  # what was read so far still counts
    return grouped


def day_stats(grouped: Dict[str, List[float]]) -> DayStats:
    """All a box plot and the averages need, however many readings a day has."""
    from matplotlib.cbook import boxplot_stats

    summary: DayStats = {}
    for box, values in grouped.items():
        stats = boxplot_stats(np.asarray(values))[0]
        stats["fliers"] = []  # not drawn, and can be thousands of values
        stats["count"] = len(values)
        summary[box] = stats
    return summary


def box_summary(stats: DayStats) -> List[Tuple[str, float]]:
    """(box, average), lowest average first."""
    return sorted(((box, float(s["mean"])) for box, s in stats.items()), key=lambda item: item[1])


class DayStatsCache:
    """Each day's summary until its file changes, so a finished day is read
    once. Holds no raw readings. Used by one chart worker at a time."""

    def __init__(self) -> None:
        self._days: Dict[Path, Tuple[Tuple[int, int], DayStats]] = {}

    def get(self, part: Optional[str], day: date) -> DayStats:
        path = daily_log_path(part, day)
        try:
            info = path.stat()
        except OSError:
            return {}
        stamp = (info.st_size, info.st_mtime_ns)
        cached = self._days.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        stats = day_stats(read_day(path))
        self._days[path] = (stamp, stats)
        return stats


class GaugeDataLog:
    """Not thread-safe: used from the UI thread."""

    def __init__(self) -> None:
        self.qr = NO_QR
        self.part: Optional[str] = None
        self._gauge_seen = False
        self._last_at = float("-inf")

    def set_part(self, part: Optional[str]) -> None:
        """Another job: its own file, and no box number carried over."""
        self.part = part
        self.qr = NO_QR
        self._gauge_seen = False
        self._last_at = float("-inf")

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
        return True
