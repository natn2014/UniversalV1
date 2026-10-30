"""OK / NG / STANDBY decision for one frame, plus the hold-time latch."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .marking.layout import check_layout
from .roi import Roi


@dataclass
class DetectionFilters:
    """The Settings tab's Detection Filters card; also what a saved job stores."""

    confidence_threshold: float = 0.0
    max_detections: int = 0  # 0 = unlimited
    target_class_1: Optional[str] = None
    target_class_2: Optional[str] = None
    target_gate: str = "OR"
    selected_classes: Set[str] = field(default_factory=set)
    # What decides the AI verdict (Settings switch): "detection" = the Target
    # Classes; "required" = Required Marks, OK only when every ticked mark is
    # found at or above its own minimum confidence (none ticked: Target Classes).
    judge_mode: str = "detection"
    required_classes: Set[str] = field(default_factory=set)
    mark_conf: Dict[str, float] = field(default_factory=dict)  # per required mark; missing = confidence_threshold
    # Mark inspection: warn (no NG) when marks are out of the usual AGC layout.
    check_layout: bool = False
    roi: Optional[Roi] = None  # detection area; None = whole frame
    perspective: Optional[list] = None  # top-view corners; None = camera view as is

    @property
    def by_marks(self) -> bool:
        return self.judge_mode == "required"

    def class_threshold(self, class_name: str) -> float:
        """Minimum confidence for a class: its own for a required mark, else the slider's."""
        if self.by_marks and class_name in self.required_classes:
            return self.mark_conf.get(class_name, self.confidence_threshold)
        return self.confidence_threshold

    def apply(self, detections: list) -> list:
        """Drops unselected / low-confidence detections, highest-confidence first."""
        filtered = [
            det for det in detections
            if det.get("class_name", "unknown") in self.selected_classes
            and det.get("conf", 0.0) >= self.class_threshold(det.get("class_name", "unknown"))
        ]
        # Highest-confidence first; cap to the configured maximum (0 = unlimited)
        filtered.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
        if self.max_detections > 0:
            filtered = filtered[: self.max_detections]
        return filtered

    def missing(self, filtered_detections: list) -> List[str]:
        """Required classes not among the detections (none when nothing is detected: no part)."""
        if not self.by_marks or not filtered_detections:
            return []
        present = {d.get("class_name") for d in filtered_detections}
        return sorted(self.required_classes - present)

    def mark_results(self, detections: list) -> List[dict]:
        """Required Marks mode, one row per ticked mark from the unfiltered
        detections: its best confidence (None = not found), its minimum and
        ok; ok is None while nothing at all is found (no part: STANDBY)."""
        if not self.by_marks:
            return []
        best: Dict[str, float] = {}
        for d in detections:
            name = d.get("class_name")
            if name in self.required_classes and name in self.selected_classes:
                best[name] = max(best.get(name, 0.0), d.get("conf", 0.0))
        part = bool(self.apply(detections))
        rows = []
        for name in sorted(self.required_classes):
            conf, minimum = best.get(name), self.class_threshold(name)
            ok = (conf is not None and conf >= minimum) if part else None
            rows.append({"class_name": name, "conf": conf, "min_conf": minimum, "ok": ok})
        return rows

    def layout_warnings(self, filtered_detections: list) -> List[str]:
        """Mark inspection messages; [] when off or nothing is detected."""
        if not self.by_marks or not self.check_layout or not filtered_detections:
            return []
        return check_layout(filtered_detections)

    def judge(self, filtered_detections: list) -> str:
        present_classes = {d.get("class_name") for d in filtered_detections}
        if self.by_marks and self.required_classes:
            if not filtered_detections:
                return "YELLOW"
            return "RED" if self.missing(filtered_detections) else "GREEN"
        targets = [c for c in (self.target_class_1, self.target_class_2) if c]
        if targets:
            if self.target_gate == "AND":
                is_match = all(t in present_classes for t in targets)
            else:
                is_match = any(t in present_classes for t in targets)
            if is_match:
                return "GREEN"
        if filtered_detections:
            return "RED"
        return "YELLOW"


class JudgementLatch:
    """Holds each STANDBY/OK/NG judgement for at least hold_ms, so a
    flickering detection can't make the banner and tower light chatter.
    No queueing: after the hold, the current raw state is taken as-is."""

    def __init__(self) -> None:
        self.state: Optional[str] = None
        self._since = 0.0

    def reset(self) -> None:
        self.state = None

    def update(self, raw_state: str, hold_ms: int) -> bool:
        """Returns True when the held judgement changed (a new cycle)."""
        now = time.monotonic()
        if self.state is None or (
            raw_state != self.state and (now - self._since) * 1000 >= hold_ms
        ):
            self.state = raw_state
            self._since = now
            return True
        return False
