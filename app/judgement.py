"""OK / NG / STANDBY decision for one frame, plus the hold-time latch."""

import time
from dataclasses import dataclass, field
from typing import Optional, Set

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
    roi: Optional[Roi] = None  # detection area; None = whole frame
    perspective: Optional[list] = None  # top-view corners; None = camera view as is

    def apply(self, detections: list) -> list:
        """Drops unselected / low-confidence detections, highest-confidence first."""
        filtered = [
            det for det in detections
            if det.get("class_name", "unknown") in self.selected_classes
            and det.get("conf", 0.0) >= self.confidence_threshold
        ]
        # Highest-confidence first; cap to the configured maximum (0 = unlimited)
        filtered.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
        if self.max_detections > 0:
            filtered = filtered[: self.max_detections]
        return filtered

    def judge(self, filtered_detections: list) -> str:
        present_classes = {d.get("class_name") for d in filtered_detections}
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
