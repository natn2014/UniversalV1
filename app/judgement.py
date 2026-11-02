"""OK / NG / STANDBY decision for one frame, plus the hold-time latch."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .marking.layout import check_layout
from .marking.short_name import valid_short_name
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
    # Required Marks: the short name must read as this (app/marking/short_name.py,
    # the worker puts each read in det["ocr"]); "" = only check it is there.
    short_name_expected: str = ""
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

    @property
    def checks_short_name(self) -> bool:
        return self.by_marks and "short_name" in self.required_classes and valid_short_name(self.short_name_expected)

    def short_name_ocr(self, filtered_detections: list) -> Optional[dict]:
        """The read of the most confident short name in view ({text, conf,
        expected, state}); None when the check is off or none is in view.
        A box without a read yet counts as pending."""
        if not self.checks_short_name:
            return None
        boxes = [d for d in filtered_detections if d.get("class_name") == "short_name"]
        if not boxes:
            return None
        ocr = max(boxes, key=lambda d: d.get("conf", 0.0)).get("ocr")
        if not ocr or ocr.get("expected") != self.short_name_expected:
            return {"text": "", "conf": 0.0, "expected": self.short_name_expected, "state": "pending"}
        return ocr

    def short_name_issue(self, filtered_detections: list) -> str:
        """"Short name: read TA ≠ T1" / "Short name: unreadable" while it is NG, else ""."""
        ocr = self.short_name_ocr(filtered_detections)
        if ocr is None or ocr["state"] in ("ok", "pending"):
            return ""
        if ocr["state"] == "unreadable":
            return "Short name: unreadable"
        return f"Short name: read {ocr['text']} ≠ {ocr['expected']}"

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
        filtered = self.apply(detections)
        ocr = self.short_name_ocr(filtered)
        rows = []
        for name in sorted(self.required_classes):
            conf, minimum = best.get(name), self.class_threshold(name)
            ok = (conf is not None and conf >= minimum) if filtered else None
            row = {"class_name": name, "conf": conf, "min_conf": minimum, "ok": ok}
            if name == "short_name" and ocr is not None and ok:
                row["read"], row["expected"], row["ocr_state"] = ocr["text"], ocr["expected"], ocr["state"]
                row["ok"] = None if ocr["state"] == "pending" else ocr["state"] == "ok"
            rows.append(row)
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
            if self.missing(filtered_detections):
                return "RED"
            ocr = self.short_name_ocr(filtered_detections)
            if ocr is not None and ocr["state"] != "ok":
                return "YELLOW" if ocr["state"] == "pending" else "RED"  # still reading / wrong or unreadable
            return "GREEN"
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


class InspectionCycle:
    """What the Monitor shows, sends to the tower / relay and counts.

    Continuous (no Trigger input): the held judgement from JudgementLatch;
    a cycle is counted each time it becomes OK or NG.
    Trigger mode: STANDBY until trigger(); the next frame's judgement is then
    held until the next trigger, and counted once.
    Confirm NG (a Confirm NG input exists): an NG stays latched, and triggers
    and new results are ignored, until confirm()."""

    def __init__(self) -> None:
        self.latch = JudgementLatch()
        self.held: Optional[str] = None  # trigger mode: the last triggered result
        self.armed = False  # trigger mode: judge the next frame
        self.awaiting_confirm = False

    def reset(self) -> None:
        self.latch.reset()
        self.held = None
        self.armed = False
        self.awaiting_confirm = False

    @property
    def state(self) -> Optional[str]:
        if self.awaiting_confirm:
            return "RED"
        return self.held if self.held is not None else self.latch.state

    def trigger(self) -> bool:
        """False while an NG waits for confirmation."""
        if self.awaiting_confirm:
            return False
        self.armed = True
        return True

    def confirm(self) -> bool:
        """Clears a latched NG to STANDBY; False when nothing waited."""
        if not self.awaiting_confirm:
            return False
        self.awaiting_confirm = False
        self.held = None
        self.latch.reset()
        self.latch.update("YELLOW", 0)
        return True

    def update(self, raw_state: str, hold_ms: int, trigger_mode: bool = False,
               confirm_ng: bool = False, pending: bool = False) -> Tuple[str, Optional[str]]:
        """(state to show, state to count or None) for this frame. `pending`:
        the part is there but not judged yet (short name still being read);
        a trigger then waits for the next frame instead of holding STANDBY."""
        if self.awaiting_confirm:
            return "RED", None
        counted = None
        if trigger_mode:
            if self.armed and not pending:
                self.armed = False
                self.held = raw_state
                counted = raw_state if raw_state in ("GREEN", "RED") else None
            state = self.held or "YELLOW"
        else:
            self.held = None
            if self.latch.update(raw_state, hold_ms) and self.latch.state in ("GREEN", "RED"):
                counted = self.latch.state
            state = self.latch.state
        if confirm_ng and counted == "RED":
            self.awaiting_confirm = True
        return state, counted
