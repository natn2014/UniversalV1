"""Total judgement: combines the tool results and the YOLO state into the
GREEN / RED / YELLOW state the Monitor already understands."""

from typing import List, Optional

from .base import ToolResult


def total_state(results: List[ToolResult], yolo_state: Optional[str], gate: str = "AND",
                position_found: bool = True) -> str:
    """YELLOW = no part (Position Adjustment lost it, YOLO sees nothing, a
    slow tool has not read this part yet, or there is nothing to judge). Otherwise the AND / OR of every input.

    The runner only returns enabled tools, and position results are left out
    here: they decide `position_found`, not OK / NG. `DetectionFilters.judge`
    is unchanged; its state arrives as one more input.
    """
    if not position_found or yolo_state == "YELLOW":
        return "YELLOW"
    # A slow tool (OCR, code) still waiting for its first read of this part.
    if any(r.extra.get("pending") for r in results):
        return "YELLOW"
    oks = [r.ok for r in results if r.kind != "position"]
    if yolo_state is not None:
        oks.append(yolo_state == "GREEN")
    if not oks:
        return "YELLOW"
    passed = any(oks) if gate == "OR" else all(oks)
    return "GREEN" if passed else "RED"
