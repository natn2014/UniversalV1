"""Runs a job's tools on one frame: Position Adjustment first, then every
other enabled tool with the offset it found."""

import time
from typing import List, Optional

import numpy as np

from .base import Offset, Tool, ToolResult
from .position import PositionTool, offset_from_result
from .store import set_runtime


def _timed_run(tool: Tool, frame: np.ndarray, offset: Offset) -> ToolResult:
    """One tool, timed; an exception becomes an NG result so a bad tool
    cannot stop the stream."""
    start = time.perf_counter()
    try:
        result = tool.run(frame, offset)
    except Exception as exc:
        result = ToolResult(tool.name, tool.kind, 0.0, False, [], str(exc) or type(exc).__name__)
    result.extra["ms"] = (time.perf_counter() - start) * 1000.0
    return result


def run_tools(frame: np.ndarray, tools: List[Tool], mm_per_px: Optional[float] = None,
              part_no: Optional[str] = None) -> List[ToolResult]:
    """Results of the enabled tools, position results first.

    The first position tool gives the offset the others follow (tools with
    `follow_position=False` get none). When it loses the part, only the
    position results are returned: the other ROIs would be in the wrong
    place, and the total judgement is "no part" anyway.
    """
    active = [t for t in tools if t.enabled]
    set_runtime(active, mm_per_px, part_no)
    results: List[ToolResult] = []
    offset = Offset()
    positions = [t for t in active if isinstance(t, PositionTool)]
    for i, tool in enumerate(positions):
        res = _timed_run(tool, frame, Offset())
        results.append(res)
        if i == 0:
            offset = offset_from_result(res)
    if positions and not offset.found:
        return results
    for tool in active:
        if not isinstance(tool, PositionTool):
            results.append(_timed_run(tool, frame, offset if tool.follow_position else Offset()))
    return results
