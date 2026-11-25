"""Runs the slow tools (OCR, code reading) in their own thread at a few Hz,
so the camera loop and the fast tools keep the live view's full frame rate.

The mailbox holds one frame: a newer request replaces one not yet started,
so the thread never falls behind the camera. A result only counts for the
part it was read from (`epoch`), so the previous part's OCR can never pass
a new part; until the new part is read its tools report "pending", which
the total judgement shows as STANDBY.
"""

import dataclasses
import threading
import time
from typing import List, Optional

import numpy as np

from .base import Offset, Tool, ToolResult
from .runner import run_tools_with_offset

PENDING_TEXT = "waiting…"


def pending_result(tool: Tool) -> ToolResult:
    return ToolResult(tool.name, tool.kind, 0.0, False, [], PENDING_TEXT, {"pending": True, "slow": True})


class SlowToolRunner:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._request: Optional[tuple] = None
        self._busy = False
        self._latest: Optional[tuple] = None  # (results, epoch, submitted_at)
        self._last_submit = float("-inf")
        self._generation = 0  # bumped by clear(), drops the result of a run in flight
        self._stopping = False
        self._thread: Optional[threading.Thread] = None
        self.last_run_s = 0.0

    def due(self, now: float, hz: float) -> bool:
        """True when idle and at least 1/hz seconds after the last submit."""
        with self._cond:
            idle = not self._busy and self._request is None
            return idle and now - self._last_submit >= 1.0 / max(hz, 0.01)

    def submit(self, frame: np.ndarray, offset: Offset, tools: List[Tool], epoch: int, now: float,
               mm_per_px: Optional[float] = None, part_no: Optional[str] = None) -> None:
        """Hands over a frame the caller no longer touches (pass a copy)."""
        with self._cond:
            self._request = (frame, offset, list(tools), mm_per_px, part_no, epoch, now, self._generation)
            self._last_submit = now
            self._stopping = False  # a thread still finishing after stop() keeps serving
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="slow-tools", daemon=True)
                self._thread.start()
            self._cond.notify()

    def results_for(self, slow_tools: List[Tool], epoch: int, now: float, max_age: float) -> List[ToolResult]:
        """The latest slow results, tagged with their age, if they were read
        from this part and are recent enough; otherwise a pending result per
        slow tool."""
        with self._cond:
            latest = self._latest
        if latest is not None:
            results, result_epoch, submitted_at = latest
            age = now - submitted_at
            if result_epoch == epoch and age <= max_age:
                return [dataclasses.replace(r, extra={**r.extra, "slow": True, "age_ms": age * 1000.0})
                        for r in results]
        return [pending_result(t) for t in slow_tools]

    def clear(self) -> None:
        """Forgets cached results and any queued frame (tools changed)."""
        with self._cond:
            self._generation += 1
            self._request = None
            self._latest = None
            self._last_submit = float("-inf")

    def stop(self, timeout: float = 1.0) -> None:
        self.clear()
        with self._cond:
            self._stopping = True
            self._cond.notify()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._request is None and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                frame, offset, tools, mm_per_px, part_no, epoch, submitted_at, generation = self._request
                self._request = None
                self._busy = True
            start = time.perf_counter()
            results: List[ToolResult] = []
            try:
                results, _ = run_tools_with_offset(frame, tools, mm_per_px, part_no, slow=True, offset=offset)
            finally:
                with self._cond:
                    self._busy = False
                    self.last_run_s = time.perf_counter() - start
                    if generation == self._generation:
                        self._latest = (results, epoch, submitted_at)
