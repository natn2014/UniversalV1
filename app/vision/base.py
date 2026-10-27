"""Tool framework: the Offset from Position Adjustment, ToolResult, the Tool
base class and the kind registry.

Overlays are plain dicts so the UI draws them and this package never imports
Qt: {"type", "pts", "ok", "label"} in frame pixels, where `pts` is
  rect / line / mask_box  [x1, y1, x2, y2]
  circle                  [cx, cy, r]
  text                    [x, y]
  poly                    [[x, y], ...]
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from ..roi import Roi, roi_to_pixels
from .preprocess import apply_chain, to_gray

FULL_FRAME: Roi = (0.0, 0.0, 1.0, 1.0)
_DEFAULT = object()  # "use the class's default limit" (None means no bound)


@dataclass
class Offset:
    """How far the part moved from where it was taught (frame pixels, degrees)."""

    dx: float = 0.0
    dy: float = 0.0
    angle: float = 0.0
    score: float = 1.0
    found: bool = True


@dataclass
class ToolResult:
    name: str
    kind: str
    value: float
    ok: bool
    overlay: List[dict] = field(default_factory=list)
    text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def overlay_item(kind: str, pts: Sequence, ok: bool = True, label: str = "") -> dict:
    return {"type": kind, "pts": list(pts), "ok": ok, "label": label}


def rotate_points(pts: Sequence[Sequence[float]], center: Tuple[float, float], angle_deg: float) -> List[Tuple[float, float]]:
    """Rotates (x, y) points about `center`; positive angle is counter-clockwise
    on screen (image y points down)."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    cx, cy = center
    out = []
    for x, y in pts:
        rx, ry = x - cx, y - cy
        out.append((cx + rx * cos_a + ry * sin_a, cy - rx * sin_a + ry * cos_a))
    return out


def shift_overlay(items: List[dict], x0: float, y0: float) -> List[dict]:
    """Copies of overlay items moved from patch to frame coordinates."""
    out = []
    for item in items:
        pts = item["pts"]
        if item["type"] == "poly":
            moved = [[x + x0, y + y0] for x, y in pts]
        elif item["type"] == "circle":
            moved = [pts[0] + x0, pts[1] + y0, pts[2]]
        else:
            moved = [v + (x0 if i % 2 == 0 else y0) for i, v in enumerate(pts)]
        out.append({**item, "pts": moved})
    return out


class Tool:
    """One inspection tool: an ROI, a measured value and an OK range.

    Subclasses set `PARAMS` (their own keys with defaults) and implement
    `process(patch, origin)`; `run` does the shared work (crop, empty-ROI
    guard, ROI overlay, judging). Taught data must stay JSON-safe in
    `params()`; a taught image lives in `template` and is saved as a PNG by
    `vision.store`.
    """

    kind = ""
    PARAMS: Dict[str, Any] = {}
    DEFAULT_LOW: Optional[float] = None
    DEFAULT_HIGH: Optional[float] = None
    DEFAULT_PRE: List[dict] = []  # pre-filter chain every tool accepts as `pre`
    # Too slow for every frame (OCR, code reading): the camera worker runs
    # these in their own thread at a few Hz (vision.slow_runner).
    SLOW = False

    # Set by the runner for each frame; never saved.
    mm_per_px: Optional[float] = None
    part_no: Optional[str] = None

    def __init__(self, name: str = "", roi: Roi = FULL_FRAME, low: Any = _DEFAULT, high: Any = _DEFAULT,
                 enabled: bool = True, follow_position: bool = True, **params: Any) -> None:
        self.name = name or self.kind
        self.roi: Roi = tuple(roi)  # type: ignore[assignment]
        self.low: Optional[float] = self.DEFAULT_LOW if low is _DEFAULT else low
        self.high: Optional[float] = self.DEFAULT_HIGH if high is _DEFAULT else high
        self.enabled = enabled
        self.follow_position = follow_position
        self.template: Optional[np.ndarray] = None
        self.template_file = ""
        for key, default in self._all_params().items():
            setattr(self, key, [dict(s) if isinstance(s, dict) else s for s in default] if isinstance(default, list) else default)
        self.set_params(params)

    def judge(self, value: float) -> bool:
        if self.low is not None and value < self.low:
            return False
        if self.high is not None and value > self.high:
            return False
        return True

    def crop(self, frame: np.ndarray, offset: Optional[Offset] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        """The ROI patch and its top-left corner. The ROI follows the part's
        offset but keeps its size, sliding back inside the frame at the edges."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi_to_pixels(self.roi, w, h)
        bw, bh = x2 - x1, y2 - y1
        if self.follow_position and offset is not None and offset.found:
            x1 += int(round(offset.dx))
            y1 += int(round(offset.dy))
        x1 = max(0, min(x1, w - bw))
        y1 = max(0, min(y1, h - bh))
        return frame[y1:y1 + bh, x1:x1 + bw], (x1, y1)

    def prepared(self, patch: np.ndarray) -> np.ndarray:
        """The patch after the `pre` chain (still BGR if it was)."""
        return apply_chain(patch, self.pre)

    def gray(self, patch: np.ndarray) -> np.ndarray:
        return apply_chain(to_gray(patch), self.pre)

    def teach(self, frame: np.ndarray, offset: Optional[Offset] = None, **kw: Any) -> None:
        """Registers the master from a frame. Default: nothing to learn."""

    def process(self, patch: np.ndarray, origin: Tuple[int, int]) -> ToolResult:
        raise NotImplementedError

    def run(self, frame: np.ndarray, offset: Optional[Offset] = None) -> ToolResult:
        """Never raises on an empty or odd ROI; returns ok=False with a text."""
        if frame is None or frame.size == 0:
            return self.fail("empty frame")
        patch, origin = self.crop(frame, offset)
        if patch.size == 0:
            return self.fail("empty ROI")
        result = self.process(patch, origin)
        x0, y0 = origin
        h, w = patch.shape[:2]
        frame_box = overlay_item("rect", [x0, y0, x0 + w, y0 + h], result.ok, self.name)
        result.overlay = [frame_box] + shift_overlay(result.overlay, x0, y0)
        return result

    def scale(self, px: float) -> float:
        """Pixels -> mm when the job is calibrated, otherwise unchanged."""
        return px * self.mm_per_px if self.mm_per_px else px

    @property
    def unit(self) -> str:
        return "mm" if self.mm_per_px else "px"

    def result(self, value: float, overlay: Optional[List[dict]] = None, text: str = "",
               extra: Optional[dict] = None, ok: Optional[bool] = None) -> ToolResult:
        value = float(value)
        return ToolResult(self.name, self.kind, value, self.judge(value) if ok is None else ok,
                          overlay or [], text, extra or {})

    def fail(self, text: str) -> ToolResult:
        return ToolResult(self.name, self.kind, 0.0, False, [], text, {})

    def _all_params(self) -> Dict[str, Any]:
        return {"pre": self.DEFAULT_PRE, **self.PARAMS}

    def params(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self._all_params()}

    def set_params(self, params: Dict[str, Any]) -> None:
        known = self._all_params()
        for key, value in params.items():
            if key in known:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "kind": self.kind, "name": self.name, "roi": list(self.roi),
            "low": self.low, "high": self.high,
            "enabled": self.enabled, "follow_position": self.follow_position,
        }
        if self.template_file:
            d["template_file"] = self.template_file
        d.update(self.params())
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Tool":
        d = dict(d)
        tool = create(d.pop("kind"), **{k: v for k, v in d.items() if k != "template_file"})
        tool.template_file = d.get("template_file", "")
        return tool


TOOL_KINDS: Dict[str, Type[Tool]] = {}


def register(kind: str):
    """Class decorator: makes `kind` creatable from a saved job."""
    def deco(cls: Type[Tool]) -> Type[Tool]:
        cls.kind = kind
        TOOL_KINDS[kind] = cls
        return cls
    return deco


def create(kind: str, **kw: Any) -> Tool:
    """Raises KeyError for an unknown kind."""
    return TOOL_KINDS[kind](**kw)
