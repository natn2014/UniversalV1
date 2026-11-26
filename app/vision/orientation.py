"""Orientation (AI): which way the part faces, from a YOLO-pose model.

The model (trained in the Train tab on a pose dataset, app/pose_labels.py)
finds the part and two keypoints on it, by default `tail` (index 1) and
`head` (index 0). The direction from `from_kpt` to `to_kpt` is the part's
angle: 0° = pointing right, 90° = up, counter-clockwise positive (the same
sense as `base.rotate_points`). A box or OBB can't tell head from tail; two
named keypoints can, so this covers the full 360°.

The value is how far the angle is from `expected_angle` (-180..180), judged
against low / high (default ±10°). Teach sets `expected_angle` from the
current frame.
"""

import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from ..paths import ROOT_DIR
from .base import Offset, Tool, overlay_item, register

Detection = Tuple[float, np.ndarray]  # (box confidence, keypoints (K, 3): x, y, conf)


class PosePredictor(Protocol):
    def __call__(self, image: np.ndarray, conf: float) -> List[Detection]:
        """Detections in `image` pixels, best first."""


def direction_angle(start: Tuple[float, float], end: Tuple[float, float]) -> float:
    """Angle (degrees, -180..180) of the arrow start -> end on screen:
    0 = right, 90 = up (image y points down)."""
    return math.degrees(math.atan2(-(end[1] - start[1]), end[0] - start[0]))


def angle_difference(angle: float, expected: float) -> float:
    """Signed angle - expected, wrapped into -180..180."""
    return (angle - expected + 180.0) % 360.0 - 180.0


def resolve_model_path(path: str) -> Optional[Path]:
    """The model file to load: relative paths are from the app folder, and
    a TensorRT .engine next to a .pt is preferred (built on this device)."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = ROOT_DIR / p
    if p.suffix == ".pt" and p.with_suffix(".engine").exists():
        return p.with_suffix(".engine")
    return p if p.exists() else None


class YoloPosePredictor:
    """An Ultralytics pose model; calls are serialised (the slow-tool thread
    and the Tools tab's frozen-frame run may overlap)."""

    def __init__(self, model) -> None:
        self._model = model
        self._lock = threading.Lock()

    def __call__(self, image: np.ndarray, conf: float) -> List[Detection]:
        with self._lock:
            results = self._model.predict(image, conf=conf, verbose=False)
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return out
        data = r.keypoints.data.cpu().numpy()  # (N, K, 3), or (N, K, 2) without confidences
        box_conf = r.boxes.conf.cpu().numpy()
        for i in np.argsort(-box_conf):
            kpts = data[i]
            if kpts.shape[1] == 2:
                kpts = np.concatenate([kpts, np.ones((len(kpts), 1), kpts.dtype)], axis=1)
            out.append((float(box_conf[i]), kpts.astype(np.float32)))
        return out


_MODELS: Dict[str, YoloPosePredictor] = {}
_MODELS_LOCK = threading.Lock()


def get_predictor(path: Path) -> YoloPosePredictor:
    """One loaded model per file, shared by every Orientation tool."""
    key = str(path.resolve())
    with _MODELS_LOCK:
        if key not in _MODELS:
            from ..optional_deps import YOLO
            if YOLO is None:
                raise RuntimeError("Ultralytics not installed")
            model = YOLO(key, task="pose")
            if getattr(model, "task", "pose") != "pose":
                raise RuntimeError(f"{path.name} is not a pose model: train it on a pose dataset")
            _MODELS[key] = YoloPosePredictor(model)
        return _MODELS[key]


def clear_models() -> None:
    with _MODELS_LOCK:
        _MODELS.clear()


@register("orientation")
class OrientationTool(Tool):
    """Its ROI is where the part is looked for. The value is the turn (°)
    from the expected direction."""

    PARAMS = {
        "model": "", "expected_angle": 90.0, "from_kpt": 1, "to_kpt": 0,
        "conf": 0.5, "kpt_conf": 0.5, "max_count": 1,
    }
    DEFAULT_LOW = -10.0
    DEFAULT_HIGH = 10.0
    SLOW = True  # a network per run: in the slow-tool thread, like OCR
    READ_IN_LABEL = True
    predictor: Optional[PosePredictor] = None  # set to override (tests)

    def _predictor(self) -> PosePredictor:
        if self.predictor is not None:
            return self.predictor
        path = resolve_model_path(self.model)
        if path is None:
            raise RuntimeError("choose a pose model" if not self.model else f"model not found: {self.model}")
        return get_predictor(path)

    def measure(self, patch: np.ndarray) -> Tuple[List[dict], str]:
        """The parts found in the patch: [{angle, start, end, conf}], and why
        none counts when the list is empty."""
        detections = self._predictor()(patch, float(self.conf))
        if not detections:
            return [], "no part"
        a, b = int(self.from_kpt), int(self.to_kpt)
        parts, reason = [], "keypoints unsure"
        for box_conf, kpts in detections:
            if max(a, b) >= len(kpts) or min(a, b) < 0:
                return [], f"model has {len(kpts)} keypoints: check From kpt / To kpt"
            start, end = kpts[a], kpts[b]
            if min(start[2], end[2]) < float(self.kpt_conf):
                continue
            s, e = (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))
            if s == e:
                continue
            parts.append({"angle": direction_angle(s, e), "start": s, "end": e,
                          "conf": box_conf, "kpt_conf": float(min(start[2], end[2]))})
        return parts, reason

    def teach(self, frame: np.ndarray, offset: Optional[Offset] = None, **kw: Any) -> None:
        """Sets the expected direction to the part's direction in this frame."""
        patch, _ = self.crop(frame, offset)
        parts, reason = self.measure(patch)
        if not parts:
            raise RuntimeError(reason)
        self.expected_angle = round(parts[0]["angle"], 1)

    def process(self, patch, origin):
        try:
            parts, reason = self.measure(patch)
        except RuntimeError as exc:
            return self.fail(str(exc))
        if not parts:
            return self.fail(reason)
        expected = float(self.expected_angle)
        for part in parts:
            part["deviation"] = angle_difference(part["angle"], expected)
        worst = max(parts, key=lambda p: abs(p["deviation"]))
        too_many = int(self.max_count) > 0 and len(parts) > int(self.max_count)
        ok = self.judge(worst["deviation"]) and not too_many
        overlay = []
        for part in parts:
            part_ok = self.judge(part["deviation"]) and not too_many
            (sx, sy), (ex, ey) = part["start"], part["end"]
            overlay.append(overlay_item("line", [sx, sy, ex, ey], part_ok))
            overlay.append(overlay_item("circle", [ex, ey, 6], part_ok))  # the head end
            overlay.append(overlay_item("circle", [sx, sy, 3], part_ok))
        text = f"{worst['angle']:.1f}°"
        if too_many:
            text += f" ({len(parts)} parts)"
        extra = {"angle": worst["angle"], "deviation": worst["deviation"], "count": len(parts),
                 "angles": [round(p["angle"], 2) for p in parts], "reason": "too many parts" if too_many else ""}
        return self.result(worst["deviation"], overlay=overlay, text=text, extra=extra, ok=ok)
