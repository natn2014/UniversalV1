"""Part Check (AI seg): which side of the part faces the camera, and how many
small parts sit on it, from a YOLO segmentation model.

Made for the D92 quarter glass (classes glass_front / glass_back / sponge):
the best glass mask tells the side, and the clip-seal sponges whose centre
lies on that glass are counted. Sponges elsewhere in the area (the parts bin
next to the station) are ignored.

The value is the item count, judged against low / high (default 3..3). The
side must match `expected_side` ("back", "front" or "any"); on the front side
the items cannot be seen, so the count is not judged there.

With `both_sides` (default on) the part is OK only once both sides were seen
in turn: the front, and the back with the right count (either order). After
one good side the result waits (STANDBY, "turn over"); a back with a wrong
count is NG at once. `expected_side` is not used then.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .base import Tool, overlay_item, register
from .orientation import resolve_model_path

# (class name, confidence, polygon (N, 2) float32 in patch pixels)
Detection = Tuple[str, float, np.ndarray]

SIDES = ["back", "front", "any"]


class SegPredictor(Protocol):
    def __call__(self, image: np.ndarray, conf: float) -> List[Detection]:
        """Detections in `image` pixels, best first."""


class YoloSegPredictor:
    """An Ultralytics segmentation model; calls are serialised (the slow-tool
    thread and the Tools tab's frozen-frame run may overlap)."""

    def __init__(self, model, imgsz: Optional[int] = None) -> None:
        self._model = model
        self._imgsz = imgsz
        self._lock = threading.Lock()

    def __call__(self, image: np.ndarray, conf: float) -> List[Detection]:
        kw: Dict[str, Any] = {"conf": conf, "verbose": False, "retina_masks": True}
        if self._imgsz:
            kw["imgsz"] = self._imgsz
        with self._lock:
            results = self._model.predict(image, **kw)
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return out
        names = r.names
        cls = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        polys = r.masks.xy  # list of (N, 2) arrays in original image pixels
        for i in np.argsort(-confs):
            poly = np.asarray(polys[i], np.float32)
            if len(poly) < 3:
                continue
            out.append((str(names[int(cls[i])]), float(confs[i]), poly))
        return out


_MODELS: Dict[str, YoloSegPredictor] = {}
_MODELS_LOCK = threading.Lock()


def _train_imgsz(model) -> Optional[int]:
    """The image size the model was trained at (small parts need it)."""
    try:
        size = model.overrides.get("imgsz") or model.ckpt.get("train_args", {}).get("imgsz")
    except Exception:
        return None
    if isinstance(size, (list, tuple)):
        size = max(size)
    return int(size) if size else None


def get_predictor(path) -> YoloSegPredictor:
    """One loaded model per file, shared by every Part Check tool."""
    key = str(path.resolve())
    with _MODELS_LOCK:
        if key not in _MODELS:
            from ..optional_deps import YOLO
            if YOLO is None:
                raise RuntimeError("Ultralytics not installed")
            model = YOLO(key, task="segment")
            if getattr(model, "task", "segment") != "segment":
                raise RuntimeError(f"{path.name} is not a segmentation model")
            _MODELS[key] = YoloSegPredictor(model, _train_imgsz(model))
        return _MODELS[key]


def clear_models() -> None:
    with _MODELS_LOCK:
        _MODELS.clear()


def poly_centroid(poly: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(poly.astype(np.float32))
    if abs(m["m00"]) < 1e-6:
        x, y = poly.mean(axis=0)
        return float(x), float(y)
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def inside(poly: np.ndarray, point: Tuple[float, float], margin: float) -> bool:
    """`point` is inside `poly`, or within `margin` pixels outside it."""
    return cv2.pointPolygonTest(poly.astype(np.float32), point, True) >= -margin


@register("segment")
class SegmentTool(Tool):
    """Its ROI is where the part is looked for. The value is the number of
    items (sponges) found on the part."""

    PARAMS = {
        "model": "", "conf": 0.5, "expected_side": "back",
        "front_class": "glass_front", "back_class": "glass_back", "item_class": "sponge",
        "item_conf": 0.4, "both_sides": True, "reset_s": 3.0,
    }
    DEFAULT_LOW = 3.0
    DEFAULT_HIGH = 3.0
    SLOW = True  # a network per run: in the slow-tool thread, like OCR
    predictor: Optional[SegPredictor] = None  # set to override (tests)

    def _predictor(self) -> SegPredictor:
        if self.predictor is not None:
            return self.predictor
        path = resolve_model_path(self.model)
        if path is None:
            raise RuntimeError("choose a segmentation model" if not self.model else f"model not found: {self.model}")
        return get_predictor(path)

    def measure(self, patch: np.ndarray) -> Dict[str, Any]:
        """{side, glass (poly, conf), items [(poly, conf, centre)], outside
        (items not on the glass)}; side is None when no part is found."""
        conf = min(float(self.conf), float(self.item_conf))
        detections = self._predictor()(patch, conf)
        glasses = [d for d in detections
                   if d[0] in (self.front_class, self.back_class) and d[1] >= float(self.conf)]
        if not glasses:
            return {"side": None, "glass": None, "items": [], "outside": []}
        name, gconf, gpoly = max(glasses, key=lambda d: d[1])
        side = "front" if name == self.front_class else "back"
        # The convex outline plus a margin: a seg mask is coarse at the thin
        # corners where the clips sit, so a sponge's centre can land just outside it.
        hull = cv2.convexHull(gpoly.astype(np.float32)).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(hull)
        margin = 0.04 * float(np.hypot(w, h))
        items, outside = [], []
        for cls, c, poly in detections:
            if cls != self.item_class or c < float(self.item_conf):
                continue
            centre = poly_centroid(poly)
            (items if inside(hull, centre, margin) else outside).append((poly, c, centre))
        return {"side": side, "glass": (gpoly, gconf), "items": items, "outside": outside}

    # --- Both sides of one part ------------------------------------------
    # With `both_sides`, one part is OK only after its front AND its back
    # (with the right item count) were seen. Until then the result "waits"
    # (STANDBY, not NG). The memory clears when the part leaves the area:
    # at once after it was OK, else after `reset_s` seconds without a part.

    def _sides(self) -> Dict[str, Any]:
        if not hasattr(self, "_side_state"):
            self.reset_sides()
        return self._side_state

    def reset_sides(self) -> None:
        # back_count: items found on this part's back (None = back not seen yet)
        self._side_state = {"front": False, "back": False, "back_count": None, "done": False, "last": 0.0}

    def _want(self, default: int) -> int:
        """How many items the back must have (the High limit, else Low)."""
        return int(self.high) if self.high is not None else (int(self.low) if self.low is not None else default)

    def _no_part(self, now: float) -> None:
        s = self._sides()
        if s["done"] or now - s["last"] > float(self.reset_s):
            self.reset_sides()

    def process(self, patch, origin):
        now = time.monotonic()
        try:
            m = self.measure(patch)
        except RuntimeError as exc:
            return self.fail(str(exc))
        if m["side"] is None:
            res = self.fail("no part")
            if self.both_sides:
                self._no_part(now)
                st = self._sides()
                want = self._want(0)
                bc = st["back_count"]
                started = st["front"] or bc is not None
                res.extra["checklist"] = {
                    "front": True if st["front"] else None,
                    "back": None if bc is None else True,
                    "items": [None if bc is None else bc >= k for k in range(1, want + 1)],
                    "extra_items": 0 if bc is None else max(0, bc - want),
                    "want": want, "verdict": "WAIT" if started else "NONE", "next": "",
                }
            return res
        side, (gpoly, gconf), items = m["side"], m["glass"], m["items"]
        n = len(items)
        count_ok = side == "front" or self.judge(n)
        waiting = False
        want = self._want(n)
        if side == "front":
            text = "FRONT"
        else:
            text = f"BACK {n}/{want}"
        if self.both_sides:
            s = self._sides()
            s["last"] = now
            side_ok = True
            if side == "back":
                s["back_count"] = n
            if count_ok:
                s[side] = True
            else:
                s["back"] = False  # a back with a missing item must be fixed and shown again
            done = s["front"] and s["back"]
            s["done"] = done and count_ok
            ok = done and count_ok
            if count_ok:
                text += " ✓"
                missing = [k.upper() for k in ("front", "back") if not s[k]]
                if missing:
                    waiting = True
                    text += " · turn over: " + missing[0]
                else:
                    text = "FRONT ✓ BACK ✓"
        else:
            side_ok = self.expected_side == "any" or side == self.expected_side
            ok = side_ok and count_ok
            if not side_ok:
                text += f" (want {self.expected_side})"
        overlay = [{**overlay_item("poly", gpoly.tolist(), ok or waiting, f"{side} {gconf:.2f}"), "fill": "green"}]
        for poly, c, (cx, cy) in items:
            overlay.append({**overlay_item("poly", poly.tolist(), count_ok), "fill": "blue"})
            overlay.append(overlay_item("circle", [cx, cy, 6], count_ok))
        if self.both_sides:
            seen_front, back_count = s["front"], s["back_count"]
        else:
            seen_front, back_count = side == "front", (n if side == "back" else None)
        checklist = {
            # True = checked and right, False = checked and wrong, None = not checked yet
            "front": True if seen_front else (False if not side_ok and side == "front" else None),
            "back": None if back_count is None else (side_ok or side == "front"),
            "items": [None if back_count is None else back_count >= k for k in range(1, want + 1)],
            "extra_items": 0 if back_count is None else max(0, back_count - want),
            "want": want,
            "verdict": "PASS" if ok else ("WAIT" if waiting else "NG"),
            "next": text.split("turn over: ")[1] if "turn over: " in text else "",
        }
        extra = {"side": side, "side_conf": gconf, "count": n, "waiting": waiting, "checklist": checklist,
                 "item_confs": [round(c, 3) for _, c, _ in items], "outside": len(m["outside"]),
                 "reason": "" if ok else ("wrong side" if not side_ok else "item count" if not count_ok
                                           else "other side not checked yet")}
        return self.result(n, overlay=overlay, text=text, extra=extra, ok=ok)
