"""Synthetic glass markings (made-up makers, real kinds of standard marks).

One fixed marking to test the OCR tool, with a JSON of each text line and its
box in the picture:

    .venv/bin/python tools/make_synthetic_mark.py [OUT.png] [--tilt DEG] [--blur K] [--seed N]

A YOLO dataset for the standard-mark model (classes: STANDARD_CLASSES), in the
Train tab's layout (images/, labels/*.txt, classes.txt) plus truth.json with
the marks present and left out of each image:

    .venv/bin/python tools/make_synthetic_mark.py --dataset datasets/synthetic_marks --count 500 [--seed 1]

Like a camera photo: grey glass with uneven light, the print tilted, warped,
blurred and noisy. Every maker, name and number is invented; the logos are
simple drawings of the kind of mark, not the official artwork, so add real
labelled photos before relying on the model.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.marking.classes import STANDARD_CLASSES, write_classes_file  # noqa: E402
from app.yolo_labels import write_yolo_label_file  # noqa: E402

W, H = 1280, 720
INK = 45
FONT = cv2.FONT_HERSHEY_DUPLEX

# (key, text, left x, baseline y, font scale, thickness)
LINES = [
    ("maker", "ACME GLASS", 330, 150, 2.2, 4),
    ("glass_maker", "NOVA SAFETY GLASS", 300, 215, 1.2, 2),
    ("e_mark", "43R-001234", 470, 300, 1.4, 3),
    ("dot", "DOT-123 M456 AS1", 300, 375, 1.4, 3),
    ("tis", "TIS 2602-2556", 470, 455, 1.4, 3),
    ("laminated", "LAMINATED AS1", 470, 525, 1.2, 2),
    ("lot", "LOT 24A0915", 300, 605, 1.2, 2),
]


def _glass(rng: np.random.Generator) -> np.ndarray:
    """Grey background, brighter to the left, with a dark frame edge on the right."""
    x = np.linspace(0, 1, W)[None, :]
    y = np.linspace(0, 1, H)[:, None]
    bg = 200 - 35 * x - 15 * y
    bg[:, int(W * 0.82):] = 70  # black ceramic band
    return np.clip(bg + rng.normal(0, 2, (H, W)), 0, 255)


def _draw_marks(img: np.ndarray) -> None:
    """The symbols around the text: E-mark circle and a TIS-style circle."""
    cv2.circle(img, (400, 288), 42, INK, 3, cv2.LINE_AA)
    cv2.putText(img, "E6", (370, 304), FONT, 1.2, INK, 3, cv2.LINE_AA)
    cv2.circle(img, (400, 445), 42, INK, 3, cv2.LINE_AA)
    pts = [(400 + 30 * math.cos(math.radians(a)), 445 + 30 * math.sin(math.radians(a))) for a in range(-90, 270, 72)]
    star = [pts[i % 5] for i in (0, 2, 4, 1, 3)]
    cv2.polylines(img, [np.int32(star)], True, INK, 3, cv2.LINE_AA)
    cv2.putText(img, "//", (305, 300), FONT, 1.2, INK, 3, cv2.LINE_AA)


def make(tilt_deg: float = -3.0, blur: int = 3, seed: int = 1):
    rng = np.random.default_rng(seed)
    img = _glass(rng)
    print_layer = np.full((H, W), 255, np.uint8)
    _draw_marks(print_layer)
    boxes = {}
    for key, text, x, y, scale, thick in LINES:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        cv2.putText(print_layer, text, (x, y), FONT, scale, INK, thick, cv2.LINE_AA)
        boxes[key] = (text, [x, y - th, x + tw, y + base])

    # Tilt the print, then blend it onto the glass so the light stays uneven.
    m = cv2.getRotationMatrix2D((W / 2, H / 2), tilt_deg, 1.0)
    print_layer = cv2.warpAffine(print_layer, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=255).astype(np.float64)
    ink = 1.0 - print_layer / 255.0
    img = img * (1 - 0.78 * ink)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur / 2.0)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)

    lines = []
    for key, (text, (x1, y1, x2, y2)) in boxes.items():
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], np.float64) @ m.T
        bx1, by1 = corners.min(0)
        bx2, by2 = corners.max(0)
        lines.append({"key": key, "text": text,
                      "box_px": [int(bx1), int(by1), int(math.ceil(bx2)), int(math.ceil(by2))]})
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), {"size": [W, H], "tilt_deg": tilt_deg, "lines": lines}


# --- Standard-mark dataset ---------------------------------------------
# Each mark is drawn black (0) on its own white (255) canvas at a base size
# (capitals ~22 px), so its box is the canvas; the canvases are laid out in
# rows, scaled, warped like a camera view and blended onto glass.

FONTS = (cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX)
BACKGROUND_SHARE = 0.05  # images with no mark (empty label file)
MIN_INSIDE = 0.6         # a mark less than this much inside the frame is re-placed
BASE_CHAR_PX = 22


class Pen:
    """Text and shapes for one image: one font and stroke for all its marks."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.font = FONTS[rng.integers(len(FONTS))]
        self.thick = int(rng.integers(2, 4))

    def text(self, text: str, scale: float = 1.0, thick: int = 0) -> np.ndarray:
        thick = thick or self.thick
        (w, h), base = cv2.getTextSize(text, self.font, scale, thick)
        pad = thick + 1
        canvas = np.full((h + base + 2 * pad, w + 2 * pad), 255, np.uint8)
        cv2.putText(canvas, text, (pad, pad + h), self.font, scale, 0, thick, cv2.LINE_AA)
        return _trim(canvas)

    def digits(self, n: int) -> str:
        return "".join(str(d) for d in self.rng.integers(0, 10, n))


def _trim(canvas: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.nonzero(canvas < 128)
    if len(xs) == 0:
        return canvas
    c = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def _hcat(parts: List[np.ndarray], gap: int = 10) -> np.ndarray:
    h = max(p.shape[0] for p in parts)
    row = [cv2.copyMakeBorder(p, (h - p.shape[0]) // 2, h - p.shape[0] - (h - p.shape[0]) // 2, 0, 0,
                              cv2.BORDER_CONSTANT, value=255) for p in parts]
    spacer = np.full((h, gap), 255, np.uint8)
    out = row[0]
    for p in row[1:]:
        out = np.hstack([out, spacer, p])
    return out


def _vcat(parts: List[np.ndarray], gap: int = 4) -> np.ndarray:
    w = max(p.shape[1] for p in parts)
    col = [cv2.copyMakeBorder(p, 0, 0, (w - p.shape[1]) // 2, w - p.shape[1] - (w - p.shape[1]) // 2,
                              cv2.BORDER_CONSTANT, value=255) for p in parts]
    spacer = np.full((gap, w), 255, np.uint8)
    out = col[0]
    for p in col[1:]:
        out = np.vstack([out, spacer, p])
    return out


def _framed(inner: np.ndarray, shape: str, thick: int, pad: int = 8) -> np.ndarray:
    """`inner` inside a circle, ellipse, box or rounded box."""
    h, w = inner.shape
    if shape == "circle":
        side = int(math.hypot(w, h)) + 2 * pad
        canvas = np.full((side, side), 255, np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = inner
        cv2.circle(canvas, (side // 2, side // 2), side // 2 - thick, 0, thick, cv2.LINE_AA)
        return canvas
    cw, ch = w + 2 * pad + (int(0.4 * w) if shape == "ellipse" else 0), h + 2 * pad + (int(0.5 * h) if shape == "ellipse" else 0)
    canvas = np.full((ch, cw), 255, np.uint8)
    y0, x0 = (ch - h) // 2, (cw - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = inner
    t = thick
    if shape == "ellipse":
        cv2.ellipse(canvas, (cw // 2, ch // 2), (cw // 2 - t, ch // 2 - t), 0, 0, 360, 0, t, cv2.LINE_AA)
    elif shape == "rounded":
        r = min(ch, cw) // 4
        for a, b in (((r, t), (cw - r, t)), ((r, ch - 1 - t), (cw - r, ch - 1 - t)),
                     ((t, r), (t, ch - r)), ((cw - 1 - t, r), (cw - 1 - t, ch - r))):
            cv2.line(canvas, a, b, 0, t, cv2.LINE_AA)
        for cx, cy, ang in ((r, r, 180), (cw - r, r, 270), (cw - r, ch - r, 0), (r, ch - r, 90)):
            cv2.ellipse(canvas, (cx, cy), (r - t, r - t), ang, 0, 90, 0, t, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (t, t), (cw - 1 - t, ch - 1 - t), 0, t, cv2.LINE_AA)
    return canvas


def _emblem(pen: Pen, size: int = 44) -> np.ndarray:
    """A round emblem with a four-arrow sign inside (TIS-like)."""
    c = np.full((size, size), 255, np.uint8)
    m, t = size // 2, pen.thick
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        tip = (m + dx * (m - 6), m + dy * (m - 6))
        cv2.arrowedLine(c, (m, m), tip, 0, t, cv2.LINE_AA, tipLength=0.35)
    return _framed(c, "circle", t, pad=2)


def mark_e(pen):
    inner = pen.text(f"E{int(pen.rng.choice([1, 2, 4, 6, 11, 43]))}", 0.9)
    return _hcat([_framed(inner, "circle", pen.thick, 4), pen.text(f"43R-00{pen.digits(4)}")], 12)


def mark_dot(pen):
    return pen.text(f"DOT-{pen.digits(3)} M{pen.digits(3)}")


def mark_as(pen):
    return pen.text(f"AS{int(pen.rng.integers(1, 4))}")


def mark_ccc(pen):
    return _hcat([_framed(pen.text("CCC", 0.9), "ellipse", pen.thick, 4), pen.text(f"E000{pen.digits(3)}")], 10)


def mark_tis(pen):
    return _hcat([_emblem(pen), pen.text(f"TIS {pen.digits(4)}-{pen.digits(4)}")], 12)


def mark_sni(pen):
    code = f"{'ABCDEFGHJKLMNPRSTUVWXY'[pen.rng.integers(22)]}{pen.rng.integers(1, 10)}"
    return _hcat([pen.text(code), _framed(pen.text("SNI", 0.9), "box", pen.thick, 4)], 10)


def mark_inmetro(pen):
    return _framed(pen.text("INMETRO", 0.8), "rounded", pen.thick, 6)


def mark_asnzs(pen):
    return pen.text("AS/NZS 2080")


def mark_bis(pen):
    isi = _framed(pen.text("ISI", 0.9), "box", pen.thick, 4)
    return _hcat([_vcat([isi, pen.text("IS 2553", 0.6, 1)]), pen.text(f"CM/L-{pen.digits(7)}", 0.8)], 10)


def mark_laminated(pen):
    return pen.text("LAMINATED")


def mark_glass_class(pen):
    return pen.text(str(pen.rng.choice(["//", "II", "V", "V-VI"])))


def mark_date(pen):
    """Digit with dots on either side (dot-pattern date code)."""
    left, right = int(pen.rng.integers(0, 4)), int(pen.rng.integers(0, 4))
    digit = pen.text(str(pen.rng.integers(0, 10)))
    h = digit.shape[0]
    dot = np.full((h, h // 2), 255, np.uint8)
    cv2.circle(dot, (h // 4, h // 2), max(2, h // 7), 0, -1, cv2.LINE_AA)
    parts = [dot] * left + [digit] + [dot] * right
    return _hcat(parts, 4) if len(parts) > 1 else _hcat([dot, digit], 4)


def logo_acme(pen):
    word = pen.text("ACME", 1.6, pen.thick + 2)
    w = word.shape[1]
    swoosh = np.full((14, w), 255, np.uint8)
    cv2.ellipse(swoosh, (w // 2, -10), (w // 2 - 2, 20), 0, 20, 160, 0, 3, cv2.LINE_AA)
    return _vcat([word, swoosh], 2)


def logo_nova(pen):
    word = pen.text("NOVA", 1.3, pen.thick + 1)
    h = word.shape[0]
    star = np.full((h, h), 255, np.uint8)
    pts = [(h / 2 + (h / 2 - 2) * math.cos(math.radians(a)), h / 2 + (h / 2 - 2) * math.sin(math.radians(a)))
           for a in range(-90, 270, 72)]
    cv2.fillPoly(star, [np.int32([pts[i % 5] for i in (0, 2, 4, 1, 3)])], 0, cv2.LINE_AA)
    return _hcat([star, word], 6)


def logo_orion(pen):
    word = pen.text("ORION", 1.2, pen.thick + 1)
    h = word.shape[0]
    belt = np.full((h, h), 255, np.uint8)
    cv2.circle(belt, (h // 2, h // 2), h // 2 - 2, 0, 2, cv2.LINE_AA)
    for i in range(3):
        cv2.circle(belt, (h // 4 + i * h // 4, h // 4 + i * h // 4), max(2, h // 10), 0, -1, cv2.LINE_AA)
    return _hcat([belt, word], 6)


MARKS: Dict[str, Callable[[Pen], np.ndarray]] = {
    "e_mark": mark_e, "dot": mark_dot, "as_class": mark_as, "ccc_mark": mark_ccc,
    "tis_mark": mark_tis, "sni_mark": mark_sni, "inmetro": mark_inmetro, "asnzs": mark_asnzs,
    "bis_isi": mark_bis, "laminated": mark_laminated, "glass_class": mark_glass_class,
    "date_code": mark_date, "logo_acme": logo_acme, "logo_nova": logo_nova, "logo_orion": logo_orion,
}
assert set(MARKS) == set(STANDARD_CLASSES)
LOGOS = [c for c in STANDARD_CLASSES if c.startswith("logo_")]


def pick_marks(rng: np.random.Generator) -> List[str]:
    """One maker logo (mostly) and a random set of standards, at least two marks."""
    if rng.random() < BACKGROUND_SHARE:
        return []
    while True:
        marks = [LOGOS[rng.integers(len(LOGOS))]] if rng.random() < 0.95 else []
        for c in STANDARD_CLASSES:
            if c in LOGOS:
                continue
            if rng.random() < (0.9 if c == "e_mark" else rng.uniform(0.5, 0.75)):
                marks.append(c)
        if len(marks) >= 2:
            return marks


def _layout(canvases: List[Tuple[str, np.ndarray]], rng: np.random.Generator, max_w: int):
    """Rows of marks (logo first, the rest shuffled); returns the block and
    each mark's (x1, y1, x2, y2) in it."""
    order = [c for c in canvases if c[0] in LOGOS] + [canvases[i] for i in rng.permutation(len(canvases))
                                                     if canvases[i][0] not in LOGOS]
    rows, row, width = [], [], 0
    for item in order:
        w = item[1].shape[1]
        if row and (width + w > max_w or item[0] in LOGOS or row[0][0] in LOGOS):
            rows.append(row)
            row, width = [], 0
        row.append(item)
        width += w + 30
    if row:
        rows.append(row)
    centred = rng.random() < 0.5
    gap_x, gap_y = int(rng.integers(20, 45)), int(rng.integers(10, 28))
    row_sizes = [(sum(c.shape[1] for _, c in r) + gap_x * (len(r) - 1), max(c.shape[0] for _, c in r)) for r in rows]
    bw = max(w for w, _ in row_sizes)
    bh = sum(h for _, h in row_sizes) + gap_y * (len(rows) - 1)
    block = np.full((bh, bw), 255, np.uint8)
    boxes, y = [], 0
    for r, (rw, rh) in zip(rows, row_sizes):
        x = (bw - rw) // 2 if centred else 0
        for key, c in r:
            h, w = c.shape
            yy = y + (rh - h) // 2
            block[yy:yy + h, x:x + w] = np.minimum(block[yy:yy + h, x:x + w], c)
            boxes.append((key, (x, yy, x + w, yy + h)))
            x += w + gap_x
        y += rh + gap_y
    return block, boxes


def _glass_bg(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Grey glass with light falling off in a random direction."""
    ang = rng.uniform(0, 2 * math.pi)
    x = np.linspace(-0.5, 0.5, w)[None, :]
    y = np.linspace(-0.5, 0.5, h)[:, None]
    ramp = math.cos(ang) * x + math.sin(ang) * y
    base = rng.uniform(150, 215)
    return base + rng.uniform(20, 60) * ramp + rng.normal(0, 2, (h, w))


def _camera_matrix(rng: np.random.Generator, w: int, h: int, center: Tuple[float, float]) -> np.ndarray:
    """Tilt about the block centre, then a slight perspective (3x3)."""
    rot = np.vstack([cv2.getRotationMatrix2D(center, rng.uniform(-8, 8), 1.0), [0, 0, 1]])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.03, 0.03, (4, 2)) * [w, h]
    persp = cv2.getPerspectiveTransform(src, np.float32(src + jitter))
    return persp @ rot


def make_dataset_image(rng: np.random.Generator, size: Tuple[int, int] = (1280, 720)):
    """(BGR image, [(class, (x1, y1, x2, y2))] in pixels, marks left out)."""
    w, h = size
    marks = pick_marks(rng)
    bg = _glass_bg(rng, w, h)
    boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
    layer = np.full((h, w), 255, np.uint8)
    if marks:
        pen = Pen(rng)
        block, block_boxes = _layout([(m, MARKS[m](pen)) for m in marks], rng, max_w=int(rng.uniform(450, 800)))
        scale = rng.uniform(12, 40) / BASE_CHAR_PX
        scale = min(scale, 0.85 * w / block.shape[1], 0.85 * h / block.shape[0])
        block = cv2.resize(block, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        bh, bw = block.shape
        for _ in range(20):
            x0, y0 = int(rng.integers(0, w - bw + 1)), int(rng.integers(0, h - bh + 1))
            m = _camera_matrix(rng, w, h, (x0 + bw / 2, y0 + bh / 2))
            placed = []
            for key, (x1, y1, x2, y2) in block_boxes:
                pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) * scale + [x0, y0]
                q = cv2.perspectiveTransform(pts[None], m)[0]
                (qx1, qy1), (qx2, qy2) = q.min(0), q.max(0)
                cx1, cy1, cx2, cy2 = max(qx1, 0), max(qy1, 0), min(qx2, w), min(qy2, h)
                inside = max(0, cx2 - cx1) * max(0, cy2 - cy1) / max(1e-6, (qx2 - qx1) * (qy2 - qy1))
                placed.append((key, (cx1, cy1, cx2, cy2), inside))
            if all(p[2] >= MIN_INSIDE for p in placed):
                break
        layer[y0:y0 + bh, x0:x0 + bw] = block
        layer = cv2.warpPerspective(layer, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        boxes = [(k, b) for k, b, inside in placed if inside >= MIN_INSIDE]
        if rng.random() < 0.4:  # dark ceramic band on the side away from the print
            xs = [b[0] for _, b in boxes] + [b[2] for _, b in boxes]
            if min(xs) > 0.25 * w:
                bg[:, :int(rng.uniform(0.05, min(xs) / w - 0.05) * w)] = rng.uniform(40, 80)
            elif max(xs) < 0.75 * w:
                bg[:, int(rng.uniform(max(xs) / w + 0.05, 0.95) * w):] = rng.uniform(40, 80)
    ink = 1.0 - layer.astype(np.float64) / 255.0
    img = bg * (1 - rng.uniform(0.55, 0.85) * ink)
    sigma = rng.uniform(0, 2.0)
    if sigma > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = np.clip(img + rng.normal(0, rng.uniform(2, 6), img.shape), 0, 255).astype(np.uint8)
    present = {k for k, _ in boxes}
    absent = [c for c in STANDARD_CLASSES if c not in present and c not in LOGOS]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), boxes, absent


def make_dataset(root: Path, count: int, seed: int = 1, size: Tuple[int, int] = (1280, 720)) -> List[dict]:
    """Writes images/, labels/, classes.txt and truth.json under `root`."""
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    write_classes_file(root, STANDARD_CLASSES)
    w, h = size
    truth = []
    for i in range(1, count + 1):
        img, boxes, absent = make_dataset_image(rng, size)
        stem = f"mark_{i:04d}"
        cv2.imwrite(str(root / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
        write_yolo_label_file(root / "labels" / f"{stem}.txt", [
            {"cls_id": STANDARD_CLASSES.index(k), "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h,
             "w": (x2 - x1) / w, "h": (y2 - y1) / h} for k, (x1, y1, x2, y2) in boxes])
        truth.append({"image": f"images/{stem}.jpg", "present": sorted({k for k, _ in boxes}), "absent": absent})
    (root / "truth.json").write_text(json.dumps(truth, indent=1))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="samples/synthetic_mark.png")
    ap.add_argument("--tilt", type=float, default=-3.0)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dataset", type=Path, help="write a YOLO dataset to this folder instead")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--size", default="1280x720")
    args = ap.parse_args()
    if args.dataset:
        size = tuple(int(v) for v in args.size.split("x"))
        truth = make_dataset(args.dataset, args.count, args.seed, size)
        print(f"wrote {len(truth)} images to {args.dataset} ({len(STANDARD_CLASSES)} classes)")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img, truth = make(args.tilt, args.blur, args.seed)
    cv2.imwrite(str(out), img)
    out.with_suffix(".json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
