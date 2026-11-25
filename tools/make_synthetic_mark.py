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
BACKGROUND_SHARE = 0.1   # images with no mark (empty label file), often with unlabelled print
MIN_INSIDE = 0.6         # a mark less than this much inside the frame is re-placed
BASE_CHAR_PX = 22
# Bold sans like real glass print; marks fall back to Hershey when none exist.
TTF_BOLD = [f for f in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
) if Path(f).exists()]
TTF_SHARE = 0.6          # images whose text is a TTF font instead of Hershey
_ttf_cache: Dict[Tuple[str, int], object] = {}


def _ttf_text(path: str, text: str, cap_px: float, narrow: float = 1.0) -> np.ndarray:
    """Black `text` on white, capitals about `cap_px` tall, x squeezed by `narrow`."""
    from PIL import Image, ImageDraw, ImageFont
    size = max(6, int(round(cap_px / 0.73)))
    font = _ttf_cache.get((path, size))
    if font is None:
        font = _ttf_cache[(path, size)] = ImageFont.truetype(path, size)
    x0, y0, x1, y1 = font.getbbox(text)
    im = Image.new("L", (x1 - x0 + 8, y1 - y0 + 8), 255)
    ImageDraw.Draw(im).text((4 - x0, 4 - y0), text, font=font, fill=0)
    a = np.array(im)
    if narrow != 1.0:
        a = cv2.resize(a, (max(1, int(a.shape[1] * narrow)), a.shape[0]), interpolation=cv2.INTER_AREA)
    return _trim(a)


class Pen:
    """Text and shapes for one image: one font and stroke for all its marks."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.font = FONTS[rng.integers(len(FONTS))]
        self.thick = int(rng.integers(2, 4))
        self.ttf = TTF_BOLD[rng.integers(len(TTF_BOLD))] if TTF_BOLD and rng.random() < TTF_SHARE else None
        self.bold_ttf = TTF_BOLD[rng.integers(len(TTF_BOLD))] if TTF_BOLD else None

    def text(self, text: str, scale: float = 1.0, thick: int = 0) -> np.ndarray:
        if self.ttf:
            return _ttf_text(self.ttf, text, BASE_CHAR_PX * scale)
        thick = thick or self.thick
        (w, h), base = cv2.getTextSize(text, self.font, scale, thick)
        pad = thick + 1
        canvas = np.full((h + base + 2 * pad, w + 2 * pad), 255, np.uint8)
        cv2.putText(canvas, text, (pad, pad + h), self.font, scale, 0, thick, cv2.LINE_AA)
        return _trim(canvas)

    def bold(self, text: str, scale: float = 1.0, narrow: float = 1.0) -> np.ndarray:
        """Heavy text for logo letters (E of the E-mark, SNI)."""
        if self.bold_ttf:
            return _ttf_text(self.bold_ttf, text, BASE_CHAR_PX * scale, narrow)
        t = self.text(text, scale, self.thick + 2)
        return cv2.resize(t, (max(1, int(t.shape[1] * narrow)), t.shape[0])) if narrow != 1.0 else t

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
    pad = max(pad, 2 * thick + 3, h // 5)  # keep bold letters clear of the frame
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
    """TIS-like emblem: a ring around an up/down arrow between two split
    triangles pointing left and right."""
    k = 4
    S = size * k
    c = np.full((S, S), 255, np.uint8)
    m = S // 2
    ring = int(S * pen.rng.uniform(0.045, 0.07))
    cv2.circle(c, (m, m), m - ring, 0, ring, cv2.LINE_AA)
    r = m - 2 * ring - int(S * 0.06)          # reach of the sign inside the ring
    shaft, head = int(S * 0.035), int(r * 0.32)
    cv2.rectangle(c, (m - shaft, m - r + head), (m + shaft, m + r - head), 0, -1)
    for sgn in (-1, 1):                        # arrow heads up and down
        tip = m + sgn * r
        cv2.fillPoly(c, [np.int32([(m, tip), (m - head * 0.75, tip - sgn * head), (m + head * 0.75, tip - sgn * head)])], 0, cv2.LINE_AA)
    gap, half = int(S * 0.025), int(r * 0.72)
    for sgn in (-1, 1):                        # left and right triangles, split by a white bar
        tip, base = m + sgn * r, m + sgn * int(S * 0.09)
        cv2.fillPoly(c, [np.int32([(tip, m), (base, m - half), (base, m + half)])], 0, cv2.LINE_AA)
        cv2.rectangle(c, (min(tip, base) - 2, m - gap), (max(tip, base) + 2, m + gap), 255, -1)
        notch = base + sgn * int(r * 0.07)
        cv2.fillPoly(c, [np.int32([(base, m - half), (notch, m), (base, m + half)])], 255, cv2.LINE_AA)
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)


def _circled(inner: np.ndarray, thick: int, pad: int = 3) -> np.ndarray:
    """`inner` in a circle that hugs it (the E-mark ring)."""
    h, w = inner.shape
    side = int(max(w, h) * 1.12) + 2 * (pad + thick)
    canvas = np.full((side, side), 255, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = np.minimum(canvas[y0:y0 + h, x0:x0 + w], inner)
    cv2.circle(canvas, (side // 2, side // 2), side // 2 - thick, 0, thick, cv2.LINE_AA)
    return canvas


def mark_e(pen):
    """ECE R43: big bold E with a smaller country number in a ring, then 43R-00xxxx."""
    n = str(int(pen.rng.choice([1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 43])))
    number = pen.bold(f"43R-00{pen.digits(4)}") if pen.rng.random() < 0.6 else pen.text(f"43R-00{pen.digits(4)}")
    if pen.rng.random() < 0.25:  # older style: E and number the same size
        return _hcat([_framed(pen.text(f"E{n}", 0.9), "circle", pen.thick, 4), number], 12)
    e = pen.bold("E", 1.45)
    num = pen.bold(n, 0.75)
    num = cv2.copyMakeBorder(num, int(e.shape[0] * 0.18), 0, 0, 0, cv2.BORDER_CONSTANT, value=255)
    ring = _circled(_hcat([e, num], 1), max(2, e.shape[0] // 11))
    return _hcat([ring, number], int(pen.rng.integers(8, 20)))


_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


def mark_dot(pen):
    return pen.text(f"DOT-{pen.digits(3)}")


def mark_m_code(pen):
    """Model code: M + 3-4 digits / capitals, sometimes ending in E or S (M874, M8U74E, M1H4)."""
    r = pen.rng
    body = "".join(str(r.integers(0, 10)) if r.random() < 0.7 else _LETTERS[r.integers(24)]
                   for _ in range(int(r.integers(2, 5))))
    return pen.text(f"M{r.integers(1, 10)}{body}{str(r.choice(['', '', 'E', 'S']))}")


def mark_short_name(pen):
    """Two characters (58, U8, Y2); a third start with T (T1, T2) so they are
    not taken for the 11T plant code. Sometimes a frame line runs through it."""
    r = pen.rng
    first = "T" if r.random() < 0.35 else (_LETTERS[r.integers(24)] if r.random() < 0.5 else str(r.integers(0, 10)))
    out = pen.bold(f"{first}{r.integers(0, 10)}", r.uniform(1.1, 1.5))
    if r.random() < 0.3:
        out = out.copy()
        x = int(r.uniform(0.1, 0.9) * out.shape[1])
        cv2.line(out, (x, 0), (x, out.shape[0] - 1), int(r.uniform(90, 170)), max(1, out.shape[0] // 12))
    return out


def mark_type_code(pen):
    r = pen.rng
    return pen.bold(_LETTERS[r.integers(24)] + _LETTERS[r.integers(24)], r.uniform(1.3, 1.8))


def mark_plant_code(pen):
    return pen.text(f"{pen.rng.integers(10, 20)}{_LETTERS[pen.rng.integers(24)] if pen.rng.random() < 0.3 else 'T'}")


def mark_letter_l(pen):
    return pen.bold("L", pen.rng.uniform(1.0, 1.4))


def mark_letter_t(pen):
    return pen.bold("T", pen.rng.uniform(1.0, 1.4))


def mark_as(pen):
    return pen.text(f"AS{int(pen.rng.integers(1, 4))}")


def _ccc_logo(pen, h: int = 44) -> np.ndarray:
    """CCC-like logo: three cascading C arcs (open to the right) in an oval."""
    k = 4
    H, W = h * k, int(h * 1.45) * k
    c = np.full((H, W), 255, np.uint8)
    t = max(k, int(H * pen.rng.uniform(0.035, 0.05)))
    cv2.ellipse(c, (W // 2, H // 2), (W // 2 - t, H // 2 - t), 0, 0, 360, 0, t, cv2.LINE_AA)
    r, d = int(H * 0.24), int(H * 0.16)
    stroke = max(k, int(H * pen.rng.uniform(0.035, 0.05)))
    x0 = W // 2 - d
    for i in range(3):
        start = 55 if i == 2 else 75
        cv2.ellipse(c, (x0 + i * d, H // 2), (r, int(r * 1.05)), 0, start, 360 - start, 0, stroke, cv2.LINE_AA)
    return cv2.resize(c, (W // k, H // k), interpolation=cv2.INTER_AREA)


def mark_ccc(pen):
    code = pen.text(f"E000{pen.digits(3)}")
    if pen.rng.random() < 0.2:  # older drawing: the letters in an oval
        return _hcat([_framed(pen.text("CCC", 0.9), "ellipse", pen.thick, 4), code], 10)
    return _hcat([_ccc_logo(pen, int(code.shape[0] * pen.rng.uniform(1.8, 2.6))), code], int(pen.rng.integers(8, 20)))


def mark_tis(pen):
    text = f"TIS {pen.digits(4)}-{pen.digits(4)}"
    words = pen.bold(text) if pen.rng.random() < 0.5 else pen.text(text)
    size = int(words.shape[0] * pen.rng.uniform(2.2, 3.4))
    return _hcat([_emblem(pen, size), words], int(pen.rng.integers(8, 20)))


def mark_sni(pen):
    """SNI logo: heavy condensed letters with a bar above and below, sometimes
    the older boxed form. The maker code beside it is not part of the box."""
    if pen.rng.random() < 0.2:
        return _framed(pen.text("SNI", 0.9), "box", pen.thick, 4)
    letters = pen.bold("SNI", pen.rng.uniform(1.2, 1.6), narrow=pen.rng.uniform(0.6, 0.8))
    h, w = letters.shape
    bar = max(2, int(h * pen.rng.uniform(0.12, 0.18)))
    rule = np.zeros((bar, w), np.uint8)
    return _vcat([rule, letters, rule], max(2, int(h * 0.12)))


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


def _date_dots(pen):
    """Digit with dots on either side (dot-pattern date code)."""
    left, right = int(pen.rng.integers(0, 4)), int(pen.rng.integers(0, 4))
    digit = pen.text(str(pen.rng.integers(0, 10)))
    h = digit.shape[0]
    dot = np.full((h, h // 2), 255, np.uint8)
    cv2.circle(dot, (h // 4, h // 2), max(2, h // 7), 0, -1, cv2.LINE_AA)
    parts = [dot] * left + [digit] + [dot] * right
    return _hcat(parts, 4) if len(parts) > 1 else _hcat([dot, digit], 4)


def _date_rows(pen):
    """Widely spaced digit rows (month / year code, e.g. 1 2 4 8 over 1 2 4 8)."""
    n = int(pen.rng.integers(3, 7))
    digits = sorted(set(int(d) for d in pen.rng.integers(0, 10, n + 2)))[:n] if pen.rng.random() < 0.5 \
        else [1, 2, 4, 8][:max(3, min(n, 4))]
    gap = int(BASE_CHAR_PX * pen.rng.uniform(0.8, 2.0))
    rows = [_hcat([pen.text(str(d)) for d in digits], gap) for _ in range(int(pen.rng.choice([1, 2, 2])))]
    return _vcat(rows, int(BASE_CHAR_PX * pen.rng.uniform(0.3, 0.8))) if len(rows) > 1 else rows[0]


def mark_date(pen):
    return _date_rows(pen) if pen.rng.random() < 0.6 else _date_dots(pen)


def _spaced(pen, text: str, scale: float = 1.0, bold: bool = False, track: float = 0.18) -> np.ndarray:
    """Letter-spaced capitals (AGC AUTOMOTIVE, LAMISAFE)."""
    draw = pen.bold if bold else pen.text
    h = int(BASE_CHAR_PX * scale)
    words = []
    for word in text.split(" "):
        words.append(_hcat([draw(ch, scale) for ch in word], max(1, int(h * track))))
    return _hcat(words, int(h * (0.6 + track)))


def _stretch(img: np.ndarray, fx: float) -> np.ndarray:
    return cv2.resize(img, (max(1, int(img.shape[1] * fx)), img.shape[0]), interpolation=cv2.INTER_AREA)


def logo_toyota(pen):
    word = pen.bold("TOYOTA", pen.rng.uniform(1.4, 1.9))
    heavy = cv2.erode(word, np.ones((1, max(2, word.shape[0] // 9)), np.uint8))  # extra-heavy strokes
    return _trim(_stretch(heavy, pen.rng.uniform(1.0, 1.2)))


def logo_nissan(pen):
    return _stretch(_spaced(pen, "NISSAN", pen.rng.uniform(1.3, 1.7), bold=True, track=0.08), pen.rng.uniform(1.1, 1.35))


def _honda_solid(h: int) -> np.ndarray:
    """Newer Honda-style H: legs flared at the top, a low crossbar with a curved top."""
    k = 4
    H, W = h * k, int(h * 1.35) * k
    c = np.full((H, W), 255, np.uint8)
    def P(pts):
        return np.int32([(x * W, y * H) for x, y in pts])
    left = [(0.0, 0.0), (0.24, 0.0), (0.30, 1.0), (0.12, 1.0)]
    cv2.fillPoly(c, [P(left)], 0, cv2.LINE_AA)
    cv2.fillPoly(c, [P([(1 - x, y) for x, y in left])], 0, cv2.LINE_AA)
    top = [(x, 0.64 - 0.10 * ((x - 0.5) / 0.25) ** 2) for x in np.linspace(0.25, 0.75, 21)]
    cv2.fillPoly(c, [P(top + [(0.76, 0.84), (0.24, 0.84)])], 0, cv2.LINE_AA)
    return cv2.resize(c, (W // k, H // k), interpolation=cv2.INTER_AREA)


def _rounded_fill(c: np.ndarray, x1: int, y1: int, x2: int, y2: int, r: int, value: int) -> None:
    cv2.rectangle(c, (x1 + r, y1), (x2 - r, y2), value, -1)
    cv2.rectangle(c, (x1, y1 + r), (x2, y2 - r), value, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(c, (cx, cy), r, value, -1, cv2.LINE_AA)


def _honda_classic(h: int) -> np.ndarray:
    """Classic Honda-style H: legs widening upward inside a rounded square."""
    k = 4
    H, W = h * k, int(h * 1.25) * k
    c = np.full((H, W), 255, np.uint8)
    t, r = int(H * 0.07), int(H * 0.25)
    _rounded_fill(c, 0, 0, W - 1, H - 1, r, 0)
    _rounded_fill(c, t, t, W - 1 - t, H - 1 - t, r - t, 255)
    def P(pts):
        return np.int32([(x * W, y * H) for x, y in pts])
    leg = [(0.20, 0.18), (0.40, 0.18), (0.36, 0.84), (0.27, 0.84)]
    cv2.fillPoly(c, [P(leg)], 0, cv2.LINE_AA)
    cv2.fillPoly(c, [P([(1 - x, y) for x, y in leg])], 0, cv2.LINE_AA)
    cv2.fillPoly(c, [P([(0.36, 0.56), (0.64, 0.56), (0.64, 0.66), (0.36, 0.66)])], 0, cv2.LINE_AA)
    return cv2.resize(c, (W // k, H // k), interpolation=cv2.INTER_AREA)


def logo_honda(pen):
    h = int(BASE_CHAR_PX * pen.rng.uniform(1.8, 2.8))
    return _trim(_honda_solid(h) if pen.rng.random() < 0.45 else _honda_classic(h))


def logo_agc(pen):
    return _spaced(pen, "AGC AUTOMOTIVE", pen.rng.uniform(0.9, 1.2), bold=pen.rng.random() < 0.4, track=pen.rng.uniform(0.0, 0.12))


def mark_lamisafe(pen):
    return _spaced(pen, "LAMISAFE", pen.rng.uniform(0.9, 1.2), bold=pen.rng.random() < 0.4, track=pen.rng.uniform(0.0, 0.1))


def mark_temperlite(pen):
    return _spaced(pen, "TEMPERLITE", pen.rng.uniform(0.9, 1.2), bold=pen.rng.random() < 0.4, track=pen.rng.uniform(0.0, 0.1))


MARKS: Dict[str, Callable[[Pen], np.ndarray]] = {
    "e_mark": mark_e, "dot": mark_dot, "as_class": mark_as, "ccc_mark": mark_ccc,
    "tis_mark": mark_tis, "sni_mark": mark_sni, "inmetro": mark_inmetro, "asnzs": mark_asnzs,
    "bis_isi": mark_bis, "laminated": mark_laminated, "glass_class": mark_glass_class,
    "date_code": mark_date, "logo_toyota": logo_toyota, "logo_honda": logo_honda, "logo_nissan": logo_nissan,
    "logo_agc": logo_agc, "lamisafe": mark_lamisafe, "temperlite": mark_temperlite,
    "short_name": mark_short_name, "type_code": mark_type_code, "plant_code": mark_plant_code,
    "m_code": mark_m_code, "letter_l": mark_letter_l, "letter_t": mark_letter_t,
}
assert set(MARKS) == set(STANDARD_CLASSES)
LOGOS = [c for c in STANDARD_CLASSES if c.startswith("logo_")]
BRANDS = [c for c in LOGOS if c != "logo_agc"]  # car brand: at most one per marking
GLASS_NAMES = ("lamisafe", "temperlite")      # at most one per marking
GLASS_LETTERS = ("letter_l", "letter_t")       # at most one per marking


# --- Unlabelled print: other things on a marking the model must ignore ----

EXTRA_WORDS = ["TEMPERED", "SECURIT", "SAFETY GLASS", "TOUGHENED", "MADE IN THAILAND",
               "MADE IN CHINA", "SOLAR", "ACOUSTIC", "PRIVACY"]


def _qr(pen):
    """QR-like square: three finder squares and random modules."""
    n = int(pen.rng.choice([21, 25, 29]))
    grid = pen.rng.random((n, n)) < 0.5
    for y, x in ((0, 0), (0, n - 7), (n - 7, 0)):
        grid[max(0, y - 1):y + 8, max(0, x - 1):x + 8] = False
        grid[y:y + 7, x:x + 7] = True
        grid[y + 1:y + 6, x + 1:x + 6] = False
        grid[y + 2:y + 5, x + 2:x + 5] = True
    cell = max(2, int(BASE_CHAR_PX * pen.rng.uniform(2.5, 5.0) / n))
    img = np.where(np.kron(grid, np.ones((cell, cell), bool)), 0, 255).astype(np.uint8)
    return cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)


def _part_number(pen):
    r = pen.rng
    a = "ABCDEFGHJKLMNPRSTUVWXYZ"
    return pen.text(str(r.choice([
        f"{pen.digits(7)}-{a[r.integers(23)]}{a[r.integers(23)]}{pen.digits(2)}-{a[r.integers(23)]}{a[r.integers(23)]} {r.integers(1, 9)}.{r.integers(0, 9)}",
        f"{a[r.integers(23)]}{pen.digits(5)}",
        f"M{pen.digits(3)}{a[r.integers(23)]}",
        f"{pen.digits(2)}{a[r.integers(23)]}",
        a[r.integers(23)],
    ])))


def extra_print(pen) -> np.ndarray:
    k = pen.rng.random()
    if k < 0.25:
        return _qr(pen)
    if k < 0.5:
        return pen.bold(str(pen.rng.choice(EXTRA_WORDS))) if pen.rng.random() < 0.5 else pen.text(str(pen.rng.choice(EXTRA_WORDS)))
    return _part_number(pen)


EXTRA_SHARE = 0.7       # images with unlabelled print (QR, words, numbers) among the marks
EXTRA = "_extra"        # layout key for unlabelled print; never written to a label


def pick_marks(rng: np.random.Generator) -> List[str]:
    """One maker logo (mostly) and a random set of standards, at least two marks."""
    if rng.random() < BACKGROUND_SHARE:
        return []
    while True:
        marks = [BRANDS[rng.integers(len(BRANDS))]] if rng.random() < 0.9 else []
        if rng.random() < 0.6:
            marks.append("logo_agc")
        if rng.random() < 0.5:
            marks.append(GLASS_NAMES[rng.integers(2)])
        if rng.random() < 0.5:
            marks.append(GLASS_LETTERS[rng.integers(2)])
        for c in STANDARD_CLASSES:
            if c in LOGOS or c in GLASS_NAMES or c in GLASS_LETTERS:
                continue
            if rng.random() < (0.9 if c == "e_mark" else rng.uniform(0.5, 0.75)):
                marks.append(c)
        if len(marks) >= 2:
            return marks


def _layout(canvases: List[Tuple[str, np.ndarray]], rng: np.random.Generator, max_w: int):
    """Rows of marks (logo first, the rest shuffled); returns the block and
    each mark's (x1, y1, x2, y2) in it."""
    order = [c for c in canvases if c[0] in BRANDS] + [canvases[i] for i in rng.permutation(len(canvases))
                                                      if canvases[i][0] not in BRANDS]
    rows, row, width = [], [], 0
    for item in order:
        w = item[1].shape[1]
        if row and (width + w > max_w or item[0] in BRANDS or row[0][0] in BRANDS):
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


# --- AGC marking layout ---------------------------------------------------
# The order of a real AGC Automotive marking: car brand, AGC AUTOMOTIVE,
# [// or V] E-mark, DOT line or M-code, [// or V] TIS emblem with L / T and
# 11T, LAMISAFE / TEMPERLITE, [TIS number line], code + SNI next to the
# 1 2 4 8 date rows, lot code. Parts are left out at random.

AGC_SHARE = 0.7          # marked images in the AGC layout; the rest use _layout
Row = Tuple[np.ndarray, List[Tuple[str, Tuple[int, int, int, int]]]]


def _row(items: List[Tuple[str, np.ndarray]], gaps) -> Row:
    """Items side by side, centred vertically; `gaps` is one gap or one per joint."""
    gaps = list(gaps) if isinstance(gaps, (list, tuple)) else [gaps] * len(items)
    h = max(c.shape[0] for _, c in items)
    w = sum(c.shape[1] for _, c in items) + sum(gaps[:len(items) - 1])
    out = np.full((h, w), 255, np.uint8)
    boxes, x = [], 0
    for i, (key, c) in enumerate(items):
        y = (h - c.shape[0]) // 2
        out[y:y + c.shape[0], x:x + c.shape[1]] = c
        boxes.append((key, (x, y, x + c.shape[1], y + c.shape[0])))
        x += c.shape[1] + (gaps[i] if i < len(gaps) else 0)
    return out, boxes


def _stack(rows: List[Row], align: str, gap: int) -> Row:
    w = max(r[0].shape[1] for r in rows)
    h = sum(r[0].shape[0] for r in rows) + gap * (len(rows) - 1)
    out = np.full((h, w), 255, np.uint8)
    boxes, y = [], 0
    for canvas, row_boxes in rows:
        rh, rw = canvas.shape
        x = {"right": w - rw, "center": (w - rw) // 2}.get(align, 0)
        out[y:y + rh, x:x + rw] = np.minimum(out[y:y + rh, x:x + rw], canvas)
        boxes += [(k, (x1 + x, y1 + y, x2 + x, y2 + y)) for k, (x1, y1, x2, y2) in row_boxes]
        y += rh + gap
    return out, boxes


def _dashed_frame(block: np.ndarray, boxes, rng: np.random.Generator) -> Row:
    """A light dashed rectangle round the print, as on the drawings (unlabelled)."""
    m = int(BASE_CHAR_PX * rng.uniform(0.4, 1.0))
    out = cv2.copyMakeBorder(block, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    h, w = out.shape
    dash, t, ink = int(BASE_CHAR_PX * rng.uniform(0.8, 1.6)), max(1, int(rng.integers(1, 3))), int(rng.uniform(90, 170))
    for i in range(0, w, 2 * dash):                      # top and bottom edges
        for y in (0, h - 1):
            cv2.line(out, (i, y), (min(w - 1, i + dash), y), ink, t)
    for i in range(0, h, 2 * dash):                      # left and right edges
        for x in (0, w - 1):
            cv2.line(out, (x, i), (x, min(h - 1, i + dash)), ink, t)
    return out, [(k, (x1 + m, y1 + m, x2 + m, y2 + m)) for k, (x1, y1, x2, y2) in boxes]


def agc_layout(pen: Pen, rng: np.random.Generator) -> Row:
    """One AGC-style marking: (block, [(class or EXTRA, box)])."""
    gap = int(BASE_CHAR_PX * rng.uniform(0.3, 0.9))
    prefix = str(rng.choice(["//", "V"], p=[0.75, 0.25]))
    tempered = rng.random() < 0.35
    rows: List[Row] = []
    if rng.random() < 0.7:
        brand = str(rng.choice(BRANDS, p=[0.4, 0.4, 0.2]))
        rows.append(_row([(brand, MARKS[brand](pen))], 0))
    if rng.random() < 0.92:
        rows.append(_row([("logo_agc", logo_agc(pen))], 0))
    if rng.random() < 0.35:  # part number (5403010-ER51-AA 3.1) and serial (S27413): unlabelled
        L = _LETTERS
        part = (f"{pen.digits(7)}-{L[rng.integers(24)]}{L[rng.integers(24)]}{pen.digits(2)}-"
                f"{L[rng.integers(24)]}{L[rng.integers(24)]} {rng.integers(1, 10)}.{rng.integers(0, 10)}")
        rows.append(_row([(EXTRA, pen.text(part))], 0))
        if rng.random() < 0.8:  # serial: often S + digits, printed just above the E-mark like the M-code below it
            rows.append(_row([(EXTRA, pen.text(f"{'S' if rng.random() < 0.5 else L[rng.integers(24)]}{pen.digits(5)}"))], 0))
    if rng.random() < 0.95:
        e = [("e_mark", mark_e(pen))]
        rows.append(_row(([("glass_class", pen.text(prefix))] if rng.random() < 0.8 else []) + e, gap))
    if rng.random() < 0.45:
        word = int(BASE_CHAR_PX * rng.uniform(0.3, 0.7))
        rows.append(_row([("dot", mark_dot(pen)), ("m_code", mark_m_code(pen)),
                          ("as_class", pen.text(f"AS{int(rng.integers(1, 4))}"))], word))
    elif rng.random() < 0.8:
        rows.append(_row([("m_code", mark_m_code(pen))], 0))
    has_tis = rng.random() < 0.92
    number = f"TIS {pen.digits(4) if rng.random() < 0.3 else '2602-2556'}"
    letter = ("letter_t", mark_letter_t(pen)) if tempered else ("letter_l", mark_letter_l(pen))
    plant_items = [("plant_code", mark_plant_code(pen))]
    if rng.random() < 0.4:
        plant_items.insert(0, (EXTRA, pen.text("THA")))
    number_line = None
    if has_tis:
        words = pen.text(number)
        emblem = _emblem(pen, int(words.shape[0] * rng.uniform(2.2, 3.2)))
        lead = [("glass_class", pen.text(prefix))] if rng.random() < 0.8 else []
        if rng.random() < 0.5:   # number beside the emblem, letter and plant code below
            rows.append(_row(lead + [("tis_mark", _hcat([emblem, words], gap))], gap))
            far = int(BASE_CHAR_PX * rng.uniform(3, 8))
            rows.append(_row([letter] + plant_items, [far] + [gap] * (len(plant_items) - 1)))
        else:                    # emblem with the letter; the number on its own line further down
            far = int(BASE_CHAR_PX * rng.uniform(3, 8))
            rows.append(_row(lead + [("tis_mark", emblem), letter] + plant_items,
                             [gap] * len(lead) + [gap // 2, far] + [gap] * (len(plant_items) - 1)))
            number_line = _row([("tis_mark", pen.text(number + (" (2013)" if rng.random() < 0.7 else "")))], 0)
    if rng.random() < 0.2:   # QR code under the TIS line (unlabelled)
        rows.append(_row([(EXTRA, _qr(pen))], 0))
    if rng.random() < 0.85:
        name = "temperlite" if tempered else "lamisafe"
        rows.append(_row([(name, MARKS[name](pen))], 0))
    if number_line:
        rows.append(number_line)
    low: List[Tuple[str, np.ndarray]] = []
    if rng.random() < 0.85:
        low.append(("short_name", mark_short_name(pen)))
    if rng.random() < 0.65:
        low.append(("sni_mark", mark_sni(pen)))
    if rng.random() < 0.92:
        low.append(("date_code", _date_rows(pen)))
    if low:
        rows.append(_row(low, [gap // 2] * (len(low) - 2) + [int(BASE_CHAR_PX * rng.uniform(1.5, 4))] if len(low) > 1 else 0))
    if rng.random() < 0.6:
        rows.append(_row([("type_code", mark_type_code(pen))], 0))
    if not rows:
        rows.append(_row([("e_mark", mark_e(pen))], 0))
    block, boxes = _stack(rows, "right" if rng.random() < 0.7 else "center", int(BASE_CHAR_PX * rng.uniform(0.25, 0.7)))
    return _dashed_frame(block, boxes, rng) if rng.random() < 0.3 else (block, boxes)


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
    pen = Pen(rng)
    block = None
    if marks and rng.random() < AGC_SHARE:
        block, block_boxes = agc_layout(pen, rng)
    else:
        canvases = [(m, MARKS[m](pen)) for m in marks]
        if rng.random() < EXTRA_SHARE:
            canvases += [(EXTRA, extra_print(pen)) for _ in range(int(rng.integers(1, 5)))]
        if canvases:
            block, block_boxes = _layout(canvases, rng, max_w=int(rng.uniform(450, 800)))
    if block is not None:
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
            if all(p[2] >= MIN_INSIDE for p in placed if p[0] != EXTRA):
                break
        layer[y0:y0 + bh, x0:x0 + bw] = block
        layer = cv2.warpPerspective(layer, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        boxes = [(k, b) for k, b, inside in placed if k != EXTRA and inside >= MIN_INSIDE]
        if rng.random() < 0.4:  # dark ceramic band on the side away from the print
            xs = [b[0] for _, b, _ in placed] + [b[2] for _, b, _ in placed]
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
