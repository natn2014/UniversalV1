"""Draws a synthetic glass marking (made-up maker, several standards) to test
the OCR tool, with a JSON of each text line and its box in the picture.

    .venv/bin/python tools/make_synthetic_mark.py [OUT.png] [--tilt DEG] [--blur K] [--seed N]

Like a camera photo: grey glass with uneven light, the print slightly
tilted, blurred and noisy. Every name and number is invented.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="samples/synthetic_mark.png")
    ap.add_argument("--tilt", type=float, default=-3.0)
    ap.add_argument("--blur", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img, truth = make(args.tilt, args.blur, args.seed)
    cv2.imwrite(str(out), img)
    out.with_suffix(".json").write_text(json.dumps(truth, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
