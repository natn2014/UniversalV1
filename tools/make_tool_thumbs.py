"""Draw the example thumbnails shown on the Add Tool tiles.

Each picture is a small wire-harness scene that shows the case a tool suits,
with its area in green and an OK / NG tag, like a sensor catalogue. They are
drawn (not photographed) so they can be regenerated:

    .venv/bin/python tools/make_tool_thumbs.py

Output: app/assets/tool_thumbs/<kind>.png (240 x 150).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "app" / "assets" / "tool_thumbs"
W, H = 480, 300  # drawn at 2x, saved at half size for smooth edges

GREEN = (60, 215, 60)
RED = (50, 50, 235)
YELLOW = (40, 215, 255)
WHITE = (245, 245, 245)
AA = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_DUPLEX


def canvas(gray: int = 70) -> np.ndarray:
    img = np.full((H, W, 3), gray, np.uint8)
    # Soft vignette so the scene reads like a camera image, not a flat icon.
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot((xx - W / 2) / W, (yy - H / 2) / H)
    return np.clip(img * (1.15 - 0.6 * d)[..., None], 0, 255).astype(np.uint8)


def box(img, x0, y0, x1, y1, color=GREEN, t=4):
    cv2.rectangle(img, (x0, y0), (x1, y1), color, t, AA)


def tag(img, ok: bool, text=None):
    text = text or ("OK" if ok else "NG")
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.1, 2)
    cv2.rectangle(img, (0, 0), (tw + 28, th + 22), (20, 140, 20) if ok else (30, 30, 205), -1)
    cv2.putText(img, text, (14, th + 11), FONT, 1.1, WHITE, 2, AA)


def label(img, text, org, scale=0.9, color=YELLOW, t=2):
    cv2.putText(img, text, org, FONT, scale, (20, 20, 20), t + 3, AA)
    cv2.putText(img, text, org, FONT, scale, color, t, AA)


def arrow2(img, p0, p1, color=YELLOW, t=3):
    cv2.arrowedLine(img, p0, p1, color, t, AA, tipLength=0.12)
    cv2.arrowedLine(img, p1, p0, color, t, AA, tipLength=0.12)


def housing(img, x0, y0, x1, y1, shade=185):
    cv2.rectangle(img, (x0, y0), (x1, y1), (shade, shade, shade), -1, AA)
    cv2.rectangle(img, (x0, y0), (x1, y1), (shade - 70,) * 3, 2, AA)


def wire(img, y, x0, x1, color, t=22):
    cv2.line(img, (x0, y), (x1, y), color, t, AA)
    hi = tuple(min(255, c + 70) for c in color)
    cv2.line(img, (x0, y - t // 4), (x1, y - t // 4), hi, 3, AA)


# --- one scene per tool kind -------------------------------------------------

def position(img):
    pts = cv2.boxPoints(((250, 160), (190, 110), 0)).astype(np.int32)
    for i in range(4):  # taught place, dashed
        p, q = pts[i], pts[(i + 1) % 4]
        for s in np.linspace(0, 1, 12)[::2]:
            a = p + (q - p) * s
            b = p + (q - p) * min(1, s + 1 / 11)
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (150, 150, 150), 2, AA)
    moved = cv2.boxPoints(((285, 175), (190, 110), 14)).astype(np.int32)
    cv2.fillPoly(img, [moved], (190, 190, 190), AA)
    cv2.polylines(img, [moved], True, GREEN, 4, AA)
    for dx in (-50, -15, 20, 55):  # cavities
        c = (int(285 + dx * np.cos(np.radians(14))), int(175 + dx * np.sin(np.radians(14))))
        cv2.circle(img, c, 11, (90, 90, 90), -1, AA)
    cv2.arrowedLine(img, (250, 160), (285, 175), YELLOW, 3, AA, tipLength=0.4)
    cv2.drawMarker(img, (285, 175), GREEN, cv2.MARKER_CROSS, 34, 3, AA)
    tag(img, True)


def color_area(img):
    housing(img, 40, 50, 150, 260)
    for y, c in zip((85, 130, 175, 220), ((40, 40, 210), (200, 90, 30), (30, 200, 230), (50, 160, 50))):
        wire(img, y, 150, 460, c)
    box(img, 250, 112, 420, 148)
    label(img, "red 96%", (270, 104), 0.8)
    tag(img, True)


def color_average(img):
    housing(img, 60, 70, 300, 240, 200)
    cv2.circle(img, (180, 155), 52, (20, 120, 240), -1, AA)  # orange cap
    cv2.circle(img, (165, 138), 14, (120, 200, 255), -1, AA)
    box(img, 120, 95, 240, 215)
    cv2.rectangle(img, (340, 110), (430, 200), (20, 120, 240), -1)
    cv2.rectangle(img, (340, 110), (430, 200), WHITE, 2)
    label(img, "dE 1.4", (330, 240), 0.9)
    tag(img, True)


def color_prohibit(img):
    housing(img, 70, 60, 410, 250, 205)
    for x in range(110, 400, 60):
        cv2.circle(img, (x, 155), 16, (120, 120, 120), -1, AA)
    cv2.circle(img, (300, 205), 9, (30, 40, 150), -1, AA)  # contamination
    cv2.circle(img, (300, 205), 22, RED, 3, AA)
    box(img, 80, 70, 400, 240)
    tag(img, False)


def area(img):
    img[:] = (img * 0.45).astype(np.uint8)
    pts = np.array([[120, 120], [300, 120], [330, 100], [400, 100], [400, 200],
                    [330, 200], [300, 180], [120, 180]], np.int32)
    cv2.fillPoly(img, [pts], (230, 230, 235), AA)
    cv2.rectangle(img, (150, 135), (290, 165), (190, 190, 200), -1)
    box(img, 100, 85, 420, 215)
    label(img, "area 18400 px", (150, 250), 0.8)
    tag(img, True)


def brightness(img):
    img[:] = (img * 0.5).astype(np.uint8)
    housing(img, 120, 60, 360, 240, 90)
    glow = np.zeros_like(img)
    cv2.circle(glow, (240, 150), 60, (140, 255, 140), -1, AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 18)
    cv2.circle(glow, (240, 150), 22, (230, 255, 230), -1, AA)
    img[:] = cv2.add(img, glow)
    box(img, 175, 85, 305, 215)
    label(img, "avg 212", (190, 270), 0.8)
    tag(img, True)


def edge_pixels(img):
    cv2.rectangle(img, (90, 70), (390, 230), (235, 235, 230), -1)
    for i, line in enumerate(("2410", "A-07")):
        cv2.putText(img, line, (130, 135 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (40, 40, 40), 5, AA)
    roi = img[80:220, 110:370]
    edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 60, 150)
    roi[cv2.dilate(edges, None) > 0] = GREEN
    box(img, 105, 78, 375, 222)
    tag(img, True)


def shading(img):
    yy, xx = np.mgrid[0:H, 0:W]
    surf = 150 + 25 * np.sin(xx / 90.0)
    surf -= 90 * np.exp(-(((xx - 310) / 28.0) ** 2 + ((yy - 170) / 22.0) ** 2))  # dent
    img[:] = np.clip(surf, 0, 255).astype(np.uint8)[..., None]
    x0, y0, cw, ch = 70, 60, 85, 60
    for r in range(3):
        for c in range(4):
            bad = c == 2 and r == 1
            if not bad:
                box(img, x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch, GREEN, 2)
    box(img, x0 + 2 * cw, y0 + ch, x0 + 3 * cw, y0 + 2 * ch, RED, 5)
    tag(img, False)


def width(img):
    wire(img, 150, 20, 270, (200, 90, 30), 44)  # insulation
    cv2.line(img, (270, 150), (440, 150), (60, 140, 215), 30, AA)  # stripped copper
    for y in range(138, 164, 6):
        cv2.line(img, (272, y), (438, y), (90, 180, 240), 1, AA)
    for x in (270, 440):
        cv2.line(img, (x, 95), (x, 205), GREEN, 3, AA)
    arrow2(img, (272, 105), (438, 105))
    label(img, "5.0 mm", (300, 245), 0.9)
    tag(img, True)


def edge_presence(img):
    housing(img, 220, 70, 440, 230)
    cv2.rectangle(img, (240, 125), (430, 175), (60, 60, 60), -1)  # cavity
    cv2.rectangle(img, (60, 132), (330, 168), (60, 160, 220), -1, AA)  # inserted terminal
    cv2.line(img, (60, 150), (20, 150), (200, 90, 30), 26, AA)
    cv2.arrowedLine(img, (420, 110), (240, 110), YELLOW, 3, AA, tipLength=0.08)
    cv2.line(img, (330, 95), (330, 190), GREEN, 4, AA)
    cv2.circle(img, (330, 150), 10, GREEN, -1, AA)
    label(img, "edge found", (250, 260), 0.8)
    tag(img, True)


def pitch(img):
    housing(img, 40, 60, 440, 240)
    xs = [90 + 60 * i for i in range(6)]
    for x in xs:
        cv2.rectangle(img, (x - 12, 90), (x + 12, 230), (90, 200, 235), -1, AA)
    cv2.line(img, (60, 160), (420, 160), YELLOW, 2, AA)
    for x in xs:
        cv2.circle(img, (x, 160), 7, GREEN, -1, AA)
    arrow2(img, (xs[1], 75), (xs[2], 75))
    label(img, "2.54", (xs[1] + 2, 280 - 15), 0.9)
    tag(img, True)


def diameter(img):
    cv2.circle(img, (240, 150), 110, (200, 200, 200), -1, AA)  # washer / ring terminal
    cv2.circle(img, (240, 150), 110, (120, 120, 120), 2, AA)
    cv2.circle(img, (240, 150), 52, (35, 35, 35), -1, AA)
    cv2.circle(img, (240, 150), 52, GREEN, 4, AA)
    arrow2(img, (190, 150), (290, 150))
    label(img, "D 6.2", (330, 70), 0.9)
    tag(img, True)


def ocr(img):
    cv2.rectangle(img, (40, 55), (440, 245), (238, 238, 232), -1)
    cv2.putText(img, "LOT 2410A", (70, 125), FONT, 1.5, (30, 30, 30), 3, AA)
    cv2.putText(img, "24V  3A", (70, 205), FONT, 1.3, (30, 30, 30), 3, AA)
    box(img, 58, 78, 360, 142)
    label(img, "LOT 2410A", (240, 285), 0.7, GREEN)
    tag(img, True)


def code(img):
    cv2.rectangle(img, (30, 40), (450, 260), (238, 238, 232), -1)
    qr = cv2.QRCodeEncoder.create().encode("UV1-2410A")
    qr = cv2.resize(qr, (170, 170), interpolation=cv2.INTER_NEAREST)
    img[65:235, 55:225] = qr[..., None]
    rng = np.random.default_rng(3)
    x = 260
    while x < 425:
        bw = int(rng.integers(2, 8))
        cv2.rectangle(img, (x, 80), (x + bw, 200), (25, 25, 25), -1)
        x += bw + int(rng.integers(3, 8))
    cv2.putText(img, "2410A", (290, 232), FONT, 0.8, (30, 30, 30), 2, AA)
    box(img, 48, 58, 232, 242)
    tag(img, True)


def outline(img):
    shape = np.array([[110, 90], [180, 90], [180, 70], [300, 70], [300, 90], [370, 90],
                      [370, 220], [300, 220], [300, 240], [180, 240], [180, 220], [110, 220]], np.int32)
    cv2.fillPoly(img, [shape], (180, 180, 180), AA)
    for x in (150, 210, 270, 330):
        cv2.rectangle(img, (x - 16, 130), (x + 16, 180), (90, 90, 90), -1)
    cv2.polylines(img, [shape], True, GREEN, 4, AA)
    label(img, "match 94%", (300, 280), 0.8)
    tag(img, True)


def blob_count(img):
    cv2.rectangle(img, (40, 50), (440, 260), (95, 95, 100), -1)
    missing = (3, 1)
    n = 0
    for r in range(2):
        for c in range(5):
            cx, cy = 85 + c * 78, 105 + r * 100
            if (c, r) == missing:
                cv2.circle(img, (cx, cy), 30, RED, 3, AA)
                continue
            cv2.circle(img, (cx, cy), 26, (210, 215, 220), -1, AA)
            cv2.circle(img, (cx, cy), 10, (120, 120, 125), -1, AA)
            cv2.circle(img, (cx, cy), 30, GREEN, 3, AA)
            n += 1
    tag(img, False, f"NG {n}/10")


SCENES = {
    "position": position, "color_area": color_area, "color_average": color_average,
    "color_prohibit": color_prohibit, "area": area, "brightness": brightness,
    "edge_pixels": edge_pixels, "shading": shading, "width": width,
    "edge_presence": edge_presence, "pitch": pitch, "diameter": diameter,
    "ocr": ocr, "code": code, "outline": outline, "blob_count": blob_count,
}


def render(kind: str) -> np.ndarray:
    img = canvas()
    SCENES[kind](img)
    return cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in SCENES:
        cv2.imwrite(str(OUT_DIR / f"{kind}.png"), render(kind))
    print(f"wrote {len(SCENES)} thumbnails to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
