"""Minimal Code 39 (incl. Full-ASCII / "Extended") barcode renderer, used to
make mock part labels for demos. No dependencies beyond Pillow."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

# 9 elements per char: bar, space, bar, ... ; 'w' = wide, 'n' = narrow
PATTERNS = {
    "0": "nnnwwnwnn", "1": "wnnwnnnnw", "2": "nnwwnnnnw", "3": "wnwwnnnnn",
    "4": "nnnwwnnnw", "5": "wnnwwnnnn", "6": "nnwwwnnnn", "7": "nnnwnnwnw",
    "8": "wnnwnnwnn", "9": "nnwwnnwnn", "A": "wnnnnwnnw", "B": "nnwnnwnnw",
    "C": "wnwnnwnnn", "D": "nnnnwwnnw", "E": "wnnnwwnnn", "F": "nnwnwwnnn",
    "G": "nnnnnwwnw", "H": "wnnnnwwnn", "I": "nnwnnwwnn", "J": "nnnnwwwnn",
    "K": "wnnnnnnww", "L": "nnwnnnnww", "M": "wnwnnnnwn", "N": "nnnnwnnww",
    "O": "wnnnwnnwn", "P": "nnwnwnnwn", "Q": "nnnnnnwww", "R": "wnnnnnwwn",
    "S": "nnwnnnwwn", "T": "nnnnwnwwn", "U": "wwnnnnnnw", "V": "nwwnnnnnw",
    "W": "wwwnnnnnn", "X": "nwnnwnnnw", "Y": "wwnnwnnnn", "Z": "nwwnwnnnn",
    "-": "nwnnnnwnw", ".": "wwnnnnwnn", " ": "nwwnnnwnn", "*": "nwnnwnwnn",
    "$": "nwnwnwnnn", "/": "nwnwnnnwn", "+": "nwnnnwnwn", "%": "nnnwnwnwn",
}

# Full-ASCII escapes for characters Code 39 can't encode directly.
EXTENDED = {
    " ": "/A", "!": "/B", '"': "/C", ",": "/D", "%": "/E", "&": "/F", "'": "/G",
    "(": "/H", ")": "/I", "*": "/J", "+": "/K", "/": "/L", ":": "/M", ";": "/N",
    "<": "/O", "=": "/P", ">": "/Q", "?": "/R", "@": "/S", "[": "/T", "\\": "/U",
    "]": "/V", "^": "/W", "_": "/X", "`": "/Y", "{": "/Z",
}


def to_extended(text: str) -> str:
    out = []
    for ch in text:
        if ch in EXTENDED:
            out.append(EXTENDED[ch])
        elif ch.islower():
            out.append("+" + ch.upper())
        else:
            out.append(ch)
    return "".join(out)


def units(payload: str, ratio: int = 3) -> int:
    """Width of the barcode in narrow-module units (without quiet zones)."""
    return (len(payload) + 2) * (6 + 3 * ratio + 1)


def render(payload: str, caption: str, narrow: int = 3, ratio: int = 3,
           height: int = 120, font_size: int = 30) -> Image.Image:
    """White label: bars for `payload` (already Code 39 characters) and a
    human-readable `caption` underneath."""
    data = f"*{payload}*"
    widths = []
    for ch in data:
        pattern = PATTERNS[ch]
        widths += [narrow * ratio if e == "w" else narrow for e in pattern]
        widths.append(narrow)                      # inter-character gap
    quiet = narrow * 12
    bar_w = sum(widths)
    try:
        font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    text_box = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), caption, font=font)
    width = max(bar_w, text_box[2]) + 2 * quiet
    image = Image.new("RGB", (width, height + font_size + 40), "white")
    draw = ImageDraw.Draw(image)
    x = (width - bar_w) // 2
    for i, w in enumerate(widths):
        if i % 10 < 9 and i % 10 % 2 == 0:         # bars at even positions
            draw.rectangle([x, 16, x + w - 1, 16 + height], fill="black")
        x += w
    draw.text(((width - text_box[2]) // 2, height + 24), caption, fill="black", font=font)
    return image
