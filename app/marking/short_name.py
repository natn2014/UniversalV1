"""OCR check of the short name (58, U8, T1, Y2...): two characters, an
uppercase letter then a digit, or two digits. The job presets the expected
value; read_short_name() reads the detected box and compares.

Plain Tesseract on the whole box misread a third of the drawings (a red
frame line through the characters, a letter read as another letter), so:
- gray = the brightest colour channel, so coloured lines (the drawings' red
  dashed frame) vanish while black or dark print stays;
- long thin lines left over are removed;
- the box is split into its two characters, and each is read alone with
  only the characters its position may hold (from the expected value's
  format), so a letter position can't come back as a digit and vice versa.
A wrong short name still reads as itself (its own letter / digit), so it
does not match the expected value: NG.

ShortNameReader runs reads in a background thread at a few Hz, so the
camera loop keeps its frame rate; a result belongs to one part (epoch).
"""

import re
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

SHORT_NAME_RE = re.compile(r"^([A-Z][0-9]|[0-9]{2})$")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
CHAR_PX = 48  # character height the crop is scaled to before reading
MIN_CONF = 50.0  # below this (0-100) a read is "unreadable"
AGREED_CONF = 60.0  # an unsure character read the same way twice
UNREADABLE_AFTER_S = 1.5  # a part with no clear read by then is NG
PART_GONE_S = 0.5  # short name not seen this long: the next one is a new part
# Look-alikes, for the whole-box fallback read: what a letter position
# means by a digit and the other way round.
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
TO_DIGIT = {v: k for k, v in TO_LETTER.items()}


@dataclass(frozen=True)
class ShortNameRead:
    text: str  # what was read ("" when nothing usable)
    conf: float  # 0-100, the weaker character's
    state: str  # "ok" (= expected), "ng" (another name) or "unreadable"


def valid_short_name(text: str) -> bool:
    return bool(SHORT_NAME_RE.match(text or ""))


def position_sets(expected: str) -> Tuple[str, str]:
    """Characters each position may hold, from the expected value's format."""
    return (LETTERS if expected[:1].isalpha() else DIGITS), DIGITS


def pad_box(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Detection box grown so a character cut by a tight box is whole:
    35 % of the height on every side (wider print needs it in x)."""
    x1, y1, x2, y2 = box
    pad = 0.35 * (y2 - y1)
    return (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
            min(width, int(x2 + pad)), min(height, int(y2 + pad)))


def _line_mask(bgr: np.ndarray) -> np.ndarray:
    """Coloured line pixels (the drawings' red frame), even dark red ones."""
    b, g, r = (bgr[..., i].astype(np.int16) for i in range(3))
    reddish = r - np.maximum(g, b) > 40
    saturated = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2) > 60
    return ((reddish | saturated) * 255).astype(np.uint8)


def _ink(bgr: np.ndarray) -> np.ndarray:
    """Binary (print = 255) at about CHAR_PX character height, coloured lines gone."""
    scale = CHAR_PX / max(1, 0.7 * bgr.shape[0])  # the print is ~70 % of the padded box
    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if bgr.ndim == 3:
        line = _line_mask(bgr)
        gray = bgr.min(axis=2)
        gray[line > 0] = 255  # the line is not print
    else:
        line, gray = np.zeros(bgr.shape, np.uint8), bgr
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if line.any():
        # Where the line crossed a stroke it left a gap: bridge it, only there.
        # Across the line only: sideways for an upright line, up-down for a flat one.
        near = cv2.dilate(line, np.ones((5, 5), np.uint8)) > 0  # its soft edges too
        ys, xs = np.nonzero(line)
        upright = np.ptp(ys) >= np.ptp(xs)
        # As wide as the line (its pixels per row / column), plus a little.
        rows = np.count_nonzero(line, axis=1 if upright else 0)
        width = int(np.median(rows[rows > 0])) + 6
        kernel = (width, 1) if upright else (1, width)
        bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        ink[near] = bridged[near]
    # Long thin lines (frame / border left in black) are not print.
    for size in ((max(3, int(CHAR_PX * 1.6)), 1), (1, max(3, int(CHAR_PX * 1.6)))):
        line = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))
        ink = cv2.subtract(ink, cv2.dilate(line, np.ones((3, 3), np.uint8)))
    return ink


def split_characters(ink: np.ndarray) -> List[np.ndarray]:
    """The two characters, left to right, each as dark-on-light with a margin;
    [] when the box does not hold two clear characters."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    boxes, pieces = [], []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if area < 0.012 * CHAR_PX * CHAR_PX:
            continue  # specks (an arm cut off by a line is larger)
        (boxes if hh >= 0.35 * CHAR_PX else pieces).append([x, y, x + w, y + hh])
    # A short piece next to a character is part of it (an arm cut off by a
    # line): join it to the nearest one within a quarter character.
    for piece in pieces:
        near = [b for b in boxes if max(b[0], piece[0]) - min(b[2], piece[2]) < 0.25 * CHAR_PX]
        if near:
            b = min(near, key=lambda b: abs((b[0] + b[2]) - (piece[0] + piece[2])))
            b[:] = [min(b[0], piece[0]), min(b[1], piece[1]), max(b[2], piece[2]), max(b[3], piece[3])]
    # Pieces of one character (a broken stroke) overlap in x: merge them.
    boxes.sort()
    merged: List[list] = []
    for b in boxes:
        if merged and b[0] < merged[-1][2] - 0.3 * min(b[2] - b[0], merged[-1][2] - merged[-1][0]):
            m = merged[-1]
            merged[-1] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
        else:
            merged.append(b)
    if len(merged) < 2:
        return []
    # The print: the two tallest, kept left to right.
    merged = sorted(sorted(merged, key=lambda b: b[3] - b[1], reverse=True)[:2])
    chars = []
    for x1, y1, x2, y2 in merged:
        glyph = 255 - ink[y1:y2, x1:x2]
        chars.append(cv2.copyMakeBorder(glyph, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255))
    return chars


def _variants(glyph: np.ndarray):
    """Ways to show Tesseract one character, as (image, psm, copies): alone,
    as a pair (it reads words better than lone glyphs: a "9" alone often
    comes back empty, "99" does not), and slightly bolder."""
    return ((glyph, 10, 1), (cv2.hconcat([glyph, glyph]), 7, 2),
            (cv2.erode(glyph, np.ones((2, 2), np.uint8)), 10, 1))


def _read_variant(engine, image: np.ndarray, psm: int, copies: int, allowed: str) -> Tuple[str, float]:
    text, conf = engine.read(image, psm=psm, whitelist=allowed)
    text = "".join(c for c in text.upper() if c in allowed)
    if len(text) == copies and len(set(text)) == 1:
        return text[0], conf
    return "", 0.0


def _best(engine, glyph: np.ndarray, allowed: str) -> Tuple[str, float]:
    """One character. Read unrestricted first: a clear digit where a letter
    belongs stays a digit (so "20" never passes as "Q0"); only a letter read
    where a digit belongs is mapped to its look-alike digit (Y0 read "YO").
    When nothing is clear, read again limited to this position's characters
    and take a character two ways of reading agree on."""
    variants = _variants(glyph)
    for image, psm, copies in variants:
        text, conf = _read_variant(engine, image, psm, copies, LETTERS + DIGITS)
        if text and conf >= MIN_CONF:
            if text in allowed:
                return text, conf
            if allowed is DIGITS and text in TO_DIGIT:
                return TO_DIGIT[text], conf
            return text, conf  # the other kind: does not match the expected value
    votes = []
    for image, psm, copies in variants + ((glyph, 8, 1), (glyph, 13, 1)):
        text, conf = _read_variant(engine, image, psm, copies, allowed)
        if not text:
            continue
        if conf >= MIN_CONF:
            return text, conf
        votes.append(text)
        if votes.count(text) >= 2:
            return text, AGREED_CONF
    return (votes[0], 0.0) if votes else ("", 0.0)


def read_short_name(bgr_crop: np.ndarray, expected: str, engine) -> ShortNameRead:
    """Reads the two characters in a (padded) short-name crop and compares
    them with `expected`."""
    if not valid_short_name(expected) or bgr_crop is None or bgr_crop.size == 0:
        return ShortNameRead("", 0.0, "unreadable")
    sets = position_sets(expected)
    ink = _ink(bgr_crop)
    chars = split_characters(ink)
    if len(chars) == 2:
        reads = [_best(engine, glyph, allowed) for glyph, allowed in zip(chars, sets)]
        text = "".join(t for t, _ in reads)
        conf = min(c for _, c in reads)
    else:
        # Fallback: the whole box, look-alikes mapped to each position's kind.
        raw, conf = engine.read(255 - ink, psm=7, whitelist=LETTERS + DIGITS)
        raw = "".join(c for c in raw.upper() if c.isalnum())
        text = "".join((c if c in allowed else (TO_LETTER if allowed is LETTERS else TO_DIGIT).get(c, ""))
                       for c, allowed in zip(raw, sets))
    if len(text) != 2 or conf < MIN_CONF:
        return ShortNameRead(text, conf, "unreadable")
    return ShortNameRead(text, conf, "ok" if text == expected else "ng")


class ShortNameReader:
    """Reads the short name in a background thread, newest crop only.

    submit() at most `hz` times a second while the box is in view; result()
    gives this part's read with its state: "pending" until the first clear
    read, "unreadable" when none came within UNREADABLE_AFTER_S of the part
    arriving. A new part (box gone for PART_GONE_S) starts over, so the
    previous part's read can never pass it.
    """

    def __init__(self, engine_factory) -> None:
        self._engine_factory = engine_factory
        self._engine = None
        self._cond = threading.Condition()
        self._request: Optional[tuple] = None
        self._thread: Optional[threading.Thread] = None
        self._epoch = 0
        self._part_since: Optional[float] = None
        self._last_seen = float("-inf")
        self._last_submit = float("-inf")
        self._read: Optional[ShortNameRead] = None  # latest of this part
        self._clear: Optional[ShortNameRead] = None  # latest clear (ok / ng) of this part

    def seen(self, now: float) -> None:
        """The short name is in view this frame."""
        with self._cond:
            if now - self._last_seen > PART_GONE_S:
                self._epoch += 1
                self._part_since = now
                self._read = self._clear = None
                self._last_submit = float("-inf")
            self._last_seen = now

    def due(self, now: float, hz: float) -> bool:
        with self._cond:
            return self._request is None and now - self._last_submit >= 1.0 / max(hz, 0.1)

    def submit(self, crop: np.ndarray, expected: str, now: float) -> None:
        with self._cond:
            self._request = (crop, expected, self._epoch)
            self._last_submit = now
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="short-name", daemon=True)
                self._thread.start()
            self._cond.notify()

    def result(self, expected: str, now: float) -> dict:
        """{text, conf, expected, state} for the part in view."""
        with self._cond:
            read = self._clear or self._read
            if read is not None and read.state != "unreadable":
                state = "ok" if read.text == expected else "ng"  # the preset may have changed since
            elif self._part_since is not None and now - self._part_since >= UNREADABLE_AFTER_S:
                state = "unreadable"
            else:
                state = "pending"
            return {"text": read.text if read else "", "conf": read.conf if read else 0.0,
                    "expected": expected, "state": state}

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._request is None:
                    if not self._cond.wait(timeout=5.0):
                        self._thread = None
                        return
                crop, expected, epoch = self._request
                self._request = None
            if self._engine is None:
                self._engine = self._engine_factory()
            if self._engine is None:
                read = ShortNameRead("", 0.0, "unreadable")
            else:
                try:
                    read = read_short_name(crop, expected, self._engine)
                except Exception:
                    read = ShortNameRead("", 0.0, "unreadable")
            with self._cond:
                if epoch == self._epoch:
                    self._read = read
                    if read.state != "unreadable":
                        self._clear = read
            time.sleep(0)
