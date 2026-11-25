"""Defect checks on a marking: missing elements, misprinted or faint
characters, smeared / broken print against the master image, and elements
printed out of position."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .reader import ReadLine
from .standards import DEFECT, Issue

Box = Tuple[int, int, int, int]


def _defect(code: str, message: str, element: Optional[str] = None) -> Issue:
    return Issue(DEFECT, code, message, element)


def diff_text(expected: str, read: str) -> List[Tuple[str, int, str, str]]:
    """Character edit script from expected to read, as (op, position in
    expected, expected char, read char); op is replace / missing / extra."""
    n, m = len(expected), len(read)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j - 1] + 1,
                             dist[i - 1][j - 1] + (expected[i - 1] != read[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (expected[i - 1] != read[j - 1]):
            if expected[i - 1] != read[j - 1]:
                ops.append(("replace", i - 1, expected[i - 1], read[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(("missing", i - 1, expected[i - 1], ""))
            i -= 1
        else:
            ops.append(("extra", i, "", read[j - 1]))
            j -= 1
    return ops[::-1]


def _compact(text: str) -> str:
    return "".join(text.upper().split())


def check_line(element: str, expected: str, line: Optional[ReadLine],
               min_conf: float = 0.5) -> List[Issue]:
    """Misprint (character diff against the master text, spaces ignored) and
    faint / broken characters (low char-model confidence)."""
    if line is None or not line.chars:
        return [_defect("LINE_MISSING", f"{element}: no text read (expected {expected!r})", element)]
    issues = []
    ops = diff_text(_compact(expected), _compact(line.text))
    if ops:
        details = []
        for op, pos, want, got in ops:
            if op == "replace":
                details.append(f"#{pos + 1} {want!r} read as {got!r}")
            elif op == "missing":
                details.append(f"#{pos + 1} {want!r} missing")
            else:
                details.append(f"extra {got!r}")
        issues.append(_defect("MISPRINT", f"{element}: {line.text!r} vs {expected!r} ({'; '.join(details)})", element))
    faint = [c.text for c in line.chars if c.conf < min_conf]
    if faint:
        issues.append(_defect("FAINT_CHAR", f"{element}: unclear print on {' '.join(faint)} (conf < {min_conf:.2f})", element))
    return issues


def check_missing(required: Iterable[str], detected: Iterable[str]) -> List[Issue]:
    present = set(detected)
    return [_defect("ELEMENT_MISSING", f"{name} not printed / not found", name)
            for name in required if name not in present]


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def print_similarity(master: np.ndarray, crop: np.ndarray, margin: float = 0.08,
                     diff_threshold: int = 60) -> Tuple[float, float, np.ndarray]:
    """Compares a crop with the master crop of the same element.

    The crop is resized to the master's size, the master's centre (minus
    `margin` on each side) is searched inside it to absorb small shifts,
    and the aligned difference is thresholded. Returns (NCC score,
    defect area ratio, defect mask in master-centre coordinates)."""
    m = _gray(master).astype(np.float32)
    c = cv2.resize(_gray(crop), (m.shape[1], m.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Normalise brightness / contrast so lighting drift isn't a defect.
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX)
    c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
    dy, dx = int(m.shape[0] * margin), int(m.shape[1] * margin)
    template = m[dy:m.shape[0] - dy, dx:m.shape[1] - dx]
    scores = cv2.matchTemplate(c, template, cv2.TM_CCOEFF_NORMED)
    _, ncc, _, (bx, by) = cv2.minMaxLoc(scores)
    aligned = c[by:by + template.shape[0], bx:bx + template.shape[1]]
    diff = cv2.absdiff(cv2.GaussianBlur(aligned, (3, 3), 0), cv2.GaussianBlur(template, (3, 3), 0))
    mask = (diff > diff_threshold).astype(np.uint8) * 255
    # Drop single-pixel edge noise from sub-pixel misalignment.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(ncc), float(np.count_nonzero(mask)) / mask.size, mask


def check_print(element: str, master: np.ndarray, crop: np.ndarray,
                min_ncc: float = 0.7, max_defect_ratio: float = 0.02) -> List[Issue]:
    """Smear / broken / extra ink against the master image."""
    if crop is None or crop.size == 0:
        return [_defect("ELEMENT_MISSING", f"{element} not found", element)]
    ncc, ratio, _ = print_similarity(master, crop)
    issues = []
    if ncc < min_ncc:
        issues.append(_defect("PRINT_MISMATCH", f"{element}: print differs from master (match {ncc:.2f} < {min_ncc:.2f})", element))
    if ratio > max_defect_ratio:
        issues.append(_defect("PRINT_FLAW", f"{element}: smear / broken print on {ratio:.1%} of area", element))
    return issues


def relative_offset(box: Box, anchor: Box) -> Tuple[float, float]:
    """Centre offset of `box` from `anchor`, in anchor heights (so the
    check doesn't depend on camera distance)."""
    ah = max(1, anchor[3] - anchor[1])
    return (((box[0] + box[2]) - (anchor[0] + anchor[2])) / 2 / ah,
            ((box[1] + box[3]) - (anchor[1] + anchor[3])) / 2 / ah)


def check_positions(boxes: Dict[str, Box], anchor: str,
                    master_offsets: Dict[str, Sequence[float]],
                    tolerance: float = 0.5) -> List[Issue]:
    """Elements whose offset from the anchor element (usually the E-mark)
    moved more than `tolerance` anchor heights from the master layout."""
    if anchor not in boxes:
        return []
    issues = []
    for name, (mx, my) in master_offsets.items():
        if name not in boxes or name == anchor:
            continue
        ox, oy = relative_offset(boxes[name], boxes[anchor])
        shift = ((ox - mx) ** 2 + (oy - my) ** 2) ** 0.5
        if shift > tolerance:
            issues.append(_defect("POSITION", f"{name}: printed {shift:.2f} x E-mark height off master position", name))
    return issues
