"""Mark inspection: the usual AGC Automotive marking layout, checked on the
standard-mark model's detections.

Top to bottom: car brand logo, AGC AUTOMOTIVE, the E-mark, then the M-code
or TIS; the plant code (11T) is right-most, SNI is in the left half and the
short name (58, U8, T1) is left-most. A rule is checked only when the marks
it names were detected (missing marks are Required Marks' job). Tolerances
are in the mark's own height, so they don't depend on camera distance.
Pure Python: detections are dicts with x1, y1, x2, y2, class_name.
"""

from typing import Dict, List

# Top-to-bottom order: each group's centre is below the previous group's.
ORDER = [("logo_toyota", "logo_honda", "logo_nissan"), ("logo_agc",), ("e_mark",), ("m_code", "tis_mark")]
RIGHT_MOST = "plant_code"
LEFT_HALF = "sni_mark"
LEFT_MOST = "short_name"
RIGHT_TOLERANCE = 1.5  # x mark height (the AGC AUTOMOTIVE box can overshoot the right edge)
LEFT_TOLERANCE = 1.0


def _centre_y(d: dict) -> float:
    return (d["y1"] + d["y2"]) / 2


def _height(d: dict) -> float:
    return max(1.0, d["y2"] - d["y1"])


def check_layout(detections: List[dict]) -> List[str]:
    """One short message per broken rule; [] when the layout looks right."""
    by_class: Dict[str, List[dict]] = {}
    for d in detections:
        by_class.setdefault(d.get("class_name"), []).append(d)
    issues = []

    # Vertical order: the lowest box of a group must lie above the highest box of the next.
    groups = [(names, [d for n in names for d in by_class.get(n, [])]) for names in ORDER]
    groups = [(names, boxes) for names, boxes in groups if boxes]
    for (above_names, above), (below_names, below) in zip(groups, groups[1:]):
        upper = max(above, key=_centre_y)
        lower = min(below, key=_centre_y)
        if _centre_y(lower) <= _centre_y(upper):
            issues.append(f"{lower['class_name']} not below {upper['class_name']}")

    others = [d for d in detections if d.get("class_name") not in (RIGHT_MOST, LEFT_MOST)]
    if by_class.get(RIGHT_MOST) and others:
        right = max(d["x2"] for d in others)
        if all(d["x2"] < right - RIGHT_TOLERANCE * _height(d) for d in by_class[RIGHT_MOST]):
            issues.append(f"{RIGHT_MOST} not right-most")
    if by_class.get(LEFT_HALF) and detections:
        middle = (min(d["x1"] for d in detections) + max(d["x2"] for d in detections)) / 2
        if all((d["x1"] + d["x2"]) / 2 >= middle for d in by_class[LEFT_HALF]):
            issues.append(f"{LEFT_HALF} not on left")
    if by_class.get(LEFT_MOST) and others:
        left = min(d["x1"] for d in others)
        if all(d["x1"] > left + LEFT_TOLERANCE * _height(d) for d in by_class[LEFT_MOST]):
            issues.append(f"{LEFT_MOST} not left-most")
    return issues
