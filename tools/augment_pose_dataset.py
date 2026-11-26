"""Grow a small pose (head / tail) dataset with slightly turned copies.

    .venv/bin/python tools/augment_pose_dataset.py SOURCE OUT [--angles 1,2,3,4,5] [--auto-head-up]
                                                   [--light=-30,30]

SOURCE is a folder of photos, or a Train-tab dataset folder (with images/,
labels/, keypoints.txt). Each photo is copied into OUT/images, plus one copy
per angle in both directions (default ±1..5° = 10 copies), with its box and
keypoints turned by exactly the same angle, into a pose dataset the Train tab
opens directly (OUT/images, OUT/labels, classes.txt, keypoints.txt).

Labels: a photo that already has a label in SOURCE/labels (e.g. checked in
the Train tab) keeps it. With --auto-head-up, an unlabelled photo is labelled
automatically: the one part near the centre is found, its box drawn, the head
put at the middle of its top edge and the tail at the middle of its bottom
edge. Use it only for photos taken with the head at the top, and check the
result in the Train tab before training.

--light adds lighting copies: every image (each turned copy too) again
brighter and darker by those percentages, e.g. -30,30 = 30 % darker and
30 % brighter, so the model also knows the part under dimmer or stronger
light. Labels are unchanged (light does not move the part).

Phone photos stored sideways with an EXIF rotation flag are read upright
(as the trainer reads them) and saved upright, so the flag no longer matters.
Re-run it when new photos are added: OUT is rebuilt from SOURCE.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pose_labels import (  # noqa: E402
    DEFAULT_KEYPOINTS,
    NOT_LABELLED,
    VISIBLE,
    read_keypoints_file,
    read_pose_label_file,
    write_keypoints_file,
    write_pose_label_file,
)
from app.marking.classes import read_classes_file, write_classes_file  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}  # no app.paths: it needs Qt, a GPU server may not have it
MAX_SIDE = 1280  # saved size: plenty for training at 640, far smaller than a 4000 px photo
FIND_SIDE = 800  # the part is found on a copy this size


# --- finding the part (auto labels) -----------------------------------------

def part_mask(img: np.ndarray) -> Optional[np.ndarray]:
    """Mask of the one part near the centre: its edges closed into a solid
    blob; the blob covering (or else nearest to) the centre wins."""
    h, w = img.shape[:2]
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    edges = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 2), 30, 90)
    window = np.zeros_like(edges)  # the part is centred: ignore table edges and glare outside
    window[int(h * .2):int(h * .8), int(w * .15):int(w * .85)] = 255
    edges = cv2.bitwise_and(edges, window)
    k = max(3, round(min(h, w) / 24)) | 1
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centre = (w / 2, h / 2)

    def score(cnt) -> float:
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 0.002 * w * h:
            return -1e9
        inside = cv2.pointPolygonTest(cnt, centre, True)  # > 0 when the centre is inside
        return inside if inside > 0 else inside * 3 + cv2.contourArea(cnt) * 1e-4

    if not contours:
        return None
    best = max(contours, key=score)
    if score(best) <= -1e9:
        return None
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [best], -1, 255, -1)
    return mask


def auto_head_up(img: np.ndarray) -> Optional[Tuple[np.ndarray, List[Tuple[float, float]]]]:
    """(part mask at full size, [head, tail] in pixels) for a head-up photo."""
    h, w = img.shape[:2]
    s = FIND_SIDE / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    mask = part_mask(small)
    if mask is None:
        return None
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    x, y, bw, bh = cv2.boundingRect(mask)
    cx = x + bw / 2
    return mask, [(cx, float(y)), (cx, float(y + bh))]


# --- turning ----------------------------------------------------------------

def turn(img: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """The image turned `angle`° counter-clockwise about its centre (edges
    filled by replicating), and the 2x3 matrix that moves points with it."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), m


def move(points, m: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    return pts @ m[:, :2].T + m[:, 2]


def box_of(points: np.ndarray, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1 = np.clip(points.min(axis=0), 0, [w - 1, h - 1])
    x2, y2 = np.clip(points.max(axis=0), 0, [w - 1, h - 1])
    return float(x1), float(y1), float(x2), float(y2)


def to_label(cls_id: int, box, kpts, w: int, h: int) -> dict:
    x1, y1, x2, y2 = box
    return {"cls_id": cls_id, "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h,
            "w": (x2 - x1) / w, "h": (y2 - y1) / h,
            "kpts": [(min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0), v) for x, y, v in kpts]}


# --- one photo ----------------------------------------------------------------

def photo_parts(img: np.ndarray, label_path: Optional[Path], k: int, auto: bool):
    """[(cls_id, outline points px, [(x, y, v)] px)] for the photo, or None.
    The outline is what gets turned to make each copy's box: the part's mask
    outline for auto labels (a tight box at any angle), else the box corners."""
    h, w = img.shape[:2]
    if label_path is not None and label_path.exists():
        parts = []
        for b in read_pose_label_file(label_path, k):
            x1, y1 = (b["cx"] - b["w"] / 2) * w, (b["cy"] - b["h"] / 2) * h
            x2, y2 = (b["cx"] + b["w"] / 2) * w, (b["cy"] + b["h"] / 2) * h
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            parts.append((b["cls_id"], corners, [(x * w, y * h, v) for x, y, v in b["kpts"]]))
        return parts or None
    if not auto:
        return None
    found = auto_head_up(img)
    if found is None:
        return None
    mask, (head, tail) = found
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    kpts = [(head[0], head[1], VISIBLE), (tail[0], tail[1], VISIBLE)]
    kpts += [(0.0, 0.0, NOT_LABELLED)] * (k - 2)
    return [(0, outline, kpts)]


def relit(img: np.ndarray, percent: float) -> np.ndarray:
    """The image under `percent` more (or, negative, less) light. Darker:
    every pixel scaled down. Brighter: a gamma lift, so the shadows and mid
    tones rise by about `percent` while bright metal keeps its detail rather
    than clipping to flat white."""
    x = img.astype(np.float32) / 255.0
    if percent < 0:
        y = x * (1.0 + percent / 100.0)
    else:
        y = np.power(x, 1.0 / (1.0 + percent / 100.0) ** 2)
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


def light_list(text: str) -> List[float]:
    return [0.0] + sorted({float(v) for v in text.split(",") if v.strip()} - {0.0})


def angles_list(text: str) -> List[float]:
    values = sorted({abs(float(a)) for a in text.split(",") if a.strip()} - {0.0})
    return [0.0] + [s * a for a in values for s in (1, -1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="folder of photos, or a Train-tab dataset folder")
    ap.add_argument("out", type=Path, help="new dataset folder (rebuilt)")
    ap.add_argument("--angles", default="1,2,3,4,5", help="degrees, each used both ways (default 1,2,3,4,5)")
    ap.add_argument("--light", default="", help="lighting copies in %%, e.g. --light=-30,30 (default none)")
    ap.add_argument("--auto-head-up", action="store_true",
                    help="label unlabelled photos automatically: head = top of the centred part")
    args = ap.parse_args(argv)

    src = args.source
    images_dir = src / "images" if (src / "images").is_dir() else src
    labels_dir = images_dir.parent / "labels" if images_dir != src else None
    names = read_keypoints_file(images_dir.parent if images_dir != src else src) or list(DEFAULT_KEYPOINTS)
    classes = read_classes_file(images_dir.parent if images_dir != src else src) or ["part"]
    k = len(names)
    angles = angles_list(args.angles)
    lights = light_list(args.light)

    out = args.out
    if out.resolve() in (src.resolve(), images_dir.resolve()):
        ap.error("OUT must be a new folder, not the source")
    for sub in ("images", "labels"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True)
    write_keypoints_file(out, names)
    write_classes_file(out, classes)

    photos = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    written, skipped = 0, []
    for photo in photos:
        img = cv2.imread(str(photo))  # applies the EXIF rotation, like the trainer
        if img is None:
            skipped.append(f"{photo.name} (unreadable)")
            continue
        label = labels_dir / f"{photo.stem}.txt" if labels_dir else None
        parts = photo_parts(img, label, k, args.auto_head_up)
        if parts is None:
            skipped.append(f"{photo.name} (no label{'' if args.auto_head_up else '; use --auto-head-up'})")
            continue
        s = min(1.0, MAX_SIDE / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            parts = [(c, outline * s, [(x * s, y * s, v) for x, y, v in kp]) for c, outline, kp in parts]
        h, w = img.shape[:2]
        for angle in angles:
            turned, m = turn(img, angle)
            boxes = []
            for cls_id, outline, kpts in parts:
                moved = move([(x, y) for x, y, _ in kpts], m)
                new_kpts = [(x, y, v) if v != NOT_LABELLED else (0.0, 0.0, NOT_LABELLED)
                            for (x, y), (_, _, v) in zip(moved, kpts)]
                boxes.append(to_label(cls_id, box_of(move(outline, m), w, h), new_kpts, w, h))
            base = photo.stem if angle == 0 else f"{photo.stem}_r{angle:+g}"
            for light in lights:
                stem = base if light == 0 else f"{base}_l{light:+g}"
                shot = turned if light == 0 else relit(turned, light)
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), shot, [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_pose_label_file(out / "labels" / f"{stem}.txt", boxes, k)
                written += 1
    light_text = f", light {', '.join(f'{v:+g}%' for v in lights)}" if len(lights) > 1 else ""
    print(f"{written} images from {len(photos) - len(skipped)} photos "
          f"(angles {', '.join(f'{a:+g}' for a in angles)}{light_text}) -> {out}")
    for line in skipped:
        print(f"  skipped {line}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
