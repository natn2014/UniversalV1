"""Builds a YOLO segmentation dataset from labelme / X-AnyLabeling polygons.

    .venv/bin/python tools/labelme_to_yoloseg.py D92-Image/D92 datasets/d92_seg \\
        [--classes glass_front,glass_back,sponge] [--val 0.2] [--max-side 2048] [--seed 0]

SRC holds the photos and, next to each, the <stem>.json that X-AnyLabeling or
labelme saved (polygon shapes; a rectangle counts as a 4-point polygon).
Images without a .json are skipped; a .json with no shapes is a valid
"nothing here" image.

Writes OUT/images/{train,val}, OUT/labels/{train,val} (cls x1 y1 x2 y2 ...
normalised), OUT/classes.txt and OUT/data.yaml. Images are shrunk so their
longest side is --max-side (labels are normalised, so they don't change):
4000 px phone photos load ~4x faster and the model trains at 1280 anyway.

The split keeps photos taken in the same second ("..._163121.jpg",
"..._163121(0).jpg") together, so a near-copy never lands in val, and takes
val from every session (the minute in the file name: ..._1631 / _1632 / ...)
so each class is checked.
"""

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

DEFAULT_CLASSES = ["glass_front", "glass_back", "sponge"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def group_key(stem: str) -> str:
    """Photos of the same second share a key: '20261127_163121(1)' -> '20261127_163121'."""
    return re.sub(r"\(\d+\)$", "", stem)


def session_key(stem: str) -> str:
    """The capture session: the name up to the minute ('20261127_1631'), or
    the whole group key when the name has no timestamp."""
    m = re.match(r"^(\d{8}_\d{4})\d{2}", stem)
    return m.group(1) if m else group_key(stem)


def shape_polygon(shape: dict):
    pts = shape.get("points") or []
    kind = shape.get("shape_type", "polygon")
    if kind == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    if kind in ("polygon", "rotation") and len(pts) >= 3:
        return pts
    return None


def label_lines(data: dict, classes: list, w: int, h: int, warn) -> list:
    lines = []
    for shape in data.get("shapes", []):
        name = (shape.get("label") or "").strip()
        if name not in classes:
            warn(f"unknown label {name!r}")
            continue
        poly = shape_polygon(shape)
        if poly is None:
            warn(f"{name}: {shape.get('shape_type')} with {len(shape.get('points') or [])} points skipped")
            continue
        coords = []
        for x, y in poly:
            coords += [min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0)]
        lines.append(f"{classes.index(name)} " + " ".join(f"{v:.6f}" for v in coords))
    return lines


def split(stems: list, val_fraction: float, seed: int) -> set:
    """Stems for val: per session, whole same-second groups, at least one group."""
    rng = random.Random(seed)
    sessions = defaultdict(lambda: defaultdict(list))
    for s in stems:
        sessions[session_key(s)][group_key(s)].append(s)
    val = set()
    for _, groups in sorted(sessions.items()):
        keys = sorted(groups)
        rng.shuffle(keys)
        if len(keys) < 2:
            continue
        n = max(1, round(len(keys) * val_fraction))
        for k in keys[:n]:
            val.update(groups[k])
    return val


def build(src: Path, out: Path, classes: list, val_fraction: float = 0.2, max_side: int = 2048, seed: int = 0) -> dict:
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.with_suffix(".json").exists())
    if not images:
        raise SystemExit(f"no labelled images (image + .json) in {src}")
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        for part in ("train", "val"):
            (out / sub / part).mkdir(parents=True, exist_ok=True)
    val = split([p.stem for p in images], val_fraction, seed)
    counts = {"train": defaultdict(int), "val": defaultdict(int)}
    n_images = {"train": 0, "val": 0}
    for path in images:
        part = "val" if path.stem in val else "train"
        data = json.loads(path.with_suffix(".json").read_text())
        img = cv2.imread(str(path))
        if img is None:
            print(f"  {path.name}: cannot read, skipped")
            continue
        h, w = img.shape[:2]
        jw, jh = data.get("imageWidth") or w, data.get("imageHeight") or h
        if (jw, jh) != (w, h):
            print(f"  {path.name}: json says {jw}x{jh}, image is {w}x{h} (EXIF turn?); using the json size")
            w, h = jw, jh
            if (img.shape[1], img.shape[0]) == (h, w):
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        lines = label_lines(data, classes, w, h, lambda m, p=path: print(f"  {p.name}: {m}"))
        scale = max_side / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "images" / part / f"{path.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / part / f"{path.stem}.txt").write_text("".join(f"{l}\n" for l in lines))
        n_images[part] += 1
        for l in lines:
            counts[part][classes[int(l.split()[0])]] += 1
    (out / "classes.txt").write_text("".join(f"{c}\n" for c in classes))
    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                 "nc": len(classes), "names": {i: c for i, c in enumerate(classes)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return {"images": n_images, "instances": {k: dict(v) for k, v in counts.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--max-side", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    stats = build(args.src, args.out, classes, args.val, args.max_side, args.seed)
    for part in ("train", "val"):
        inst = ", ".join(f"{c} {stats['instances'][part].get(c, 0)}" for c in classes)
        print(f"{part}: {stats['images'][part]} images ({inst})")
    print(f"wrote {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
