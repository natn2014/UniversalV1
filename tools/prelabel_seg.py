"""Pre-labels photos with a segmentation model, for checking in X-AnyLabeling.

    .venv/bin/python tools/prelabel_seg.py MODEL FOLDER [--conf 0.3] [--imgsz 1280] [--overwrite]

Runs MODEL (a YOLO *-seg .pt, e.g. a first model trained on ~15 hand-labelled
photos) on every image in FOLDER and writes <stem>.json next to it in the
labelme format X-AnyLabeling opens: one polygon per detection, labelled with
the model's class name. Images that already have a .json are skipped unless
--overwrite, so hand-made labels are never replaced by accident.

Every polygon still needs a look: open FOLDER in X-AnyLabeling, fix what the
model got wrong (missing sponge, loose outline), and save.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def simplify(poly: np.ndarray, frac: float = 0.002) -> list:
    """Fewer points (easier to edit): Douglas-Peucker at `frac` of the perimeter."""
    p = poly.astype(np.float32).reshape(-1, 1, 2)
    eps = max(1.0, frac * cv2.arcLength(p, True))
    out = cv2.approxPolyDP(p, eps, True).reshape(-1, 2)
    return out.tolist() if len(out) >= 3 else poly.tolist()


def labelme_doc(image_name: str, w: int, h: int, shapes: list) -> dict:
    return {"version": "2.4.0", "flags": {}, "shapes": shapes, "imagePath": image_name,
            "imageData": None, "imageHeight": h, "imageWidth": w}


def prelabel(model, folder: Path, conf: float = 0.3, imgsz: int = 1280, overwrite: bool = False) -> dict:
    """{name: number of polygons written} for the images labelled this run."""
    written = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        out = path.with_suffix(".json")
        if out.exists() and not overwrite:
            continue
        r = model.predict(str(path), conf=conf, imgsz=imgsz, retina_masks=True, verbose=False)[0]
        h, w = r.orig_shape
        shapes = []
        if r.masks is not None:
            for poly, c, k in zip(r.masks.xy, r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                if len(poly) < 3:
                    continue
                shapes.append({"label": r.names[int(k)], "points": simplify(np.asarray(poly)),
                               "group_id": None, "description": f"conf {c:.2f}",
                               "shape_type": "polygon", "flags": {}})
        out.write_text(json.dumps(labelme_doc(path.name, w, h, shapes), indent=1))
        written[path.name] = len(shapes)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    from ultralytics import YOLO
    written = prelabel(YOLO(str(args.model)), args.folder, args.conf, args.imgsz, args.overwrite)
    for name, n in written.items():
        print(f"  {name}: {n} polygons")
    print(f"pre-labelled {len(written)} images in {args.folder}: check them in X-AnyLabeling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
