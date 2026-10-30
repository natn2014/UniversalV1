"""Pre-labels a dataset folder with a model, for checking in the Train tab.

    .venv/bin/python tools/prelabel_marks.py weights/standard_marks_v3.pt datasets/real_marks [--conf 0.3] [--overwrite]

Runs the model on <dir>/images and writes <dir>/labels/<stem>.txt (YOLO
cls cx cy w h) with the model's class ids, and <dir>/classes.txt with its
class names when the folder has none. Images that already have a label file
are skipped unless --overwrite. Every box still needs a look: open
<dir>/images in the Train tab and fix what the model got wrong.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.marking.classes import read_classes_file, write_classes_file  # noqa: E402
from app.yolo_labels import write_yolo_label_file  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def prelabel(model, root: Path, conf: float = 0.3, overwrite: bool = False, imgsz: int = 640) -> dict:
    """{stem: number of boxes written} for the images labelled this run."""
    names = [model.names[i] for i in sorted(model.names)]
    classes = read_classes_file(root)
    if classes is None:
        write_classes_file(root, names)
    elif classes != names:
        raise SystemExit(f"{root}/classes.txt differs from the model's classes; the ids would not match")
    labels = root / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    written = {}
    for path in sorted((root / "images").iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        out = labels / f"{path.stem}.txt"
        if out.exists() and not overwrite:
            continue
        r = model.predict(str(path), conf=conf, imgsz=imgsz, verbose=False)[0]
        h, w = r.orig_shape
        boxes = [{"cls_id": int(b.cls), "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h,
                  "w": (x2 - x1) / w, "h": (y2 - y1) / h}
                 for b in r.boxes for x1, y1, x2, y2 in [b.xyxy[0].tolist()]]
        write_yolo_label_file(out, boxes)
        written[path.stem] = len(boxes)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from ultralytics import YOLO
    written = prelabel(YOLO(str(args.model)), args.folder, args.conf, args.overwrite)
    print(f"labelled {len(written)} images, {sum(written.values())} boxes, in {args.folder}/labels")


if __name__ == "__main__":
    main()
