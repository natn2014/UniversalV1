"""Train a head / tail pose model for the Orientation tool on a GPU machine
(e.g. a DGX Spark), without the app's Qt user interface.

    python tools/train_pose.py DATASET [--model weights/yolo11n-pose.pt] [--name model_a]
                               [--epochs 100] [--imgsz 640] [--batch 16] [--device 0]
                               [--preset light] [--no-rotate]

DATASET is a pose dataset folder made by the Train tab or by
tools/augment_pose_dataset.py (images/, labels/, keypoints.txt). It writes
DATASET/labelled.txt and DATASET/data.yaml exactly as the Train tab does
(app.pose_labels.dataset_yaml) and trains with the same augmentation preset
(app.augmentation.build_hyp), with "Any rotation (0-360°)" on unless
--no-rotate. The model is saved to DATASET/runs/<name>/weights/best.pt; copy
that file to the Jetson and pick it with Browse... in the Orientation tool.

--model is the starting point: the pose base model (downloaded once if
missing), or a best.pt from an earlier run to fine-tune it (e.g. model B from
model A). If DATASET/runs/<name>/weights/last.pt exists from an interrupted
run, training resumes from it instead of starting over.

Only images whose every box has all its keypoints are used.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from app.augmentation import PRESETS, build_hyp  # noqa: E402
from app.marking.classes import read_classes_file  # noqa: E402
from app.pose_labels import dataset_yaml, label_is_complete, read_keypoints_file, read_pose_label_file  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def write_dataset_files(ds: Path) -> Path:
    """labelled.txt (complete images only) + data.yaml; returns data.yaml."""
    names = read_keypoints_file(ds)
    if not names:
        raise SystemExit(f"{ds} is not a pose dataset (no keypoints.txt)")
    classes = read_classes_file(ds) or ["part"]
    k = len(names)
    images = sorted(p for p in (ds / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    ready = [p for p in images if label_is_complete(read_pose_label_file(ds / "labels" / f"{p.stem}.txt", k))]
    if not ready:
        raise SystemExit(f"no image in {ds / 'images'} has all its keypoints")
    print(f"{len(ready)} of {len(images)} images ready (keypoints: {', '.join(names)}; classes: {', '.join(classes)})")
    (ds / "labelled.txt").write_text("".join(f"{p.resolve()}\n" for p in ready))
    path = ds / "data.yaml"
    path.write_text(yaml.safe_dump(dataset_yaml(ds, "labelled.txt", classes, names), sort_keys=False))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--model", default=str(ROOT / "weights" / "yolo11n-pose.pt"),
                    help="start from this .pt (pose base model, or an earlier best.pt to fine-tune)")
    ap.add_argument("--name", default="pose", help="run folder: DATASET/runs/<name>")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="GPU index, or cpu")
    ap.add_argument("--patience", type=int, default=50, help="stop when no better for this many epochs")
    ap.add_argument("--preset", default="light", choices=sorted(PRESETS), help="augmentation preset")
    ap.add_argument("--no-rotate", action="store_true", help="turn off Any rotation (0-360°)")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    ds = args.dataset.resolve()
    data_yaml = write_dataset_files(ds)
    run_dir = ds / "runs" / args.name
    last = run_dir / "weights" / "last.pt"
    t0 = time.time()
    if last.exists():
        print(f"resuming the interrupted run from {last}")
        results = YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(args.model)
        if getattr(model, "task", "pose") != "pose":
            raise SystemExit(f"{args.model} is not a pose model")
        results = model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, project=str(ds / "runs"), name=args.name, exist_ok=True,
            patience=args.patience, optimizer="auto", plots=True, val=True, amp=True,
            **build_hyp(args.preset, rotate_any=not args.no_rotate),
        )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"done in {(time.time() - t0) / 60:.1f} min: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
