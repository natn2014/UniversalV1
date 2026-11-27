"""Train a YOLO segmentation model (e.g. D92 quarter glass side + clip-seal
sponges) on a GPU machine, without the app's Qt user interface.

    .venv/bin/python tools/train_seg.py datasets/d92_seg [--model yolo26n-seg.pt] [--name d92_v1]
                                       [--epochs 150] [--imgsz 1280] [--batch 8] [--device 0]
                                       [--preset strong] [--no-rotate] [--mirror] [--flip-ud]
                                       [--hsv-v 0.4]

DATASET is made by tools/labelme_to_yoloseg.py (data.yaml, images/, labels/).
Augmentation comes from app.augmentation.build_hyp (the Train tab's presets):
mirroring stays off unless --mirror / --flip-ud (a mirrored back side
looks like the other-hand part, so only turn them on when the model need not
tell left from right) and "Any rotation" is on unless --no-rotate, because
the part may lie at any angle. --hsv-v sets the brightness change; for more
lighting conditions build the dataset with tools/light_aug_seg.py. Small parts (sponges ~100 px in a 4000 px photo) need --imgsz 1280.

The model is saved to DATASET/runs/<name>/weights/best.pt; copy it to
weights/ and pick it with Browse... in the Part Check (AI Seg) tool. If
DATASET/runs/<name>/weights/last.pt exists from an interrupted run, training
resumes from it instead of starting over. --model may be an earlier best.pt
to fine-tune it (e.g. on line-camera images).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.augmentation import PRESETS, build_hyp  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--model", default="yolo26n-seg.pt",
                    help="start from this .pt (seg base model, downloaded once, or an earlier best.pt)")
    ap.add_argument("--name", default="seg", help="run folder: DATASET/runs/<name>")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0", help="GPU index, or cpu")
    ap.add_argument("--patience", type=int, default=50, help="stop when no better for this many epochs")
    ap.add_argument("--preset", default="strong", choices=sorted(PRESETS), help="augmentation preset")
    ap.add_argument("--no-rotate", action="store_true", help="turn off Any rotation (0-360°)")
    ap.add_argument("--mirror", action="store_true", help="left-right flip half the images")
    ap.add_argument("--flip-ud", action="store_true", help="up-down flip half the images")
    ap.add_argument("--hsv-v", type=float, default=None, help="brightness change (0-1), overrides the preset")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    ds = args.dataset.resolve()
    data_yaml = ds / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} missing: run tools/labelme_to_yoloseg.py first")
    run_dir = ds / "runs" / args.name
    last = run_dir / "weights" / "last.pt"
    hyp = build_hyp(args.preset, mirror=args.mirror, flip_ud=args.flip_ud, rotate_any=not args.no_rotate)
    if args.hsv_v is not None:
        hyp["hsv_v"] = args.hsv_v
    t0 = time.time()
    if last.exists():
        print(f"resuming the interrupted run from {last}")
        results = YOLO(str(last)).train(resume=True)
    else:
        model_path = args.model
        local = ROOT / "weights" / model_path
        if not Path(model_path).exists() and local.exists():
            model_path = str(local)
        model = YOLO(model_path)
        if getattr(model, "task", "segment") != "segment":
            raise SystemExit(f"{args.model} is not a segmentation model (use a *-seg.pt)")
        results = model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, project=str(ds / "runs"), name=args.name, exist_ok=True,
            patience=args.patience, optimizer="auto", plots=True, val=True, amp=True,
            **hyp,
        )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"done in {(time.time() - t0) / 60:.1f} min: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
