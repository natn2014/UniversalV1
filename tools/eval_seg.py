"""Checks a Part Check (AI seg) model the way the app judges: side + sponge count.

    .venv/bin/python tools/eval_seg.py MODEL FOLDER [--expect 20261127_1631=back:0,...]
                                      [--out runs/d92_eval] [--conf 0.5] [--item-count 3]

For every image in FOLDER it runs the app's own SegmentTool
(app/vision/segment.py) on the whole photo and compares the result with what
was expected:
  * from <stem>.json (X-AnyLabeling / labelme) when it exists: the glass
    label gives the side, the number of sponge shapes the count;
  * else from --expect: file-name prefix = side:count (side front|back),
    default the D92 sessions (16:31 back no sponge, 16:32 back 3 sponges,
    16:33 front).
Prints one line per image and a summary, writes an overlay image per photo to
--out (green = OK, red = NG, text = got / expected), and exits 1 if any image
is judged differently from what was expected.

Also run Ultralytics val for mAP: yolo segment val model=MODEL data=DATASET/data.yaml imgsz=1280
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.vision import create  # noqa: E402
from app.vision.segment import YoloSegPredictor, _train_imgsz  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
D92_SESSIONS = "20261127_1631=back:0,20261127_1632=back:3,20261127_1633=front:0"


def parse_expect(text: str) -> dict:
    out = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        prefix, value = item.split("=")
        side, _, count = value.partition(":")
        out[prefix] = (side, int(count or 0))
    return out


def expected_for(path: Path, rules: dict, front: str, back: str, item: str):
    js = path.with_suffix(".json")
    if js.exists():
        labels = [s.get("label") for s in json.loads(js.read_text()).get("shapes", [])]
        side = "back" if back in labels else "front" if front in labels else None
        return side, labels.count(item), "json"
    for prefix, (side, count) in sorted(rules.items(), key=lambda kv: -len(kv[0])):
        if path.name.startswith(prefix):
            return side, count, "name"
    return None, None, "-"


def draw(frame: np.ndarray, res, text: str) -> np.ndarray:
    img = frame.copy()
    for o in res.overlay:
        color = (60, 200, 60) if o["ok"] else (40, 40, 230)
        if o["type"] == "poly":
            if o.get("fill"):  # same see-through paint as the app (BGR)
                layer = img.copy()
                cv2.fillPoly(layer, [np.int32(o["pts"])], (60, 220, 40) if o["fill"] == "green" else (255, 110, 30))
                img = cv2.addWeighted(layer, 0.35 if o["fill"] == "green" else 0.45, img, 1 - (0.35 if o["fill"] == "green" else 0.45), 0)
            cv2.polylines(img, [np.int32(o["pts"])], True, color, 6, cv2.LINE_AA)
        elif o["type"] == "circle":
            x, y, r = o["pts"]
            cv2.circle(img, (int(x), int(y)), max(int(r), 14), color, -1, cv2.LINE_AA)
    color = (60, 200, 60) if res.ok else (40, 40, 230)
    cv2.rectangle(img, (0, 0), (img.shape[1], 150), (25, 25, 25), -1)
    cv2.putText(img, text, (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 6, cv2.LINE_AA)
    scale = 1280 / max(img.shape[:2])
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--expect", default=D92_SESSIONS)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "d92_eval")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--item-conf", type=float, default=0.4)
    ap.add_argument("--item-count", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=None, help="default: the size the model was trained at, like the app")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(str(args.model), task="segment")
    predictor = YoloSegPredictor(model, args.imgsz or _train_imgsz(model))
    rules = parse_expect(args.expect)
    args.out.mkdir(parents=True, exist_ok=True)

    tool = create("segment", conf=args.conf, item_conf=args.item_conf, expected_side="any", both_sides=False,
                  low=args.item_count, high=args.item_count)
    tool.predictor = predictor
    rows, wrong = [], []
    for path in sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        side, count, source = expected_for(path, rules, tool.front_class, tool.back_class, tool.item_class)
        res = tool.run(frame)
        got_side = res.extra.get("side")
        got_count = int(res.value) if got_side else None
        good_part = side == "front" or (side == "back" and count == args.item_count)
        app_ok = res.ok  # what the line would say with expected_side "any"
        match = side is not None and got_side == side and (side == "front" or got_count == count)
        rows.append((path.name, side, count, got_side, got_count, app_ok, good_part, match, source))
        if side is not None and not match:
            wrong.append(path.name)
        exp = "?" if side is None else ("front" if side == "front" else f"back {count}")
        got = res.text or "no part"
        mark = "ok " if match else ("?? " if side is None else "XX ")
        print(f"{mark}{path.name:32s} expected {exp:8s} got {got:14s} ({source})")
        cv2.imwrite(str(args.out / f"{path.stem}.jpg"), draw(frame, res, f"{got}  | want {exp}"),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    known = [r for r in rows if r[1] is not None]
    side_right = sum(1 for r in known if r[3] == r[1])
    print(f"\n{len(rows)} images, {len(known)} with an expected result")
    print(f"side right:        {side_right}/{len(known)}")
    print(f"side + count right: {len(known) - len(wrong)}/{len(known)}")
    # OK/NG as the line would see it (good part = front, or back with all sponges).
    tp = sum(1 for r in known if r[6] and r[5])
    fn = sum(1 for r in known if r[6] and not r[5])
    fp = sum(1 for r in known if not r[6] and r[5])
    tn = sum(1 for r in known if not r[6] and not r[5])
    print(f"OK/NG (any side): good->OK {tp}, good->NG {fn} (false reject), "
          f"bad->OK {fp} (ESCAPE), bad->NG {tn}")
    if wrong:
        print("wrong: " + ", ".join(wrong))
    print(f"overlays in {args.out}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
