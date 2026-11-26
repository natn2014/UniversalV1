"""YOLO-pose datasets: the keypoint names file and pose label files.

A pose dataset is a normal dataset folder (<root>/images, <root>/labels) that
also holds <root>/keypoints.txt, one keypoint name per line, in order. Each
label line is a detect box followed by one (x, y, v) per keypoint, all
normalised to 0..1:

    <cls> <cx> <cy> <w> <h>  <x0> <y0> <v0>  <x1> <y1> <v1> ...

v: 0 = not labelled, 1 = labelled but hidden, 2 = labelled and visible.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .marking.classes import validate_class_name

KEYPOINTS_FILE = "keypoints.txt"
# No kpt_oks_sigmas in data.yaml, on purpose: Ultralytics then uses 1/K per
# keypoint (0.5 for head/tail). A tight value such as COCO's 0.025-0.107 was
# tried (2026-11-26) and the model learned no keypoints at all, not even on
# its own 16 training photos (worst error ~240 px vs ~49 px with the default):
# the loss is already at its maximum while the points are still far off, so
# it gives no direction to move them. The default's only cost is a lenient
# pose mAP score; judge a model by its measured angle error instead.
DEFAULT_KEYPOINTS = ["head", "tail"]

NOT_LABELLED, HIDDEN, VISIBLE = 0, 1, 2

Keypoint = Tuple[float, float, int]


def read_keypoints_file(dataset_root: Path) -> Optional[List[str]]:
    """The keypoint names in order; None when this is not a pose dataset."""
    path = Path(dataset_root) / KEYPOINTS_FILE
    if not path.exists():
        return None
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return names or None


def write_keypoints_file(dataset_root: Path, names: Sequence[str]) -> None:
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / KEYPOINTS_FILE).write_text("\n".join(names) + "\n")


def is_pose_dataset(dataset_root: Path) -> bool:
    return (Path(dataset_root) / KEYPOINTS_FILE).exists()


def parse_keypoint_names(text: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """'head, tail' -> (['head', 'tail'], None), or (None, why it is not valid)."""
    names: List[str] = []
    for raw in text.replace("\n", ",").split(","):
        name = raw.strip()
        if not name:
            continue
        if " " in name:
            return None, f'Keypoint name "{name}" must be one word.'
        problem = validate_class_name(name, names)
        if problem is not None:
            return None, problem.replace("class", "keypoint")
        names.append(name)
    if not names:
        return None, "Give at least one keypoint name."
    return names, None


def read_pose_label_file(label_path: Path, k: int) -> List[dict]:
    """Boxes with `kpts`: k (x, y, v) normalised. A detect-only line (5
    values) gets k unlabelled points, so a detect dataset can become pose."""
    boxes: List[dict] = []
    label_path = Path(label_path)
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5 and len(parts) != 5 + 3 * k:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
            kpts: List[Keypoint] = []
            for i in range(k):
                if len(parts) == 5:
                    kpts.append((0.0, 0.0, NOT_LABELLED))
                    continue
                x, y, v = parts[5 + 3 * i: 8 + 3 * i]
                kpts.append((float(x), float(y), int(float(v))))
        except ValueError:
            continue
        boxes.append({"cls_id": cls_id, "cx": cx, "cy": cy, "w": w, "h": h, "kpts": kpts})
    return boxes


def write_pose_label_file(label_path: Path, boxes: List[dict], k: int) -> None:
    lines = []
    for box in boxes:
        kpts = list(box.get("kpts") or [])[:k]
        kpts += [(0.0, 0.0, NOT_LABELLED)] * (k - len(kpts))
        fields = [f"{box['cls_id']}", f"{box['cx']:.6f}", f"{box['cy']:.6f}",
                  f"{box['w']:.6f}", f"{box['h']:.6f}"]
        for x, y, v in kpts:
            if v == NOT_LABELLED:
                fields += ["0.000000", "0.000000", "0"]
            else:
                fields += [f"{x:.6f}", f"{y:.6f}", str(int(v))]
        lines.append(" ".join(fields))
    Path(label_path).write_text("\n".join(lines) + ("\n" if lines else ""))


def kpts_to_pixel(kpts: Sequence[Keypoint], img_w: int, img_h: int) -> List[Keypoint]:
    return [(x * img_w, y * img_h, int(v)) for x, y, v in kpts]


def kpts_to_normalized(kpts: Sequence[Keypoint], img_w: int, img_h: int) -> List[Keypoint]:
    out: List[Keypoint] = []
    for x, y, v in kpts:
        if v == NOT_LABELLED:
            out.append((0.0, 0.0, NOT_LABELLED))
        else:
            out.append((min(max(x / img_w, 0.0), 1.0), min(max(y / img_h, 0.0), 1.0), int(v)))
    return out


def box_is_complete(box: dict) -> bool:
    """Every keypoint was placed or marked hidden."""
    return all(v != NOT_LABELLED for _, _, v in box.get("kpts") or [])


def label_is_complete(boxes: List[dict]) -> bool:
    return bool(boxes) and all(box_is_complete(b) for b in boxes)


def dataset_yaml(dataset_root: Path, list_name: str, classes: Sequence[str],
                 keypoints: Optional[Sequence[str]] = None) -> Dict:
    """data.yaml contents for Ultralytics; pose adds kpt_shape and flip_idx.
    flip_idx is the identity: a head mirrored is still the head (swap pairs
    such as left_pin/right_pin by hand if a dataset ever has them)."""
    data = {
        "path": str(Path(dataset_root).resolve()),
        "train": list_name,
        "val": list_name,
        "nc": len(classes),
        "names": list(classes),
    }
    if keypoints:
        data["kpt_shape"] = [len(keypoints), 3]
        data["flip_idx"] = list(range(len(keypoints)))
        # Saved in the model, so the Orientation tool can show the names.
        data["kpt_names"] = {i: list(keypoints) for i in range(len(classes))}
    return data
