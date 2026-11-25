from pathlib import Path
from typing import List


def read_yolo_label_file(label_path: Path) -> List[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, w, h = parts
        boxes.append(
            {
                "cls_id": int(cls_id),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
            }
        )
    return boxes


def write_yolo_label_file(label_path: Path, boxes: List[dict]) -> None:
    lines = [
        f"{box['cls_id']} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}"
        for box in boxes
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def yolo_box_to_pixel(box: dict, img_w: int, img_h: int) -> tuple:
    cx, cy, w, h = box["cx"] * img_w, box["cy"] * img_h, box["w"] * img_w, box["h"] * img_h
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return x1, y1, x2, y2


def pixel_box_to_yolo(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> dict:
    return {
        "cx": ((x1 + x2) / 2) / img_w,
        "cy": ((y1 + y2) / 2) / img_h,
        "w": (x2 - x1) / img_w,
        "h": (y2 - y1) / img_h,
    }
