"""Augmented sample images for the Training Options preview, made by
Ultralytics' own training data pipeline so they match what the trainer sees."""

import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

BOX_COLOR = (0, 210, 255)  # BGR amber, like the app's ROI outline
# Keypoint colours by index (BGR), the same order as the Train tab's dots.
KPT_COLORS = [(80, 80, 255), (255, 200, 60), (80, 220, 80), (220, 80, 220), (60, 200, 255)]


def preview_samples(data_yaml: Path, hyp: dict, count: int = 6, imgsz: int = 640,
                    seed: int = None) -> List[np.ndarray]:
    """`count` augmented training images (BGR, boxes and class names drawn).
    Raises on an unreadable dataset; Ultralytics is imported lazily because
    loading it takes a few seconds."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    pose = "kpt_shape" in data
    overrides = {**hyp, "imgsz": imgsz, "cache": False}
    if pose:
        overrides["task"] = "pose"  # loads and augments the keypoints too
    cfg = get_cfg(overrides=overrides)
    dataset = build_yolo_dataset(cfg, data["train"], batch=1, data=data, mode="train")
    if len(dataset) == 0:
        raise ValueError("no labelled images")
    rng = random.Random(seed)
    names = data["names"]
    samples = []
    for _ in range(count):
        item = dataset[rng.randrange(len(dataset))]
        # Ultralytics returns CHW RGB; boxes as normalised centre x, y, w, h.
        img = cv2.cvtColor(np.ascontiguousarray(item["img"].numpy().transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        for (cx, cy, bw, bh), cls in zip(item["bboxes"].tolist(), item["cls"].flatten().tolist()):
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
            label = names[int(cls)] if isinstance(names, (list, dict)) else str(int(cls))
            cv2.putText(img, str(label), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)
        if pose and "keypoints" in item:
            draw_keypoints(img, item["keypoints"].numpy())
        samples.append(img)
    return samples


def draw_keypoints(img: np.ndarray, keypoints: np.ndarray) -> None:
    """Dots for (N, K, 2 or 3) normalised keypoints, with an arrow from the
    second point to the first (tail -> head) so a turned sample shows its
    direction."""
    h, w = img.shape[:2]
    for kpts in keypoints:
        pts = []
        for i, kp in enumerate(kpts):
            visible = len(kp) < 3 or kp[2] > 0
            x, y = int(kp[0] * w), int(kp[1] * h)
            pts.append((x, y) if visible and (kp[0] > 0 or kp[1] > 0) else None)
            if pts[-1] is not None:
                cv2.circle(img, pts[-1], 4, KPT_COLORS[i % len(KPT_COLORS)], -1, cv2.LINE_AA)
        if len(pts) >= 2 and pts[0] is not None and pts[1] is not None:
            cv2.arrowedLine(img, pts[1], pts[0], BOX_COLOR, 2, cv2.LINE_AA, tipLength=0.25)
