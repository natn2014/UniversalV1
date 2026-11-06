"""Augmented sample images for the Training Options preview, made by
Ultralytics' own training data pipeline so they match what the trainer sees."""

import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

BOX_COLOR = (0, 210, 255)  # BGR amber, like the app's ROI outline


def preview_samples(data_yaml: Path, hyp: dict, count: int = 6, imgsz: int = 640,
                    seed: int = None) -> List[np.ndarray]:
    """`count` augmented training images (BGR, boxes and class names drawn).
    Raises on an unreadable dataset; Ultralytics is imported lazily because
    loading it takes a few seconds."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    cfg = get_cfg(overrides={**hyp, "imgsz": imgsz, "cache": False})
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
        samples.append(img)
    return samples
