"""Add lighting-changed copies of the training images to a YOLO-seg dataset.

    .venv/bin/python tools/light_aug_seg.py datasets/d92_seg datasets/d92_seg_light [--copies 4] [--seed 0]

Ultralytics' own colour augmentation (hsv_v) only scales brightness evenly.
A line camera also sees dim and over-bright frames, low contrast, uneven
light (one side in shadow), glare spots, warm/cool lamps and sensor noise.
Each copy gets 2-3 of those changes at random. They change only the pixels,
never the geometry, so the label file is copied unchanged.

OUT gets: images/train = the originals + COPIES variants each (name
`<stem>_L<k>.jpg`), images/val = the originals only (validation stays real
photos), labels to match, and its own data.yaml. Train on OUT with
tools/train_seg.py.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def gamma(img, rng):
    g = rng.choice([rng.uniform(0.45, 0.8), rng.uniform(1.3, 2.2)])  # brighter / darker mid-tones
    lut = ((np.arange(256) / 255.0) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def brightness_contrast(img, rng):
    alpha, beta = rng.uniform(0.5, 1.5), rng.uniform(-60, 60)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gradient_shadow(img, rng):
    """Light falling off across the image in a random direction."""
    h, w = img.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    low = rng.uniform(0.3, 0.7)
    gain = low + (1 - low) * t
    return np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def glare(img, rng):
    """A bright lamp reflection somewhere on the image."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
    r = rng.uniform(0.15, 0.45) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(60, 150)
    return np.clip(img.astype(np.float32) + spot[..., None], 0, 255).astype(np.uint8)


def tint(img, rng):
    """Warm or cool lamp colour (BGR gains)."""
    k = rng.uniform(0.08, 0.25) * rng.choice([-1, 1])
    gains = np.array([1 - k, 1.0, 1 + k], np.float32)
    return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def noise_blur(img, rng):
    """Low light: sensor noise, sometimes a little focus/motion blur."""
    out = img.astype(np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(
        0, rng.uniform(4, 14), img.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def clahe(img, rng):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=rng.uniform(1.5, 4.0), tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


OPS = [gamma, brightness_contrast, gradient_shadow, glare, tint, noise_blur, clahe]


def relight(img, rng):
    for op in rng.sample(OPS, rng.choice([2, 3])):
        img = op(img, rng)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--copies", type=int, default=4, help="lighting variants per training image")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    src, out = args.src.resolve(), args.out.resolve()
    rng = random.Random(args.seed)
    if out.exists():
        shutil.rmtree(out / "images", ignore_errors=True)
        shutil.rmtree(out / "labels", ignore_errors=True)
    n = 0
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for im in sorted((src / "images" / split).glob("*.jpg")):
            lab = src / "labels" / split / (im.stem + ".txt")
            shutil.copy2(im, out / "images" / split / im.name)
            shutil.copy2(lab, out / "labels" / split / lab.name)
            if split != "train":
                continue
            img = cv2.imread(str(im))
            for k in range(1, args.copies + 1):
                name = f"{im.stem}_L{k}"
                cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), relight(img, rng),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                shutil.copy2(lab, out / "labels" / split / f"{name}.txt")
                n += 1
    data = yaml.safe_load((src / "data.yaml").read_text())
    data["path"] = str(out)
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    if (src / "classes.txt").exists():
        shutil.copy2(src / "classes.txt", out / "classes.txt")
    print(f"{out}: +{n} lighting copies in train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
