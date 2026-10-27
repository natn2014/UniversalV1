"""Pre-filters and shading correction. Pure functions: uint8 image in,
uint8 image out (gray or BGR). Tools run them through `apply_chain` using
their `pre` param, e.g. [{"op": "flat_field", "kernel": 51}]."""

from typing import Callable, Dict, List

import cv2
import numpy as np


def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _as_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def to_hsv(img: np.ndarray) -> np.ndarray:
    """BGR (or gray, promoted) -> HSV. A 3-channel input is assumed BGR."""
    return cv2.cvtColor(_as_bgr(img), cv2.COLOR_BGR2HSV)


def to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_as_bgr(img), cv2.COLOR_BGR2LAB)


def _per_channel(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[..., np.ndarray]:
    """Lets a gray-only filter accept BGR by filtering each channel."""
    def wrapper(img: np.ndarray, *args, **kw) -> np.ndarray:
        if img.ndim == 2:
            return fn(img, *args, **kw)
        return cv2.merge([fn(c, *args, **kw) for c in cv2.split(img)])
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


@_per_channel
def flat_field(gray: np.ndarray, kernel: int = 51) -> np.ndarray:
    """Divides out the slow brightness gradient (uneven lighting, shading)
    and restores the mean level."""
    k = _odd(kernel)
    bg = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32)
    out = gray.astype(np.float32) / np.maximum(bg, 1.0) * float(bg.mean())
    return np.clip(out, 0, 255).astype(np.uint8)


@_per_channel
def top_hat(gray: np.ndarray, kernel: int = 15) -> np.ndarray:
    """Small bright features (dots, specks) on a smooth background."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(kernel),) * 2)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)


@_per_channel
def black_hat(gray: np.ndarray, kernel: int = 15) -> np.ndarray:
    """Small dark features (pits, dirt)."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(kernel),) * 2)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)


@_per_channel
def clahe(gray: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Local contrast equalisation."""
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles))).apply(gray)


OPS: Dict[str, Callable[..., np.ndarray]] = {
    "gray": to_gray, "hsv": to_hsv, "lab": to_lab,
    "flat_field": flat_field, "top_hat": top_hat, "black_hat": black_hat, "clahe": clahe,
}


def apply_chain(img: np.ndarray, steps: List[dict]) -> np.ndarray:
    """Runs the steps in order. An unknown op raises ValueError."""
    for step in steps or []:
        step = dict(step)
        op = step.pop("op", None)
        if op not in OPS:
            raise ValueError(f"unknown pre-filter op: {op!r}")
        img = OPS[op](img, **step)
    return img
