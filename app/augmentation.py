"""Training augmentation presets, passed to Ultralytics' model.train().

Ultralytics augments by default, including a left-right mirror on half the
images (fliplr=0.5) and mosaic. Mirroring breaks printed text and parts
whose orientation matters, so build_hyp() always sets both flips
explicitly and the default here is no mirror.
"""

from typing import Dict

# Every key any preset uses, so switching presets never leaves an
# Ultralytics default (e.g. mosaic=1.0) active by omission.
_ZERO = {
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "erasing": 0.0,
}

PRESETS: Dict[str, Dict[str, float]] = {
    "off": dict(_ZERO),
    # Fixed camera station: mostly lighting changes, small shifts.
    "light": {**_ZERO, "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.3, "degrees": 3.0,
              "translate": 0.05, "scale": 0.2, "mosaic": 0.5, "erasing": 0.2},
    # Few images or changing setup: Ultralytics-like strength plus tilt.
    "strong": {**_ZERO, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 10.0,
               "translate": 0.1, "scale": 0.5, "shear": 2.0, "perspective": 0.0005,
               "mosaic": 1.0, "mixup": 0.1, "erasing": 0.4},
}

PRESET_LABELS = {"off": "Off", "light": "Light", "strong": "Strong"}
PRESET_HINTS = {
    "off": "No changes: trains on the images exactly as captured.",
    "light": "Small brightness, colour, shift and zoom changes. Best for a fixed camera station.",
    "strong": "Bigger changes, tilt and image mixing. For few images or a changing setup.",
}
DEFAULT_PRESET = "light"
FLIP_PROBABILITY = 0.5


def build_hyp(preset: str, mirror: bool = False, flip_ud: bool = False) -> Dict[str, float]:
    """Ultralytics hyperparameters for a preset; unknown names use the default."""
    hyp = dict(PRESETS.get(preset, PRESETS[DEFAULT_PRESET]))
    hyp["fliplr"] = FLIP_PROBABILITY if mirror else 0.0
    hyp["flipud"] = FLIP_PROBABILITY if flip_ud else 0.0
    return hyp


def describe(preset: str, mirror: bool, flip_ud: bool) -> str:
    """Short status text, e.g. 'Augment: Light, no mirror'."""
    name = PRESET_LABELS.get(preset, PRESET_LABELS[DEFAULT_PRESET])
    flips = [f for f, on in (("mirror", mirror), ("upside-down", flip_ud)) if on]
    return f"Augment: {name}, " + ("+".join(flips) if flips else "no mirror")
