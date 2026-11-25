from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QPushButton, QSlider

from ...context import AppContext
from ...image_adjust import ImageAdjust
from ...theme import make_icon
from ...widgets.layout import make_card, row, tip_label

# (setting suffix, caption, low, high, step, format of the shown value, hint)
SLIDERS: Tuple = (
    ("brightness", "Brightness", -100, 100, 5, "{:+d}", "Lighter or darker overall."),
    ("contrast", "Contrast", 50, 200, 5, "{} %", "Stretches dark and light apart. 100 % = as the camera gives it."),
    ("saturation", "Colour", 0, 200, 5, "{} %", "Colour strength. 0 % = gray, 100 % = as the camera gives it."),
    ("gamma", "Shadows", 50, 200, 5, "{:.2f}", "Above 1.00 lifts dark areas without washing out the light ones."),
)


def make_image_card(ctx: AppContext):
    """Camera > Image card: mirror / flip tiles, enhancement sliders, local
    contrast and Reset. Every change goes to app settings at once (the worker
    picks it up on the next frame); returns (card, reload) where reload()
    puts the widgets back to the settings after Revert."""
    settings = ctx.settings
    card, body = make_card("Image", icon="camera")

    mirror = QPushButton("Mirror  ⇆")
    flip = QPushButton("Flip  ⇅")
    for button, key, hint in ((mirror, "camera_mirror", "Swap left and right (camera looks via a mirror or is mounted reversed)."),
                              (flip, "camera_flip", "Turn the image upside down (camera mounted upside down).")):
        button.setCheckable(True)
        button.setMinimumHeight(48)
        button.setToolTip(hint)
        button.toggled.connect(lambda checked, k=key: settings.set(**{k: checked}))
    body.addLayout(row(mirror, flip))

    widgets: List = []
    for key, caption, low, high, step, fmt, hint in SLIDERS:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setSingleStep(step)
        slider.setPageStep(step * 4)
        value_label = QLabel()
        value_label.setMinimumWidth(64)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setStyleSheet("color: #ffcf73; font-weight: 700;")
        scale = 100.0 if key == "gamma" else 1

        def changed(value, k=key, f=fmt, s=scale, label=value_label):
            shown = value / s if s != 1 else value
            label.setText(f.format(shown))
            settings.set(**{f"camera_{k}": round(shown, 2) if s != 1 else int(value)})

        slider.valueChanged.connect(changed)
        body.addLayout(row(tip_label(caption, hint, slider), None, value_label))
        body.addWidget(slider)
        widgets.append((key, slider, scale, changed))

    clahe = QCheckBox("Local contrast (even out uneven light)")
    clahe.setToolTip("Brings out print in dark or bright corners (CLAHE). Costs a little speed.")
    clahe.toggled.connect(lambda checked: settings.set(camera_clahe=checked))
    reset = QPushButton("Reset to original")
    reset.setIcon(make_icon("refresh", "#e0e0e0"))
    reset.setToolTip("Mirror / flip off and every slider back to the camera's own image.")
    reset.clicked.connect(lambda: settings.set(**ImageAdjust().to_settings()))
    body.addWidget(clahe)
    body.addLayout(row(None, reset))
    hint = QLabel("Mirror / flip moves where ROIs and tools sit. The AI was trained on the camera's own "
                  "image: after enhancing, check the confidences in the preview.")
    hint.setObjectName("hint")
    hint.setWordWrap(True)
    body.addWidget(hint)

    def reload(_keys=None) -> None:
        mirror.setChecked(settings["camera_mirror"])
        flip.setChecked(settings["camera_flip"])
        clahe.setChecked(settings["camera_clahe"])
        for key, slider, scale, changed in widgets:
            slider.setValue(int(round(settings[f"camera_{key}"] * scale)))
            changed(slider.value())  # the value label even when the slider did not move

    reload()
    settings.changed.connect(lambda keys: reload() if any(k.startswith("camera_") for k in keys) else None)
    card.mirror_button, card.flip_button, card.clahe_check, card.reset_button = mirror, flip, clahe, reset
    card.sliders = {key: slider for key, slider, _, _ in widgets}
    return card, reload
