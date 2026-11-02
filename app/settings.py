import json

from PySide6.QtCore import QObject, Signal

from .optional_deps import BUZZER_ON
from .paths import APP_SETTINGS_FILE

# Buzzer defaults to on so NG buzzes out of the box — the app auto-starts on
# boot, so an unpersisted default-off would silently disable the alarm.
DEFAULT_APP_SETTINGS = {
    "buzzer_enabled": True,
    "buzzer_pattern": BUZZER_ON,
    "ok_beep_enabled": True,
    "ok_beep_ms": 50,
    # Minimum time a STANDBY/OK/NG judgement is held before it may change.
    "state_hold_ms": 1000,
    # Camera capture size "WxH"; falls back to 640x480 if the camera can't.
    "camera_resolution": "1920x1080",
    # OK/NG cycle counters, persisted so a kiosk restart doesn't lose them.
    "ok_count": 0,
    "ng_count": 0,
    # Auto capture: stop at this many images per part, save when the frame
    # is within this many bits (of 64) of a master, at most once per interval.
    "autocapture_max_images": 200,
    "autocapture_threshold": 10,
    "autocapture_interval_ms": 1000,
    # After a save, wait until no master matches (part removed) before the
    # next one, so one part held in view is saved once, not every interval.
    "autocapture_wait_clear": True,
    # OCR / code tools run at most this often, in their own thread, so the
    # live view stays smooth.
    "slow_tool_hz": 2.0,
    # Camera image adjustments (app/image_adjust.py): how the camera is
    # mounted (mirror / flip) and colour enhancement, before anything sees it.
    "camera_mirror": False,
    "camera_flip": False,
    "camera_brightness": 0,
    "camera_contrast": 100,
    "camera_saturation": 100,
    "camera_gamma": 1.0,
    "camera_clahe": False,
    # Relay I/O board (app/relay_io.py, Waveshare Modbus POE ETH Relay 8DI/8DO).
    # Roles per channel "1".."8": DI none / trigger / confirm_ng, DO none / ok / ng.
    "relay_enabled": False,
    "relay_host": "192.168.1.201",
    "relay_port": 4196,
    "relay_address": 1,
    "relay_protocol": "rtu_tcp",
    "relay_di_roles": {},
    "relay_do_roles": {},
    "relay_do_pulse_ms": 0,  # 0 = OK / NG outputs stay on while the result lasts
    "language": "en",  # UI language: "en" / "th" (flags in the tab bar corner)
}


def load_app_settings() -> dict:
    settings = dict(DEFAULT_APP_SETTINGS)
    try:
        data = json.loads(APP_SETTINGS_FILE.read_text())
        for key, default in DEFAULT_APP_SETTINGS.items():
            if key in data:
                settings[key] = type(default)(data[key])
    except (OSError, ValueError, TypeError):
        pass
    return settings


def save_app_settings(settings: dict) -> None:
    data = {}
    try:
        data = json.loads(APP_SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        pass
    data.update(settings)
    APP_SETTINGS_FILE.write_text(json.dumps(data, indent=2))


# Written to disk at once: counters and operator toggles outside Settings.
# Every other key is a Settings value: live in memory at once, on disk only
# after commit() (the Settings tab's Save), back to the saved value on revert().
IMMEDIATE_KEYS = {"ok_count", "ng_count", "buzzer_enabled", "language"}


class AppSettings(QObject):
    """In-memory copy of app_settings.json, shared by every tab."""

    changed = Signal(list)  # keys whose value changed (set or revert)

    def __init__(self) -> None:
        super().__init__()
        self._values = load_app_settings()
        self._saved = dict(self._values)

    def __getitem__(self, key: str):
        return self._values[key]

    def set(self, **values) -> None:
        keys = [k for k, v in values.items() if self._values.get(k) != v]
        self._values.update(values)
        immediate = {k: v for k, v in values.items() if k in IMMEDIATE_KEYS}
        if immediate:
            self._saved.update(immediate)
            save_app_settings(immediate)
        if keys:
            self.changed.emit(keys)

    def unsaved_keys(self) -> list:
        return sorted(k for k, v in self._values.items() if self._saved.get(k) != v)

    def commit(self) -> None:
        if self.unsaved_keys():
            self._saved = dict(self._values)
            save_app_settings(self._saved)

    def revert(self) -> list:
        """Back to the saved values; returns the keys that changed."""
        keys = self.unsaved_keys()
        self._values.update({k: self._saved[k] for k in keys})
        if keys:
            self.changed.emit(keys)
        return keys
