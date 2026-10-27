import json

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


class AppSettings:
    """In-memory copy of app_settings.json, shared by every tab; each change
    is written straight back to disk."""

    def __init__(self) -> None:
        self._values = load_app_settings()

    def __getitem__(self, key: str):
        return self._values[key]

    def set(self, **values) -> None:
        self._values.update(values)
        save_app_settings(self._values)
