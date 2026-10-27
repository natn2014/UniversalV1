import re
from pathlib import Path

from PySide6.QtCore import QSize

# Runtime data lives next to main.py (the repo root), not inside this package.
ROOT_DIR = Path(__file__).resolve().parent.parent

CAPTURES_DIR = ROOT_DIR / "captures"
CAPTURES_IMAGES_DIR = CAPTURES_DIR / "images"
CAPTURES_LABELS_DIR = CAPTURES_DIR / "labels"
# NG log: ng_logs/<part>/<timestamp>.jpg (annotated frame) + <timestamp>_thumb.jpg
NG_LOGS_DIR = ROOT_DIR / "ng_logs"
NG_THUMB_SUFFIX = "_thumb.jpg"
NG_THUMB_SIZE = QSize(160, 120)  # 5 across fit the 1024-wide kiosk screen
NG_THUMBS_PER_PART = 5
NO_PART_NAME = "No Part"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PART_CONFIGS_DIR = ROOT_DIR / "part_configs"
# Auto capture (app/autocapture.py): dataset per part, master images per job.
AUTOCAPTURE_DIR = ROOT_DIR / "_autocapture_"
MASTERS_DIR = PART_CONFIGS_DIR / "masters"
# Rule-based vision tools (app/vision): taught template images per job.
TOOLS_DIR = PART_CONFIGS_DIR / "tools"
LAST_PART_FILE = PART_CONFIGS_DIR / ".last_part"
APP_SETTINGS_FILE = ROOT_DIR / "app_settings.json"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_part_no(part_no: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", part_no.strip())
    return cleaned.strip("_") or "unknown_part"
