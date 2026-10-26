"""On-disk NG image log: ng_logs/<part>/<timestamp>.jpg plus a thumbnail."""

import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from .paths import NG_LOGS_DIR, NG_THUMB_SIZE, NG_THUMB_SUFFIX, sanitize_part_no


def save_ng_image(pixmap: QPixmap, part: str) -> None:
    """Raises OSError if the part folder can't be written."""
    part_dir = NG_LOGS_DIR / sanitize_part_no(part)
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / ".part_name").write_text(part)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    pixmap.save(str(part_dir / f"{stamp}.jpg"), "JPG", 90)
    pixmap.scaled(NG_THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation).save(
        str(part_dir / f"{stamp}{NG_THUMB_SUFFIX}"), "JPG", 85)


def collect_ng_log() -> list:
    """[(part_name, [thumb paths newest first], total count)], parts with
    the most recent NG first."""
    groups = []
    if not NG_LOGS_DIR.is_dir():
        return groups
    for part_dir in NG_LOGS_DIR.iterdir():
        if not part_dir.is_dir():
            continue
        # Timestamped names sort chronologically.
        thumbs = sorted(part_dir.glob(f"*{NG_THUMB_SUFFIX}"), reverse=True)
        if not thumbs:
            continue
        name_file = part_dir / ".part_name"
        name = name_file.read_text().strip() if name_file.exists() else part_dir.name
        groups.append((name, thumbs, len(thumbs)))
    groups.sort(key=lambda g: g[1][0].name, reverse=True)
    return groups


def ng_stamp_text(thumb_path: Path) -> str:
    stamp = thumb_path.name[: -len(NG_THUMB_SUFFIX)]
    try:
        t = time.strptime(stamp[:15], "%Y%m%d_%H%M%S")
        return time.strftime("%Y-%m-%d %H:%M:%S", t)
    except ValueError:
        return stamp


def full_image_for_thumb(thumb_path: Path) -> Path:
    return thumb_path.with_name(thumb_path.name[: -len(NG_THUMB_SUFFIX)] + ".jpg")
