"""On-disk NG image log: ng_logs/<part>/<timestamp>.jpg plus a thumbnail.

The log is kept under a size limit (Settings, default 5 GB): when a save takes
it over, the oldest images (full + thumbnail, any part) are deleted until it
is at 90 % of the limit, so it is not trimmed on every save.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from .paths import NG_LOGS_DIR, NG_THUMB_SIZE, NG_THUMB_SUFFIX, sanitize_part_no


PRUNE_TO = 0.9  # after a clean-up the log is at this share of the limit

_total_bytes: Optional[int] = None  # the log's size, scanned once, then kept up to date


def save_ng_image(pixmap: QPixmap, part: str, max_bytes: int = 0) -> int:
    """Saves the NG image and its thumbnail; with `max_bytes` > 0 the log is
    then kept under it. Returns how many old images were deleted. Raises
    OSError if the part folder can't be written."""
    global _total_bytes
    part_dir = NG_LOGS_DIR / sanitize_part_no(part)
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / ".part_name").write_text(part)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    full, thumb = part_dir / f"{stamp}.jpg", part_dir / f"{stamp}{NG_THUMB_SUFFIX}"
    pixmap.save(str(full), "JPG", 90)
    pixmap.scaled(NG_THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation).save(str(thumb), "JPG", 85)
    if _total_bytes is None:
        _total_bytes = ng_log_size()
    else:
        _total_bytes += sum(f.stat().st_size for f in (full, thumb) if f.exists())
    if max_bytes > 0 and _total_bytes > max_bytes:
        return prune_ng_log(int(max_bytes * PRUNE_TO))[0]
    return 0


def _images() -> List[Tuple[str, Path, List[Path], int]]:
    """(stamp, part folder, [full, thumb], bytes) of every NG image; stamps
    are timestamps, so they sort oldest first across parts."""
    images = []
    if not NG_LOGS_DIR.is_dir():
        return images
    for part_dir in NG_LOGS_DIR.iterdir():
        if not part_dir.is_dir():
            continue
        for full in part_dir.glob("*.jpg"):
            if full.name.endswith(NG_THUMB_SUFFIX):
                continue
            stamp = full.stem
            files = [f for f in (full, part_dir / f"{stamp}{NG_THUMB_SUFFIX}") if f.exists()]
            images.append((stamp, part_dir, files, sum(f.stat().st_size for f in files)))
    images.sort(key=lambda i: i[0])
    return images


def ng_log_size() -> int:
    """Bytes of NG images (full + thumbnail) in the log."""
    return sum(i[3] for i in _images())


def prune_ng_log(max_bytes: int) -> Tuple[int, int]:
    """Deletes the oldest NG images until the log is at most `max_bytes`;
    a part folder left empty goes too. Returns (images deleted, bytes freed)."""
    global _total_bytes
    images = _images()
    total = sum(i[3] for i in images)
    deleted = freed = 0
    for _, part_dir, files, size in images:
        if total <= max_bytes:
            break
        for f in files:
            try:
                f.unlink()
            except OSError:
                pass
        total -= size
        freed += size
        deleted += 1
        if not any(part_dir.glob("*.jpg")):
            for leftover in part_dir.iterdir():
                leftover.unlink(missing_ok=True)
            part_dir.rmdir()
    _total_bytes = total
    return deleted, freed


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
