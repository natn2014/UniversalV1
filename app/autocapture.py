"""Auto capture: save live frames that look like any of the job's master
images into one folder per job, for labelling by hand in the Train tab
(object detection). No labels are written here: an image only joins
training once its boxes are drawn.

_autocapture_/<part>/images/<stamp>_m<N>.jpg   (N = master that matched)
_autocapture_/<part>/labels/                   (written by the Train tab)
part_configs/masters/<part>/master_<stamp>.jpg
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from PySide6.QtGui import QImage

from .paths import AUTOCAPTURE_DIR, IMAGE_EXTENSIONS, MASTERS_DIR, NO_PART_NAME, sanitize_part_no

MAX_MASTERS = 9
# A frame this close to the last saved one is the same view again (part not
# moved yet): skipped, so a still part doesn't fill the dataset with copies.
DEDUP_BITS = 4
# "Wait for part to leave": after a save, the next one needs this many frames
# in a row farther than threshold + REARM_MARGIN from every master (part
# removed), or matching a different master (part swapped with no gap).
# The margin and the frame count stop a part right at the threshold (or a
# hand passing through) from re-arming the gate.
CLEAR_FRAMES = 3
REARM_MARGIN = 4


def image_hash(image: np.ndarray) -> int:
    """64-bit difference hash (dHash): grayscale, shrink to 9x8, one bit per
    'left pixel brighter than right'. Insensitive to size, small noise and
    overall brightness; sensitive to layout."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, :-1] > small[:, 1:]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def qimage_to_array(image: QImage) -> np.ndarray:
    """RGB uint8 array (a copy) of a QImage."""
    image = image.convertToFormat(QImage.Format_RGB888)
    width, height = image.width(), image.height()
    buffer = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes())
    rows = buffer.reshape(height, image.bytesPerLine())
    return rows[:, : width * 3].reshape(height, width, 3).copy()


@dataclass
class Master:
    image: str  # path relative to MASTERS_DIR, e.g. "<part>/master_<stamp>.jpg"
    hash: int

    @property
    def path(self) -> Path:
        return MASTERS_DIR / self.image

    def to_entry(self) -> dict:
        return {"image": self.image}


def save_master(part: str, image: QImage) -> str:
    """Writes a master image under a new unique name (so a removed master's
    file is never reused by another); returns its entry path. Old files are
    left alone: a job saved as a copy of this one may still use them.
    Raises OSError if it can't be written."""
    stem = f"{sanitize_part_no(part)}/master_{capture_stem()}"
    rel, n = f"{stem}.jpg", 1
    while (MASTERS_DIR / rel).exists():  # several saved in the same millisecond
        rel, n = f"{stem}_{n}.jpg", n + 1
    path = MASTERS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(path), "JPG", 95):
        raise OSError(f"cannot write {path}")
    return rel


def load_masters(entries: Optional[Sequence]) -> List[Master]:
    """Masters from a job config's "autocapture_masters" list; empty entries
    (older configs had fixed slots) and unreadable images are dropped, and
    an old "class_name" key is ignored. Hashes are recomputed from the
    files, so they can't go stale."""
    masters: List[Master] = []
    for entry in list(entries or [])[:MAX_MASTERS]:
        if not isinstance(entry, dict) or not entry.get("image"):
            continue
        image = cv2.imread(str(MASTERS_DIR / entry["image"]))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masters.append(Master(entry["image"], image_hash(image)))
    return masters


def autocapture_part_dir(part: Optional[str]) -> Path:
    return AUTOCAPTURE_DIR / sanitize_part_no(part or NO_PART_NAME)


def autocapture_images_dir(part: Optional[str]) -> Path:
    """Where a job's auto captures go; a Train-tab dataset (labels/ beside it)."""
    return autocapture_part_dir(part) / "images"


def count_images(part_dir: Path) -> int:
    """Images already captured for a part."""
    images_dir = part_dir / "images"
    if not images_dir.is_dir():
        return 0
    return sum(1 for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def capture_stem() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"


class AutoCaptureGate:
    """Decides, frame by frame, whether to save (and which master matched)."""

    def __init__(self, threshold: int, interval_s: float, max_images: int, count: int = 0,
                 require_clear: bool = False) -> None:
        self.threshold = threshold
        self.interval_s = interval_s
        self.max_images = max_images
        self.count = count
        self.require_clear = require_clear
        self._last_saved_at: Optional[float] = None
        self._last_saved_hash: Optional[int] = None
        self._armed = True
        self._clear_frames = 0
        self._last_master: Optional[Master] = None

    @property
    def full(self) -> bool:
        return self.count >= self.max_images

    @property
    def waiting(self) -> bool:
        """Saved one; waiting for the part to leave before the next."""
        return not self._armed

    @staticmethod
    def nearest(frame_hash: int, masters: Sequence[Master]):
        """(master, distance) of the closest master, (None, None) if none."""
        best, best_distance = None, None
        for master in masters:
            if master is None:
                continue
            distance = hamming(frame_hash, master.hash)
            if best_distance is None or distance < best_distance:
                best, best_distance = master, distance
        return best, best_distance

    def check(self, frame_hash: int, masters: Sequence[Master], now: float) -> Optional[Master]:
        if self.full:
            return None
        master, distance = self.nearest(frame_hash, masters)
        if not self._armed:
            clear = (distance is None or distance > self.threshold + REARM_MARGIN
                     or (distance <= self.threshold and master is not self._last_master))
            self._clear_frames = self._clear_frames + 1 if clear else 0
            if self._clear_frames >= CLEAR_FRAMES:
                # Part left: the next one may sit exactly where this one did.
                self._armed = True
                self._last_saved_hash = None
            return None
        if self._last_saved_at is not None and now - self._last_saved_at < self.interval_s:
            return None
        if self._last_saved_hash is not None and hamming(frame_hash, self._last_saved_hash) < DEDUP_BITS:
            return None
        if master is None or distance > self.threshold:
            return None
        return master

    def record_saved(self, frame_hash: int, now: float, master: Optional[Master] = None) -> None:
        self.count += 1
        self._last_saved_at = now
        self._last_saved_hash = frame_hash
        self._last_master = master
        if self.require_clear:
            self._armed = False
            self._clear_frames = 0
