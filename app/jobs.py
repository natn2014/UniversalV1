"""Saved jobs: one part_configs/<part>.json per part number, plus the
.last_part marker used to auto-load the previous job on boot."""

import json
from typing import Optional

from .paths import LAST_PART_FILE, PART_CONFIGS_DIR, sanitize_part_no


def list_saved_jobs() -> list:
    """[(part_no, config_path, config)] for every readable saved job."""
    jobs = []
    config_paths = sorted(PART_CONFIGS_DIR.glob("*.json")) if PART_CONFIGS_DIR.exists() else []
    for config_path in config_paths:
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        jobs.append((config.get("part_no") or config_path.stem, config_path, config))
    return jobs


def save_part_config(part_no: str, config: dict) -> None:
    PART_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = PART_CONFIGS_DIR / f"{sanitize_part_no(part_no)}.json"
    path.write_text(json.dumps(config, indent=2))


def load_part_config(part_no: str) -> Optional[dict]:
    path = PART_CONFIGS_DIR / f"{sanitize_part_no(part_no)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_last_part() -> Optional[str]:
    if not LAST_PART_FILE.exists():
        return None
    return LAST_PART_FILE.read_text().strip() or None


def write_last_part(part_no: str) -> None:
    PART_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    LAST_PART_FILE.write_text(part_no)


def clear_last_part() -> None:
    try:
        LAST_PART_FILE.unlink()
    except OSError:
        pass
