"""Saving tools in a job: the job JSON keeps each tool as a dict; taught
template images are PNG files under `tools_dir/<part>/<tool>.png` and the
dict stores their path relative to `tools_dir`.

`tools_dir` is passed in (the app uses `paths.TOOLS_DIR`) because `app.paths`
needs Qt and this package does not.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import cv2

from .base import Tool

log = logging.getLogger(__name__)
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(name: str, fallback: str) -> str:
    return _SAFE_RE.sub("_", name.strip()).strip("_") or fallback


def tools_to_config(tools: List[Tool], part_no: str, tools_dir: Path) -> List[dict]:
    """JSON-safe dicts for the job file. Writes each tool's template to PNG
    and removes this job's PNGs that no tool uses any more."""
    part_dir = Path(tools_dir) / _safe(part_no, "unknown_part")
    used, kept = set(), []
    entries = []
    for i, tool in enumerate(tools):
        entry = tool.to_dict()
        if tool.template is not None:
            stem = _safe(tool.name, f"tool{i}")
            while stem in used:
                stem += f"_{i}"
            used.add(stem)
            part_dir.mkdir(parents=True, exist_ok=True)
            path = part_dir / f"{stem}.png"
            if cv2.imwrite(str(path), tool.template):
                entry["template_file"] = f"{part_dir.name}/{path.name}"
                kept.append(path)
            else:
                log.warning("could not write template for tool %r", tool.name)
                entry.pop("template_file", None)
        else:
            entry.pop("template_file", None)
        entries.append(entry)
    if part_dir.is_dir():
        for old in part_dir.glob("*.png"):
            if old not in kept:
                try:
                    old.unlink()
                except OSError:
                    pass
    return entries


def tools_from_config(entries: Optional[list], part_no: str, tools_dir: Path) -> List[Tool]:
    """Tools from a job's "tools" list (None / missing = no tools). A broken
    entry is logged and skipped; a missing template leaves the tool untaught."""
    tools = []
    for entry in entries or []:
        try:
            tool = Tool.from_dict(entry)
        except Exception as exc:
            log.warning("skipping broken tool entry %r: %s", entry, exc)
            continue
        if tool.template_file:
            image = cv2.imread(str(Path(tools_dir) / tool.template_file), cv2.IMREAD_UNCHANGED)
            if image is None:
                log.warning("template missing for tool %r: %s", tool.name, tool.template_file)
            tool.template = image
        tools.append(tool)
    return tools
