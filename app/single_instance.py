"""One copy of the app at a time: the autostart service and a manual
`python3 main.py` would otherwise fight over the camera and tower light."""

import fcntl
import os
from pathlib import Path
from typing import IO, Optional

LOCK_NAME = "universalv1.lock"


def lock_path() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / LOCK_NAME


def acquire(path: Optional[Path] = None) -> Optional[IO]:
    """Returns the open lock file (keep a reference for the app's lifetime;
    the OS releases it when the process exits, even after a crash), or None
    if another copy already holds it."""
    handle = open(path or lock_path(), "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle
