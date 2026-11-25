"""Test package.

PySide6 destroys the Qt objects still alive when Python exits, and several
test modules leave windows behind (deleteLater with no event loop afterwards).
On the Jetson (PySide6 6.12) that cleanup segfaults in QWidget::destroy after
every test has passed, so the run ends with exit code 139 and the installer
reports "some tests failed". Closing and deleting the windows first, in an
atexit hook registered after QtCore's own (atexit runs last-in, first-out),
lets Python exit cleanly.
"""

import atexit
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402  (registers PySide's cleanup first)


def _close_windows() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for widget in QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


atexit.register(_close_windows)
