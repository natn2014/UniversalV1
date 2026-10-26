"""Entry point; the application lives in the app/ package."""

import os

# Before OpenCV loads: an unplugged camera then fails a frame grab after 2 s
# instead of 10 s, so the reconnect logic in app/camera.py reacts quickly.
os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "2")

from app.main_window import main  # noqa: E402

if __name__ == "__main__":
    main()
