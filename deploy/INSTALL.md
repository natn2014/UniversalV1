# Installing UniversalV1 on a Jetson Orin Nano

Tested on Ubuntu 24.04 (JetPack R39.2, CUDA 13.2), Python 3.12.

## New machine

```bash
git clone git@github.com:natn2014/UniversalV1.git ~/dev/UniversalV1
cd ~/dev/UniversalV1
git checkout imageprocessing
# copy the model(s) that are not in git, e.g. weights/MarkV9_best.pt
./deploy/install.sh
sudo reboot
```

Run it as the user the kiosk runs as, not with `sudo`. It asks for the sudo
password for the system steps, and asks before turning on automatic login and
changing the power mode (`--yes` skips the questions).

After the reboot the desktop logs in by itself, and the app starts the last job
and begins inspecting.

## The 12 steps

| # | Step | What it does |
|---|------|--------------|
| 1 | preflight | Jetson, Ubuntu 24.04, CUDA, Python 3.12, 10 GB free, internet, sudo |
| 2 | system | apt: Python build tools, Qt/X11 libraries, zbar, Tesseract (+ headers), HID, fonts with Thai |
| 3 | devices | tower light udev rule; user in `video`, `plugdev`, `dialout` |
| 4 | venv | `.venv` with pip / setuptools / wheel |
| 5 | torch | PyTorch with CUDA **first**, then checked on the GPU |
| 6 | opencv | `opencv-python-headless` |
| 7 | ultralytics | YOLO; removes the `opencv-python` it pulls in (its Qt breaks PySide6) and reinstalls headless |
| 8 | pyside6 | Qt user interface |
| 9 | extras | hidapi, pyzbar, pytesseract, tesserocr (fast OCR; optional) |
| 10 | verify | every import, the GPU, the model file, the test suite |
| 11 | power | nvpmodel 25 W (MAXN SUPER trips over-current on this board) |
| 12 | autostart | GDM automatic login, no screen lock / blanking, systemd user service |

Versions are pinned in `deploy/constraints-jetson.txt` (frozen from the working
machine). Each step checks its result before the next starts. On a failure, fix
the cause and resume with `./deploy/install.sh --from N`. The log is in
`logs/install.log`.

## Day to day

```bash
./deploy/install.sh --check              # what is installed / missing (no changes)
systemctl --user status universalv1      # is the app running
systemctl --user restart universalv1     # restart the app
journalctl --user -u universalv1 -f      # service messages
tail -f logs/app.log                     # the app's own output
./deploy/install.sh --remove-service     # no more auto start
```

- **Restarts:** the service restarts the app if it crashes, up to 5 times in 2 minutes.
- **Closing on purpose:** closing the window doesn't restart it; it starts again at the next boot / login.
- **One copy at a time:** a second start (for example from VS Code while the service runs) says
  "already running" and exits.

`deploy/wire-harness-detector.service` is from the earlier wire-harness project
(other paths) and is not used.
