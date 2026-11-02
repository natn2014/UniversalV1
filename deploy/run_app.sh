#!/usr/bin/env bash
# Starts the UniversalV1 inspection app on the desktop. Run by the systemd user
# service universalv1.service (deploy/install.sh installs it); can also be run
# by hand.
#   - waits (up to 120 s) for the desktop's X display after boot / login
#   - only one copy: a second start (service + VS Code) would fight over the
#     camera and the tower light, so it exits at once if one is running
#   - output goes to logs/app.log (kept to the last 5 MB, one old copy)
set -uo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/logs"
LOG="$LOG_DIR/app.log"
LOCK="${XDG_RUNTIME_DIR:-/tmp}/universalv1.lock"
mkdir -p "$LOG_DIR"

log() { echo "$(date '+%F %T') [run_app] $*" | tee -a "$LOG"; }

# One copy only.
exec 9>"$LOCK"
if ! flock -n 9; then
    log "already running (lock $LOCK held), not starting a second copy"
    exit 0
fi

# Keep the log small: past 5 MB it becomes app.log.1.
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt $((5 * 1024 * 1024)) ]; then
    mv -f "$LOG" "$LOG.1"
fi

# The desktop's display: GNOME passes DISPLAY to the user service; when it
# hasn't (yet), take the X server this user's session runs.
if [ -z "${DISPLAY:-}" ]; then
    for sock in /tmp/.X11-unix/X*; do
        [ -S "$sock" ] && [ -O "$sock" ] && DISPLAY=":${sock##*/X}" && break
    done
    DISPLAY="${DISPLAY:-:0}"
fi
export DISPLAY
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export QT_QPA_PLATFORM=xcb

for i in $(seq 1 120); do
    if [ -S "/tmp/.X11-unix/X${DISPLAY#:}" ] && "$APP_DIR/.venv/bin/python" - <<'EOF' 2>/dev/null
import os, sys
from PySide6.QtGui import QGuiApplication
app = QGuiApplication(sys.argv)   # fails while the display isn't ready
EOF
    then
        break
    fi
    [ "$i" -eq 1 ] && log "waiting for display $DISPLAY ..."
    sleep 1
done

cd "$APP_DIR" || exit 1
log "starting (display $DISPLAY)"
"$APP_DIR/.venv/bin/python" "$APP_DIR/main.py" >>"$LOG" 2>&1
code=$?
log "app exited with code $code"
exit $code
