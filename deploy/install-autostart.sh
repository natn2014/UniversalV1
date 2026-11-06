#!/usr/bin/env bash
# Kiosk autostart for UniversalV1: a systemd *user* service that starts the
# app when the desktop session comes up, plus "never blank / never lock"
# screen settings. Run as the station user (jetson), not with sudo.
#
#   ./deploy/install-autostart.sh            install + enable (starts at next login/boot)
#   ./deploy/install-autostart.sh --now      install + enable + start now
#   ./deploy/install-autostart.sh uninstall  stop, disable and remove the service
#   ./deploy/install-autostart.sh status     service state and recent log
set -euo pipefail

UNIT=wire-harness-detector.service
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as the station user (e.g. jetson), not root/sudo: it installs a *user* service." >&2
    exit 1
fi

kiosk_screen() {
    # Keep the Monitor visible all shift: no blanking, dimming, lock or sleep.
    if ! command -v gsettings >/dev/null; then
        echo "  gsettings not found; skipped screen settings." >&2
        return
    fi
    gsettings set org.gnome.desktop.session idle-delay 0
    gsettings set org.gnome.desktop.screensaver lock-enabled false
    gsettings set org.gnome.desktop.screensaver idle-activation-enabled false
    gsettings set org.gnome.settings-daemon.plugins.power idle-dim false
    gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
    echo "  Screen: never blank, dim, lock or sleep."
}

check_autologin() {
    local conf=/etc/gdm3/custom.conf
    if grep -qiE "^\s*AutomaticLoginEnable\s*=\s*true" "$conf" 2>/dev/null \
        && grep -qE "^\s*AutomaticLogin\s*=\s*$USER\s*$" "$conf" 2>/dev/null; then
        echo "  GDM autologin: on for $USER."
    else
        echo "  WARNING: GDM autologin for $USER is not set, so the app only starts after someone logs in." >&2
        echo "  Add under [daemon] in $conf (needs sudo):" >&2
        echo "      AutomaticLoginEnable=true" >&2
        echo "      AutomaticLogin=$USER" >&2
    fi
}

install() {
    [ -x "$REPO/.venv/bin/python3" ] || { echo "No $REPO/.venv; create the venv first (see README)." >&2; exit 1; }
    mkdir -p "$UNIT_DIR"
    sed "s|@REPO@|$REPO|g" "$REPO/deploy/$UNIT" > "$UNIT_DIR/$UNIT"
    systemctl --user daemon-reload
    systemctl --user enable "$UNIT" >/dev/null
    echo "Installed $UNIT_DIR/$UNIT (app: $REPO/main.py), enabled at login."
    kiosk_screen
    check_autologin
    if [ "${1:-}" = "--now" ]; then
        systemctl --user restart "$UNIT"
        sleep 3
        echo "Started: $(systemctl --user is-active "$UNIT")"
    else
        echo "It starts at the next boot/login. Start it now with: systemctl --user start $UNIT"
    fi
}

uninstall() {
    systemctl --user disable --now "$UNIT" 2>/dev/null || true
    rm -f "$UNIT_DIR/$UNIT"
    systemctl --user daemon-reload
    echo "Removed $UNIT. Screen settings were left as they are."
}

status() {
    systemctl --user --no-pager status "$UNIT" || true
    echo
    # User-service logs land in the system journal on this image.
    journalctl --user-unit "$UNIT" -n 20 --no-pager || true
}

case "${1:-install}" in
    install|--now) install "${1:-}" ;;
    uninstall) uninstall ;;
    status) status ;;
    *) echo "Usage: $0 [--now | uninstall | status]" >&2; exit 2 ;;
esac
