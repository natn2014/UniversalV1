#!/usr/bin/env bash
# UniversalV1 installer for a Jetson Orin Nano (Ubuntu 24.04 / JetPack, CUDA 13).
#
#   ./deploy/install.sh                  install everything, step by step
#   ./deploy/install.sh --from 6         resume at step 6 (after fixing a failure)
#   ./deploy/install.sh --check          only check an install (no changes)
#   ./deploy/install.sh --no-autologin   leave the login screen as it is
#   ./deploy/install.sh --remove-service stop and remove the auto start
#   ./deploy/install.sh --yes            don't ask before system changes
#
# Run it as the user the kiosk runs as (not root, not with sudo); it asks for
# the sudo password for the system steps. Every step checks its own result
# before the next starts, so a failure stops at the step that caused it with
# the reason; the log is logs/install.log. Safe to run again: done steps are
# quick no-ops.
#
# Order, and why:
#   1  preflight      Jetson, Ubuntu 24.04, CUDA 13, Python 3.12, disk, network
#   2  system         apt packages: Python build, Qt/X11 libs, zbar, Tesseract
#                     (+ dev headers for tesserocr), HID, fonts with Thai
#   3  devices        tower light udev rule; user in video / plugdev / dialout
#   4  venv           .venv with pip / setuptools / wheel
#   5  torch          PyTorch with CUDA first, checked on the GPU, so nothing
#                     later can pull a CPU-only or mismatched build
#   6  opencv         opencv-python-headless (no own Qt: plain opencv-python's
#                     Qt plugins break PySide6)
#   7  ultralytics    YOLO; it pulls opencv-python, which is removed again; that
#                     also deletes the shared cv2 folder, so headless is reinstalled
#   8  pyside6        Qt for the UI
#   9  extras         hidapi, pyzbar, pytesseract, tesserocr (built against the
#                     system Tesseract; skipped if it can't build: pytesseract works)
#   10 verify         every import, torch on the GPU, the model files, the tests
#   11 power          nvpmodel 25 W (MAXN SUPER trips this board's over-current
#                     protection and reboots it); fan profile "cool", so long
#                     inference and training runs stay cooler
#   12 autostart      GDM automatic login, no screen lock, the systemd user service

set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$APP_DIR/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
CONSTRAINTS="$APP_DIR/deploy/constraints-jetson.txt"
LOG_DIR="$APP_DIR/logs"
LOG="$LOG_DIR/install.log"
SERVICE_NAME="universalv1.service"
SERVICE_DIR="$HOME/.config/systemd/user"
TORCH_INDEX="https://pypi.org/simple"   # CUDA 13 aarch64 wheels that see the Orin GPU (see requirements.txt)
MARK_MODEL="weights/MarkV9_best.pt"     # app/paths.py MARK_MODEL_PATH
FAN_CONF=/etc/nvfancontrol.conf
FAN_STATUS=/var/lib/nvfancontrol/status

FROM_STEP=1
CHECK_ONLY=0
AUTOLOGIN=1
ASSUME_YES=0

mkdir -p "$LOG_DIR"

# --- Output -------------------------------------------------------------------

if [ -t 1 ]; then
    BOLD=$'\e[1m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
    BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
log()  { echo "$(date '+%F %T') $*" >>"$LOG"; }
say()  { echo "$*"; log "$*"; }
ok()   { echo "  ${GREEN}OK${RESET}  $*"; log "OK $*"; }
warn() { echo "  ${YELLOW}!!${RESET}  $*"; log "WARN $*"; }
die()  {
    echo "  ${RED}FAILED${RESET}  $*"
    log "FAILED $*"
    echo
    echo "Fix the cause, then resume with:  ./deploy/install.sh --from ${CURRENT_STEP:-1}"
    echo "Details: $LOG"
    exit 1
}
trap 'die "unexpected error at line $LINENO: $BASH_COMMAND"' ERR

CURRENT_STEP=0
step() {
    CURRENT_STEP=$1
    echo
    say "${BOLD}[$1/12] $2${RESET}"
}
run() {  # a command with its output in the log; dies with the last lines shown
    log "\$ $*"
    if ! "$@" >>"$LOG" 2>&1; then
        tail -n 15 "$LOG" | sed 's/^/      /'
        die "command failed: $*"
    fi
}
confirm() {
    [ "$ASSUME_YES" -eq 1 ] && return 0
    read -r -p "  $1 [y/N] " answer
    [[ "$answer" =~ ^[Yy] ]]
}
pip_install() { run "$PIP" install --no-input -c "$CONSTRAINTS" "$@"; }
pyver() { "$PY" -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || echo "-"; }

# --- Steps --------------------------------------------------------------------

step_preflight() {
    step 1 "Preflight checks"
    [ "$(id -u)" -ne 0 ] || die "run as the kiosk user, not root / sudo (it asks for sudo itself)"
    [ "$(uname -m)" = "aarch64" ] || die "not an ARM64 (Jetson) machine: $(uname -m)"
    [ -f /etc/nv_tegra_release ] || die "no /etc/nv_tegra_release: not a Jetson with JetPack"
    ok "Jetson: $(head -1 /etc/nv_tegra_release | cut -d, -f1-2 | tr -d '#')"
    . /etc/os-release
    [ "${VERSION_ID:-}" = "24.04" ] || warn "Ubuntu $VERSION_ID (tested on 24.04)"
    ok "$PRETTY_NAME"
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        ok "CUDA $(/usr/local/cuda/bin/nvcc --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
    else
        warn "nvcc not found in /usr/local/cuda (PyTorch's wheels bring their own CUDA runtime)"
    fi
    command -v python3.12 >/dev/null || die "python3.12 not found"
    ok "Python $(python3.12 --version | cut -d' ' -f2)"
    local free_gb
    free_gb=$(df -BG --output=avail "$APP_DIR" | tail -1 | tr -dc '0-9')
    [ "$free_gb" -ge 10 ] || die "only ${free_gb} GB free; need 10 GB (PyTorch + CUDA libraries ~6 GB)"
    ok "${free_gb} GB free"
    curl -fsS --max-time 10 -o /dev/null https://pypi.org/simple/pip/ || die "no internet (pypi.org)"
    ok "internet"
    [ -f "$CONSTRAINTS" ] || die "missing $CONSTRAINTS"
    sudo -v || die "sudo needed for the system steps"
    ok "sudo"
}

step_system() {
    step 2 "System packages (apt)"
    local pkgs=(
        # Python and building tesserocr
        python3.12-venv python3.12-dev build-essential pkg-config git curl
        # Qt (PySide6) on X11
        libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1
        libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 libxcb-shape0 libegl1
        # barcode (pyzbar), OCR (Tesseract + headers for tesserocr)
        libzbar0t64 tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev
        # tower light (hidapi), camera tools
        libhidapi-hidraw0 libhidapi-libusb0 libusb-1.0-0 v4l-utils
        # fonts: Noto incl. Thai for the TH language
        fonts-noto-core fonts-dejavu-core
    )
    local missing=()
    for p in "${pkgs[@]}"; do
        dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p")
    done
    if [ ${#missing[@]} -eq 0 ]; then
        ok "all ${#pkgs[@]} packages already installed"
        return
    fi
    say "  to install: ${missing[*]}"
    run sudo apt-get update
    run sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
    ok "installed ${#missing[@]} packages"
}

step_devices() {
    step 3 "Devices: tower light rule, user groups"
    local rule=/etc/udev/rules.d/99-patlite-lr6.rules
    if ! cmp -s "$APP_DIR/deploy/99-patlite-lr6.rules" "$rule"; then
        run sudo install -m 644 "$APP_DIR/deploy/99-patlite-lr6.rules" "$rule"
        run sudo udevadm control --reload-rules
        run sudo udevadm trigger
        ok "tower light rule installed (unplug / replug the tower once)"
    else
        ok "tower light rule in place"
    fi
    local added=()
    for g in video plugdev dialout; do
        getent group "$g" >/dev/null || continue
        id -nG "$USER" | tr ' ' '\n' | grep -qx "$g" || { run sudo usermod -aG "$g" "$USER"; added+=("$g"); }
    done
    if [ ${#added[@]} -gt 0 ]; then
        warn "added $USER to: ${added[*]} (takes effect after the reboot at the end)"
    else
        ok "$USER in video, plugdev, dialout"
    fi
    ls /dev/video* >/dev/null 2>&1 && ok "camera: $(ls /dev/video* | tr '\n' ' ')" || warn "no camera found now (plug it in before starting)"
}

step_venv() {
    step 4 "Python virtual environment"
    if [ ! -x "$PY" ]; then
        run python3.12 -m venv "$VENV"
        ok "created $VENV"
    else
        ok "$VENV exists ($("$PY" --version))"
    fi
    pip_install --upgrade pip setuptools wheel
    ok "pip $(pyver pip)"
}

step_torch() {
    step 5 "PyTorch with CUDA (first, so nothing later replaces it)"
    pip_install --index-url "$TORCH_INDEX" torch torchvision
    say "  checking the GPU (first use compiles kernels for the Orin: up to a minute) ..."
    local out
    out=$("$PY" - 2>>"$LOG" <<'EOF'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.ones(256, 256, device="cuda")
y = (x @ x).sum().item()
print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, matmul ok ({y:.0f})")
EOF
    ) || die "PyTorch does not see the GPU (see $LOG)"
    ok "$out"
}

step_opencv() {
    step 6 "OpenCV (headless)"
    pip_install opencv-python-headless
    "$PY" -c "import cv2; cv2.imdecode" >>"$LOG" 2>&1 || die "cv2 does not import"
    ok "opencv-python-headless $(pyver opencv-python-headless)"
}

step_ultralytics() {
    step 7 "Ultralytics (YOLO)"
    local torch_before
    torch_before=$(pyver torch)
    pip_install ultralytics
    [ "$(pyver torch)" = "$torch_before" ] || die "ultralytics changed torch $torch_before -> $(pyver torch)"
    if [ "$(pyver opencv-python)" != "-" ]; then
        # Same cv2 folder as headless: removing it deletes cv2, so headless goes back in.
        run "$PIP" uninstall -y opencv-python
        pip_install --force-reinstall --no-deps opencv-python-headless
        ok "removed opencv-python (pulled by ultralytics), headless reinstalled"
    fi
    "$PY" -c "import cv2, ultralytics; print(ultralytics.__version__, cv2.__version__)" >>"$LOG" 2>&1 \
        || die "ultralytics / cv2 do not import"
    ok "ultralytics $(pyver ultralytics), torch still $(pyver torch)"
}

step_pyside() {
    step 8 "PySide6 (Qt user interface)"
    pip_install PySide6
    QT_QPA_PLATFORM=offscreen "$PY" -c "from PySide6.QtWidgets import QApplication; QApplication([])" >>"$LOG" 2>&1 \
        || die "PySide6 does not start (missing Qt libraries? see $LOG)"
    ok "PySide6 $(pyver PySide6)"
}

step_extras() {
    step 9 "Tower light, barcode and OCR modules"
    pip_install hidapi pyzbar pytesseract
    ok "hidapi $(pyver hidapi), pyzbar $(pyver pyzbar), pytesseract $(pyver pytesseract)"
    # tesserocr: Tesseract inside the app (~10 ms a read instead of ~190 ms). Builds
    # from source on ARM64; when it can't, the app uses pytesseract.
    if "$PIP" install --no-input -c "$CONSTRAINTS" tesserocr >>"$LOG" 2>&1; then
        ok "tesserocr $(pyver tesserocr) (fast OCR)"
    else
        warn "tesserocr did not build (see $LOG); OCR falls back to pytesseract (slower)"
    fi
}

step_verify() {
    step 10 "Verify the install"
    local out
    out=$(cd "$APP_DIR" && QT_QPA_PLATFORM=offscreen "$PY" - 2>>"$LOG" <<'EOF'
import importlib
for name in ("cv2", "numpy", "torch", "ultralytics", "PySide6.QtWidgets", "hid", "pyzbar.pyzbar"):
    importlib.import_module(name)
from app.optional_deps import TESSERACT, TESSEROCR, TowerLight, YOLO, zbar_decode
from app.vision.ocr import get_engine
missing = [n for n, v in (("YOLO", YOLO), ("tower light driver", TowerLight), ("zbar", zbar_decode),
                          ("Tesseract", TESSERACT or TESSEROCR)) if v is None]
assert not missing, "not available: " + ", ".join(missing)
import torch
assert torch.cuda.is_available()
print(f"imports ok, OCR engine {type(get_engine()).__name__}, GPU {torch.cuda.get_device_name(0)}")
EOF
    ) || die "a module is missing (see $LOG)"
    ok "$out"
    if [ -f "$APP_DIR/$MARK_MODEL" ]; then
        ok "mark model $MARK_MODEL"
    else
        warn "mark model $MARK_MODEL missing: copy it in (it is not in git) before using Required Marks"
    fi
    say "  running the test suite (about a minute) ..."
    if (cd "$APP_DIR" && QT_QPA_PLATFORM=offscreen "$PY" -m unittest discover -s tests >>"$LOG" 2>&1); then
        ok "test suite passed"
    else
        warn "some tests failed (see the end of $LOG); the app may still run"
    fi
}

step_power() {
    step 11 "Power mode 25 W and fan"
    set_power_mode
    set_fan_profile
}

set_power_mode() {
    command -v nvpmodel >/dev/null || { warn "nvpmodel not found: set the power mode by hand"; return; }
    local current
    current=$(nvpmodel -q 2>/dev/null | head -1)
    if echo "$current" | grep -q "25W"; then
        ok "already $current"
        return
    fi
    local id
    id=$(grep -oP '< POWER_MODEL ID=\K[0-9]+(?= NAME=25W >)' /etc/nvpmodel.conf || true)
    [ -n "$id" ] || { warn "no 25W mode in /etc/nvpmodel.conf; keeping $current"; return; }
    if confirm "Set power mode 25W (now: $current)? MAXN SUPER can trip over-current and reboot."; then
        echo "yes" | sudo nvpmodel -m "$id" >>"$LOG" 2>&1 || die "nvpmodel -m $id failed"
        ok "power mode: $(nvpmodel -q 2>/dev/null | head -1)"
    else
        warn "power mode left at $current"
    fi
}

fan_profile() { grep -oP '^\s*FAN_DEFAULT_PROFILE\s+\K\S+' "$FAN_CONF" 2>/dev/null || echo "-"; }

set_fan_profile() {
    # nvfancontrol stays in charge (the fan still follows the temperature), on
    # its faster "cool" curve. It keeps the profile it last ran in its status
    # file, which wins over the config, so that file is removed while it is
    # stopped; the start runs even if the removal fails, so the fan is never left
    # without its controller.
    [ -f "$FAN_CONF" ] || { warn "no $FAN_CONF: fan left as it is"; return; }
    local current
    current=$(fan_profile)
    if [ "$current" = "cool" ]; then
        ok "fan profile cool"
        return
    fi
    grep -qP '^\s*FAN_PROFILE\s+cool\b' "$FAN_CONF" || { warn "no cool profile in $FAN_CONF; fan left at $current"; return; }
    if confirm "Set the fan profile to cool (now: $current)? Faster fan, cooler board, a little louder."; then
        run sudo cp -n "$FAN_CONF" "$FAN_CONF.before-universalv1"
        run sudo sed -i -E 's/^(\s*FAN_DEFAULT_PROFILE\s+)\S+/\1cool/' "$FAN_CONF"
        run sudo sh -c "systemctl stop nvfancontrol; rm -f $FAN_STATUS; systemctl start nvfancontrol"
        systemctl is-active --quiet nvfancontrol || die "nvfancontrol did not start again: sudo systemctl start nvfancontrol"
        ok "fan profile cool (old file kept as $FAN_CONF.before-universalv1)"
    else
        warn "fan profile left at $current"
    fi
}

step_autostart() {
    step 12 "Start at boot"
    # 1. Log in automatically, so the desktop (and the app) comes up after boot.
    local conf=/etc/gdm3/custom.conf
    if [ "$AUTOLOGIN" -eq 1 ] && [ -f "$conf" ]; then
        if grep -q "^AutomaticLogin=$USER$" "$conf" && grep -q "^AutomaticLoginEnable=true" "$conf"; then
            ok "automatic login as $USER"
        elif confirm "Log in automatically as $USER at boot (needed to start without a keyboard)?"; then
            run sudo cp -n "$conf" "$conf.before-universalv1"
            run sudo sed -i -e '/^AutomaticLoginEnable=/d' -e '/^AutomaticLogin=/d' \
                -e "s/^\[daemon\]$/[daemon]\nAutomaticLoginEnable=true\nAutomaticLogin=$USER/" "$conf"
            grep -q "^WaylandEnable=false" "$conf" || run sudo sed -i 's/^\[daemon\]$/[daemon]\nWaylandEnable=false/' "$conf"
            ok "automatic login as $USER (X11); old file kept as $conf.before-universalv1"
        else
            warn "no automatic login: the app starts once someone logs in"
        fi
    elif [ "$AUTOLOGIN" -eq 0 ]; then
        warn "automatic login left as it is (--no-autologin)"
    fi
    # 2. A kiosk screen never locks or blanks.
    if command -v gsettings >/dev/null && [ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ]; then
        gsettings set org.gnome.desktop.screensaver lock-enabled false 2>>"$LOG" || true
        gsettings set org.gnome.desktop.session idle-delay 0 2>>"$LOG" || true
        gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>>"$LOG" || true
        ok "screen lock / blanking / sleep off"
    else
        warn "not in the desktop session: turn off screen lock and blanking in Settings by hand"
    fi
    # 3. The service.
    chmod +x "$APP_DIR/deploy/run_app.sh"
    mkdir -p "$SERVICE_DIR"
    sed "s#@APP_DIR@#$APP_DIR#g" "$APP_DIR/deploy/universalv1.service" >"$SERVICE_DIR/$SERVICE_NAME"
    run systemctl --user daemon-reload
    run systemctl --user enable "$SERVICE_NAME"
    ok "service $SERVICE_NAME enabled (starts with the desktop session)"
}

check_only() {
    echo "${BOLD}UniversalV1 install check${RESET} ($APP_DIR)"
    [ -x "$PY" ] && ok "venv ($("$PY" --version))" || warn "no venv: run ./deploy/install.sh"
    for p in torch ultralytics opencv-python-headless PySide6 tesserocr; do
        local v
        v=$(pyver "$p")
        [ "$v" != "-" ] && ok "$p $v" || warn "$p not installed"
    done
    [ "$(pyver opencv-python)" = "-" ] && ok "no opencv-python (would break Qt)" || warn "opencv-python installed: run --from 7"
    "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && ok "torch sees the GPU" || warn "torch does not see the GPU"
    [ -f /etc/udev/rules.d/99-patlite-lr6.rules ] && ok "tower light rule" || warn "no tower light rule: run --from 3"
    nvpmodel -q 2>/dev/null | head -1 | grep -q 25W && ok "power 25W" || warn "power mode: $(nvpmodel -q 2>/dev/null | head -1)"
    [ "$(fan_profile)" = "cool" ] && ok "fan profile cool" || warn "fan profile: $(fan_profile) (cool: run --from 11)"
    grep -q "^AutomaticLoginEnable=true" /etc/gdm3/custom.conf 2>/dev/null && ok "automatic login" || warn "no automatic login"
    systemctl --user is-enabled "$SERVICE_NAME" >/dev/null 2>&1 && ok "service enabled ($(systemctl --user is-active "$SERVICE_NAME" 2>/dev/null))" \
        || warn "service not enabled: run --from 12"
    [ -f "$APP_DIR/$MARK_MODEL" ] && ok "mark model" || warn "mark model $MARK_MODEL missing"
}

remove_service() {
    systemctl --user disable --now "$SERVICE_NAME" 2>/dev/null || true
    rm -f "$SERVICE_DIR/$SERVICE_NAME"
    systemctl --user daemon-reload
    echo "Removed $SERVICE_NAME (automatic login and the packages are left as they are)."
}

# --- Main ---------------------------------------------------------------------

while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM_STEP="$2"; shift ;;
        --check) CHECK_ONLY=1 ;;
        --no-autologin) AUTOLOGIN=0 ;;
        --remove-service) remove_service; exit 0 ;;
        --yes|-y) ASSUME_YES=1 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)"; exit 2 ;;
    esac
    shift
done

if [ "$CHECK_ONLY" -eq 1 ]; then
    check_only
    exit 0
fi

log "===== install started (from step $FROM_STEP) as $USER in $APP_DIR ====="
echo "${BOLD}UniversalV1 installer${RESET}  ($APP_DIR, log: $LOG)"
STEPS=(step_preflight step_system step_devices step_venv step_torch step_opencv step_ultralytics
       step_pyside step_extras step_verify step_power step_autostart)
[ "$FROM_STEP" -ge 2 ] && sudo -v
for i in "${!STEPS[@]}"; do
    n=$((i + 1))
    [ "$n" -lt "$FROM_STEP" ] && continue
    "${STEPS[$i]}"
done

echo
say "${GREEN}${BOLD}Install complete.${RESET}"
echo "  Reboot to check the whole start-up:  sudo reboot"
echo "  After boot the desktop logs in and the app starts the last job by itself."
echo "  Status:  systemctl --user status universalv1   App log: $LOG_DIR/app.log"
log "===== install complete ====="
