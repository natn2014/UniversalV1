from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

from .optional_deps import BUZZER_OFF, BUZZER_ON, LED_ON, TowerLight
from .settings import AppSettings

try:
    import hid
except Exception:
    hid = None

PATLITE_VID, PATLITE_PID = 0x191A, 0x8003
# How often the watchdog checks the cable / retries the connection.
WATCHDOG_MS = 2000


def device_present() -> Optional[bool]:
    """True/False if the tower light is on the USB bus; None if that can't
    be checked (no hidapi), in which case only write errors detect a pull."""
    if hid is None:
        return None
    try:
        return bool(hid.enumerate(PATLITE_VID, PATLITE_PID))
    except Exception:
        return None


class TowerController(QObject):
    """PATLITE LR6 tower light + buzzer. Buzzer preferences live in
    AppSettings; every hardware error drops the connection instead of
    raising, so a yanked USB cable never takes the inspection loop down.

    Between connect_device() and shutdown(), a watchdog checks every
    WATCHDOG_MS that the device is still on the bus, reconnects it after the
    cable is plugged back in, and restores the current light / buzzer."""

    connection_text_changed = Signal(str)
    buzzer_enabled_changed = Signal(bool)
    message = Signal(str)

    def __init__(self, settings: AppSettings, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._tower = None
        self._connected = False
        self._last_state: Optional[str] = None
        # Latest state asked for, even while disconnected: re-applied on reconnect.
        self._wanted_state: Optional[str] = None
        self._ok_beep_token = 0
        self._watchdog = QTimer(self)
        self._watchdog.setInterval(WATCHDOG_MS)
        self._watchdog.timeout.connect(self._check_connection)
        self._connection_text = ""
        self.connection_text_changed.connect(self._remember_text)

    def _remember_text(self, text: str) -> None:
        self._connection_text = text

    def _set_text(self, text: str) -> None:
        """Emits connection_text_changed only when the text changes (the
        watchdog retries every 2 s)."""
        if text != self._connection_text:
            self.connection_text_changed.emit(text)

    @property
    def connected(self) -> bool:
        return self._connected and self._tower is not None

    @property
    def reconnecting(self) -> bool:
        """Wanted (watchdog running) but not connected: cable out or no access."""
        return self._watchdog.isActive() and not self.connected

    @property
    def backend_name(self) -> Optional[str]:
        return self._tower.backend_name if self._tower is not None else None

    @property
    def buzzer_enabled(self) -> bool:
        return self._settings["buzzer_enabled"]

    @property
    def buzzer_pattern(self) -> int:
        return self._settings["buzzer_pattern"]

    def connect_device(self) -> None:
        """Connects now and keeps reconnecting (watchdog) until shutdown()."""
        if TowerLight is None:
            self.connection_text_changed.emit("Tower Light: Driver Unavailable")
            return
        self._watchdog.start()
        if not self._try_open():
            self._emit_not_connected()

    def _emit_not_connected(self) -> None:
        """Says why: unplugged, or plugged in but not allowed to open it."""
        if device_present():
            self._set_text("Tower Light: Found but can't open — install deploy/99-patlite-lr6.rules (see README)")
        else:
            self._set_text("Tower Light: Not Connected (retrying)")

    def _try_open(self) -> bool:
        if device_present() is False:
            return False  # don't open (and log) a device that isn't plugged in
        try:
            self._tower = TowerLight(backend="auto")
            self._tower.open()
        except Exception:
            self._tower = None
            self._connected = False
            return False
        self._connected = True
        self.connection_text_changed.emit(f"Tower Light: Connected ({self._tower.backend_name})")
        return True

    def _check_connection(self) -> None:
        if self.connected:
            if device_present() is False:
                self._disconnected()
            return
        if not self._try_open():
            self._emit_not_connected()
            return
        self.message.emit("Tower light reconnected.")
        # Show the current judgement again, not the device's power-on state.
        self._last_state = None
        if self._wanted_state is not None:
            self.update_state(self._wanted_state)

    def shutdown(self) -> None:
        self._watchdog.stop()
        self._wanted_state = None
        if self._tower is not None:
            try:
                self._tower.reset()
            except Exception:
                pass
            try:
                self._tower.close()
            except Exception:
                pass
        self._tower = None
        self._connected = False
        self._last_state = None
        self.connection_text_changed.emit("Tower Light: Not Connected")

    def _disconnected(self) -> None:
        """Drops the dead handle; the watchdog reconnects when it's back."""
        was_connected = self._connected
        if self._tower is not None:
            try:
                self._tower.close()
            except Exception:
                pass
        self._tower = None
        self._connected = False
        self._last_state = None
        self.connection_text_changed.emit("Tower Light: Disconnected (retrying)")
        if was_connected:
            self.message.emit("Tower light disconnected — check the USB cable. Reconnecting...")

    def _compute_buzzer_pattern(self, state: Optional[str]) -> int:
        if self.buzzer_enabled and state == "RED":
            return self.buzzer_pattern
        return BUZZER_OFF

    def update_state(self, state: str) -> None:
        self._wanted_state = state
        if not self.connected:
            return

        if state == self._last_state:
            return

        try:
            # Buzzer first: set_tower() re-sends the driver's remembered buzzer
            # value, so it must already be correct or a stale one chirps briefly.
            self._tower.set_buzzer(self._compute_buzzer_pattern(state))
            if state == "GREEN":
                self._tower.set_tower(green=LED_ON)
            elif state == "RED":
                self._tower.set_tower(red=LED_ON)
            else:
                self._tower.set_tower(yellow=LED_ON)
            self._last_state = state
            # Beep after set_tower(), which re-sends the driver's remembered
            # (off) buzzer value and would otherwise cancel the tone.
            if state == "GREEN" and self.buzzer_enabled and self._settings["ok_beep_enabled"]:
                self._play_ok_beep()
        except Exception:
            self._disconnected()

    def set_buzzer_enabled(self, enabled: bool) -> None:
        if enabled == self.buzzer_enabled:
            return
        self._settings.set(buzzer_enabled=enabled)
        self.buzzer_enabled_changed.emit(enabled)
        self._sync_buzzer_now()

    def set_buzzer_pattern(self, pattern: int) -> None:
        self._settings.set(buzzer_pattern=pattern)
        self._sync_buzzer_now()

    def set_ok_beep_enabled(self, enabled: bool) -> None:
        self._settings.set(ok_beep_enabled=enabled)
        if enabled and self.buzzer_enabled:
            self._play_ok_beep()

    def set_ok_beep_ms(self, value: int) -> None:
        # No preview here: spinbox arrows fire on every step, so use Test.
        self._settings.set(ok_beep_ms=value)

    def test_ok_beep(self) -> None:
        if not self.connected:
            self.message.emit("Tower Light: Not Connected — cannot test buzzer.")
            return
        self._play_ok_beep()

    def test_alarm(self) -> None:
        """Plays one cycle of the NG pattern; the tower stops it by itself."""
        if not self.connected:
            self.message.emit("Tower Light: Not Connected — cannot test buzzer.")
            return
        try:
            self._tower.set_buzzer(self.buzzer_pattern, limit=1)
        except Exception:
            self._disconnected()

    def _play_ok_beep(self) -> None:
        """Short confirmation beep: the built-in patterns are all too long, so
        switch the tone on and turn it off again after ok_beep_ms."""
        if not self.connected:
            return
        self._ok_beep_token += 1
        token = self._ok_beep_token
        try:
            self._tower.set_buzzer(BUZZER_ON)
        except Exception:
            self._disconnected()
            return
        QTimer.singleShot(self._settings["ok_beep_ms"], lambda: self._end_ok_beep(token))

    def _end_ok_beep(self, token: int) -> None:
        # A newer beep owns the buzzer now; let its own timer end it.
        if token != self._ok_beep_token:
            return
        if not self.connected:
            return
        try:
            # Restore whatever the current state wants — silent for OK, but the
            # NG alarm if the state turned NG during the beep.
            self._tower.set_buzzer(self._compute_buzzer_pattern(self._last_state))
        except Exception:
            self._disconnected()

    def _sync_buzzer_now(self) -> None:
        """Immediately re-applies buzzer on/off, independent of whether the
        judgement state itself just changed — so toggling the checkbox or
        switching styles takes effect right away, not on the next transition.

        If NG is already active, applies the style continuously (real alarm).
        Otherwise, enabling or changing style while idle plays a short preview
        beep (limit=1, auto-stops via the tower's own repeat-count) so the
        operator can hear the selected style without needing a live NG."""
        if not self.connected:
            return
        try:
            if self._last_state == "RED" and self.buzzer_enabled:
                self._tower.set_buzzer(self.buzzer_pattern)
            elif self.buzzer_enabled:
                self._tower.set_buzzer(self.buzzer_pattern, limit=1)
            else:
                self._tower.set_buzzer(BUZZER_OFF)
        except Exception:
            self._disconnected()
