"""Relay I/O on a Waveshare Modbus POE ETH Relay (8 DI / 8 DO), the board
used by github.com/natn2014/Pokayoke (Relay_IO.py).

The board speaks Modbus over its Ethernet port in one of two modes, chosen in
its own web page: "Modbus RTU over TCP" (raw RTU frames with CRC on a TCP
socket, port 4196 by default; what Relay_IO.py uses) or "Modbus TCP" (MBAP
header, no CRC, port 502). Only three functions are needed: 0x02 read the 8
discrete inputs, 0x01 read the 8 coils (DO state), 0x05 write one coil.

ModbusRelayBoard is plain blocking sockets. RelayIOController owns one in a
background thread that polls the inputs, so the UI never waits on the network,
and maps roles onto the channels: a DI is a Trigger or Confirm NG input, a DO
follows the OK or NG result.
"""

import queue
import select
import socket
import struct
import threading
import time
from typing import Dict, Optional

from PySide6.QtCore import QObject, QTimer, Signal

CHANNELS = range(1, 9)
DI_ROLES = (("none", "Not used"), ("trigger", "Trigger"), ("confirm_ng", "Confirm NG"))
DO_ROLES = (("none", "Not used"), ("ok", "OK"), ("ng", "NG"))
PROTOCOLS = (("rtu_tcp", "Modbus RTU over TCP"), ("tcp", "Modbus TCP"))
POLL_S = 0.05
RECONNECT_S = 3.0


class RelayError(Exception):
    pass


def crc16(data: bytes) -> int:
    """Modbus CRC-16 (poly 0xA001, init 0xFFFF); sent low byte first."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


class ModbusRelayBoard:
    """One relay board. Methods raise RelayError (or OSError) on failure;
    the caller reconnects."""

    def __init__(self, host: str, port: int = 4196, address: int = 1, protocol: str = "rtu_tcp",
                 timeout: float = 0.5) -> None:
        self.host, self.port, self.address = host, int(port), int(address)
        self.protocol = protocol
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._tid = 0

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        self.close()
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def read_di(self) -> int:
        """DI1..DI8 as bits 0..7."""
        return self._read_bits(0x02)

    def read_do(self) -> int:
        """DO1..DO8 (coils) as bits 0..7."""
        return self._read_bits(0x01)

    def write_do(self, channel: int, on: bool) -> None:
        if channel not in CHANNELS:
            raise ValueError(f"DO channel must be 1-8, not {channel}")
        pdu = struct.pack(">BHH", 0x05, channel - 1, 0xFF00 if on else 0x0000)
        if self._request(pdu, 4) != pdu[1:]:
            raise RelayError("write not echoed")

    # --- Framing --------------------------------------------------------

    def _read_bits(self, function: int) -> int:
        data = self._request(struct.pack(">BHH", function, 0, 8), 2)
        if data[0] != 1:
            raise RelayError(f"byte count {data[0]}, expected 1")
        return data[1]

    def _request(self, pdu: bytes, data_len: int, retries: int = 3) -> bytes:
        """Sends a PDU, returns the reply's data (after the function code).
        A stale or corrupt reply is flushed and the request sent again."""
        if self._sock is None:
            raise RelayError("not connected")
        last: Exception = RelayError("no reply")
        for _ in range(retries):
            self._flush()
            try:
                if self.protocol == "tcp":
                    return self._request_tcp(pdu, data_len)
                return self._request_rtu(pdu, data_len)
            except (RelayError, socket.timeout) as exc:
                last = exc
        raise last if isinstance(last, RelayError) else RelayError(str(last))

    def _request_rtu(self, pdu: bytes, data_len: int) -> bytes:
        frame = bytes([self.address]) + pdu
        self._sock.sendall(frame + struct.pack("<H", crc16(frame)))
        head = self._recv(2)
        if head[1] == pdu[0] | 0x80:
            rest = self._recv(3)
            raise RelayError(f"device error code {rest[0]}")
        reply = head + self._recv(data_len + 2)
        if reply[0] != self.address or reply[1] != pdu[0]:
            raise RelayError(f"unexpected reply {reply.hex(' ')}")
        if struct.unpack("<H", reply[-2:])[0] != crc16(reply[:-2]):
            raise RelayError("bad CRC")
        return reply[2:-2]

    def _request_tcp(self, pdu: bytes, data_len: int) -> bytes:
        self._tid = (self._tid + 1) & 0xFFFF
        self._sock.sendall(struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, self.address) + pdu)
        tid, _, length, unit = struct.unpack(">HHHB", self._recv(7))
        body = self._recv(length - 1)
        if tid != self._tid or unit != self.address:
            raise RelayError("stale reply")
        if body[0] == pdu[0] | 0x80:
            raise RelayError(f"device error code {body[1]}")
        if body[0] != pdu[0] or len(body) != data_len + 1:
            raise RelayError(f"unexpected reply {body.hex(' ')}")
        return body[1:]

    def _recv(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("board closed the connection")
            data += chunk
        return data

    def _flush(self) -> None:
        """Drops bytes left from an earlier, timed-out reply."""
        while select.select([self._sock], [], [], 0)[0]:
            if not self._sock.recv(256):
                raise ConnectionError("board closed the connection")


class _PollThread(threading.Thread):
    """Talks to the board: polls the inputs, sends queued output writes,
    reconnects after errors. Reports through the controller's signals."""

    def __init__(self, controller: "RelayIOController", board: ModbusRelayBoard) -> None:
        super().__init__(daemon=True, name="relay-io")
        self.c, self.board = controller, board
        self.writes: "queue.Queue" = queue.Queue()
        self.stopping = threading.Event()

    def run(self) -> None:
        di = do = None
        while not self.stopping.is_set():
            if not self.board.connected:
                try:
                    self.board.connect()
                    do = self.board.read_do()
                    self.c._report(True, f"Relay: connected {self.board.host}:{self.board.port}", do=do)
                except (OSError, RelayError) as exc:
                    self.board.close()
                    self.c._report(False, f"Relay: not connected ({exc})")
                    self.stopping.wait(RECONNECT_S)
                    continue
            try:
                wrote = False
                while not self.writes.empty():
                    channel, on = self.writes.get_nowait()
                    self.board.write_do(channel, on)
                    wrote = True
                if wrote:
                    do = self.board.read_do()
                    self.c._report(True, None, do=do)
                value = self.board.read_di()
                if value != di:
                    di = value
                    self.c._report(True, None, di=di)
            except (OSError, RelayError) as exc:
                self.board.close()
                di = None
                self.c._report(False, f"Relay: lost ({exc}), reconnecting")
                continue
            self.stopping.wait(POLL_S)
        # Leave nothing switched on behind us.
        if self.board.connected:
            for channel in CHANNELS:
                try:
                    self.board.write_do(channel, False)
                except (OSError, RelayError):
                    break
        self.board.close()


class RelayIOController(QObject):
    """The relay board as the app sees it: connection, DI / DO state, and the
    roles from app settings (relay_*). update_state() drives the OK / NG
    outputs like the tower light; a rising DI with a role emits input_pressed."""

    connection_changed = Signal(bool, str)
    di_changed = Signal(int)
    do_changed = Signal(int)
    input_pressed = Signal(str)  # "trigger" / "confirm_ng", on the DI's rising edge
    # From the poll thread to the GUI thread (queued: emitted off the GUI thread).
    _reported = Signal(object, object, object, object)

    def __init__(self, settings, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._thread: Optional[_PollThread] = None
        self.connected = False
        self.text = "Relay: off"
        self.di = 0
        self.do = 0
        self._wanted: Dict[int, bool] = {}
        self._state: Optional[str] = None
        self._reported.connect(self._on_reported)
        settings.changed.connect(self._on_settings_changed)

    # --- Settings -------------------------------------------------------

    def di_role(self, channel: int) -> str:
        return self._settings["relay_di_roles"].get(str(channel), "none")

    def do_role(self, channel: int) -> str:
        return self._settings["relay_do_roles"].get(str(channel), "none")

    def has_input(self, role: str) -> bool:
        """A DI has this role and the relay is in use (not necessarily connected)."""
        return self._settings["relay_enabled"] and any(self.di_role(c) == role for c in CHANNELS)

    def _on_settings_changed(self, keys: list) -> None:
        if {"relay_enabled", "relay_host", "relay_port", "relay_address", "relay_protocol"} & set(keys):
            self.restart()
        elif "relay_do_roles" in keys:
            for channel, on in list(self._wanted.items()):
                if self.do_role(channel) == "none":
                    del self._wanted[channel]
                    if on:
                        self.write(channel, False)
            state, self._state = self._state, None
            self.update_state(state)

    # --- Connection -----------------------------------------------------

    def restart(self) -> None:
        self.stop()
        s = self._settings
        if not s["relay_enabled"]:
            self._set_connection(False, "Relay: off")
            return
        board = ModbusRelayBoard(s["relay_host"], s["relay_port"], s["relay_address"], s["relay_protocol"])
        self._thread = _PollThread(self, board)
        self._set_connection(False, f"Relay: connecting to {board.host}:{board.port}...")
        self._thread.start()

    def stop(self) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.stopping.set()
            thread.join(timeout=2.0)
        self._wanted.clear()

    def _report(self, connected: bool, text: Optional[str], di: Optional[int] = None, do: Optional[int] = None) -> None:
        self._reported.emit(connected, text, di, do)

    def _on_reported(self, connected, text, di, do) -> None:
        if text is not None or connected != self.connected:
            self._set_connection(connected, text or self.text)
            if connected:
                # Outputs back to the current result after a (re)connect.
                state, self._state = self._state, None
                self._wanted.clear()
                self.update_state(state)
        if do is not None and do != self.do:
            self.do = do
            self.do_changed.emit(do)
        if di is not None and di != self.di:
            rising = di & ~self.di
            self.di = di
            self.di_changed.emit(di)
            for channel in CHANNELS:
                if rising & (1 << (channel - 1)) and self.di_role(channel) != "none":
                    self.input_pressed.emit(self.di_role(channel))

    def _set_connection(self, connected: bool, text: str) -> None:
        self.connected, self.text = connected, text
        self.connection_changed.emit(connected, text)

    # --- Outputs --------------------------------------------------------

    def write(self, channel: int, on: bool) -> None:
        if self._thread is not None:
            self._thread.writes.put((channel, on))

    def update_state(self, state: Optional[str]) -> None:
        """OK DOs on while GREEN, NG DOs on while RED (or a pulse of
        relay_do_pulse_ms when it is above 0)."""
        if state == self._state:
            return
        self._state = state
        pulse = self._settings["relay_do_pulse_ms"]
        for channel in CHANNELS:
            role = self.do_role(channel)
            if role == "none":
                continue
            on = (role == "ok" and state == "GREEN") or (role == "ng" and state == "RED")
            if self._wanted.get(channel) != on:
                self._wanted[channel] = on
                self.write(channel, on)
                if on and pulse > 0:
                    QTimer.singleShot(pulse, lambda c=channel: self._end_pulse(c))

    def _end_pulse(self, channel: int) -> None:
        if self._wanted.get(channel):
            self._wanted[channel] = False
            self.write(channel, False)

    def test_pulse(self, channel: int, ms: int = 500) -> None:
        self.write(channel, True)
        QTimer.singleShot(ms, lambda: self.write(channel, False))

    def shutdown(self) -> None:
        self.stop()
