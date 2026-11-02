"""Relay I/O: the Modbus framing against a fake Waveshare board (RTU over TCP
and Modbus TCP), the controller's roles, the inspection cycle (trigger mode,
Confirm NG) and the Monitor / Settings wiring. Offscreen, settings in a temp
dir, the fake board on localhost."""

import os
import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import app.paths as paths_mod  # noqa: E402
import app.settings as settings_mod  # noqa: E402
import app.tabs.monitor_tab as monitor_mod  # noqa: E402
from app.context import AppContext  # noqa: E402
from app.judgement import InspectionCycle  # noqa: E402
from app.relay_io import ModbusRelayBoard, RelayError, crc16  # noqa: E402
from app.tabs.monitor_tab import MonitorTab  # noqa: E402
from app.tabs.settings_tab import SettingsTab  # noqa: E402

APP = QApplication.instance() or QApplication([])


class FakeBoard:
    """8 DI / 8 DO board on 127.0.0.1. `bad_crc` corrupts that many replies;
    `stale` sends junk bytes before the next reply."""

    def __init__(self, protocol="rtu_tcp", address=1):
        self.protocol, self.address = protocol, address
        self.di = 0
        self.coils = 0
        self.bad_crc = 0
        self.stale = b""
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(4)
        self.port = self.server.getsockname()[1]
        self.running = True
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self):
        self.running = False
        self.server.close()

    def _accept(self):
        while self.running:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _pdu_reply(self, pdu):
        func, start, value = struct.unpack(">BHH", pdu[:5])
        if func in (0x01, 0x02):
            return bytes([func, 1, self.coils if func == 0x01 else self.di])
        if func == 0x05:
            bit = 1 << start
            self.coils = self.coils | bit if value == 0xFF00 else self.coils & ~bit
            return pdu[:5]
        return bytes([func | 0x80, 1])

    def _serve(self, conn):
        with conn:
            while self.running:
                try:
                    if self.protocol == "rtu_tcp":
                        frame = conn.recv(8)
                        if not frame:
                            return
                        reply = bytes([frame[0]]) + self._pdu_reply(frame[1:6])
                        crc = crc16(reply)
                        if self.bad_crc:
                            self.bad_crc -= 1
                            crc ^= 0xFFFF
                        out = reply + struct.pack("<H", crc)
                    else:
                        head = conn.recv(7)
                        if not head:
                            return
                        tid, _, length, unit = struct.unpack(">HHHB", head)
                        body = self._pdu_reply(conn.recv(length - 1))
                        out = struct.pack(">HHHB", tid, 0, len(body) + 1, unit) + body
                    conn.sendall(self.stale + out)
                    self.stale = b""
                except OSError:
                    return


class Board(unittest.TestCase):
    def test_crc_matches_relay_io_example(self):
        # Relay_IO.py: read coils for slave 1 is 01 01 00 00 00 08 3D CC.
        self.assertEqual(struct.pack("<H", crc16(bytes([1, 1, 0, 0, 0, 8]))), bytes([0x3D, 0xCC]))

    def check_protocol(self, protocol):
        fake = FakeBoard(protocol)
        board = ModbusRelayBoard("127.0.0.1", fake.port, 1, protocol)
        try:
            board.connect()
            fake.di = 0b1000_0101
            self.assertEqual(board.read_di(), 0b1000_0101)
            board.write_do(3, True)
            board.write_do(8, True)
            self.assertEqual(board.read_do(), 0b1000_0100)
            board.write_do(3, False)
            self.assertEqual(fake.coils, 0b1000_0000)
            with self.assertRaises(ValueError):
                board.write_do(9, True)
        finally:
            board.close()
            fake.close()

    def test_rtu_over_tcp(self):
        self.check_protocol("rtu_tcp")

    def test_modbus_tcp(self):
        self.check_protocol("tcp")

    def test_bad_crc_is_retried_then_fails(self):
        fake = FakeBoard()
        board = ModbusRelayBoard("127.0.0.1", fake.port)
        try:
            board.connect()
            fake.di, fake.bad_crc = 5, 1
            self.assertEqual(board.read_di(), 5)  # second try is good
            fake.bad_crc = 5
            with self.assertRaises(RelayError):
                board.read_di()
        finally:
            board.close()
            fake.close()

    def test_stale_bytes_are_flushed(self):
        fake = FakeBoard()
        board = ModbusRelayBoard("127.0.0.1", fake.port)
        try:
            board.connect()
            fake.stale = b"\x01\x02\x01"  # tail of an old reply
            fake.di = 9
            self.assertEqual(board.read_di(), 9)
        finally:
            board.close()
            fake.close()

    def test_not_connected(self):
        with self.assertRaises(RelayError):
            ModbusRelayBoard("127.0.0.1", 1).read_di()


class Cycle(unittest.TestCase):
    def test_continuous_counts_each_new_result(self):
        c = InspectionCycle()
        self.assertEqual(c.update("GREEN", 0), ("GREEN", "GREEN"))
        self.assertEqual(c.update("GREEN", 0), ("GREEN", None))
        self.assertEqual(c.update("YELLOW", 0), ("YELLOW", None))
        self.assertEqual(c.update("RED", 0), ("RED", "RED"))

    def test_trigger_mode_judges_one_frame_per_trigger(self):
        c = InspectionCycle()
        self.assertEqual(c.update("GREEN", 0, trigger_mode=True), ("YELLOW", None))  # waiting
        self.assertTrue(c.trigger())
        self.assertEqual(c.update("GREEN", 0, trigger_mode=True), ("GREEN", "GREEN"))
        self.assertEqual(c.update("RED", 0, trigger_mode=True), ("GREEN", None))  # held until next trigger
        c.trigger()
        self.assertEqual(c.update("YELLOW", 0, trigger_mode=True), ("YELLOW", None))  # no part: not counted

    def test_confirm_ng_latches(self):
        c = InspectionCycle()
        self.assertEqual(c.update("RED", 0, confirm_ng=True), ("RED", "RED"))
        self.assertTrue(c.awaiting_confirm)
        self.assertEqual(c.update("GREEN", 0, confirm_ng=True), ("RED", None))
        self.assertFalse(c.trigger())
        self.assertTrue(c.confirm())
        self.assertEqual(c.state, "YELLOW")
        self.assertFalse(c.confirm())
        self.assertEqual(c.update("GREEN", 0, confirm_ng=True), ("GREEN", "GREEN"))

    def test_trigger_mode_with_confirm(self):
        c = InspectionCycle()
        c.trigger()
        self.assertEqual(c.update("RED", 0, True, True), ("RED", "RED"))
        self.assertFalse(c.trigger())
        c.confirm()
        self.assertEqual(c.update("GREEN", 0, True, True), ("YELLOW", None))


def wait_for(condition, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        APP.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


class WithBoard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(settings_mod, "APP_SETTINGS_FILE", Path(self.tmp.name) / "app_settings.json"),
            mock.patch.object(paths_mod, "MARK_MODEL_PATH", Path(self.tmp.name) / "missing.pt"),
            mock.patch.object(monitor_mod, "save_ng_image", mock.Mock()),
        ]
        for p in self.patches:
            p.start()
        self.fake = FakeBoard()
        self.ctx = AppContext()
        self.relay = self.ctx.relay
        self.ctx.settings.set(relay_host="127.0.0.1", relay_port=self.fake.port,
                              relay_di_roles={"1": "trigger", "2": "confirm_ng"},
                              relay_do_roles={"1": "ok", "2": "ng"}, relay_enabled=True)
        self.assertTrue(wait_for(lambda: self.relay.connected), self.relay.text)

    def tearDown(self):
        self.relay.shutdown()
        self.fake.close()
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_inputs_and_outputs(self):
        pressed = []
        self.relay.input_pressed.connect(pressed.append)
        self.fake.di = 0b01
        self.assertTrue(wait_for(lambda: pressed == ["trigger"]))
        self.fake.di = 0b11
        self.assertTrue(wait_for(lambda: pressed == ["trigger", "confirm_ng"]))
        self.relay.update_state("GREEN")
        self.assertTrue(wait_for(lambda: self.fake.coils == 0b01))
        self.relay.update_state("RED")
        self.assertTrue(wait_for(lambda: self.fake.coils == 0b10))
        self.assertTrue(wait_for(lambda: self.relay.do == 0b10))
        self.relay.update_state(None)
        self.assertTrue(wait_for(lambda: self.fake.coils == 0))

    def test_pulse_outputs(self):
        self.ctx.settings.set(relay_do_pulse_ms=100)
        self.relay.update_state("GREEN")
        self.assertTrue(wait_for(lambda: self.fake.coils == 0b01))
        self.assertTrue(wait_for(lambda: self.fake.coils == 0))

    def test_role_removed_switches_output_off(self):
        self.relay.update_state("RED")
        self.assertTrue(wait_for(lambda: self.fake.coils == 0b10))
        self.ctx.settings.set(relay_do_roles={"1": "ok"})
        self.assertTrue(wait_for(lambda: self.fake.coils == 0))

    def test_reconnects(self):
        port = self.fake.port
        self.fake.close()
        self.relay.restart()
        self.assertTrue(wait_for(lambda: "not connected" in self.relay.text, 5))
        self.fake = FakeBoard()
        self.ctx.settings.set(relay_port=self.fake.port)  # restarts with the new port
        self.assertTrue(wait_for(lambda: self.relay.connected, 5), (port, self.relay.text))

    def test_monitor_trigger_and_confirm(self):
        tab = MonitorTab(self.ctx)
        image = QImage(64, 48, QImage.Format_RGB888)
        image.fill(QColor("#202020"))
        self.ctx.filters.selected_classes = {"e_mark"}
        self.ctx.filters.target_class_1 = "e_mark"
        det = {"class_name": "dot", "conf": 0.9, "x1": 1, "y1": 1, "x2": 9, "y2": 9, "label": "dot"}
        self.ctx.filters.selected_classes.add("dot")
        with mock.patch.object(type(self.ctx.worker), "isRunning", return_value=True):
            tab._reset_judgement()
            tab.on_frame(image, [det], [])  # NG part, but no trigger yet
            self.assertEqual(tab.judgement_banner.text(), "STANDBY")
            self.assertEqual(tab.cycle_label.text(), "Waiting for trigger")
            tab._on_relay_input("trigger")
            tab.on_frame(image, [det], [])
            self.assertEqual(tab.judgement_banner.text(), "NG")
            self.assertFalse(tab.confirm_button.isHidden())
            self.assertTrue(wait_for(lambda: self.fake.coils == 0b10))  # NG output on
            tab._on_relay_input("trigger")  # ignored until confirmed
            self.assertFalse(tab._latch.armed)
            tab.confirm_button.click()
            self.assertEqual(tab.judgement_banner.text(), "STANDBY")
            self.assertTrue(tab.confirm_button.isHidden())
            self.assertTrue(wait_for(lambda: self.fake.coils == 0))
        tab.deleteLater()


class SettingsPage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(settings_mod, "APP_SETTINGS_FILE", Path(self.tmp.name) / "app_settings.json")
        self.patch.start()
        self.ctx = AppContext()
        self.tab = SettingsTab(self.ctx)
        self.card = self.tab.relay_card

    def tearDown(self):
        self.ctx.relay.shutdown()
        self.tab.deleteLater()
        self.patch.stop()
        self.tmp.cleanup()

    def test_roles_and_revert(self):
        self.card.di_combos[3].setCurrentIndex(self.card.di_combos[3].findData("trigger"))
        self.card.do_combos[5].setCurrentIndex(self.card.do_combos[5].findData("ng"))
        self.assertEqual(self.ctx.settings["relay_di_roles"], {"3": "trigger"})
        self.assertEqual(self.ctx.settings["relay_do_roles"], {"5": "ng"})
        self.assertEqual(self.tab.unsaved_changes(), ["relay_di_roles", "relay_do_roles"])
        self.tab.revert()
        self.assertEqual(self.card.di_combos[3].currentData(), "none")
        self.assertEqual(self.ctx.settings["relay_di_roles"], {})

    def test_bad_ip_is_not_taken(self):
        self.card.host_edit.setText("192.168.1.999")
        self.card.host_edit.editingFinished.emit()
        self.assertEqual(self.ctx.settings["relay_host"], "192.168.1.201")
        self.card.host_edit.setText("10.0.0.5")
        self.card.host_edit.editingFinished.emit()
        self.assertEqual(self.ctx.settings["relay_host"], "10.0.0.5")

    def test_protocol_follows_default_port(self):
        self.card.protocol_combo.setCurrentIndex(self.card.protocol_combo.findData("tcp"))
        self.assertEqual(self.ctx.settings["relay_port"], 502)

    def test_test_pulse_asks_first(self):
        self.ctx.relay.connected = True
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.No), \
             mock.patch.object(self.ctx.relay, "test_pulse") as pulse:
            self.card.findChildren(type(self.card.connect_button), "")[0]  # widgets exist
            buttons = [b for b in self.card.findChildren(type(self.card.connect_button)) if b.text() == "Test"]
            buttons[0].click()
            pulse.assert_not_called()
        with mock.patch.object(QMessageBox, "question", return_value=QMessageBox.Yes), \
             mock.patch.object(self.ctx.relay, "test_pulse") as pulse:
            buttons[0].click()
            pulse.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
