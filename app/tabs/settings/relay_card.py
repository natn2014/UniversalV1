import ipaddress

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...context import AppContext
from ...i18n import tr, translator
from ...relay_io import CHANNELS, DI_ROLES, DO_ROLES, PROTOCOLS
from ...theme import make_icon
from ...widgets.layout import make_card, row, tip_label

# I/O lamps: light blue when on, so they never read as the OK green / NG red.
LAMP_ON = "#4fc3f7"
LAMP_OFF = "#3a4048"
DEFAULT_PORTS = {"rtu_tcp": 4196, "tcp": 502}


def _lamp() -> QLabel:
    lamp = QLabel()
    lamp.setFixedSize(22, 22)
    _set_lamp(lamp, False)
    return lamp


def _set_lamp(lamp: QLabel, on: bool) -> None:
    lamp.setStyleSheet(f"background: {LAMP_ON if on else LAMP_OFF}; border: 2px solid #15181c; border-radius: 11px;")


def make_relay_card(ctx: AppContext, parent: QWidget):
    """Settings > Relay I/O: the board's address, a role per DI / DO with live
    lamps, and a test pulse per output. Returns (card, reload)."""
    settings, relay = ctx.settings, ctx.relay
    holder = QWidget()
    holder.setObjectName("clearBox")
    holder_layout = QVBoxLayout(holder)
    holder_layout.setContentsMargins(0, 0, 0, 0)

    # --- Board ----------------------------------------------------------
    board_card, board = make_card("Relay Board", icon="relay")
    enabled = QCheckBox("Use the relay board")
    status = QLabel()
    status.setWordWrap(True)
    status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    host = QLineEdit()
    host.setPlaceholderText("192.168.1.201")
    port = QSpinBox()
    port.setRange(1, 65535)
    address = QSpinBox()
    address.setRange(1, 247)
    protocol = QComboBox()
    for value, text in PROTOCOLS:
        protocol.addItem(text, value)
    connect = QPushButton("Connect")
    connect.setIcon(make_icon("refresh", "#e0e0e0"))
    connect.setToolTip("Connect again with these settings.")
    board.addWidget(enabled)
    board.addLayout(row(tip_label("IP address:", "The board's IP (set in its web page).", host), host))
    board.addLayout(row(tip_label("Port:", "4196 for Modbus RTU over TCP, 502 for Modbus TCP.", port), None, port))
    board.addLayout(row(tip_label("Address:", "Modbus slave address, 1 unless changed on the board.", address),
                        None, address))
    board.addWidget(tip_label("Protocol:", "Must match the board's own setting (its web page): RTU over TCP "
                              "is the board's default and what Pokayoke uses.", protocol))
    board.addWidget(protocol)
    status_row = row(connect)
    status_row.insertWidget(0, status, 1)
    board.addLayout(status_row)

    def host_done() -> None:
        text = host.text().strip()
        try:
            ipaddress.ip_address(text)
        except ValueError:
            host.setStyleSheet("border: 2px solid #c0392b;")
            host.setToolTip("Not an IP address, e.g. 192.168.1.201")
            return
        host.setStyleSheet("")
        settings.set(relay_host=text)

    def protocol_changed() -> None:
        value = protocol.currentData()
        if settings["relay_port"] == DEFAULT_PORTS.get(settings["relay_protocol"]):
            port.setValue(DEFAULT_PORTS[value])  # still on the old default: follow the protocol
        settings.set(relay_protocol=value)

    enabled.toggled.connect(lambda on: settings.set(relay_enabled=on))
    host.editingFinished.connect(host_done)
    port.valueChanged.connect(lambda v: settings.set(relay_port=v))
    address.valueChanged.connect(lambda v: settings.set(relay_address=v))
    protocol.currentIndexChanged.connect(lambda _: protocol_changed())
    connect.clicked.connect(lambda: (host_done(), relay.restart()))

    def show_status(connected: bool, text: str) -> None:
        status.setText(tr(text))
        status.setStyleSheet(f"color: {'#7ee08a' if connected else '#ffcf73'}; font-weight: 700;")

    relay.connection_changed.connect(show_status)
    translator().language_changed.connect(lambda _: show_status(relay.connected, relay.text))
    show_status(relay.connected, relay.text)

    # --- Inputs / outputs -----------------------------------------------
    def io_card(title: str, hint: str, roles, key: str, with_test: bool):
        card, body = make_card(title, icon="relay")
        note = QLabel(hint)
        note.setObjectName("hint")
        note.setWordWrap(True)
        body.addWidget(note)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        lamps, combos = {}, {}
        for i, channel in enumerate(CHANNELS):
            lamp = _lamp()
            name = QLabel(f"{key.upper()}{channel}")
            name.setMinimumWidth(44)
            combo = QComboBox()
            for value, text in roles:
                combo.addItem(text, value)
            combo.currentIndexChanged.connect(lambda _=0, c=channel: role_changed(key, c))
            grid.addWidget(lamp, i, 0)
            grid.addWidget(name, i, 1)
            grid.addWidget(combo, i, 2)
            grid.setColumnStretch(2, 1)
            if with_test:
                test = QPushButton("Test")
                test.setToolTip(f"Switch {key.upper()}{channel} on for 0.5 s.")
                test.clicked.connect(lambda _=False, c=channel: test_output(c))
                grid.addWidget(test, i, 3)
            lamps[channel], combos[channel] = lamp, combo
        body.addLayout(grid)
        return card, body, lamps, combos

    di_card, _, di_lamps, di_combos = io_card(
        "Inputs (DI)", "Trigger: judge one part per pulse. Confirm NG: an NG stays on until this input "
        "(or Confirm NG on the Monitor). Not used: ignored.", DI_ROLES, "di", False)
    do_card, do_body, do_lamps, do_combos = io_card(
        "Outputs (DO)", "OK / NG: on while the Monitor shows that result (or a pulse, below).", DO_ROLES, "do", True)
    pulse = QSpinBox()
    pulse.setRange(0, 10000)
    pulse.setSingleStep(100)
    pulse.setSuffix(" ms")
    pulse.setSpecialValueText("Hold")
    pulse.valueChanged.connect(lambda v: settings.set(relay_do_pulse_ms=v))
    do_body.addLayout(row(tip_label("OK / NG output:", "Hold = on while the result lasts. "
                                             "A time = a pulse of that length per result.", pulse), None, pulse))

    def role_changed(key: str, channel: int) -> None:
        combos = di_combos if key == "di" else do_combos
        roles = {str(c): combos[c].currentData() for c in CHANNELS if combos[c].currentData() != "none"}
        settings.set(**{f"relay_{key}_roles": roles})  # a new dict: settings compare it with the saved one

    def test_output(channel: int) -> None:
        if not relay.connected:
            QMessageBox.information(parent, "Relay Test", "The relay board is not connected.")
            return
        answer = QMessageBox.question(
            parent, "Relay Test", f"Switch DO{channel} on for 0.5 s?\n\nWhatever is wired to it will run.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            relay.test_pulse(channel)

    relay.di_changed.connect(lambda bits: [_set_lamp(di_lamps[c], bool(bits >> (c - 1) & 1)) for c in CHANNELS])
    relay.do_changed.connect(lambda bits: [_set_lamp(do_lamps[c], bool(bits >> (c - 1) & 1)) for c in CHANNELS])

    for card in (board_card, di_card, do_card):
        holder_layout.addWidget(card)

    def reload() -> None:
        enabled.setChecked(settings["relay_enabled"])
        host.setText(settings["relay_host"])
        host.setStyleSheet("")
        port.setValue(settings["relay_port"])
        address.setValue(settings["relay_address"])
        protocol.setCurrentIndex(max(protocol.findData(settings["relay_protocol"]), 0))
        pulse.setValue(settings["relay_do_pulse_ms"])
        for key, combos in (("di", di_combos), ("do", do_combos)):
            roles = settings[f"relay_{key}_roles"]
            for channel, combo in combos.items():
                combo.blockSignals(True)
                combo.setCurrentIndex(max(combo.findData(roles.get(str(channel), "none")), 0))
                combo.blockSignals(False)

    reload()
    holder.enabled_check, holder.host_edit, holder.port_spin, holder.protocol_combo = enabled, host, port, protocol
    holder.connect_button, holder.status_label, holder.pulse_spin = connect, status, pulse
    holder.di_combos, holder.do_combos, holder.di_lamps, holder.do_lamps = di_combos, do_combos, di_lamps, do_lamps
    return holder, reload
