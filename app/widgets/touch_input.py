"""On-screen keyboard and numpad for touch-only operation (no attached keyboard).

Tapping a text field pops a QWERTY keyboard (`KeyboardDialog`); tapping a
spin box's number pops a numeric keypad (`NumpadDialog`). `attach_keyboard`
and `attach_numpad` wire an existing QLineEdit / QSpinBox / QDoubleSpinBox to
the matching dialog by intercepting the tap that would otherwise just place a
caret. `get_text` / `get_double` are touch-friendly drop-ins for
`QInputDialog.getText` / `.getDouble`.
"""
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

BG = "#0b0d10"
PANEL = "#1f2329"
FIELD_BG = "#0f1114"
KEY_BG = "#2a2f36"
INK = "#e8eaed"
DIM = "#9aa5b1"
ACCENT = "#e0a526"
ACCENT_HI = "#ffcf73"
EDGE = "#3a4048"
BACK_BG = "#5c1f17"
BACK_FG = "#ff7a6b"


def _screen_scale() -> float:
    screen = QApplication.primaryScreen().geometry()
    return min(screen.width() / 1280.0, screen.height() / 800.0, 1.0)


def _place_centered(dialog: QDialog) -> None:
    """Position explicitly before show: left to the window manager, a frameless
    dialog's reported geometry can disagree with where it actually lands, so a
    touch tap hits the wrong key."""
    dialog.adjustSize()
    parent = dialog.parentWidget()
    area = (parent.window().geometry() if parent is not None
            else QApplication.primaryScreen().availableGeometry())
    center = area.center()
    dialog.move(center.x() - dialog.width() // 2, center.y() - dialog.height() // 2)


class _TouchDialog(QDialog):
    """Shared chrome for KeyboardDialog / NumpadDialog: frameless, scaled to
    the screen, centered on the parent window."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self._scale = _screen_scale()
        self.setStyleSheet(f"QDialog{{background-color:{PANEL}; border:2px solid {EDGE};}}")
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(self._px(14), self._px(14), self._px(14), self._px(14))
        self._outer.setSpacing(self._px(10))
        if title:
            title_label = QLabel(title)
            title_label.setFont(self._font(14, bold=True))
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet(f"color:{DIM};")
            self._outer.addWidget(title_label)

    def _px(self, value: float) -> int:
        return max(1, round(value * self._scale))

    def _font(self, point_size: float, bold: bool = False) -> QFont:
        font = QFont()
        font.setPointSize(max(8, round(point_size * self._scale)))
        font.setBold(bold)
        return font

    def _key_button(self, label: str, bg: str = KEY_BG, fg: str = INK) -> QPushButton:
        button = QPushButton(label)
        button.setFont(self._font(16, bold=True))
        button.setMinimumSize(self._px(64), self._px(64))
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep focus on the display field
        button.setStyleSheet(
            f"QPushButton{{background:{bg};color:{fg};border:1px solid {EDGE};border-radius:8px;}}"
            f"QPushButton:pressed{{background:{ACCENT};color:{BG};}}"
        )
        return button

    def _action_row(self):
        row = QHBoxLayout()
        row.setSpacing(self._px(8))
        cancel = QPushButton("Cancel")
        ok = QPushButton("OK")
        font = self._font(16, bold=True)
        for button in (cancel, ok):
            button.setFont(font)
            button.setMinimumHeight(self._px(60))
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cancel.setStyleSheet(
            f"QPushButton{{background:{KEY_BG};color:{INK};border:1px solid {EDGE};border-radius:8px;}}"
            f"QPushButton:pressed{{background:{PANEL};}}"
        )
        ok.setStyleSheet(
            f"QPushButton{{background:#2e7d32;color:#ffffff;border:1px solid #1b5e20;border-radius:8px;}}"
            f"QPushButton:pressed{{background:#1b5e20;}}"
        )
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(ok)
        return row


_ROWS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL-"),
    list("ZXCVBNM()_"),
]
_SYMBOLS = [".", "/", "+", "#", "&"]


class KeyboardDialog(_TouchDialog):
    """Full QWERTY on-screen keyboard for a text value."""

    def __init__(self, current_text: str = "", title: str = "Enter Text",
                 max_length: int = 100, allowed_chars: str = None, validate=None, parent=None) -> None:
        """`allowed_chars`, if given, silently rejects any key not in it (e.g.
        digits + "." for an IP). `validate(text) -> error message or None` runs
        on OK / Enter; a message keeps the dialog open and shows it instead of
        accepting, so an incomplete or malformed value can't be saved."""
        super().__init__(title, parent)
        self._upper = True
        self._letter_buttons = []
        self._allowed_chars = allowed_chars
        self._validate = validate
        self.display = QLineEdit(current_text)
        self.display.setFont(self._font(20, bold=True))
        self.display.setMinimumHeight(self._px(56))
        self.display.setMaxLength(max_length)
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._field_border = ACCENT
        self._apply_field_style()
        self.display.returnPressed.connect(self.accept)
        self._outer.addWidget(self.display)

        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setFont(self._font(11, bold=True))
        self.error_label.setStyleSheet(f"color:{BACK_FG};")
        self._outer.addWidget(self.error_label)

        for key_row in _ROWS:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(self._px(6))
            for ch in key_row:
                button = self._key_button(ch)
                if ch.isalpha():
                    self._letter_buttons.append(button)
                button.clicked.connect(lambda _checked=False, c=ch: self._insert(c))
                row_layout.addWidget(button)
            self._outer.addLayout(row_layout)

        bottom = QHBoxLayout()
        bottom.setSpacing(self._px(6))
        self.shift_button = self._key_button("⇧ ABC", bg=KEY_BG, fg=ACCENT_HI)
        self.shift_button.setCheckable(True)
        self.shift_button.setChecked(self._upper)
        self.shift_button.toggled.connect(self._on_shift)
        bottom.addWidget(self.shift_button, 2)
        for sym in _SYMBOLS:
            button = self._key_button(sym)
            button.clicked.connect(lambda _checked=False, c=sym: self._insert(c))
            bottom.addWidget(button, 1)
        space = self._key_button("Space", bg=KEY_BG, fg=ACCENT_HI)
        space.clicked.connect(lambda: self._insert(" "))
        bottom.addWidget(space, 4)
        back = self._key_button("< Back", bg=BACK_BG, fg=BACK_FG)
        back.clicked.connect(self._backspace)
        bottom.addWidget(back, 2)
        clear = self._key_button("Clear", bg=BACK_BG, fg=BACK_FG)
        clear.clicked.connect(self._clear)
        bottom.addWidget(clear, 2)
        self._outer.addLayout(bottom)

        self._outer.addLayout(self._action_row())
        self.setMinimumWidth(self._px(760))

        self.display.setFocus()
        self.display.selectAll()
        _place_centered(self)

    def _insert(self, ch: str) -> None:
        if self._allowed_chars is not None and ch not in self._allowed_chars:
            self._show_error(f'Only "{self._allowed_chars}" allowed')
            return
        self._clear_error()
        self.display.insert(ch if self._upper or not ch.isalpha() else ch.lower())
        self.display.setFocus()

    def _backspace(self) -> None:
        self._clear_error()
        if self.display.hasSelectedText():
            self.display.del_()
        else:
            self.display.backspace()
        self.display.setFocus()

    def _clear(self) -> None:
        self._clear_error()
        self.display.clear()

    def _on_shift(self, checked: bool) -> None:
        self._upper = checked
        self.shift_button.setText("⇧ ABC" if checked else "⇧ abc")
        for button in self._letter_buttons:
            button.setText(button.text().upper() if checked else button.text().lower())
        self.display.setFocus()

    def _apply_field_style(self) -> None:
        self.display.setStyleSheet(
            f"background-color:{FIELD_BG}; color:{ACCENT_HI}; padding:6px 12px;"
            f"border-radius:6px; border:2px solid {self._field_border};"
        )

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self._field_border = BACK_FG
        self._apply_field_style()

    def _clear_error(self) -> None:
        if self.error_label.text():
            self.error_label.setText("")
            self._field_border = ACCENT
            self._apply_field_style()

    def accept(self) -> None:
        if self._validate is not None:
            error = self._validate(self.display.text().strip())
            if error:
                self._show_error(error)
                return
        super().accept()

    def get_text(self) -> str:
        return self.display.text().strip()


class NumpadDialog(_TouchDialog):
    """Numeric-only keypad (0-9, decimal point, +/- sign, backspace) for a value."""

    def __init__(self, current_value: float = 0.0, decimals: int = 0, title: str = "Enter Value",
                 min_val=None, max_val=None, parent=None) -> None:
        super().__init__(title, parent)
        self.decimals = decimals
        self.min_val = min_val
        self.max_val = max_val
        self._value_str = (f"{current_value:.{decimals}f}" if decimals
                            else str(int(round(current_value))))
        self._fresh = True
        self._allow_negative = min_val is None or min_val < 0

        self.display = QLabel(self._value_str)
        self.display.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.display.setFont(self._font(26, bold=True))
        self.display.setMinimumHeight(self._px(60))
        self.display.setStyleSheet(
            f"background-color:{FIELD_BG}; color:{ACCENT_HI}; padding:6px 12px;"
            f"border-radius:6px; border:2px solid {ACCENT};"
        )
        self._outer.addWidget(self.display)

        if self._allow_negative:
            sign_row = QHBoxLayout()
            sign_row.setSpacing(self._px(6))
            sign_btn = self._key_button("±", bg=KEY_BG, fg=ACCENT_HI)
            sign_btn.clicked.connect(self._on_sign)
            sign_row.addWidget(sign_btn)
            self._outer.addLayout(sign_row)

        grid = QGridLayout()
        grid.setSpacing(self._px(8))
        keys = [
            ("7", 0, 0), ("8", 0, 1), ("9", 0, 2),
            ("4", 1, 0), ("5", 1, 1), ("6", 1, 2),
            ("1", 2, 0), ("2", 2, 1), ("3", 2, 2),
            (".", 3, 0), ("0", 3, 1), ("<", 3, 2),
        ]
        for label, row, col in keys:
            if label == "<":
                button = self._key_button("<", bg=BACK_BG, fg=BACK_FG)
            elif label == ".":
                button = self._key_button(".", bg=KEY_BG, fg=ACCENT_HI)
                button.setEnabled(decimals > 0)
            else:
                button = self._key_button(label)
            button.setMinimumSize(self._px(90), self._px(70))
            button.clicked.connect(lambda _checked=False, k=label: self._on_key(k))
            grid.addWidget(button, row, col)
        self._outer.addLayout(grid)

        self._outer.addLayout(self._action_row())
        self.setMinimumWidth(self._px(320))
        _place_centered(self)

    def _on_key(self, key: str) -> None:
        if self._fresh:
            self._value_str = ""
            self._fresh = False
        if key == "<":
            self._value_str = self._value_str[:-1]
        elif key == ".":
            if "." not in self._value_str:
                self._value_str = (self._value_str or "0") + "."
        else:
            digits_only = self._value_str[1:] if self._value_str.startswith("-") else self._value_str
            if "." in digits_only:
                after_dot = len(digits_only) - digits_only.index(".") - 1
                if after_dot >= self.decimals:
                    return
            self._value_str += key
        self.display.setText(self._value_str or "0")

    def _on_sign(self) -> None:
        text = self._value_str or "0"
        self._value_str = text[1:] if text.startswith("-") else "-" + text
        self.display.setText(self._value_str)

    def accept(self) -> None:
        try:
            value = float(self._value_str)
        except ValueError:
            value = 0.0
        if self.min_val is not None and value < self.min_val:
            value = self.min_val
        if self.max_val is not None and value > self.max_val:
            value = self.max_val
        self._value_str = f"{value:.{self.decimals}f}" if self.decimals else str(int(round(value)))
        super().accept()

    def get_value(self) -> float:
        try:
            return round(float(self._value_str), self.decimals)
        except ValueError:
            return 0.0


def get_text(parent, title: str, label: str = "", text: str = "", max_length: int = 100):
    """Touch-friendly drop-in for `QInputDialog.getText`: returns (text, ok)."""
    dialog = KeyboardDialog(current_text=text, title=title or label, max_length=max_length, parent=parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_text(), True
    return text, False


def get_double(parent, title: str, label: str = "", value: float = 0.0,
                min_val=None, max_val=None, decimals: int = 2):
    """Touch-friendly drop-in for `QInputDialog.getDouble`: returns (value, ok)."""
    dialog = NumpadDialog(current_value=value, decimals=decimals, title=title or label,
                           min_val=min_val, max_val=max_val, parent=parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_value(), True
    return value, False


class _LineEditTap(QObject):
    """Swallows the tap that would place a caret in `edit` and opens a
    KeyboardDialog instead."""

    def __init__(self, edit: QLineEdit, title: str, allowed_chars: str = None, validate=None) -> None:
        super().__init__(edit)
        self._edit = edit
        self._title = title
        self._allowed_chars = allowed_chars
        self._validate = validate

    def eventFilter(self, obj, event) -> bool:
        edit = self._edit
        if event.type() == QEvent.Type.MouseButtonPress and edit.isEnabled() and not edit.isReadOnly():
            max_length = edit.maxLength() if edit.maxLength() < 32767 else 200
            dialog = KeyboardDialog(current_text=edit.text(), title=self._title, max_length=max_length,
                                     allowed_chars=self._allowed_chars, validate=self._validate,
                                     parent=edit.window())
            if dialog.exec() == QDialog.DialogCode.Accepted:
                edit.setText(dialog.get_text())
                edit.textEdited.emit(edit.text())  # listeners bound to textEdited, not just textChanged
                edit.editingFinished.emit()  # listeners that save only on editingFinished (e.g. the relay host)
            return True
        return False


class _SpinTap(QObject):
    """Swallows the tap on a spin box's number (not its +/- arrows) and opens
    a NumpadDialog instead."""

    def __init__(self, spin, title: str) -> None:
        super().__init__(spin.lineEdit())
        self._spin = spin
        self._title = title

    def eventFilter(self, obj, event) -> bool:
        spin = self._spin
        if event.type() == QEvent.Type.MouseButtonPress and spin.isEnabled() and not spin.isReadOnly():
            decimals = spin.decimals() if isinstance(spin, QDoubleSpinBox) else 0
            dialog = NumpadDialog(current_value=spin.value(), decimals=decimals, title=self._title,
                                   min_val=spin.minimum(), max_val=spin.maximum(), parent=spin.window())
            if dialog.exec() == QDialog.DialogCode.Accepted:
                value = dialog.get_value()
                spin.setValue(int(round(value)) if isinstance(spin, QSpinBox) else value)
            return True
        return False


def attach_keyboard(edit: QLineEdit, title: str = "", allowed_chars: str = None, validate=None) -> None:
    """Make tapping `edit` open an on-screen QWERTY keyboard instead of
    placing a caret. No-op for read-only fields. `allowed_chars` / `validate`
    are forwarded to KeyboardDialog to block an incorrect value at the source
    (e.g. an IP address field)."""
    edit._touch_tap = _LineEditTap(edit, title, allowed_chars=allowed_chars, validate=validate)
    edit.installEventFilter(edit._touch_tap)


def attach_numpad(spin, title: str = "") -> None:
    """Make tapping the number in `spin` (a QSpinBox / QDoubleSpinBox) open an
    on-screen numeric keypad; the +/- step buttons keep working normally."""
    spin._touch_tap = _SpinTap(spin, title)
    spin.lineEdit().installEventFilter(spin._touch_tap)
