"""Part Check card: the Part Check (AI Seg) result as an operator checklist.

    Glass front side        ✓ OK
    Glass back side         ✓ OK
    Clip seal sponge 1/3    ✓ OK
    Clip seal sponge 2/3    ✓ OK
    Clip seal sponge 3/3    ✗ NG
    ─────────────────────────────
              NG

Each row is ✓ OK (checked, right), ✗ NG (checked, wrong) or — (not checked
yet: that side has not been shown). The banner is PASS, NG, TURN OVER (one
good side seen, show the other) or NO GLASS. It reads
`ToolResult.extra["checklist"]` (app/vision/segment.py) and is used by the
Tools and Monitor tabs.
"""

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from ..i18n import tr, translator
from ..theme import NG_RED, OK_GREEN, STANDBY_YELLOW
from ..vision.base import ToolResult
from .layout import make_card

NOT_CHECKED = "#3a4048"
CHIP_QSS = ("QLabel {{ background: {bg}; color: {fg}; border-radius: 6px; font-weight: 700;"
            " padding: 4px 8px; min-width: 58px; }}")
BANNER_QSS = ("QLabel {{ background: {bg}; color: {fg}; border-radius: 8px; font-size: 22pt;"
              " font-weight: 800; padding: 8px; }}")
CHIPS = {True: ("✓ OK", OK_GREEN, "#ffffff"), False: ("✗ NG", NG_RED, "#ffffff"), None: ("—", NOT_CHECKED, "#c8ced6")}


def item_name(item_class: str) -> str:
    """Row name for the counted parts ("sponge" -> "Clip seal sponge")."""
    return "Clip seal sponge" if item_class == "sponge" else item_class.replace("_", " ").capitalize()


class PartCheckPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("clearBox")
        card, body = make_card("Part Check", icon="check")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(6)
        self.grid.setColumnStretch(0, 1)
        self.grid.setContentsMargins(0, 0, 10, 0)  # chips clear of the panel edge / scroll bar
        body.addLayout(self.grid)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #3a4048;")
        body.addWidget(line)
        self.banner = QLabel()
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setMinimumHeight(56)
        self.banner.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.banner)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setContentsMargins(0, 0, 10, 0)
        body.addWidget(self.hint)
        self._rows: List[tuple] = []  # (name label, chip label)
        self._last: Optional[ToolResult] = None
        self._item_class = "sponge"
        translator().language_changed.connect(lambda _lang: self.show_result(self._last, self._item_class))
        self.show_result(None)

    # --- rows -----------------------------------------------------------

    def _ensure_rows(self, n: int) -> None:
        while len(self._rows) < n:
            name, chip = QLabel(), QLabel()
            name.setProperty("_i18n_skip", True)  # set with tr() below
            chip.setAlignment(Qt.AlignCenter)
            r = len(self._rows)
            self.grid.addWidget(name, r, 0)
            self.grid.addWidget(chip, r, 1)
            self._rows.append((name, chip))
        for i, (name, chip) in enumerate(self._rows):
            name.setVisible(i < n)
            chip.setVisible(i < n)

    @staticmethod
    def _set_chip(chip: QLabel, state: Optional[bool]) -> None:
        text, bg, fg = CHIPS[state]
        chip.setText(text)
        chip.setStyleSheet(CHIP_QSS.format(bg=bg, fg=fg))

    def rows(self) -> List[tuple]:
        """(name, chip text) of the visible rows (tests, screenshots)."""
        return [(n.text(), c.text()) for n, c in self._rows if not n.isHidden()]

    # --- result ---------------------------------------------------------

    def show_result(self, result: Optional[ToolResult], item_class: str = "sponge") -> None:
        self._last, self._item_class = result, item_class
        cl = (result.extra.get("checklist") if result is not None else None) or {
            "front": None, "back": None, "items": [None] * 3, "extra_items": 0, "want": 3,
            "verdict": "NONE", "next": ""}
        if result is not None and "checklist" not in result.extra:
            cl = dict(cl, verdict="NONE")  # an error / no part without memory
        items = cl["items"]
        rows = [(tr("Glass front side"), cl["front"]), (tr("Glass back side"), cl["back"])]
        name = tr(item_name(item_class))
        rows += [(f"{name} {k}/{cl['want']}", state) for k, state in enumerate(items, 1)]
        if cl.get("extra_items"):
            rows.append((tr("Extra {}: {}").format(name.lower(), cl["extra_items"]), False))
        self._ensure_rows(len(rows))
        for (label, chip), (text, state) in zip(self._rows, rows):
            label.setText(text)
            self._set_chip(chip, state)
        verdict = cl["verdict"]
        if verdict == "PASS":
            text, bg, fg, hint = tr("PASS"), OK_GREEN, "#ffffff", ""
        elif verdict == "NG":
            text, bg, fg = tr("NG"), NG_RED, "#ffffff"
            hint = tr("Wrong side for this step") if result is not None and result.extra.get("reason") == "wrong side" \
                else tr("Fit the missing parts, then show the back again")
        elif verdict == "WAIT":
            side = cl.get("next") or ("BACK" if not cl["back"] else "FRONT")
            text, bg, fg = tr("TURN OVER"), STANDBY_YELLOW, "#1a1a1a"
            hint = tr("Show the {} side").format(tr(side.lower()))
        else:
            text, bg, fg = tr("NO GLASS"), NOT_CHECKED, "#c8ced6"
            hint = result.text if result is not None and result.text and result.text != "no part" else ""
        self.banner.setText(text)
        self.banner.setStyleSheet(BANNER_QSS.format(bg=bg, fg=fg))
        self.banner.setProperty("_i18n_skip", True)
        self.hint.setText(hint)
        self.hint.setProperty("_i18n_skip", True)
        self.hint.setVisible(bool(hint))
