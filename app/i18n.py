"""English / Thai user interface.

English is written in the code; app/i18n_th.py maps each English string to
Thai. Two ways text gets translated:

- Static text (labels, buttons, tooltips, combo items, table headers, tab
  titles, placeholders, spinbox suffixes): the Translator walks the widgets
  when the language changes and whenever a widget is shown for the first
  time, keeping each widget's English source in a dynamic property. No code
  change is needed for text set once.
- Text the code changes while running (status, info strip, verdict, counts):
  call tr() / trf() where it is set, and refresh it on language_changed.

Formatted strings ("Unsaved changes (3)") are matched by the patterns in
i18n_th.PATTERNS. Class / mark names and part numbers are never in the
catalogue, so they stay as they are.
"""

import re
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTabWidget,
    QWidget,
)

from .i18n_th import PATTERNS, TH

LANGUAGES = (("en", "English"), ("th", "ไทย"))
_SRC_ROLE = Qt.UserRole + 77  # combo / header item: its English text
_OUT_ROLE = Qt.UserRole + 78  # ... and the text last set on it
_RICH_SPLIT = re.compile(r"(<[^>]+>|&nbsp;|·)")


class Translator(QObject):
    language_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.language = "en"
        self._filter_installed = False

    # --- Strings --------------------------------------------------------

    def tr(self, text: str) -> str:
        if self.language == "en" or not text:
            return text
        found = TH.get(text)
        if found is not None:
            return found
        stripped = text.strip()
        if stripped != text and stripped in TH:
            return text.replace(stripped, TH[stripped])
        for pattern, template in PATTERNS:
            match = pattern.fullmatch(text)
            if match:
                return template.format(*(self.tr(g) if g else g for g in match.groups()))
        if "<" in text or "&nbsp;" in text or "·" in text:
            return "".join(part if _RICH_SPLIT.fullmatch(part) else self._tr_segment(part)
                           for part in _RICH_SPLIT.split(text))
        return text

    def _tr_segment(self, part: str) -> str:
        core = part.strip()
        if not core:
            return part
        return part.replace(core, self.tr(core)) if self.tr(core) != core else part

    def set_language(self, language: str) -> None:
        if language not in dict(LANGUAGES):
            language = "en"
        if language == self.language:
            return
        self.language = language
        self._set_filter(language != "en")
        app = QApplication.instance()
        if app is not None:
            for widget in app.allWidgets():
                self.translate_widget(widget)
        self.language_changed.emit(language)

    def _set_filter(self, on: bool) -> None:
        """While not English, widgets are translated as they are shown (dialogs,
        rebuilt lists); each child gets its own Show event."""
        app = QApplication.instance()
        if app is None or on == self._filter_installed:
            return
        if on:
            app.installEventFilter(self)
        else:
            app.removeEventFilter(self)
        self._filter_installed = on

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Show and isinstance(obj, QWidget):
            self.translate_widget(obj)
        return False

    # --- Widgets --------------------------------------------------------

    def translate_tree(self, root: QWidget) -> None:
        self.translate_widget(root)
        for child in root.findChildren(QWidget):
            self.translate_widget(child)

    def _swap(self, obj: QObject, key: str, current: str, setter) -> None:
        """Remembers the English source of one text and sets its translation.
        Text the code set since the last swap is taken as the new source."""
        if not current:
            return
        src_key, out_key = f"_i18n_src_{key}", f"_i18n_out_{key}"
        out = obj.property(out_key)
        src = obj.property(src_key) if out is not None and current == out else current
        new = self.tr(src)
        obj.setProperty(src_key, src)
        obj.setProperty(out_key, new)
        if new != current:
            setter(new)

    def _swap_item(self, current: str, get, put, set_text) -> None:
        """_swap for a combo / header item, keeping its texts in item roles."""
        if not current:
            return
        src, out = get(_SRC_ROLE), get(_OUT_ROLE)
        if src is None or current != out:
            src = current  # new text from the code
        new = self.tr(src)
        put(src, _SRC_ROLE)
        put(new, _OUT_ROLE)
        if new != current:
            set_text(new)

    def source(self, obj: QObject, key: str, current: str) -> str:
        """The English behind a translated text (e.g. a tab title)."""
        src, out = obj.property(f"_i18n_src_{key}"), obj.property(f"_i18n_out_{key}")
        return src if src is not None and current == out else current

    def translate_widget(self, w: QWidget) -> None:
        if w.property("_i18n_skip"):
            return
        self._swap(w, "tip", w.toolTip(), w.setToolTip)
        if isinstance(w, (QLabel, QAbstractButton)):
            self._swap(w, "text", w.text(), w.setText)
        if isinstance(w, QLineEdit):
            self._swap(w, "placeholder", w.placeholderText(), w.setPlaceholderText)
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            self._swap(w, "suffix", w.suffix(), w.setSuffix)
        if isinstance(w, QAbstractSpinBox):
            self._swap(w, "special", w.specialValueText(), w.setSpecialValueText)
        if isinstance(w, QComboBox):
            for i in range(w.count()):
                self._swap_item(w.itemText(i), lambda role, i=i: w.itemData(i, role),
                                lambda value, role, i=i: w.setItemData(i, value, role),
                                lambda text, i=i: w.setItemText(i, text))
        if isinstance(w, QTableWidget):
            for i in range(w.columnCount()):
                item = w.horizontalHeaderItem(i)
                if item is not None:
                    self._swap_item(item.text(), item.data, lambda value, role, it=item: it.setData(role, value),
                                    item.setText)
        if isinstance(w, QTabWidget):
            bar = w.tabBar()
            for i in range(w.count()):
                self._swap(bar, f"tab{i}", w.tabText(i), lambda t, i=i: w.setTabText(i, t))
        title = w.windowTitle()
        if w.isWindow() and title:
            self._swap(w, "title", title, w.setWindowTitle)


_translator: Optional[Translator] = None


def translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def tr(text: str) -> str:
    return translator().tr(text)


def trf(template: str, *values, **named) -> str:
    """tr() of a template, then format() with the values: the values (a
    part number, a read) are never translated themselves."""
    return tr(template).format(*values, **named)
