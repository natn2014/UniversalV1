from typing import Callable, Optional

from PySide6.QtWidgets import QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from ..theme import RUN_BUTTON_STYLES, action_button_qss


class RemoveJobDialog(QDialog):
    """Type-to-confirm removal: the Remove button only enables once the exact
    part number is typed, so a stray tap can't delete a job."""

    def __init__(self, part_numbers: list, current_part: Optional[str], parent=None,
                 title: str = "Remove Job", item_label: str = "Job",
                 warning: Optional[Callable[[str], str]] = None) -> None:
        """`title`, `item_label` and `warning(name)` let other lists of named
        things (the vision tools) reuse the same type-to-confirm flow."""
        super().__init__(parent)
        self._warning = warning
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(560, 360)

        top_bar = QFrame()
        top_bar.setObjectName("monitorTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 6, 10, 6)
        title_label = QLabel(title)
        title_label.setObjectName("dangerTitle")
        top_layout.addWidget(title_label)
        top_layout.addStretch(1)

        self.part_combo = QComboBox()
        for part_no in part_numbers:
            self.part_combo.addItem(part_no)
        if current_part in part_numbers:
            self.part_combo.setCurrentText(current_part)

        self.warning_label = QLabel()
        self.warning_label.setObjectName("dangerText")
        self.warning_label.setWordWrap(True)
        self.prompt_label = QLabel()
        self.prompt_label.setWordWrap(True)
        self.confirm_field = QLineEdit()
        self.confirm_field.setPlaceholderText("Type the part number here")

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 10, 12, 12)
        card_layout.setSpacing(8)
        job_row = QHBoxLayout()
        job_row.addWidget(QLabel(f"{item_label}:"))
        job_row.addWidget(self.part_combo, 1)
        card_layout.addLayout(job_row)
        card_layout.addWidget(self.warning_label)
        card_layout.addWidget(self.prompt_label)
        card_layout.addWidget(self.confirm_field)
        card_layout.addStretch(1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["stop"]))
        self.remove_button.clicked.connect(self.accept)
        for button in (self.cancel_button, self.remove_button):
            button.setFixedHeight(44)
            button.setMinimumWidth(140)

        bottom_bar = QFrame()
        bottom_bar.setObjectName("monitorBottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(6, 4, 6, 4)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.cancel_button)
        bottom_layout.addWidget(self.remove_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top_bar)
        layout.addWidget(card, 1)
        layout.addWidget(bottom_bar)

        self.part_combo.currentTextChanged.connect(self._on_part_changed)
        self.confirm_field.textChanged.connect(self._update_remove_enabled)
        self._on_part_changed(self.part_combo.currentText())
        self.confirm_field.setFocus()

    def selected_part(self) -> str:
        return self.part_combo.currentText()

    def _on_part_changed(self, part_no: str) -> None:
        if self._warning is not None:
            self.warning_label.setText(self._warning(part_no))
        else:
            self.warning_label.setText(
                f"This permanently deletes the saved job \"{part_no}\" (model, filters and "
                "target classes). NG images in the Log are kept. This cannot be undone.")
        self.prompt_label.setText(f"To confirm, type  <b>{part_no}</b>  below:")
        self.confirm_field.clear()
        self._update_remove_enabled()

    def _update_remove_enabled(self) -> None:
        # Exact match (case-sensitive): part numbers like "ab-1" / "AB-1" differ.
        part_no = self.part_combo.currentText()
        self.remove_button.setEnabled(bool(part_no) and self.confirm_field.text().strip() == part_no)
