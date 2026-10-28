from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from ..theme import icon_title, set_badge_title

# Width of the panel right of the live image. Shared by the Monitor and Tools
# tabs so the camera image is the same size (and ROIs in the same place) in both.
SIDE_PANEL_WIDTH = 400
# Smallest live image, for both tabs; with the side panel it fits a 1024 screen.
VIDEO_MIN_SIZE = (520, 300)


def make_card(title: str, badge: str = None, icon: str = None):
    """Section card with a dark header strip (Keyence-style section bar).
    `badge` ("ai" / "vision") or `icon` (a make_icon kind) goes before the title."""
    card = QFrame()
    card.setObjectName("card")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    header = QLabel(title)
    header.setObjectName("cardHeader")
    if badge:
        set_badge_title(header, badge, title)
    elif icon:
        header.setTextFormat(Qt.RichText)
        header.setText(icon_title(icon, title))
    outer.addWidget(header)
    body = QVBoxLayout()
    body.setContentsMargins(12, 10, 12, 12)
    body.setSpacing(8)
    outer.addLayout(body, 1)
    return card, body


def row(*items) -> QHBoxLayout:
    """HBox from widgets; None adds a stretch."""
    layout = QHBoxLayout()
    for item in items:
        if item is None:
            layout.addStretch(1)
        else:
            layout.addWidget(item)
    return layout


def top_bar(margins=(10, 4, 6, 4)):
    """Dark info strip used at the top of the Monitor/Train/Job Change tabs."""
    bar = QFrame()
    bar.setObjectName("monitorTopBar")
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(*margins)
    return bar, layout


def bottom_bar(margins=(6, 4, 6, 4)):
    """Action bar used at the bottom of the Monitor/Train/Job Change tabs."""
    bar = QFrame()
    bar.setObjectName("monitorBottomBar")
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(*margins)
    return bar, layout


def section_heading(title: str, note: str = ""):
    """Big heading over a group of cards, with a grey note under it.
    Returns (widget, note_label) so the note can be updated."""
    box = QFrame()
    box.setObjectName("clearBox")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(4, 2, 4, 2)
    layout.setSpacing(0)
    heading = QLabel(title)
    heading.setObjectName("sectionHeading")
    note_label = QLabel(note)
    note_label.setObjectName("hint")
    note_label.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(note_label)
    return box, note_label
