from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


def make_card(title: str):
    """Section card with a dark header strip (Keyence-style section bar)."""
    card = QFrame()
    card.setObjectName("card")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    header = QLabel(title)
    header.setObjectName("cardHeader")
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
