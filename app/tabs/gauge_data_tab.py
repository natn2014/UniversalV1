"""Gauge Data tab: box plot of the recent readings for Box No.1-43, and each
box's average, lowest first, as in Gauge_reader-main's Graph tab."""

from typing import List, Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..context import AppContext
from ..gauge_log import BOX_COUNT, Row, box_labels, box_summary, values_by_box
from ..i18n import tr, translator
from ..theme import make_icon
from ..widgets.layout import bottom_bar, top_bar

REFRESH_MS = 5000
SUMMARY_WIDTH = 230
DPI = 100
INK = "#1a1a1a"
MUTED = "#5a6470"
MEDIAN = "#1f5fb8"  # the theme's blue: the one line to read first


def render_box_plot(rows: List[Row], width: int, height: int, unit: str) -> QImage:
    """Every box keeps its slot, empty or not. Matplotlib's own objects, not
    pyplot, so it is safe off the UI thread."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    labels = box_labels()
    positions = list(range(1, BOX_COUNT + 1))
    grouped = values_by_box(rows)
    data = [grouped.get(label, []) for label in labels]
    if any(data):
        axes.boxplot(data, positions=positions, widths=0.6, showfliers=False, manage_ticks=False,
                     boxprops={"color": INK}, whiskerprops={"color": INK}, capprops={"color": INK},
                     medianprops={"color": MEDIAN, "linewidth": 2})
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=90, fontsize=7, color=MUTED)
    axes.set_xlim(0.25, BOX_COUNT + 0.75)
    axes.tick_params(axis="y", labelcolor=MUTED, labelsize=8)
    axes.set_ylabel(f"Converted Value ({unit})".replace(" ()", ""), color=INK, fontsize=9)
    axes.grid(False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    figure.tight_layout(pad=0.6)
    canvas.draw()
    w, h = canvas.get_width_height()
    return QImage(bytes(canvas.buffer_rgba()), w, h, 4 * w, QImage.Format_RGBA8888).copy()


class _ChartWorker(QThread):
    """Drawing 43 boxes takes long enough to stutter the live view on the Jetson."""

    done = Signal(QImage)
    failed = Signal(str)

    def __init__(self, rows: List[Row], width: int, height: int, unit: str) -> None:
        super().__init__()
        self._args = (rows, width, height, unit)

    def run(self) -> None:
        try:
            self.done.emit(render_box_plot(*self._args))
        except Exception as exc:
            self.failed.emit(f"Gauge chart failed: {exc}")


class GaugeDataTab(QWidget):
    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._worker: Optional[_ChartWorker] = None

        self.info_label = QLabel()
        self.info_label.setObjectName("monitorInfo")
        self.chart = QLabel()
        self.chart.setAlignment(Qt.AlignCenter)
        self.chart.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # the picture follows the space
        self.summary_header = QLabel()
        self.summary_header.setObjectName("panelHeader")
        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["Box No.", "Avg Value"])
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setSelectionMode(QTableWidget.NoSelection)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setIcon(make_icon("refresh", "#e0e0e0"))
        self.refresh_button.setFixedHeight(44)
        self.refresh_button.clicked.connect(self.refresh)

        top, top_layout = top_bar()
        top_layout.addWidget(self.info_label, 1)

        frame = QFrame()
        frame.setObjectName("videoFrame")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(2, 2, 2, 2)
        frame_layout.addWidget(self.chart)

        side = QFrame()
        side.setObjectName("sidePanel")
        side.setFixedWidth(SUMMARY_WIDTH)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_layout.addWidget(self.summary_header)
        side_layout.addWidget(self.summary_table, 1)

        middle = QHBoxLayout()
        middle.addWidget(frame, 1)
        middle.addWidget(side)

        bottom, bottom_layout = bottom_bar()
        hint = QLabel("One reading a second, saved per job and day. The chart shows the last 1000.")
        hint.setObjectName("hint")
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bottom_layout.addWidget(hint, 1)
        bottom_layout.addWidget(self.refresh_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(middle, 1)
        layout.addWidget(bottom)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        translator().language_changed.connect(lambda _: self._refresh_text())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._timer.start()  # only while looked at, so it costs the Monitor nothing
        QTimer.singleShot(0, self.refresh)

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.isVisible():
            self.refresh()

    def _refresh_text(self) -> None:
        log = self._ctx.gauge_log
        self.info_label.setText(tr("Readings: {}   ·   Box now: {}").format(len(log.rows), log.qr))
        self.summary_header.setText(tr("Box Summary ({})").format(self._ctx.filters.gauge.unit))

    @Slot()
    def refresh(self) -> None:
        self._refresh_text()
        rows = list(self._ctx.gauge_log.rows)
        summary = box_summary(rows)
        self.summary_table.setRowCount(len(summary))
        for row, (box, average) in enumerate(summary):
            value = QTableWidgetItem(f"{average:.4f}")
            value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.summary_table.setItem(row, 0, QTableWidgetItem(box))
            self.summary_table.setItem(row, 1, value)
        if self._worker is not None and self._worker.isRunning():
            return  # the next refresh picks up what this one missed
        width, height = self.chart.width(), self.chart.height()
        if width < 200 or height < 150:
            return  # not laid out yet
        self._worker = _ChartWorker(rows, width, height, self._ctx.filters.gauge.unit)
        self._worker.done.connect(self._show_chart)
        self._worker.failed.connect(self._ctx.status)
        self._worker.start()

    @Slot(QImage)
    def _show_chart(self, image: QImage) -> None:
        self.chart.setPixmap(QPixmap.fromImage(image))

    def shutdown(self) -> None:
        """Before the window closes: Qt aborts on a QThread destroyed while running."""
        self._timer.stop()
        if self._worker is not None:
            self._worker.wait()
