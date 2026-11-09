"""Gauge Data tab: box plot of every reading of the chosen days for Box No.1-43,
day against day in colour, and each box's average."""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..context import AppContext
from ..gauge_log import BOX_COUNT, DayStats, DayStatsCache, available_days, box_labels, box_summary
from ..i18n import tr, translator
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon
from ..widgets.gauge_export_dialog import GaugeExportDialog
from ..widgets.layout import bottom_bar, top_bar

REFRESH_MS = 15000
SUMMARY_WIDTH = 230
DAY_ROW_HEIGHT = 44
VISIBLE_DAY_ROWS = 3
SWATCH_SIZE = 18
COMPARE_VALUE_PT = 9  # three days of 0.0000 have to fit the side panel
DPI = 100
INK = "#1a1a1a"
MUTED = "#5a6470"
# One colour per day. Not green or red, which mean OK and NG. Checked for
# colour-blind separation; add a day only with a colour that passes too.
DAY_COLORS = ("#1f5fb8", "#e08a00", "#00a3c4")
MAX_DAYS = len(DAY_COLORS)
BOX_GROUP_WIDTH = 0.8  # of one Box No. slot, shared by the days shown

ChartDay = Tuple[int, date, DayStats]  # colour index, day, its summary


class DaySelection:
    """The days on the chart. A day keeps its slot, so its colour, while it
    stays selected, whatever else is added or removed."""

    def __init__(self, day: date) -> None:
        self.slots: List[Optional[date]] = [None] * MAX_DAYS
        self.reset(day)

    def reset(self, day: date) -> None:
        self.slots = [day] + [None] * (MAX_DAYS - 1)

    def chosen(self) -> List[Tuple[int, date]]:
        return [(slot, day) for slot, day in enumerate(self.slots) if day is not None]

    @property
    def full(self) -> bool:
        return None not in self.slots

    def toggle(self, day: date) -> bool:
        """False if refused: the last day can't go, and a 4th can't come in."""
        if day in self.slots:
            if len(self.chosen()) == 1:
                return False
            self.slots[self.slots.index(day)] = None
            return True
        if self.full:
            return False
        self.slots[self.slots.index(None)] = day
        return True

    def follow(self, old: date, new: date) -> None:
        """The calendar moved on: a selected "today" stays today."""
        if old in self.slots:
            self.slots[self.slots.index(old)] = None if new in self.slots else new


def render_box_plot(days: List[ChartDay], width: int, height: int, unit: str) -> QImage:
    """Every box keeps its slot, empty or not; the days stand side by side in
    it, in legend order. Matplotlib's own objects, not pyplot, so it is safe
    off the UI thread."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    figure = Figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    labels = box_labels()
    positions = list(range(1, BOX_COUNT + 1))
    step = BOX_GROUP_WIDTH / max(len(days), 1)
    handles = []
    for index, (color_index, day, stats) in enumerate(days):
        color = DAY_COLORS[color_index]
        offset = (index - (len(days) - 1) / 2) * step
        present = [(position + offset, stats[label]) for position, label in zip(positions, labels) if label in stats]
        if present:
            axes.bxp([box for _, box in present], positions=[x for x, _ in present], widths=step * 0.85,
                     showfliers=False, manage_ticks=False, patch_artist=True,
                     boxprops={"facecolor": color, "edgecolor": color, "linewidth": 0.5},
                     whiskerprops={"color": color}, capprops={"color": color},
                     medianprops={"color": INK, "linewidth": 1.2})
        count = sum(s["count"] for s in stats.values())
        handles.append(Patch(facecolor=color, edgecolor=color, label=f"{day.isoformat()}  ({count:,})"))
    axes.legend(handles=handles, ncols=MAX_DAYS, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
                fontsize=8, labelcolor=INK, borderaxespad=0.2)
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


def summary_table(days: List[ChartDay]) -> Tuple[List[str], List[Tuple[str, List[Optional[float]]]]]:
    """(column headers, rows). One day: lowest average first. Several: in box
    order, so a row reads across the days."""
    if len(days) == 1:
        return ["Box No.", "Avg Value"], [(box, [average]) for box, average in box_summary(days[0][2])]
    seen = {box for _, _, stats in days for box in stats}
    labels = box_labels()
    ordered = [box for box in labels if box in seen] + sorted(seen - set(labels))
    rows = [(box.replace("Box No.", ""), [float(stats[box]["mean"]) if box in stats else None for _, _, stats in days])
            for box in ordered]
    return ["#", *[day.strftime("%m-%d") for _, day, _ in days]], rows


def _swatch(color: Optional[str]) -> QIcon:
    """A day's colour, or an empty outline while it isn't selected."""
    pixmap = QPixmap(SWATCH_SIZE, SWATCH_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setPen(QColor(color or MUTED))
    if color:
        painter.setBrush(QColor(color))
    painter.drawRect(1, 1, SWATCH_SIZE - 3, SWATCH_SIZE - 3)
    painter.end()
    return QIcon(pixmap)


@dataclass
class _ChartResult:
    image: QImage
    part: Optional[str]
    days: List[ChartDay]


class _ChartWorker(QThread):
    """Reading whole days and drawing their boxes takes long enough to stutter
    the live view on the Jetson."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, cache: DayStatsCache, part: Optional[str], chosen: List[Tuple[int, date]],
                 width: int, height: int, unit: str) -> None:
        super().__init__()
        self._cache = cache
        self._part = part
        self._chosen = chosen
        self._size = (width, height, unit)

    def run(self) -> None:
        try:
            days = [(slot, day, self._cache.get(self._part, day)) for slot, day in self._chosen]
            self.done.emit(_ChartResult(render_box_plot(days, *self._size), self._part, days))
        except Exception as exc:
            self.failed.emit(f"Gauge chart failed: {exc}")


class GaugeDataTab(QWidget):
    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._worker: Optional[_ChartWorker] = None
        self._cache = DayStatsCache()
        self._today = date.today()
        self._part: Optional[str] = None
        self._selection = DaySelection(self._today)
        self._days: List[date] = []
        self._result: Optional[_ChartResult] = None
        self._redraw = False  # asked for while a chart was being drawn
        self._closed = False

        self.info_label = QLabel()
        self.info_label.setObjectName("monitorInfo")
        self.chart = QLabel()
        self.chart.setAlignment(Qt.AlignCenter)
        self.chart.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)  # the picture follows the space
        days_header = QLabel("Days")
        days_header.setObjectName("panelHeader")
        self.day_list = QListWidget()
        self.day_list.setSelectionMode(QListWidget.NoSelection)
        self.day_list.setFocusPolicy(Qt.NoFocus)
        self.day_list.setIconSize(QSize(SWATCH_SIZE, SWATCH_SIZE))
        self.day_list.setFixedHeight(DAY_ROW_HEIGHT * VISIBLE_DAY_ROWS + 2)
        self.day_list.itemClicked.connect(self._on_day_clicked)
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
        self.export_button = QPushButton("Export")
        self.export_button.setIcon(make_icon("save"))
        self.export_button.setFixedSize(170, 44)
        self.export_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.export_button.clicked.connect(self._open_export)

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
        side_layout.addWidget(days_header)
        side_layout.addWidget(self.day_list)
        side_layout.addWidget(self.summary_header)
        side_layout.addWidget(self.summary_table, 1)

        middle = QHBoxLayout()
        middle.addWidget(frame, 1)
        middle.addWidget(side)

        bottom, bottom_layout = bottom_bar()
        hint = QLabel("Every reading of the day. Tap up to 3 days to compare.")
        hint.setObjectName("hint")
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        bottom_layout.addWidget(hint, 1)
        bottom_layout.addWidget(self.refresh_button)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.export_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(middle, 1)
        layout.addWidget(bottom)

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        translator().language_changed.connect(lambda _: self._on_language())

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

    def _on_language(self) -> None:
        self._refresh_text()
        self._show_days()
        if self._result is not None:
            self._fill_table(self._result.days)

    def _refresh_text(self) -> None:
        total = 0
        if self._result is not None:
            total = sum(s["count"] for _, _, stats in self._result.days for s in stats.values())
        self.info_label.setText(tr("Readings: {}   ·   Box now: {}").format(f"{total:,}", self._ctx.gauge_log.qr))
        self.summary_header.setText(tr("Box Summary ({})").format(self._ctx.filters.gauge.unit))

    def _sync_days(self) -> None:
        """Catches up with a job change, midnight and newly logged days."""
        part, today = self._ctx.gauge_log.part, date.today()
        if part != self._part:
            self._part = part
            self._selection.reset(today)
        elif today != self._today:
            self._selection.follow(self._today, today)
        self._today = today
        chosen = {day for _, day in self._selection.chosen()}
        days = sorted(set(available_days(part)) | {today} | chosen, reverse=True)
        if days != self._days:
            self._days = days
            self.day_list.clear()
            for _ in days:
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, DAY_ROW_HEIGHT))
                item.setFlags(Qt.ItemIsEnabled)
                self.day_list.addItem(item)
        self._show_days()

    def _show_days(self) -> None:
        colors = {day: DAY_COLORS[slot] for slot, day in self._selection.chosen()}
        for row, day in enumerate(self._days):
            item = self.day_list.item(row)
            text = day.isoformat()
            item.setText(f"{text}  {tr('Today')}" if day == self._today else text)
            item.setIcon(_swatch(colors.get(day)))
            font = item.font()
            font.setBold(day in colors)
            item.setFont(font)

    @Slot(QListWidgetItem)
    def _on_day_clicked(self, item: QListWidgetItem) -> None:
        if self._selection.toggle(self._days[self.day_list.row(item)]):
            self._show_days()
            self.refresh()
        elif self._selection.full:
            self._ctx.status.emit(tr("Up to {} days at once: tap one to remove it first.").format(MAX_DAYS))

    @Slot()
    def refresh(self) -> None:
        if self._closed:
            return
        self._sync_days()
        self._refresh_text()
        if self._worker is not None and self._worker.isRunning():
            self._redraw = True
            return
        width, height = self.chart.width(), self.chart.height()
        if width < 200 or height < 150:
            return  # not laid out yet
        self._redraw = False
        self._worker = _ChartWorker(self._cache, self._part, self._selection.chosen(), width, height,
                                    self._ctx.filters.gauge.unit)
        self._worker.done.connect(self._on_chart)
        self._worker.failed.connect(self._ctx.status)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    @Slot(object)
    def _on_chart(self, result: _ChartResult) -> None:
        if result.part != self._ctx.gauge_log.part:
            self._redraw = True  # drawn for the job before this one
            return
        self._result = result
        self.chart.setPixmap(QPixmap.fromImage(result.image))
        self._fill_table(result.days)
        self._refresh_text()

    @Slot()
    def _on_worker_finished(self) -> None:
        if self._redraw and self.isVisible():
            self.refresh()

    def _fill_table(self, days: List[ChartDay]) -> None:
        headers, rows = summary_table(days)
        self.summary_table.setColumnCount(len(headers))
        self.summary_table.setHorizontalHeaderLabels([tr(text) for text in headers])
        # Comparing, the values need the width; the box column holds just a number.
        self.summary_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents if len(days) > 1 else QHeaderView.Stretch)
        self.summary_table.setRowCount(len(rows))
        value_font = self.summary_table.font()
        if len(days) > 1:
            value_font.setPointSize(COMPARE_VALUE_PT)
        for row, (box, values) in enumerate(rows):
            self.summary_table.setItem(row, 0, QTableWidgetItem(box))
            for column, value in enumerate(values, start=1):
                cell = QTableWidgetItem("-" if value is None else f"{value:.4f}")
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                cell.setFont(value_font)
                self.summary_table.setItem(row, column, cell)

    @Slot()
    def _open_export(self) -> None:
        part = self._ctx.gauge_log.part
        days = available_days(part)
        if days:
            GaugeExportDialog(part, days, self).exec()
        else:
            self._ctx.status.emit(tr("No gauge data for this job yet."))

    def shutdown(self) -> None:
        """Before the window closes: Qt aborts on a QThread destroyed while running."""
        self._closed = True
        self._timer.stop()
        if self._worker is not None:
            self._worker.wait()
