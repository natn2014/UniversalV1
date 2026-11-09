"""Record the README GIFs of the image-processing tools and the gauge reader, offscreen.

The real widgets (Tool Catalog, Tools tab, Monitor, Gauge Data) are driven on
drawn scenes, so the GIFs can be remade whenever the UI changes. No camera or
tower light is used and all data goes to a temporary folder, so this can run
beside the station app:

    QT_QPA_PLATFORM=offscreen .venv/bin/python tools/make_readme_gifs.py [gauge]

Output: docs/gifs/{tool_catalog,eyedropper,pattern_search,counted_pixels}.gif
        docs/gifs/{gauge_reading,gauge_data}.gif ("gauge" makes only these)

The gauge GIFs show a drawn dial with its four points supplied directly, not
found by the model: everything after detection is the real code.
"""

import csv
import io
import math
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PySide6.QtCore import QBuffer, QIODevice, QPoint  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

APP = QApplication.instance() or QApplication([])

import app.context as context_mod  # noqa: E402
import app.jobs as jobs  # noqa: E402
import app.settings as settings_mod  # noqa: E402
from app.theme import DARK_INDUSTRIAL_QSS  # noqa: E402

OUT = ROOT / "docs" / "gifs"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GIF_WIDTH = 960


# --- frames ------------------------------------------------------------------

def settle() -> None:
    for _ in range(4):
        APP.processEvents()


def to_pil(pixmap_or_image) -> Image.Image:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    pixmap_or_image.save(buf, "PNG")
    return Image.open(io.BytesIO(bytes(buf.data()))).convert("RGB")


def tap(img: Image.Image, xy, r=26) -> Image.Image:
    """A finger-tap marker: white ring with a dark edge."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    x, y = xy
    d.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], outline=(20, 20, 20), width=3)
    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=4)
    d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 255, 255))
    return img


def captioned(img: Image.Image, text: str, width=GIF_WIDTH) -> Image.Image:
    """Scaled to `width` with a caption bar under it."""
    h = round(img.height * width / img.width)
    img = img.resize((width, h), Image.LANCZOS)
    bar = 46
    out = Image.new("RGB", (width, h + bar), (18, 21, 25))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    font = ImageFont.truetype(FONT_PATH, 20) if Path(FONT_PATH).exists() else ImageFont.load_default()
    d.text((14, h + 11), text, fill=(255, 210, 122), font=font)
    return out


def save_gif(name: str, frames, durations, palette_from=None) -> None:
    """`palette_from`: frames that between them hold every colour, when the
    first frame doesn't (OK green and NG red are never on screen together)."""
    sources = palette_from or frames[:1]
    sheet = Image.new("RGB", (sources[0].width, sum(f.height for f in sources)))
    for index, frame in enumerate(sources):
        sheet.paste(frame, (0, index * frame.height))
    palette = sheet.quantize(colors=255, method=Image.MEDIANCUT)
    images = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path = OUT / name
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"{path.relative_to(ROOT)}: {len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB")


def widget_point(widget, child, local=None) -> QPoint:
    p = local if local is not None else child.rect().center()
    return child.mapTo(widget, p)


# --- drawn scenes ------------------------------------------------------------

def harness_scene() -> np.ndarray:
    """Connector with four wires, a lot label, a terminal tray and a stain."""
    img = np.full((720, 1280, 3), 62, np.uint8)
    cv2.rectangle(img, (70, 120), (250, 520), (185, 185, 185), -1)
    cv2.rectangle(img, (70, 120), (250, 520), (110, 110, 110), 3)
    for i, color in enumerate(((40, 40, 205), (205, 95, 30), (30, 200, 230), (50, 160, 50))):
        y = 180 + i * 95
        for t in range(-22, 23):  # round wire: light on top, shadow below
            k = 1.25 - 0.75 * (t + 22) / 44
            c = tuple(int(min(255, v * k)) for v in color)
            cv2.line(img, (250, y + t), (760, y + t), c, 1)
    cv2.rectangle(img, (840, 90), (1200, 250), (236, 236, 230), -1)
    cv2.putText(img, "LOT 2410A", (870, 160), cv2.FONT_HERSHEY_DUPLEX, 1.5, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(img, "24V 3A", (870, 225), cv2.FONT_HERSHEY_DUPLEX, 1.3, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.rectangle(img, (840, 320), (1200, 520), (95, 95, 100), -1)
    for r in range(2):
        for c in range(5):
            if (c, r) == (3, 1):
                continue  # one terminal missing
            cx, cy = 885 + c * 68, 375 + r * 90
            cv2.circle(img, (cx, cy), 24, (215, 220, 225), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 9, (120, 120, 125), -1, cv2.LINE_AA)
    cv2.rectangle(img, (300, 580), (760, 690), (205, 205, 200), -1)
    cv2.circle(img, (610, 640), 13, (40, 50, 160), -1, cv2.LINE_AA)  # a stain
    return img


def mark(angle: float, size=120) -> np.ndarray:
    m = np.full((size, size), 225, np.uint8)
    c = size // 2
    cv2.circle(m, (c, c), c - 8, 40, 6, cv2.LINE_AA)
    cv2.putText(m, "E", (c - 22, c + 22), cv2.FONT_HERSHEY_DUPLEX, 2.0, 40, 5, cv2.LINE_AA)
    rot = cv2.getRotationMatrix2D((c - 0.5, c - 0.5), angle, 1.0)
    return cv2.warpAffine(m, rot, (size, size), borderValue=225)


def label_scene(angles) -> np.ndarray:
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(225, 3, (720, 1280)), 0, 255).astype(np.uint8)
    cv2.putText(img, "AGC  43R-006735", (520, 640), cv2.FONT_HERSHEY_DUPLEX, 1.4, 60, 3, cv2.LINE_AA)
    for (cx, cy), angle in zip(((260, 300), (640, 260), (1010, 380)), angles):
        m = mark(angle)
        s = m.shape[0] // 2
        img[cy - s:cy + s, cx - s:cx + s] = np.minimum(img[cy - s:cy + s, cx - s:cx + s], m)
    return cv2.merge([img] * 3)


def qimage(bgr: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def with_overlay(bgr: np.ndarray, results) -> Image.Image:
    from app.widgets.tool_overlay import draw_overlay
    img = qimage(bgr).convertToFormat(QImage.Format_RGB32)
    p = QPainter(img)
    for res in results:
        draw_overlay(p, res.overlay)
    p.end()
    return to_pil(img)


# --- the four GIFs -------------------------------------------------------------

def gif_catalog() -> None:
    from app.widgets.tool_params import AddToolDialog
    host = QWidget()
    host.resize(1120, 744)
    AddToolDialog.last_category = "Position"
    d = AddToolDialog(host)
    d.show()
    settle()
    frames, durations = [], []

    def shot(text, child=None, ms=1500):
        settle()
        img = to_pil(d.grab())
        if child is not None:
            p = widget_point(d, child)
            img = tap(img, (p.x(), p.y()))
        frames.append(captioned(img, text))
        durations.append(ms)

    shot("Tool Catalog: pick a category, then a tool", ms=1800)
    for category, kind, text in (
        ("Color", "color_prohibit", "Color → Color Prohibit: the navigator says what it checks"),
        ("Measure", "pitch", "Measure → Pitch: pin spacing, bent pins"),
        ("Pattern & Count", "pattern_search", "Pattern & Count → Pattern Search: any angle, counts copies"),
    ):
        d.category_tiles[category].click()
        shot(f"Tap a category: {category}", d.category_tiles[category], 1200)
        d.tiles[kind].click()
        shot(text, d.tiles[kind], 2000)
    shot("Add (or a double tap) puts it on the image: drag its area", d.add_button, 2200)
    d.close()
    save_gif("tool_catalog.gif", frames, durations)


def tools_tab(tmp: Path):
    from app.context import AppContext
    from app.tabs.tools_tab import ToolsTab
    jobs.save_part_config("DEMO", {"part_no": "DEMO"})
    ctx = AppContext()
    ctx.current_part_no = "DEMO"
    tab = ToolsTab(ctx)
    tab.resize(1500, 860)
    return tab


def view_point(tab, frame_xy):
    """Where a frame pixel shows on the grabbed tab."""
    view = tab.view
    rect = view._image_rect
    fh, fw = tab._frame.shape[:2]
    x = rect.x() + frame_xy[0] * rect.width() / fw
    y = rect.y() + frame_xy[1] * rect.height() / fh
    p = view.mapTo(tab, QPoint(int(x), int(y)))
    return p.x(), p.y()


def gif_eyedropper(tmp: Path) -> None:
    tab = tools_tab(tmp)
    frame = harness_scene()
    tab.set_frame(frame)
    tool = tab.add_tool("color_area", (250 / 1280, 140 / 720, 770 / 1280, 540 / 720))
    tab.show()
    settle()
    frames, durations = [], []

    def shot(text, at=None, child=None, ms=1700):
        settle()
        img = to_pil(tab.grab())
        if at is not None:
            img = tap(img, view_point(tab, at))
        if child is not None:  # a QPoint taken before the click (the button may hide)
            img = tap(img, (child.x(), child.y()))
        frames.append(captioned(img, text))
        durations.append(ms)

    shot(f"A new Color Area learns its whole area: {tool.run(frame).value:.0f} % counted, far too much", ms=2200)
    def at_button(button):
        return widget_point(tab, button)

    p = at_button(tab.pick_color_button)
    tab.pick_color_button.click()
    shot("Pick Color opens Color Extraction", child=p)
    lit = (520, 452)  # upper, lit side of the green wire
    tab.pick_at(lit)
    shot("Tap the green wire: the pixels it counts turn yellow", at=lit, ms=2200)
    shade = (520, 482)
    tab.pick_at(shade)
    shot("Tap its shadow too: that shade is added to the range", at=shade, ms=2200)
    p = at_button(tab.wider_button)
    tab.wider_button.click()
    shot("+ Wider / − Narrower adjust the range in steps", child=p)
    p = at_button(tab.done_pick_button)
    tab.done_pick_button.click()
    shot(f"Done: the green wire counts, {tool.run(frame).value:.1f} % of the area", child=p, ms=2600)
    save_gif("eyedropper.gif", frames, durations)


def gif_pattern_search() -> None:
    from app.vision.search import PatternSearchTool
    tool = PatternSearchTool(name="Pattern Search 1", roi=(0, 0, 1, 1))
    tool.teach(label_scene((0, 0, 0)), template_roi=(200 / 1280, 240 / 720, 320 / 1280, 360 / 720))
    frames, durations = [], []
    for k in range(12):
        angles = (30 * k, -25 * k, 45 * k + 10)
        scene = label_scene(angles)
        res = tool.run(scene)
        text = f"Pattern Search: {int(res.value)} found, each with its score and angle"
        frames.append(captioned(with_overlay(scene, [res]), text, 800))
        durations.append(700)
    save_gif("pattern_search.gif", frames, durations)


def gif_counted_pixels() -> None:
    from app.vision.brightness import AreaTool, EdgePixelsTool
    from app.vision.color import ColorProhibitTool
    from app.vision.count import BlobCountTool
    scene = harness_scene()
    w, h = 1280, 720

    def roi(x1, y1, x2, y2):
        return (x1 / w, y1 / h, x2 / w, y2 / h)

    shots = [
        (AreaTool(name="Area (mono) 1", roi=roi(830, 310, 1210, 530), threshold=170),
         "Area (mono): the bright pixels it counts, in see-through yellow"),
        (EdgePixelsTool(name="Edge Pixels 1", roi=roi(845, 95, 1195, 245), canny_lo=80, canny_hi=200),
         "Edge Pixels: the edges it counts (print present?)"),
        (BlobCountTool(name="Blob Count 1", roi=roi(830, 310, 1210, 530), threshold=170, area_min=300, low=10),
         "Blob Count: each counted terminal filled and numbered: 9 of 10 → NG"),
        (ColorProhibitTool(name="Color Prohibit 1", roi=roi(300, 580, 760, 690), h_lo=150, h_hi=10, s_lo=80,
                           min_blob_px=60),
         "Color Prohibit: the forbidden color in red → NG"),
    ]
    frames, durations = [], []
    for tool, text in shots:
        frames.append(captioned(with_overlay(scene, [tool.run(scene)]), text, 800))
        durations.append(2200)
    save_gif("counted_pixels.gif", frames, durations)


# --- the gauge GIFs -------------------------------------------------------------

GAUGE_GIF_WIDTH = 768  # as the other Monitor GIFs
GAUGE_JOB = "GAUGE-DEMO"
GAUGE_SPEC = {"value_at_start": 0.1, "value_at_end": 0.0, "unit": "MPa", "spec_min": 0.03, "spec_max": 0.07}
DIAL_CENTER = (640, 360)
DIAL_RADIUS = 300
SCALE_RADIUS = 270  # the Start and End points
NEEDLE_LENGTH = 185  # inside Start and End left to right, as read_gauge requires
DIAL_START, DIAL_SWEEP = 135.0, 270.0  # screen degrees, clockwise


def dial_point(percent: float, radius: float) -> tuple:
    angle = math.radians(DIAL_START + DIAL_SWEEP * percent / 100)
    return DIAL_CENTER[0] + radius * math.cos(angle), DIAL_CENTER[1] + radius * math.sin(angle)


def dial_value(percent: float) -> float:
    start, end = GAUGE_SPEC["value_at_start"], GAUGE_SPEC["value_at_end"]
    return start + (end - start) * percent / 100


def dial_scene(percent: float) -> np.ndarray:
    img = np.full((720, 1280, 3), 66, np.uint8)
    cv2.circle(img, DIAL_CENTER, DIAL_RADIUS + 22, (150, 150, 155), -1, cv2.LINE_AA)
    cv2.circle(img, DIAL_CENTER, DIAL_RADIUS + 8, (238, 238, 234), -1, cv2.LINE_AA)
    ink = (35, 35, 35)
    for step in range(0, 101, 2):
        major = step % 10 == 0
        outer = tuple(round(v) for v in dial_point(step, DIAL_RADIUS - 8))
        inner = tuple(round(v) for v in dial_point(step, DIAL_RADIUS - (42 if major else 24)))
        cv2.line(img, outer, inner, ink, 4 if major else 2, cv2.LINE_AA)
        if step % 20 == 0:
            x, y = dial_point(step, DIAL_RADIUS - 76)
            cv2.putText(img, f"{dial_value(step):.2f}", (round(x) - 32, round(y) + 9), cv2.FONT_HERSHEY_DUPLEX,
                        0.75, ink, 2, cv2.LINE_AA)
    cv2.putText(img, "MPa", (DIAL_CENTER[0] - 38, DIAL_CENTER[1] + 190), cv2.FONT_HERSHEY_DUPLEX, 1.0, ink, 2, cv2.LINE_AA)
    tip = tuple(round(v) for v in dial_point(percent, NEEDLE_LENGTH))
    cv2.line(img, DIAL_CENTER, tip, (25, 25, 25), 7, cv2.LINE_AA)
    cv2.circle(img, DIAL_CENTER, 20, (25, 25, 25), -1, cv2.LINE_AA)
    return img


def dial_points(percent: float) -> list:
    from app.gauge import CLASS_CENTER, CLASS_END, CLASS_START, CLASS_TIP

    points = ((CLASS_CENTER, "Center", DIAL_CENTER), (CLASS_START, "Start", dial_point(0, SCALE_RADIUS)),
              (CLASS_END, "End", dial_point(100, SCALE_RADIUS)), (CLASS_TIP, "Tip", dial_point(percent, NEEDLE_LENGTH)))
    size = 16
    return [{"x1": x - size, "y1": y - size, "x2": x + size, "y2": y + size, "cls_id": cls_id, "conf": 0.9,
             "class_name": name} for cls_id, name, (x, y) in points]


def gif_gauge_reading(win) -> None:
    frames, durations = [], []

    def show(percent, text, ms):
        for _ in range(5):  # what GaugeHold needs to settle on this position
            win.monitor_tab.on_frame(qimage(dial_scene(percent)), dial_points(percent), [])
        settle()
        frames.append(captioned(to_pil(win.grab()), text, GAUGE_GIF_WIDTH))
        durations.append(ms)

    def sweep(first, last, text):
        direction = 5 if last > first else -5
        for percent in range(first, last, direction):
            show(percent, text(percent), 220)

    def live(percent):
        return f"Needle at {dial_value(percent):.3f} MPa, limits 0.030 to 0.070"

    win.tabs.setCurrentWidget(win.monitor_tab)
    show(50, "In spec: OK", 2000)
    ok_frame = frames[-1]
    sweep(50, 15, live)
    show(15, "Over MAX: NG, and the image is saved tagged OVER", 2800)
    over_frame = frames[-1]
    sweep(15, 50, live)
    show(50, "Back in spec: OK", 1600)
    sweep(50, 85, live)
    show(85, "Under MIN: NG, and the image is saved tagged UNDER", 2800)
    sweep(85, 50, live)
    save_gif("gauge_reading.gif", frames, durations, palette_from=[ok_frame, over_frame])


def write_demo_days(part: str) -> None:
    """Three days of readings, the middle one running high."""
    from app import gauge_log

    rng = np.random.default_rng(7)
    today = date.today()
    for back, center in ((0, 0.050), (1, 0.056), (2, 0.046)):
        day = today - timedelta(days=back)
        path = gauge_log.daily_log_path(part, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        start = datetime(day.year, day.month, day.day, 8)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(gauge_log.CSV_HEADER)
            for i in range(43 * 200):
                box = i % 43 + 1
                value = round(float(rng.normal(center + box / 43 * 0.008, 0.004)), 4)
                writer.writerow([start + timedelta(seconds=i), value, f"Box No.{box}"])


def gif_gauge_data(win) -> None:
    tab = win.gauge_data_tab
    frames, durations = [], []

    def wait_for_chart():
        settle()
        while tab._redraw or (tab._worker is not None and tab._worker.isRunning()):
            APP.processEvents()
        settle()

    def shot(text, ms, row=None):
        img = to_pil(win.grab())
        if row is not None:
            p = widget_point(win, tab.day_list.viewport(), tab.day_list.visualItemRect(tab.day_list.item(row)).center())
            img = tap(img, (p.x(), p.y()))
        frames.append(captioned(img, text, GAUGE_GIF_WIDTH))
        durations.append(ms)

    def tap_day(row, text):
        shot(text, 700, row)
        tab._on_day_clicked(tab.day_list.item(row))
        wait_for_chart()
        shot(text, 2400)

    write_demo_days(GAUGE_JOB)
    win.tabs.setCurrentWidget(tab)
    tab.refresh()
    wait_for_chart()
    tab.refresh()  # the first one was laid out before the tab had its size
    wait_for_chart()
    shot("Every reading of the day, box by box", 2800)
    tap_day(1, "Tap another day: it is added in its own colour")
    tap_day(2, "Up to 3 days side by side, with each day's averages")
    tap_day(1, "Tap a day again to remove it")
    save_gif("gauge_data.gif", frames, durations, palette_from=[frames[4], frames[0]])  # 4: all three days


def gauge_gifs(root: Path) -> None:
    from app import ng_log, paths
    from app.tabs import settings_tab

    patches = [
        mock.patch.object(paths, "MARK_MODEL_PATH", root / "missing.pt"),
        mock.patch.object(jobs, "LAST_PART_FILE", root / ".last_part"),
        mock.patch.object(ng_log, "NG_LOGS_DIR", root / "ng"),
        mock.patch.object(paths, "GAUGE_LOGS_DIR", root / "gauge"),
        mock.patch.object(settings_tab, "find_cameras", return_value=[]),
        mock.patch("app.tower.TowerController.connect_device"),
        mock.patch("app.main_window.MainWindow.confirm_unsaved", return_value=True),
        mock.patch("app.tabs.monitor_tab.MonitorTab.ensure_live", return_value=False),
    ]
    for patch in patches:
        patch.start()
    try:
        from app.main_window import MainWindow
        win = MainWindow()
        win.resize(1024, 768)
        win.ctx.set_current_part(GAUGE_JOB)
        win.settings_tab.set_gauge(GAUGE_SPEC)
        win.settings_tab.set_judge_mode("gauge")
        win.ctx.settings.set(state_hold_ms=0, mark_hold_ms=0, ok_count=0, ng_count=0)
        win.show()
        settle()
        win.ctx.worker.isRunning = lambda: True  # no camera here: shown as it is on the line
        win.monitor_tab._refresh_run_button()
        gif_gauge_reading(win)
        del win.ctx.worker.isRunning
        gif_gauge_data(win)
        win.gauge_data_tab.shutdown()
        win.ctx.relay.shutdown()
    finally:
        for patch in patches:
            patch.stop()


def main() -> int:
    APP.setStyleSheet(DARK_INDUSTRIAL_QSS)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    with mock.patch.object(settings_mod, "APP_SETTINGS_FILE", tmp / "app_settings.json"), \
            mock.patch.object(jobs, "PART_CONFIGS_DIR", tmp), \
            mock.patch.object(context_mod, "TOOLS_DIR", tmp / "tools"):
        if sys.argv[1:] != ["gauge"]:
            gif_catalog()
            gif_eyedropper(tmp)
            gif_pattern_search()
            gif_counted_pixels()
        gauge_gifs(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
