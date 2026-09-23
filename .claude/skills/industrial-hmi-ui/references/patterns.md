# Screen recipes and behaviour patterns

Code assumes `from industrial_theme import *` plus the usual PySide6 imports. Each recipe matches a screenshot in `../assets/examples/`.

## Contents
1. [Monitor / live-inspection screen](#1-monitor--live-inspection-screen)
2. [Settings screen (two columns of cards)](#2-settings-screen-two-columns-of-cards)
3. [Job / recipe picker (touch tiles)](#3-job--recipe-picker-touch-tiles)
4. [Gallery / log with image popup](#4-gallery--log-with-image-popup)
5. [Labeling / training screen](#5-labeling--training-screen)
6. [Camera dialog (e.g. barcode scan)](#6-camera-dialog-eg-barcode-scan)
7. [Type-to-confirm delete](#7-type-to-confirm-delete)
8. [Verdict hold latch + cycle counting](#8-verdict-hold-latch--cycle-counting)
9. [Start/Stop button driven by worker state](#9-startstop-button-driven-by-worker-state)
10. [Persisted per-machine settings](#10-persisted-per-machine-settings)

---

## 1. Monitor / live-inspection screen
*(monitor_running.png)*

```python
top, top_l = make_strip("top")
self.info = make_info_label()            # "Part: X · Model: Y", refreshed on change
self.pill = make_judgement_pill()
top_l.addWidget(self.info); top_l.addStretch(1); top_l.addWidget(self.pill)

side, side_l = make_side_panel()          # light results panel on the right
side_l.addWidget(make_panel_header("Detections"))
style_panel_table(self.table); side_l.addWidget(self.table, 1)
counters = make_clear_box(); c_l = QVBoxLayout(counters)
row = QHBoxLayout()
ok_tile, self.ok_value = make_counter_tile("OK", "#5cbf60", "#2e7d32", "#4cd964")
ng_tile, self.ng_value = make_counter_tile("NG", "#e0685a", "#b83225", "#ff5a4a")
row.addWidget(ok_tile, 1); row.addWidget(ng_tile, 1); c_l.addLayout(row)
reset = QPushButton("Reset Count"); reset.setObjectName("counterReset")
c_l.addLayout(make_row(None, reset)); side_l.addWidget(counters)

content = QHBoxLayout()
content.addWidget(make_framed(self.video_label), 1)
content.addWidget(side)

bottom, bot_l = make_strip("bottom")
bot_l.addWidget(self.capture_button); bot_l.addWidget(self.buzzer_toggle)   # secondary, left
bot_l.addStretch(1)
bot_l.addWidget(self.status_chip); bot_l.addSpacing(8); bot_l.addWidget(self.run_button)  # primary, right

page = QVBoxLayout(); page.addWidget(top); page.addLayout(content, 1); page.addWidget(bottom)
```

- On/off toggles (such as the buzzer) are `setCheckable(True)`. The checked state shows the orange gloss, and a speaker icon from `make_icon("speaker_on" / "speaker_off")` carries the meaning.
- The counter is 40pt, which fits 3 digits in a ~120 px tile. Say so if counts can exceed 999.

## 2. Settings screen (two columns of cards)
*(settings.png)*

```python
model_card, b = make_card("Model");   b.addLayout(make_row(load_btn, model_label))
cam_card, b   = make_card("Camera");  b.addLayout(make_row(QLabel("Camera:"), combo, scan_btn, None))
io_card, b    = make_card("Tower Light & Buzzer")
b.addWidget(status_label); b.addWidget(enable_checkbox)
b.addLayout(make_row(QLabel("NG alarm:"), style_combo, None, test_btn))
b.addLayout(make_row(QLabel("State hold time:"), None, hold_spin))
b.addWidget(make_hint("Minimum time between STANDBY / OK / NG changes (0 = off)"))
filt_card, b  = make_card("Detection Filters")
value_label.setStyleSheet("color:#ffcf73; font-weight:700;")   # amber live value
b.addLayout(make_row(QLabel("Confidence threshold"), None, value_label)); b.addWidget(slider)
b.addLayout(make_row(QLabel("Max detections:"), None, spin))
b.addWidget(class_list_scroll, 1)

left = QVBoxLayout(); [left.addWidget(c) for c in (model_card, cam_card, io_card)]; left.addStretch(1)
cols = QHBoxLayout(); cols.addLayout(left, 1); cols.addWidget(filt_card, 1)
# Wrap in a QScrollArea (setFrameShape(QFrame.NoFrame)) as a safety net on small panels.
```

- Put controls in the same row as their label, with the value right-aligned by a `None` stretch. It scans faster than stacked label and control pairs.
- A "Test" button next to anything audible or visible (buzzer style, beep) lets people check the setting without waiting for a real event.

## 3. Job / recipe picker (touch tiles)
*(job_change.png)*

```python
top, top_l = make_strip("top")
cur = QLabel(); cur.setObjectName("currentPart")                  # big amber current value
top_l.addWidget(make_info_label("Current Part:")); top_l.addWidget(cur); top_l.addStretch(1)

card, body = make_card("Saved Jobs")
self.grid = QGridLayout(); self.grid.setSpacing(10)
grid_w = make_clear_box(); grid_w.setLayout(self.grid)
holder = make_clear_box(); h = QVBoxLayout(holder); h.setContentsMargins(0,0,0,0)
h.addWidget(grid_w); h.addStretch(1)                                # tiles stay at the top
scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
scroll.setWidget(holder); scroll.viewport().setObjectName("clearBox"); body.addWidget(scroll, 1)

def refresh_tiles(self):
    clear_layout(self.grid)
    for i, (name, subtitle) in enumerate(self.jobs()):
        t = make_tile_button(f"{name}\n{subtitle}", checked=(name == self.current))
        t.clicked.connect(lambda _=False, n=name: self.select_job(n))   # select_job calls refresh_tiles
        self.grid.addWidget(t, *divmod(i, 4))
    for c in range(4): self.grid.setColumnStretch(c, 1)

bottom, bot_l = make_strip("bottom")
bot_l.addWidget(remove_btn)                                  # destructive: secondary, far left
bot_l.addWidget(make_hint("Tap a job to switch · Scan a barcode to load or create a job"))
bot_l.addStretch(1); bot_l.addWidget(make_primary_button("Scan Barcode..."))
```

- A tile shows two lines: the key (part number) and what it implies (model name). The active tile is orange, and the highlight moves on every switch.
- An empty state is a hint in the grid ("No saved jobs yet. Scan a barcode to create one."), never a blank screen.

## 4. Gallery / log with image popup
*(log_popup.png)*

- **Groups:** one `card`-style `QFrame` per group, with a `panelHeader`-style strip ("Part: X · NG 7 · Last: 2026-09-23 17:25"). The group with the newest item is at the top.
- **Thumbnails:** a row of `QToolButton`s (`ToolButtonTextUnderIcon`, icon = thumbnail file, text = timestamp, 2 px red border for NG). Show the 5 newest, then a stretch.
- **Thumbnail files:** save a small `_thumb.jpg` beside each full image when you write it, so the gallery never decodes full frames.
- **Rebuilding:** rebuild only when the tab is visible. Otherwise set a dirty flag and rebuild on `QTabWidget.currentChanged`, which keeps the live screen fast.
- **Popup:** create it once, `self.overlay = ImageOverlay(self)`, then call `self.overlay.show_image(path, "NG · X · time")`.
  - It opens fitted at 85% of the window and supports pinch/wheel zoom (50–800%), drag, double-tap to fit, the − + Fit buttons, ✕, Esc and tap-outside to close.
  - Forward the main window's `resizeEvent` to `overlay.setGeometry(self.rect())` while it is visible.

## 5. Labeling / training screen
- **Top strip:** "Active Model: …" and "Dataset: …" on **one line**, followed by `Select Folder...`.
  - Show only the last two path parts, and put the full path in a tooltip.
- **Centre:** `make_framed(image_widget)` plus a side panel of width 260 containing:
  - the "Images" header and a list set up with `style_panel_list` (elided on the left, so dates stay visible);
  - the "Box Correction" header and a clear box with `panelText` labels, a combo and **Confirm**.
- **Bottom strip:** ◀ Prev · "Image 1 / N" · Next ▶ · status label (ignored size policy) … primary **Start Training**.

## 6. Camera dialog (e.g. barcode scan)
*(scan_dialog.png)*

- **Title strip:** the title plus a right-aligned hint ("Align the barcode with the red line").
- **Camera view:** `make_framed(preview)`. Draw guide overlays (the red scan line) in the preview's `paintEvent`, with a dark outline pen under a 2 px colour pen so they show on any background.
- **Entry card:** a large read-only amber readout (`QLineEdit` with objectName `readout`), then a manual entry row: `QLineEdit`, then **Load** (blue, `make_primary_button(..., min_width=96)`), then **Save** (gray).
- **Bottom strip:** Close at the bottom right.
- **Camera release:** stop the camera worker in both `reject()` and `closeEvent()`. The camera is shared with the main screen, so refuse to open the dialog while the main stream runs, and say why in the status text.

## 7. Type-to-confirm delete
*(remove_job.png)*

```python
dlg = ConfirmByTypingDialog(
    "Remove Job", part_no,
    f'This permanently deletes the saved job "{part_no}". NG images are kept. This cannot be undone.',
    action_text="Remove", parent=self)
if dlg.exec() == QDialog.Accepted:
    ...delete...; refresh UI; status "Removed job 'X'."
```

- If the user may pick *which* item to delete, add a combo above the field. Clearing the typed text when the selection changes forces a fresh confirmation (see the reference app's `RemoveJobDialog`).
- Deleting the *active* item should clear any "auto-load last item on boot" marker as well.

## 8. Verdict hold latch + cycle counting

```python
def _latch(self, raw: str) -> str:
    now = time.monotonic()
    if self._judged is None or (raw != self._judged and (now - self._since) * 1000 >= self.hold_ms):
        self._judged, self._since = raw, now
        if raw in ("OK", "NG"):
            self._count(raw)            # one cycle per latched transition
    return self._judged
# per frame:  state = self._latch(compute_raw(dets)); set_judgement(pill, state); drive_outputs(state)
# on Start/Stop: self._judged = None   (first frame after Start applies immediately)
```

- Nothing is queued. When the hold time is up, whatever is current wins, so outputs follow reality rather than history.
- Hardware outputs (tower light, buzzer) should act only when the **latched** state changes.
- If an event needs the annotated frame (for example saving the NG image), set a flag in `_count` and act at the end of the frame handler, after drawing.

## 9. Start/Stop button driven by worker state

```python
worker.started.connect(self._refresh_run); worker.finished.connect(self._refresh_run)

def _refresh_run(self):
    running = self.worker.isRunning()
    if not running: self._stop_requested = False
    stopping = running and self._stop_requested
    if stopping:  text, icon, kind, chip = "Stopping…", "stop", "stop", ("Stopping…", "#f4d03f")
    elif running: text, icon, kind, chip = "Stop", "stop", "stop", ("Running", "#2ecc71")
    else:         text, icon, kind, chip = "Start", "play", "start", ("Stopped", "#9aa5b1")
    self.run_button.setEnabled(not stopping)
    self.run_button.setText(text); self.run_button.setIcon(make_icon(icon))
    set_primary_button_kind(self.run_button, kind); set_status_chip(self.chip, *chip)

def start(self): self.worker.start(); self._refresh_run()      # don't wait for the queued signal
def stop(self):
    if self.worker.isRunning(): self._stop_requested = True; self.worker.stop()   # non-blocking
    self._refresh_run()
# closeEvent: stop(); self.worker.wait()   ← the only place a full join is right
```

In the worker, set `self._running = True` in an overridden `start()`, not at the top of `run()`. Otherwise a Stop clicked while the camera or model is still loading gets overwritten.

## 10. Persisted per-machine settings

```python
DEFAULTS = {"buzzer_enabled": True, "state_hold_ms": 1000, "ok_count": 0, "ng_count": 0}
def load_settings():
    s = dict(DEFAULTS)
    try:
        data = json.loads(SETTINGS_FILE.read_text())
        for k, d in DEFAULTS.items():
            if k in data: s[k] = type(d)(data[k])      # tolerate old/partial files
    except (OSError, ValueError, TypeError): pass
    return s
def save_settings(values):                             # merge, don't clobber unknown keys
    try: data = json.loads(SETTINGS_FILE.read_text())
    except (OSError, ValueError): data = {}
    data.update(values); SETTINGS_FILE.write_text(json.dumps(data, indent=2))
```

- Keep machine-local runtime files (settings, counters, logs, captures) out of git.
- Default safety features **on**, such as the NG buzzer. A kiosk that autostarts must not come up silently disarmed.
