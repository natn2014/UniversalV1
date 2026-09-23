---
name: industrial-hmi-ui
description: Keyence-style industrial HMI design system for PySide6/Qt desktop and touch-panel apps — dark glossy theme, info strip + framed camera view + light side panel + bottom action bar, big OK/NG status pill, counter tiles, touch job tiles, pinch-zoom image popup, type-to-confirm delete dialog. Includes a drop-in theme module (industrial_theme.py). Use this skill whenever building or restyling a PySide6/PyQt/Qt Widgets UI for a factory, machine vision, inspection station, kiosk, HMI, operator panel, test bench, Jetson/Raspberry Pi touch screen, or anything that should "look like Keyence/Cognex/Omron", "look industrial/professional", or match an existing tab of such an app — even if the user just says "make the UI better", "same style as the Monitor tab", or "redesign this dialog".
---

# Industrial HMI UI (PySide6)

A proven design system for operator-facing machine UIs, extracted from a production inspection kiosk (Jetson Orin Nano, 1024×768 touch panel). Screens look like a Keyence vision-system panel:
- a near-black frame;
- glossy gray buttons;
- a blue gloss for the primary action and the selected tab;
- dark gradient header strips;
- an amber-outlined image well;
- a light-gray results panel.

The design suits people standing at a machine: they glance at it, tap it with a finger (sometimes gloved), and must never misread OK vs NG.

## Quick start

1. Copy `assets/industrial_theme.py` (next to this file) into the project, and import from it. It needs PySide6 only.
2. At startup, call `apply_theme(app)`.
3. Build screens from the helpers rather than raw widgets. The helpers set the object names the stylesheet keys on:

| Helper | What it makes |
|---|---|
| `make_strip("top" / "bottom")` | Dark gradient bar; returns `(frame, hbox)` |
| `make_info_label(text)` | Bold muted caption for strips |
| `make_framed(widget)` | Black well with amber outline, for camera or image views |
| `make_side_panel(width)` + `make_panel_header(title)` | Light-gray panel with a dark header |
| `style_panel_table(t)`, `style_panel_list(l)` | Table/list setup for that panel |
| `make_card(title)` | Dark section card with a header strip. Use it instead of `QGroupBox`. |
| `make_row(a, None, b)` | HBox; `None` is a stretch |
| `make_primary_button(text, "start" / "stop", icon)` | Blue gloss (main action) or red gloss (stop / destructive) |
| `make_judgement_pill()` + `set_judgement(pill, "OK" / "NG" / "STANDBY")` | Glossy status pill |
| `make_status_chip()` + `set_status_chip(chip, "Running", "#2ecc71")` | ● state text next to the primary button |
| `make_counter_tile(title, top, bottom, digits)` | Big 40pt counter tile |
| `make_tile_button(text, checked)` | Large checkable touch tile; the checked one is orange |
| `make_hint(text)`, `make_clear_box()`, `clear_layout(layout)` | Hint text, transparent container, safe layout reset |
| `ConfirmByTypingDialog(title, name, consequence)` | Delete confirmation that requires typing the name |
| `ImageOverlay(parent).show_image(path, title)` | Full-window popup: pinch-zoom, pan, fit, ✕ |
| `make_icon("play" / "stop" / "camera" / "speaker_on" / "speaker_off" / "close")` | Vector icons drawn in code, so no image files or emoji fonts are needed |

Run `python industrial_theme.py` for a live demo. Run it with `--screenshot out.png` plus `QT_QPA_PLATFORM=offscreen` to render one headless. Before designing, look at the reference screenshots in `assets/examples/`: `monitor_running.png`, `settings.png`, `job_change.png`, `scan_dialog.png`, `log_popup.png`, `remove_job.png`. They are the look to match.

## The screen skeleton

Almost every screen uses the same three bands:

```
┌ top strip ─────────────────────────────────────────────┐
│ context: what is loaded (part · model · dataset)   [OK] │  ← status pill top-right
├────────────────────────────────────────┬───────────────┤
│                                        │ ▌Panel header  │
│   main content (framed image,          │  light panel:  │
│   grid of tiles, or cards)             │  table / list  │
│                                        │  counters      │
├────────────────────────────────────────┴───────────────┤
│ secondary actions · hint            ● status  [PRIMARY] │  ← primary action bottom-right
└ bottom strip ──────────────────────────────────────────┘
```

Why this works:
- Operators learn one layout. Every tab and dialog then feels the same: context is always at the top, "do the thing" is always at the bottom right, and results are always on the right.
- The main action sits where a right-handed operator's thumb lands, as the reference's **Run** button does.

Use it for tabs **and** dialogs, such as a barcode-scan dialog: title strip, framed camera, entry card, and a Close bar. For concrete screen recipes (monitor, settings, job picker, gallery/log, training/labeling, scan dialog), read `references/patterns.md`.

## Design rules and the reasons behind them

- **Colour encodes the action or the verdict, never both at once.**
  - The primary button's colour says what a click *does*: blue = Start, red = Stop or Delete.
  - A separate status chip (● Running / Stopped / Stopping…) says what the machine *is doing*.
  - An earlier design put a status dot inside the Start button, and users found it confusing.
- **Verdict colours are reserved:**
  - OK = green `#2e7d32`, NG = red `#c0392b`, STANDBY = yellow `#f4d03f` with dark text.
  - Keep them identical on screen, on the tower light and in counters, and don't reuse green or red for decoration.
  - Orange (`GLOSS_ORANGE`) means "selected / active" (checked toggles, the active job tile), as the highlighted tool does in Keyence UIs.
- **Touch sizes:**
  - Toolbar buttons are 40–44 px tall, the primary button about 170×44, and tiles have a 72 px minimum height.
  - Spinboxes get 36 px wide ▲▼ buttons.
  - Tiny targets cause mis-taps on a production line.
- **Check the target resolution.**
  - Kiosk panels are often 1024×768. Check the real size with `xrandr`, and render screenshots at that size.
  - Five 160×120 thumbnails fit across 1024 px; 240 px ones cause horizontal scrolling.
  - Stop long status text from widening bars: `label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)`.
- **Light panel for data, dark for chrome.** Tables, lists and counts sit in the light-gray `sidePanel`, which reads like the reference's settings pane. Text inside it needs `objectName("panelText")` to be dark.
- **Destructive actions need friction.** A single tap should never delete a job, recipe or data. Use `ConfirmByTypingDialog`: the name must be typed exactly, case-sensitive, and choosing a different item clears what was typed.
- **Resetting counts or changing state needs a yes/no.** Use `QMessageBox.question`, and show the current values in the message.
- **Keep hints short and grey (`make_hint`).** Operators ignore paragraphs, so write one line saying what to tap.

## Behaviour patterns that make it feel solid

These come from real bugs in the reference app; read `references/patterns.md` for code.

- **Status hold / debounce.**
  - Latch the displayed verdict for a minimum time, default 1000 ms and configurable.
  - Frame-by-frame verdicts flicker, which makes tower lights and buzzers chatter.
  - Count cycles on latched *transitions*, not on frames.
- **Non-blocking Stop.**
  - Never `wait()` on a worker thread in a click handler.
  - Show "Stopping…" (button disabled, chip amber) and flip back when the thread's `finished` signal arrives.
  - Refresh the button immediately after `start()`, because the queued `started` signal lags.
- **Drive buttons from real state.** Connect the worker's `started` and `finished` signals to one `_refresh_*()` method, so the UI is right even when the worker dies on its own, for example because the camera was unplugged.
- **Rebuilding widget lists:** use `clear_layout()`, which hides widgets before `deleteLater()`, or stale tiles keep painting.
- **Persist per-machine settings** (buzzer, hold time, counters) in a small JSON file. Apply saved values *before* connecting signals, so restoring them doesn't fire side effects such as preview beeps or rewrites.

## Qt stylesheet pitfalls (learned the hard way)

- Styling `QSpinBox` with a border hides its arrows. The theme supplies styled up/down buttons, plus arrow SVGs written to `~/.cache/industrial_hmi_ui/`, because QSS can only draw sub-control arrows from image files.
- `background: transparent` on a container cascades into every child and wipes their gradients. Scope it with an object name (`clearBox`), never a bare widget stylesheet.
- In QSS, `QFrame#sidePanel QLabel` outranks `QLabel#panelHeader`. Give special labels their own object names, and don't write broad descendant rules.
- Per-widget `setStyleSheet()` beats the app stylesheet, so use it for state-dependent colours (pill, primary button) and keep the rest global.
- A plain `QWidget` ignores QSS backgrounds unless `setAttribute(Qt.WA_StyledBackground, True)` is set. `ImageOverlay` does this.
- Touch: accept `QEvent.Touch*` and set `WA_AcceptTouchEvents`. Once touch events are accepted, Qt stops synthesising mouse events, so handle taps (backdrop close, double-tap) in the touch path too, as `ZoomImageView` does.

## Workflow when applying this skill

1. Read the target screen's current code and list its widgets by role:
   - context: what is loaded;
   - main content;
   - data or results;
   - secondary actions;
   - the one primary action.
2. Map the roles onto the three-band skeleton. Merge scattered group boxes into cards or strips, and put context items on one line. For example, "Active Model" and "Dataset" share the top strip.
3. Build with the helpers, reusing the app's existing widgets. Re-parenting a widget is enough, so the signal wiring is kept.
4. Render headless screenshots at the real resolution (`QT_QPA_PLATFORM=offscreen`, `widget.grab().save(...)`) and look at them before reporting. The most common bugs only show up visually:
   - clipped text;
   - horizontal scrollbars;
   - stale widgets;
   - missing arrows.
5. If the app is a kiosk or service, point test runs at scratch copies of data directories and settings files. Stop the service before touching shared hardware (camera, USB tower light), and restart it afterwards.
