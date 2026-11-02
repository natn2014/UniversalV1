# Develop Record: image processing tools (branch `imageprocessing`)

The build log for [imageprocessing_todo.md](imageprocessing_todo.md).
Add one entry each time a TODO item (T01–T28) is finished or blocked. Newest
entries go at the bottom. Do not edit old entries; add a correction entry instead.

**Test command:** `.venv/bin/python -m unittest discover -s tests`
**Baseline before T01:** 86 tests, all passing (2026-10-27).

## Status board

| T | Function | Status | Commit |
|---|---|---|---|
| T01 | vision/base.py framework | done | b0a612c |
| T02 | preprocess / shading correction | done | e6873c4 |
| T03 | Position Adjustment | done | 51e92af |
| T04 | Total judgement | done | d0cf1b0 |
| T05 | Save tools in job | done | 05392cb |
| T06 | Color Area | done | cccfc85 |
| T07 | Color Average (ΔE2000) | done | 93041a9 |
| T08 | Color Prohibit | done | 18b1d32 |
| T09 | Area (mono) | done | c74942d |
| T10 | Brightness Average | done | 738afe1 |
| T11 | Edge Pixels | done | 738afe1 |
| T12 | Shading Uniformity | done | ab0dc26 |
| T13 | Edge-profile core | done | 08fa916 |
| T14 | Width | done | b903d42 |
| T15 | Edge Presence | done | 78c14b9 |
| T16 | Pitch | done | c5d9ccb |
| T17 | Diameter | done | 61cd556 |
| T18 | mm calibration | done | 0957bc5 |
| T19 | OCR read | done | f64a814 |
| T20 | OCR verify | done | faedec9 |
| T21 | Code tool (QR / 1D) | done | 23ffed7 |
| T22 | Outline | done | 8b8d1a4 |
| T23 | Blob Count | done | 0f710f6 |
| T24 | Worker integration | done | 2b76039 |
| T25 | Monitor integration | done | e09d2f0 |
| T26 | Tools tab UI | done | 7d9832d |
| T27 | Performance + final check | done | afe7ab9 |
| T28 | Slow-tool thread (OCR / code at 2 Hz) | done | |

Status values: `todo` · `in-progress` · `done` · `blocked` (give the reason in the entry).

## Entry template

```
### Txx <name>: <done|blocked>  (YYYY-MM-DD)
- Files: app/vision/..., tests/test_vision_....py
- Tests added: N  ·  Suite: <passed>/<total> passing
- Timing: <ms per run on a 1280x720 frame, if relevant>
- Decisions / deviations from the TODO: ...
- Blocked by (if blocked): ...
- Commit: <short hash>
```

## Entries

### T01 vision/base.py framework: done  (2026-10-27)
- Files: app/vision/__init__.py, app/vision/base.py, tests/test_vision_base.py
- Tests added: 14  ·  Suite: 100/100 passing
- Decisions / deviations from the TODO:
  - `Tool.run` is a concrete template (crop, empty-ROI guard, ROI rect overlay, judging) that calls an abstract `process(patch, origin)`. Subclasses return overlay in patch coordinates; `run` shifts it to frame pixels. Position (T03) overrides `run`.
  - Subclasses declare `PARAMS = {key: default}` and `DEFAULT_LOW/DEFAULT_HIGH`; `params()`, `set_params()` and `to_dict()` are generic. `low=None` means no bound, an omitted `low` means the class default.
  - A taught image lives in `tool.template` (ndarray) with `template_file` (path) in the dict; `vision.store` (T05) reads and writes the PNG.
  - Runtime-only attributes `mm_per_px` and `part_no` are set by the runner and never saved.
  - Overlay `pts` layout is documented in the `base.py` docstring.
- Commit: b0a612c

### T02 preprocess / shading correction: done  (2026-10-27)
- Files: app/vision/preprocess.py, app/vision/base.py (adds the `pre` param), tests/test_vision_preprocess.py
- Tests added: 9  ·  Suite: 109/109 passing
- Decisions / deviations from the TODO:
  - Gray-only filters (flat_field, top_hat, black_hat, clahe) accept BGR by filtering each channel, so `apply_chain` works on both.
  - `to_hsv` / `to_lab` cannot detect "already in that space" for a 3-channel array; they assume BGR and promote a 2-D gray input.
  - Extra ops "gray", "hsv", "lab" are available inside `apply_chain`.
  - `Tool` now has a `pre` param on every tool (`DEFAULT_PRE`, overridable per class, e.g. OCR) plus `Tool.prepared()` / `Tool.gray()` helpers.
  - Test geometry: flat_field divides by a blur, so a feature close to the kernel size is partly absorbed into the background (a 40 px square with kernel 51 keeps only ~5 levels). The test uses a 20 px square; keep the kernel at least 2.5x the feature. Background std is measured away from the left/right 60 px border, where the ramp starts at 0.
- Commit: e6873c4

### T03 Position Adjustment: done  (2026-10-27)
- Files: app/vision/position.py, app/vision/__init__.py, tests/test_vision_position.py
- Tests added: 13  ·  Suite: 122/122 passing
- Timing: 2-D match, 1280x720 frame, ~400x300 px search area, angle_range=0: about 3 ms (best of 20, Orin Nano); target < 5 ms.
- Decisions / deviations from the TODO:
  - The result carries the offset in `extra` (dx, dy, angle, score, found); `offset_from_result(result)` rebuilds an `Offset`, and `tool.offset()` returns the last one. The T24 runner uses `offset_from_result`.
  - `teach(frame, template_roi=...)`: `template_roi` is in frame fractions; when omitted the central half of the search area is used.
  - The 1-axis modes use a private `_strongest_edge` (Gaussian-smoothed profile gradient, parabola sub-pixel) because T13's `find_edges` does not exist yet. Revisit at T13. The 1-axis score is edge strength / taught strength, capped at 1.
  - The half-scale pyramid is used only when the template is at least 32 px per side; smaller templates are matched at full scale.
  - Angle convention: positive = counter-clockwise on screen, the same as `cv2.getRotationMatrix2D` and `rotate_points`.
  - The speed test allows 15 ms so a busy box does not make it flaky.
- Commit: 51e92af

### T04 Total judgement: done  (2026-10-27)
- Files: app/vision/total.py, app/vision/__init__.py, tests/test_vision_total.py
- Tests added: 7  ·  Suite: 129/129 passing
- Decisions / deviations from the TODO:
  - "Ignores disabled tools" is met by the runner (T24) not returning results for disabled tools; `total_state` itself has no notion of enabled.
  - Results of kind `position` are left out of the AND/OR; they feed `position_found` instead (a lost part is YELLOW, not RED).
- Commit: d0cf1b0

### T05 Save tools in job: done  (2026-10-27)
- Files: app/vision/store.py, app/context.py, app/paths.py (TOOLS_DIR), app/tabs/settings_tab.py, tests/test_vision_store.py
- Tests added: 9  ·  Suite: 138/138 passing
- Decisions / deviations from the TODO:
  - `store` functions take `tools_dir` as an argument instead of importing `app.paths`, because `app.paths` imports QtCore and `app/vision` must stay Qt-free. `AppContext` passes `paths.TOOLS_DIR` (`part_configs/tools/`).
  - Template PNG path is stored relative to `tools_dir` (`<part>/<tool>.png`) in `template_file`. PNGs no longer used by any tool are deleted on save.
  - The job is loaded and saved in `settings_tab.collect_config` / `apply_config` (that is where `set_masters` is called from, not `job_change_tab`). Without the `collect_config` change a Settings-tab "save job" would have dropped the tools.
  - `AppContext` gained `tools`, `tools_gate` (saved as `"tools_gate"`, default AND), `tools_changed`, `set_tools`, `tool_entries`, `load_tools(config)`. `set_tools` writes the dict list directly instead of going through `_save_to_job`.
  - A missing template PNG keeps the tool (untaught, so it reports NG "not taught") instead of dropping it; unknown kinds and malformed entries are logged and skipped.
  - `AppContext` is built in tests with `__new__` to avoid the camera, tower and settings file.
- Commit: 05392cb

### T06 Color Area: done  (2026-10-27)
- Files: app/vision/color.py, app/vision/__init__.py, tests/test_vision_color.py
- Tests added: 11  ·  Suite: 149/149 passing
- Decisions / deviations from the TODO:
  - Shared helpers in color.py for T08 / T23: `hsv_mask`, `learn_hsv_range`, `HSV_PARAMS`, and the `HsvTool` base (teach + mask).
  - "Minimum widths" (h 8, s 40, v 40) are read as minimum half-widths around the mean. A hue half-width of 90 or more means the whole hue circle.
  - `open_px` is the opening kernel size in px (values below 2 do nothing).
  - `teach(frame, point=(x, y) | teach_roi=(fractions))`; default samples the tool's own ROI.
  - Brightness test uses a mid-brightness color (V about 120): the minimum V half-width of 40 would not cover +20% on V=220.
- Commit: cccfc85

### T07 Color Average: done  (2026-10-27)
- Files: app/vision/color.py, tests/test_vision_color.py
- Tests added: 5  ·  Suite: 154/154 passing
- Decisions / deviations from the TODO:
  - `delta_e_2000` is pure Python `math` on one pair (not vectorised): it is called once per frame on the mean color. Checked against 5 Sharma 2005 pairs (2 more than asked), in both argument orders, within 0.01.
  - `result.text` shows the measured L*a*b*; `extra["lab"]` holds the floats.
- Commit: 93041a9

### T08 Color Prohibit: done  (2026-10-27)
- Files: app/vision/color.py, tests/test_vision_color.py
- Tests added: 5  ·  Suite: 159/159 passing
- Decisions / deviations from the TODO:
  - The opening in `hsv_mask` erodes with a constant 0 border. OpenCV's default (+inf) border let single pixels on the ROI edge survive `MORPH_OPEN`. This also applies to Color Area when `open_px` is set.
  - Default `high = min_blob_px - 1` is set in `__init__` unless `high` is passed (None still means no bound).
  - `extra["blobs"]` counts the blobs at or above `min_blob_px`.
- Commit: 18b1d32

### T09 Area (mono): done  (2026-10-27)
- Files: app/vision/brightness.py, app/vision/__init__.py, tests/test_vision_brightness.py
- Tests added: 5  ·  Suite: 164/164 passing
- Decisions / deviations from the TODO:
  - `binarize(gray, method, threshold, polarity)` is a module function so Blob Count (T23) can reuse it. For Otsu the dark side includes the threshold itself (`<=`), the bright side excludes it (`>`), so the two always add up to 100%.
- Commit: c74942d

### T10 Brightness Average: done  (2026-10-27)
- Files: app/vision/brightness.py, tests/test_vision_brightness.py
- Tests added: 3  ·  Suite: 170/170 passing (shared with T11, see below)
- Decisions / deviations from the TODO: none. `extra` holds both `mean` and `std`.
- Commit: 738afe1 (same commit as T11)

### T11 Edge Pixels: done  (2026-10-27)
- Files: app/vision/brightness.py, tests/test_vision_brightness.py
- Tests added: 3  ·  Suite: 170/170 passing
- Decisions / deviations from the TODO:
  - T10 and T11 were written together and went into one commit (the process rule is one commit per item; this is the one exception, caught too late to split cleanly).
  - The scratch test accepts +100..+320 edge pixels for a 140 px line, because a 1 px line has an edge on each side.
- Commit: 738afe1

### T12 Shading Uniformity: done  (2026-10-27)
- Files: app/vision/brightness.py, tests/test_vision_brightness.py
- Tests added: 4  ·  Suite: 174/174 passing
- Decisions / deviations from the TODO: none. The darkest and brightest cells are `mask_box` overlays labelled `min N` / `max N`; `extra` has `cells`, `argmin`, `argmax` as [row, col]. The ramp with `flat_field` (kernel 51) drops below 10.
- Commit: ab0dc26

### T13 Edge-profile core: done  (2026-10-27)
- Files: app/vision/edges.py, app/vision/position.py (1-axis modes now use `find_edges`), tests/test_vision_edges.py
- Tests added: 9  ·  Suite: 183/183 passing
- Decisions / deviations from the TODO:
  - Positions are in pixel-edge coordinates (a sharp step between pixel 39 and 40 is at 40.0), so the spec's "step at x = 40.0 gives 40 +- 0.2" holds. Widths and pitches are differences, so they are unaffected.
  - `find_edges` returns `(position, signed strength)`: positive for rising, negative for falling (Width needs the sign). `threshold` applies to the magnitude. It has an extra `smooth=1.0` argument.
  - A central-difference gradient merges two steps closer than about 3 px into one plateau, so the "2 px apart" case is merged by the gradient itself; the min_distance behaviour is tested with steps 4 px apart (kept at min_distance 3, merged at 6).
  - Replaced T03's private edge finder in position.py with `find_edges` (the T03 note is now resolved).
- Commit: 08fa916

### T14 Width: done  (2026-10-27)
- Files: app/vision/measure.py, app/vision/__init__.py, tests/test_vision_measure.py
- Tests added: 8  ·  Suite: 191/191 passing
- Decisions / deviations from the TODO:
  - Shared `ScanTool` base (scan, mm scaling via runtime `mm_per_px`, edge line overlay) for Width / Edge Presence / Pitch. The value is in mm only when the runner has set `mm_per_px`.
  - Pairing: `outer` = first edge matching `polarity_1` and last matching `polarity_2`; `strongest` = the two strongest edges by position, which must match both polarities; `first_rising_falling` as in the spec.
  - Overlay is two edge lines plus a dimension line labelled with the value (a `line` item).
- Commit: b903d42

### T15 Edge Presence: done  (2026-10-27)
- Files: app/vision/measure.py, tests/test_vision_measure.py
- Tests added: 4  ·  Suite: 195/195 passing
- Decisions / deviations from the TODO: none. The default `low` is 20 (gray levels per px of smoothed gradient), matching Width's default threshold; there is no separate `threshold` param, `low` is the threshold.
- Commit: 78c14b9

### T16 Pitch: done  (2026-10-27)
- Files: app/vision/measure.py, tests/test_vision_measure.py
- Tests added: 5  ·  Suite: 200/200 passing
- Decisions / deviations from the TODO:
  - `extra` has `count`, `mean`, `min`, `max` (the last three only with 2 or more edges, scaled to mm when calibrated) and `positions`. `count` is never scaled. A mean/min/max measure with fewer than 2 edges returns ok=False "not enough edges"; count with no edges is value 0.
- Commit: c5d9ccb

### T17 Diameter: done  (2026-10-27)
- Files: app/vision/measure.py, app/vision/base.py (`scale` / `unit` moved up to `Tool`), tests/test_vision_measure.py
- Tests added: 4  ·  Suite: 204/204 passing
- Decisions / deviations from the TODO:
  - The refinement step drops points more than 2 px from the *median* distance to the first fit's center, not from the first fit's radius: a Kasa fit pulled off by outliers has a bad radius, the median is stable. Still one refinement pass.
  - "10% outliers" is tested as isolated bright specks that add stray edge points (about 10% of the circle's edge points), not 10% salt-and-pepper pixels, which would destroy any edge-based circle fit.
  - Edge image is Gaussian-blurred (3x3) before Canny, as in Edge Pixels.
- Commit: 61cd556

### T18 mm calibration: done  (2026-10-27)
- Files: app/vision/calibration.py, app/vision/store.py (`set_runtime`), app/context.py, app/tabs/settings_tab.py, tests/test_vision_calibration.py, tests/test_vision_store.py
- Tests added: 6  ·  Suite: 210/210 passing
- Decisions / deviations from the TODO:
  - `AppContext.mm_per_px` (+ `mm_per_px_changed`, `set_mm_per_px(value, save_to_job)`) is saved as `"mm_per_px"` next to `"tools"`; `collect_config`, `set_tools` and `load_tools` all carry it.
  - `store.set_runtime(tools, mm_per_px, part_no)` hands the job-level values to the tools before each run (T24 calls it); they are never in a tool's dict.
  - `calibrate_from_width` returns None when no width is found or `known_mm <= 0`.
- Commit: 0957bc5

### T19 OCR read: done  (2026-10-27)
- Files: app/vision/ocr.py, app/vision/__init__.py, app/optional_deps.py (`TESSERACT` guard), requirements.txt (`pytesseract` + system `tesseract-ocr` note), tests/test_vision_ocr.py
- Tests added: 14 (3 need tesseract)  ·  Suite: 224/224 passing, tesseract tests run (not skipped)
- Environment: the coordinator lifted the no-install rule; tesseract 5.3.4 (system) and `pytesseract` (in .venv) were installed by them. Nothing was installed by this item.
- Decisions / deviations from the TODO:
  - Deskew uses the principal axis (PCA) of the dark pixels instead of `minAreaRect`: the `minAreaRect` angle convention differs between OpenCV versions (this venv has 5.0). It is applied only when the pixels are elongated (eigenvalue ratio at least 2) and the angle is under 15 degrees, so a tall multi-line block is left alone.
  - Auto character height = median height of the dark blobs; the upscale is capped at 4x and never shrinks.
  - `OcrTool.engine` (class attribute) can be set to a fake or another `OcrEngine`; `get_engine()` wraps `TESSERACT` otherwise. Missing engine gives ok=False "OCR engine not installed".
  - Confidence = mean of Tesseract word confidences; multi-line text is joined with newlines.
  - Importing `app.optional_deps` from `vision.ocr` is lazy (inside `get_engine`) so the module stays light.
  - Tesseract costs roughly 50-200 ms per read, far over the 10 ms budget; the runner (T24) should not OCR every frame. See the T27 notes.
- Commit: f64a814

### T20 OCR verify: done  (2026-10-27)
- Files: app/vision/ocr.py, tests/test_vision_ocr.py
- Tests added: 10 (2 need tesseract)  ·  Suite: 234/234 passing, tesseract tests run
- Decisions / deviations from the TODO:
  - `verify(text, conf, expected, pattern, char_count, low, date_format, now) -> (ok, reason)` is a plain function. `{date}` is today's date in `date_format`; inside `pattern` it is regex-escaped. `expected` is an exact match after stripping the ends of the read text.
  - Orientation check runs only when `expected` or `pattern` is set (without one there is nothing to "match"). It flags NG "upside down" when the 180-degree read passes the checks with higher confidence and the normal read does not.
  - A failed check keeps the read text in `text` and puts the reason in `extra["reason"]`; only the orientation case replaces `text` with "upside down", as the spec says.
- Commit: faedec9

### T21 Code tool (QR / 1D): done  (2026-10-27)
- Files: app/vision/code.py, app/vision/__init__.py, tests/test_vision_code.py
- Tests added: 7  ·  Suite: 241/241 passing
- Decisions / deviations from the TODO:
  - Order: pyzbar, then the cv2 QR detector (only when the symbology allows QR), then both again on a 2x upscale. Polygon points are scaled back to patch pixels.
  - `symbology` filters by a case-insensitive substring of the pyzbar type (`qrcode`, `code39`, `code128`, `ean13`, ...); `any` accepts all.
  - Value is 1 when read; default `low=1` so an unread code is NG. A read code with the wrong text keeps value 1 but ok=False, with the reason in `extra["reason"]`.
  - `use_job_part_no` compares with `tool.part_no` (set by `store.set_runtime`); if there is none, the tool is NG with reason "no job part number".
  - `optional_deps.zbar_decode` is looked up at call time so a test (or a machine without pyzbar) can patch it.
- Commit: 23ffed7

### T22 Outline: done  (2026-10-27)
- Files: app/vision/pattern.py, app/vision/__init__.py, tests/test_vision_pattern.py
- Tests added: 10  ·  Suite: 251/251 passing
- Decisions / deviations from the TODO:
  - The taught master is the gray ROI crop (after `pre`), saved as a viewable PNG; the edge template and contour are derived from it and cached.
  - Template mode: the template is the master's edges inset by `margin_px` (new param, default 8), so the ROI is slightly larger than the template and a small shift is tolerated. Edges of the master and of the live ROI are both dilated 3x3. With only the template dilated, `TM_CCORR_NORMED` tops out near 0.58 for an identical part.
  - Shape mode: the TODO formula `100 - matchShapes x 100` cannot separate a circle from a square (Hu I1 distance is only about 0.03, so it scores 97). Used `100 * exp(-distance * 25)` instead: circle vs square 45, rotated rectangle 99, notched rectangle 83.
  - The largest contour comes from an Otsu mask whose minority class is taken as the part.
  - If the ROI size changes after teaching, the tool reports "ROI size changed: teach again".
- Commit: 8b8d1a4

### T23 Blob Count: done  (2026-10-27)
- Files: app/vision/count.py, app/vision/color.py (`open_mask` split out of `hsv_mask`), app/vision/__init__.py, tests/test_vision_count.py
- Tests added: 8  ·  Suite: 259/259 passing
- Decisions / deviations from the TODO:
  - Circularity is computed per kept blob from `findContours` on its own mask, with the contour smoothed by `approxPolyDP(1 px)` before taking the perimeter: the raw pixel staircase made a digital circle score only 0.77-0.87 and `circ_min=0.8` dropped real circles. Now circles score 0.87-0.98, a square 0.785, a bar 0.2. Circularity is only computed when `circ_min > 0`.
  - `BlobCountTool` derives from `HsvTool`, so the color source can be taught like Color Area (`k` param added).
  - The open (default `open_px=1`, which does nothing) uses the 0-border erosion shared with the color tools.
- Commit: 0f710f6

### T24 Worker integration: done  (2026-10-27)
- Files: app/vision/runner.py, app/camera.py, app/context.py, app/tabs/monitor_tab.py, tests/test_reconnect.py, tests/test_vision_runner.py, tests/test_vision_store.py, tests/test_vision_calibration.py
- Tests added: 10  ·  Suite: 269/269 passing
- Decisions / deviations from the TODO:
  - `frame_ready` is now `Signal(QImage, list, list)`. Connections found by grep: `app/tabs/monitor_tab.py` (`on_frame` now takes `tool_results=None`, ignored until T25), `tests/test_reconnect.py` (lambda takes 3 args). `app/widgets/barcode_scan.py` connects to its own `BarcodeWorker.frame_ready` (a different class and signal), so it needed no change. `tools/record_demo_gifs.py` uses `lambda *a` and is unaffected.
  - `run_tools(frame, tools, mm_per_px, part_no)`: first position tool gives the offset; if it loses the part, only the position results are returned (other ROIs would be in the wrong place; the total is YELLOW anyway). Every result gets `extra["ms"]`; an exception becomes ok=False with the exception text. It calls `store.set_runtime`, so mm_per_px and part_no reach the tools.
  - Also added `VideoWorker.set_part_no` (for the Code tool's `use_job_part_no`); `AppContext.set_current_part` pushes it. `set_tools` and `set_mm_per_px` push to the worker.
  - With an empty tools list the worker does nothing extra (no `run_tools` call).
  - Context tests build `AppContext` with `__new__` plus a real, never-started `VideoWorker`.
- Commit: 2b76039

### T25 Monitor integration: done  (2026-10-27)
- Files: app/tabs/monitor_tab.py, app/widgets/tool_overlay.py, app/theme.py, tests/test_vision_monitor.py, tests/test_reconnect.py
- Tests added: 11  ·  Suite: 280/280 passing
- Decisions / deviations from the TODO:
  - `app/theme.py` had no verdict color constants (the banner hard-coded them), so `OK_GREEN`, `NG_RED`, `STANDBY_YELLOW` were added and the banner now uses them (same values, no visual change). Overlays use lighter tints of the first two so they show on a camera image.
  - `on_frame(image, detections, tool_results=None)`: with no tool results (no tools, or all disabled) the raw state is `filters.judge(detections)` exactly as before. With results: `total_state(results, yolo_state, ctx.tools_gate, position_found)`. YOLO counts as an input only when a model path is set; with no model the tools alone decide. A YOLO "no detection" (YELLOW) still makes the total STANDBY when a model is loaded, as the TODO specifies.
  - Overlay drawing lives in `app/widgets/tool_overlay.py` (`draw_overlay`, `tool_value_text`). `_render_frame` has a third optional argument, so the NG log image also carries the overlays.
  - The tool table (Tool / Value / OK-NG cell with verdict background) sits under the detection table, hidden when the job has no tools. The "pill" is a colored, bold table cell rather than a separate widget (cheaper at 30 fps). The mm unit is added only for `width` and `diameter`; OCR and Code rows show their text.
  - `tests/test_reconnect.py` now creates a `QApplication` (offscreen) instead of `QCoreApplication`, because the Monitor smoke test needs widgets in the same process. The monitor test uses a real `AppContext` with the settings file and `save_ng_image` patched to a temp dir; the tower and camera are never opened.
- Commit: e09d2f0

### T26 Tools tab UI: done  (2026-10-27)
- Files: app/tabs/tools_tab.py, app/widgets/tool_params.py (Add Tool tiles, parameter form), app/widgets/video_view.py (hint text, point mode), app/widgets/remove_job_dialog.py (title / item label / warning options), app/theme.py (QDoubleSpinBox styled like QSpinBox), app/main_window.py (tab between Settings and Job Change), tests/test_vision_tools_tab.py; unused imports cleaned in app/vision/ocr.py
- Tests added: 23  ·  Suite: 303/303 passing
- Checked visually: rendered offscreen at 1024x700 with `DARK_INDUSTRIAL_QSS` and looked at the screenshot (tool list, buttons, limits, Add Tool tiles). Only `QT_QPA_PLATFORM=offscreen` was used; the kiosk, camera and tower were not touched. Tests write only to a temp `PART_CONFIGS_DIR` / `TOOLS_DIR` / settings file.
- Decisions / deviations from the TODO:
  - The frozen frame comes from a Snapshot button (the newest `frame_ready` image; falls back to `worker.last_raw_frame` through the top-view warp). The tab shows all tool ROIs (selected one orange, thick), the live result overlays, and a total OK / NG / STANDBY pill on the frozen frame.
  - Outline does not ask for a second rectangle: its master is the ROI crop (with the margin inset), so the ROI itself is the template box. Position (2-D mode) does ask for the template box. Color Area, Color Average, Outline and Position learn from their ROI right after it is drawn when a frame is frozen; Color Prohibit and Blob Count do not (their ROI is not the color to learn).
  - Pick Color (shown for Color Area / Color Prohibit / Blob Count) uses a new point mode in `VideoView` (`point_picked`, fractions).
  - Delete reuses `RemoveJobDialog` (type the tool name), now parameterised with `title`, `item_label` and `warning`; the Job Change use is unchanged.
  - Structural changes (add, delete, teach, ROI, gate, limit button) save at once; spin box / combo / check box edits re-run the tools at once and save after 300 ms (debounce) so typing does not write the job file on every keystroke. The `Tools` tab also has Enabled / Follow position toggles.
  - Calibrate mm uses `QInputDialog.getDouble` for the known size (no numeric keypad widget exists in the app yet), then a drag across the feature; the direction follows the ROI's aspect.
  - Parameter form: bool -> check box, int -> spin box, float -> double spin box, known choices -> combo box, other str -> line edit; `pre` and taught values (template_roi, taught_center, ref_*) are hidden.
  - `QDoubleSpinBox` was not covered by the theme QSS; it now shares the `QSpinBox` rules.
- Commit: 7d9832d

### T27 Performance + final check: done  (2026-10-27)
- Files: tools/bench_vision.py, core_function.md (section 4 status), tests/test_vision_code.py, app/vision/code.py
- Tests added: 1  ·  Suite: 304/304 passing (baseline was 86; +218 tests, none skipped on this machine)
- Timing (`.venv/bin/python tools/bench_vision.py`, Orin Nano, 1280x720 frame, 300x200 px ROI, 200 runs per tool, OCR 20 runs; the position tool gets the same area as its search area):

| tool | mean ms | p95 ms |
|---|---|---|
| area | 0.11 | 0.13 |
| blob_count | 0.46 | 0.48 |
| brightness | 0.09 | 0.09 |
| code (no code in view) | 16.78 | 16.82 |
| color_area | 0.36 | 0.38 |
| color_average | 0.46 | 0.49 |
| color_prohibit | 0.82 | 0.85 |
| diameter | 1.27 | 1.31 |
| edge_pixels | 0.39 | 0.41 |
| edge_presence | 0.25 | 0.28 |
| ocr (Tesseract) | 183.41 | 189.75 |
| outline (template mode) | 3.28 | 4.46 |
| pitch | 0.30 | 0.32 |
| position (2-D, no angle) | 1.94 | 2.01 |
| shading | 0.19 | 0.20 |
| width | 0.35 | 0.38 |
| **total, one of each kind, no OCR** | **27.05** | **28.61** |
| total, no OCR and no Code | 10.27 | 10.81 |
| total, with OCR | 210.46 | 218.36 |

- Against the < 10 ms target: the 14 tools other than Code and OCR add up to 10.3 ms (just over; Outline 3.3 ms and Position 1.9 ms are the largest, and a job rarely has one of every kind). The two slow tools are the decoders:
  - Code: pyzbar at 1x and 2x costs about 15 ms on a miss. The OpenCV QR detector (16 ms) is now used only when `symbology` is set to QR, instead of on every "any" miss (which had cost 49 ms in the first run).
  - OCR: about 185 ms per read (Tesseract), plus the three preprocessing steps.
  - Suggested fixes (not done, per the TODO): run Code / OCR on a trigger or every Nth frame, or only while the Position tool reports a part; a smaller ROI or the half-scale pyramid for Outline; skip the 2x retry for Code on large ROIs.
- Decisions / deviations from the TODO:
  - `core_function.md` section 4 now has a Status column with what is built and what is still unmeasured on real parts (the 100-frame repeatability and 200-frame OCR criteria need real samples).
  - Not built from the IV4 list: AI Differentiate, AI Identify, AI Count, AI Trigger, Sorting and AI Through Count modes.
- Commit: afe7ab9

### T28 Slow-tool thread (OCR / code at 2 Hz): done  (2026-10-27)
- Files: app/vision/base.py, ocr.py, code.py (`SLOW` flag), app/vision/runner.py (`run_tools_with_offset`, fast/slow split), app/vision/slow_runner.py (new), app/vision/total.py (pending gives STANDBY), app/camera.py (`_run_vision`, part epoch, `set_slow_hz`), app/settings.py + app/context.py + app/tabs/settings_tab.py (`slow_tool_hz` = 2.0, spin box), app/widgets/tool_overlay.py + app/tabs/monitor_tab.py ("…" pending pill, result age), app/tabs/tools_tab.py (hint to add a Position tool), tools/bench_vision.py (worker-loop case), tests/test_vision_slow_runner.py, tests/test_vision_total.py
- Tests added: 11  ·  Suite: 315/315 passing
- Timing (`tools/bench_vision.py`, worker loop, every tool, 5 s at 24 fps): loop fps **24.0**, slow reads **2.0 /s**, OCR no longer adds anything to the camera thread.
  - The fast tools take about 22 ms per frame in this paced loop and 11 ms when run back to back, with or without the slow thread (checked: continuous OCR or Code alongside changes them by less than 0.5 ms). The difference is the `schedutil` CPU governor clocking down between frames. On the station, `sudo jetson_clocks` (or the performance governor) should bring it back to about 11 ms. This was not changed here; it is a device setting.
- How it works:
  - The camera loop runs Position and the fast tools on every frame. When the part is found and the slow thread is idle and due (1 / `slow_tool_hz`), it hands a copy of the frame and the offset to `SlowToolRunner`.
  - The mailbox holds one frame (a newer request replaces one not yet started), so the slow thread never falls behind.
  - Each frame joins in the latest slow results, tagged `slow` and `age_ms`.
  - `part_epoch` goes up when Position goes from lost to found and when the tools change. A slow result only counts within its own epoch and while younger than 3 / hz + 2 × the last run time. Otherwise the tool reports `pending`, and the total judgement is STANDBY. The previous part's OCR can never pass a new part. Without a Position tool only the age limit applies; the Tools tab hints at this.
- Decisions / deviations from the plan: none worth noting. `Tool.SLOW` is a class attribute, so tool kinds are not hard-coded in the runner. The Tools tab still runs every tool on its frozen snapshot synchronously (one frame, on a button press).
- Commit: 9397828

### Add Tool thumbnails
- What changed: each Add Tool tile shows an example picture of the case the tool suits (wire-harness scenes with the area in green and an OK / NG tag, as in the IV4 catalogue), the tool name, and a one-line "use for" text (`KIND_USES`). Drag scrolls the dialog on the touch screen (QScroller).
- Pictures: drawn by `tools/make_tool_thumbs.py` into `app/assets/tool_thumbs/<kind>.png` (240 × 150). They are not photographs, so rerun the script after changing a scene.
- Test: `test_every_tile_shows_an_example_and_its_use`.

### Touch delete for tools
- What changed: Delete on the Tools tab opens `DeleteToolDialog` (the tool's picture, name and kind, a big Cancel on the left and a big red Delete on the right, 64 px tall) instead of the type-the-name dialog. Delete is never the default button, so Enter or a stray tap does nothing. The typed confirm stays for removing a whole job.
- Tests: `test_delete_is_a_touch_confirm`, `test_delete_dialog_needs_no_typing_and_is_not_default`.

### Counted area shown as a yellow region
- What changed: Area (mono) and Color Area now show the pixels they count as a see-through yellow fill with a darker 2 px amber edge, instead of a dashed box around them. Holes stay clear.
- How: new overlay type `region` (`region_item` in `app/vision/base.py`): `pts` is the mask's top-left in frame pixels and `item["mask"]` is the patch-sized mask itself. `draw_region` in `app/widgets/tool_overlay.py` turns it into one premultiplied ARGB image (threshold → erode → LUT, all in OpenCV) and draws it in a single call.
- Why a mask, not outlines: outlines of a speckled mask need tens of thousands of Python points (about 60 ms per frame). The mask costs the same however speckled it is: the tool takes about 1 ms and the drawing about 5 ms for a 1024 × 576 area. Overlays are never saved, so a numpy array in the item is fine.
- Tests: `test_gray_input_and_counted_region`, `test_region_follows_the_roi`, `test_region_is_see_through_yellow`, plus the region cases in `test_all_overlay_types_draw_without_error`.

### Region fill for Edge Pixels, Blob Count, Color Prohibit
- Edge Pixels: the counted Canny edge pixels are drawn as solid bright yellow lines (`color="lines"`: a one-pixel edge map has no inside to fill).
- Blob Count: only the blobs that pass the area / aspect / circularity filters are filled yellow, so a merged or split blob is visible; each has its number (a `text` item) in place of a box. Rejected blobs stay unfilled.
- Color Prohibit: the forbidden-color blobs at least `min_blob_px` large are filled see-through red (`color="red"`); the red boxes and `Npx` labels stay so a tiny spot is still easy to find.
- `region_item(..., color=)` with palettes yellow / red / lines in `tool_overlay.REGION_PALETTES`. Text overlay labels get a dark one-pixel shadow so they read on a fill.
- Bench (ms, unchanged): blob_count 0.73, color_prohibit 1.09, edge_pixels 0.39.
- Tests: `EdgePixelsOverlay` (2), `test_only_counted_blobs_are_filled`, `test_nothing_counted_has_no_region`, the red region in `test_spot_is_ng`, `test_region_palettes`.

### Eyedropper: Color / Brightness Extraction (Keyence style)
- Study: on the IV, the Color Area tool's [Color Extraction] screen lets you tap the reference color on the image, tap more spots to add them, widen or narrow the range with [+] / [-], [Clear] and start again, and check on the live image; Advanced Color Extraction gives exact HSB numbers. On mono models, [Brightness Extraction] does the same with a gray level.
- What changed:
  - Tool API (`app/vision/base.py`): `PICK` / `pick_mode` ("color", "gray" or None), `pick(frame, point, add)` and `widen(steps)`.
  - Color (Color Area, Color Prohibit, Blob Count with source color): `merge_hsv_range` (the shorter hue arc covering both ranges and the union of the S/V spans) and `widen_hsv_range` (±3 hue, ±10 S/V per step, never below the minimum widths; a full hue circle stays full).
  - Gray (Area (mono), Blob Count with source gray): `pick_threshold` sets polarity (bright when the tap is in the upper half of the ROI's 5–95 % range) and a fixed threshold halfway between the tapped gray and the median of the ROI pixels on the other side. More taps only lower (bright) or raise (dark) the threshold, so every tapped spot counts; ±5 per step.
  - Tools tab: Pick Color / Pick Gray opens a session with an Extraction card (− Narrower, + Wider, Clear, Done). Each tap adds to the range and is saved, and the yellow/red fill shows the result straight away. The session survives its own saves and ends on Done, on selecting another tool, or on Resume Live. Clear goes back to the range from before the first tap.
- Not done: showing the hue in degrees (0–359°) like Keyence; the form still uses OpenCV's 0–179.
- Tests: `EyedropperRange` (5), `GrayEyedropper` (3), and Tools tab tests for the multi-tap session, the gray pick, ending the session on a tool change, and the button label.

### Pattern Search (ShapeTrax-style outline pattern tool)
- Keyence's name for this is ShapeTrax (CV-X: ShapeTrax2, VS: ShapeTrax 3A), an edge-based shape search. No Keyence tool called "PatternTrax" was found. The existing Outline tool scores one part at a fixed angle; this tool finds and counts every copy of a taught pattern at any angle.
- `app/vision/search.py`, kind `pattern_search`, in "Pattern & Count". The ROI is the search area and `template_roi` is the pattern dragged inside it. The value is the number of copies found (`DEFAULT_LOW = 1`); `extra["matches"]` holds x, y, angle and score for each copy.
- How it works:
  - Matching runs on edge strength (smoothed Sobel magnitude, with texture below `edge_min` set to zero), so lighting and background shade matter little.
  - The pattern is padded to a square as wide as its diagonal, turned, and matched with a mask of its own footprint (masked TM_CCOEFF_NORMED). A 3 px border band of the template is left out of the mask, because the edge filter sees the crop's border there.
  - Coarse pass: downscaled until the template is about 12 px, slightly blurred, at the coarsest angle step the pattern's size allows (at most 12°), with the angles matched in a 6-thread pool.
  - Each candidate is refined at full scale over ± half a coarse step. Copies are kept above `min_score` (0.88) and when they don't overlap a better one, up to `max_count`.
  - Each copy is drawn as the pattern's thin Canny edges, turned and placed where it was found (`region`, color "lines"), with "n: score% angle°" above it.
- `SLOW = True`: it runs in the slow-tool thread (2 Hz), like OCR.
- Bugs found and fixed while testing on `sample_mark.jpg` (386 × 208, a windshield marking):
  - The zero padding was compared with the text around each copy, so a mark scored only 0.64 against itself. Fixed with the masked match.
  - `roi_to_pixels` grows a box to its 32 px minimum, which moved small marks by about 11 px. The template box now uses exact pixels.
  - Masked TM_CCORR_NORMED scores about 1.0 in empty background (0/0 on mostly-zero edge maps). Both passes now use TM_CCOEFF_NORMED; its infinite values on flat windows are set to 0 and scores are clamped to ±1.
  - Downscaling the coarse pass to an 8 px template lost small and long patterns; 12 px is kept.
- Sample results: E6 circle, TIS logo and the word TOYOTA found exactly once at 0°, 30°, 90°, 145° and −120° (image turned on a larger canvas), within 1–2 px and about 4° (score 0.92–0.99). The 15 × 19 px "SN" box is found at every angle, plus one look-alike at 0.88 at two angles. Tiny one-stroke marks ("//", "2") match many letter strokes; teach a distinctive pattern.
- Time:
  - Sample (441 × 441 canvas): 130–220 ms, and about 330–520 ms for the tiny SN box, which cannot be downscaled for the coarse pass.
  - 1280 × 720 frame with a 640 × 432 search area and a ~100 px pattern: 94 ms at ±180°, 30 ms at ±30°, 15 ms at 0°.
- Tools tab: after the search area is drawn, it asks straight away for a box around one copy of the pattern; Teach does the same later. Thumbnail `pattern_search.png`: three turned "E" marks found, OK 3.
- Tests: `tests/test_vision_search.py` (12) on drawn marks, plus a Tools tab test for the area → pattern box flow.

### CV-X style Tool Catalog, fit and contrast pass
- Reference: Keyence CV-X "Tool Catalog" plus "Application Navigator". The Add Tool dialog is now that catalog:
  - a dark title strip;
  - a light catalog panel with category icon tiles in 2 rows of 3 (light gloss, orange when selected);
  - a question bar for the category ("▼ What do you want to measure?") and that category's tool tiles;
  - a gray Application Navigator on the right: the tool's name, what it does (`KIND_HELP`), its use and a large example picture;
  - an info bar with an "i" tip ("add Position Adjust first …"), and Cancel / Add <tool>.
- Behaviour: a tap selects a tool and explains it; Add (or a double tap) confirms. The catalog reopens on the category used last. It sizes itself to the window (at most 1180 × 720, the window minus 24 px), so it fits 1024 × 768 without scrolling. It was 820 px tall before.
- Fit:
  - Tools tab live value: wraps, and its horizontal size policy is Ignored. A long reason ("search area smaller than …") used to widen the side panel and push the table, Draw ROI, Delete and "Set from current" off screen (seen on the station).
  - Pattern Search pads its search area with blank margins, so a tight search area no longer fails with "search area smaller than the turned pattern".
- Contrast:
  - Image overlays: labels are bold, about 1.6 % of the frame width (min 13 px), on a dark chip and kept inside the image. Line widths scale with the frame. They used to be about 9 pt in frame pixels, about 5 px on a 1024 screen, and green straight on the image.
  - The same helpers draw the YOLO box labels and the Monitor "ROI" label. The selected tool in the Tools tab is no longer named twice.
  - Hint text #9aa5b1 → #c4ccd5; disabled button text #7a838d → #949da7 (about 5:1); the Train classes note is 10 pt #2f353c.
- Checked by rendering every tab and the catalog offscreen at 1920 × 1080 (the station's panel, `xrandr`) and 1024 × 768.
- Tests: the catalog (a tap selects and fills the navigator, Add confirms, categories switch, double tap adds, fits 1024 × 768). 357 pass.

### Same live-image size on Monitor and Tools
- The Monitor side panel took its natural width (~290 px) and the Tools panel was fixed at 430 px, so the camera image was a different size in each tab (1606 vs 1434 px wide at 1920x1080; 710 vs 538 at 1024x768).
- `app/widgets/layout.py` now holds `SIDE_PANEL_WIDTH = 400` and `VIDEO_MIN_SIZE = (520, 300)`, used by both tabs: the image is 1464x843 at 1920x1080 and 568x531 at 1024x768 in both, and the window still fits 1024 wide (minimum 996 px).

### AI vs image-processing badges; Settings split into Machine / This Job
- Two small SVG badges in `app/theme.py` (written to `.ui_assets/`): purple chip with "AI" for YOLO model settings and results, cyan magnifier for the image-processing tools. Helpers: `badge_icon`, `badge_title`, `set_badge_title`; `make_card(title, badge=...)`.
- Used on: tab icons (Tools = image, Train = AI), Monitor side panel ("AI Detections" / "Image Tools"), Tools tab list header ("Image Tools"), Settings cards ("AI Model", "AI Detection Filters").
- Settings now has two headed columns: **Machine** (Camera, Tower Light & Buzzer, Auto Capture; same for every job) and **This Job** (AI Model, AI Detection Filters; "Saved with job <part>", updated on part change). Nothing moved between tabs.

### Icons across the UI
- `make_icon` gained 20 outline icons on the same 20x20 grid (monitor, gear, swap, list, folder, refresh, trash, plus, check, pencil, eyedropper, ruler, gate, target, barcode, sliders, range, save, tower, snapshot). `icon_file` / `icon_title` put one in a rich-text header; `make_card(title, icon=...)`.
- Tabs: Monitor, Settings, Job Change, Log get icons (Tools / Train keep their image / AI badges).
- Card headers: Camera, Tower Light & Buzzer, Auto Capture, Saved Jobs, Limits, Parameters.
- Buttons: Settings (Load model, Scan Cameras, Test), Tools (Teach, Draw ROI, Pick Color, Delete, Gate, Calibrate, Set from current, Clear, Done, Snapshot / Resume Live, Add Tool), Monitor (Clear ROI, Reset Count), Job Change (Remove Job, Scan Barcode), Train (Select Folder, Add Class, Rename, Confirm, Draw, Delete), Log (Refresh), and the dialogs (Add / Cancel / Delete / Remove / Save / Close, Set from live, Reset Corners).
- Same meaning everywhere: trash = delete, X = cancel / clear, check = confirm, plus = add, refresh = reload / reset. Train's box buttons are now "Draw" / "Delete" (under the "Box Editing" header) so they fit with icons.
- Window minimum width 1016 px, still fits 1024x768.

### Hover hints
- Every tool parameter has a hint (`PARAM_HINTS` in `app/widgets/tool_params.py`, `(kind, key)` entries override plain keys); `ToolParamsForm` puts it on the row's label and its control. A check confirms every visible parameter of every tool kind has one.
- Tool names: `kind_hint(kind)` = what the tool does + "Good for: ..." on catalog tiles and the Name / Kind cells of the Tools and Monitor tables.
- Buttons and controls: Tools tab (Teach, Draw ROI, Pick Color, Wider / Narrower, Gate, Calibrate, Low / High, Enabled, Follow position, Snapshot, Add Tool...), Monitor bottom bar, Settings labels via `tip_label(text, hint, *controls)`.
- `QToolTip` styled light cream on an orange border, 11pt, for contrast on the dark theme. `HintWrapper` (installed app-wide in `main()`) re-shows plain-text tooltips as rich text so Qt word-wraps them, including table cells.
- Hover needs a mouse: on the touch screen Qt shows no tooltips.

### Overlays painted on screen, not into the frame
- Before: boxes and labels were painted (QPainter) into the camera frame at its own resolution, and the frame was then shrunk to fit the view, like cv2 drawing would. Text and line size followed the camera resolution, text was resampled (soft), and OCR text sat at a fixed 14 frame px, hitting the box edge.
- Now `VideoView.set_frame(pixmap, overlay)` paints the frame, then calls `overlay(painter)` with a `ScreenPainter` scaled to frame pixels. `overlay_pen` gives fixed-width (cosmetic) lines, `draw_label` maps its point to the screen and draws the chip unscaled at `SCREEN_FONT_PX` (15 px), elided to the image width. Same size and sharp at any camera resolution (test: `ScreenOverlay`).
- Monitor: `_show_frame` (live, screen-painted) vs `_render_frame` (burned in, only for the NG log image, sized as before); both use `_paint_overlay`. Tools tab: `_paint_overlay`. The live view no longer re-renders a full-size pixmap every frame.
- Text items take `place`: "above" (default; Pattern Search), "inside" (OCR / code text at the ROI's top-left, under the tool name), "center" (Blob Count numbers).

### OCR on a glass marking: cleaner prep, read text in the name chip
- Tested on `sample_mark.jpg` (a real windshield mark, ~386x208, print 8-14 px tall) and on a new synthetic mark from `tools/make_synthetic_mark.py` (made-up maker, E-mark, DOT, TIS, laminated and lot lines; tilted, blurred, noisy, with a JSON of each line's text and box). The PNG is generated into `samples/` (gitignored, rebuilt by the script).
- `flat_field(..., ignore_dark=True)` closes away dark print before blurring the background, so a dense line of bold text no longer lightens its own surroundings (that made Otsu split a bright halo from the paper and read white-on-black). The OCR default pre uses it; saved tools keep their own `pre`.
- `prepare_for_ocr` now also runs `clean_binary` before and after deskew (drops pieces of the lines above / below cut by the ROI edge and specks, keeps hyphens) and adds a 20 px white margin (`PAD_PX`); Tesseract misread glyphs touching the edge.
- Synthetic mark (default): 7/7 lines read exactly, confidence 78-96 (was 4/7 with hyphens lost and one line at confidence 0). Harder versions (tilt 4-6 deg, heavy blur): 4-6/7; misses are mostly a stray last character. Real photo: TOYOTA / AGC AUTOMOTIVE / 43R 006735 read, the 8 px DOT / TIS lines are unreliable (too small and soft for Tesseract). Test: `SyntheticMark`.
- `Tool.READ_IN_LABEL` (OCR, code): the read text goes in the tool's name chip ("Lot no.: LOT 24A0915") instead of a second chip inside the ROI, which covered the print and, on stacked lines, the next tool's name.
- Tools tab: "Live value" now follows the selected row at once (it only changed on the next run).

### Standard-mark dataset and Required Marks per job
- `STANDARD_CLASSES` (`app/marking/classes.py`): one YOLO class per standard so the detector alone says which standards a windshield carries: `e_mark` (ECE R43 circle + 43R number), `dot`, `as_class`, `ccc_mark`, `tis_mark`, `sni_mark`, `inmetro`, `asnzs`, `bis_isi`, `laminated`, `glass_class`, `date_code`, and made-up maker logos `logo_acme` / `logo_nova` / `logo_orion` (rename to the line's brands). Order = class id; append only. `tools/init_marking_dataset.py standard <dir>` makes an empty dataset with them.
- `tools/make_synthetic_mark.py --dataset DIR --count N [--seed S] [--size WxH]` writes the Train tab layout (`images/*.jpg`, `labels/*.txt`, `classes.txt`) plus `truth.json` (marks present / left out per image). Per image: one maker logo (95 %), e-mark 90 %, every other standard 50-75 %, at least two marks; 5 % blank backgrounds with an empty label file. Each mark is drawn on its own canvas (tight box), laid out in rows, scaled to 12-40 px capitals, tilted +-8 deg with a slight perspective, blended on glass with uneven light, sometimes a dark ceramic band, blur, noise, JPEG 70-95. Boxes are the warped corners, axis-aligned; a placement leaving a mark < 60 % in frame is retried. Same seed = same dataset. The logos are simple drawings of the kind of mark, not official artwork: add real labelled photos before relying on the model.
- `DetectionFilters.required_classes` / `missing()`: when any are required the AI verdict is OK only if all are detected, NG if one is missing, STANDBY if nothing is detected; with none required the Target Classes logic is unchanged. Joins the tools through `total_state` as before.
- Settings > This Job > **Required Marks**: a checkbox per model class, "Set from current part" (ticks exactly the classes in the Monitor's latest filtered detections, `ctx.last_detections`, after a yes/no listing them). Saved with the job as `required_classes`; older jobs require nothing.
- Monitor top strip: red "Missing: tis_mark, sni_mark" next to the judgement pill while a required mark is absent.
- `datasets/` is gitignored (generated data).
- First model (yolo26n, 300 synthetic images, 30 epochs, batch 4, imgsz 640, 25W mode; MAXN SUPER tripped the Jetson's over-current protection and rebooted it): mAP50 0.97-0.995 for every class on 60 separately generated images (seed 99). At conf 0.5 the detected set of marks equals the truth on 49/60 images with no false marks; every error is a miss, on the smallest / blurriest marks (letters ~10 px), and ISI was once read as SNI (both are letters in a box). Missing list right on 49/57. Saved as `weights/standard_marks_synthetic.pt` (not committed). Not tested on real windshields yet.

### Synthetic marks closer to real print
- The first model on a clean real marking (`mark_sample2.png`: AGC, E6, TIS, SNI, QR, TEMPERLITE, date rows): only the TIS mark right; E6 called `tis_mark`, SNI and the date code missed, 9 false marks on the QR code, TEMPERLITE, part numbers and the dashed border.
- Generator changes: text in bold TTF sans (DejaVu / Liberation, 60 % of images; Hershey otherwise); E-mark as a big bold E with a small country number in a ring (25 % older same-size style); TIS emblem with an up/down arrow between two split triangles; SNI as heavy condensed letters between two bars (20 % boxed, 40 % with a maker code); date code as one or two widely spaced digit rows (1 2 4 8) or the dot pattern; frames keep bold letters clear of the border.
- Unlabelled print the model must ignore (70 % of images): QR-like squares, words (TEMPERLITE, SECURIT, MADE IN ...), part numbers and short codes; blank images rose to 10 %, often with this print only.
- Second model (`weights/standard_marks_synthetic_v2.pt`, not committed): trained from the first, 400 + 80 regenerated images (seeds 1 / 99), 30 epochs, batch 4, 25W. Validation mAP50 0.988 (e_mark / tis_mark 0.995, sni_mark 0.993, date_code 0.984). On `mark_sample2.png` at imgsz 640: e_mark 0.84, tis_mark 0.93, sni_mark 0.91, date_code 0.50, one false `dot` on the part number at 0.33, nothing else; at imgsz 960 the date code is missed. Use conf ~0.4 for this job until real photos are added.
- OCR on the detected boxes (the OcrTool's prep, psm 6, skipping the emblem left of the E-mark / TIS number, first line only): `43R-008574`, `TIS 2602-2556`, `SNI`, `1248` / `1248`, all right.

### Real AGC drawings: real brands, CCC, LAMISAFE / TEMPERLITE, and a real validation set
- `image_mark.zip` (27 AGC Automotive marking drawings, CAD and scans): TOYOTA 8, Honda 9 (new solid H 4, classic H in a rounded square 5), NISSAN 2, no brand 8; AGC AUTOMOTIVE, `//` or `V` + E6 43R, the TIS emblem with L / T, LAMISAFE or TEMPERLITE and the 1 2 4 8 rows on all; DOT-207 Mxxx ASn on 12; SNI on 17 (the rest have the maker code without the logo: real "missing SNI" cases); CCC once.
- `STANDARD_CLASSES`: ids 12-14 renamed `logo_toyota`, `logo_honda` (both styles), `logo_nissan` (the placeholders were never trained on real data); appended `logo_agc`, `lamisafe`, `temperlite` (18 classes).
- Generator: drawings of these logos (approximations, not the official artwork), the CCC logo as three cascading C arcs in an oval, and an AGC layout for 70 % of marked images that follows the real order (brand, AGC, [// or V] E-mark, DOT line or M-code, [// or V] TIS emblem with L / T and 11T, LAMISAFE / TEMPERLITE, TIS number line, code + SNI next to the date rows, lot code), each part left out at random, sometimes in a dashed frame. Labels: `glass_class` on the `//` / `V` alone; `dot` on DOT-207 Mxxx and `as_class` on ASn; `tis_mark` on emblem + number when they share a line, else on the emblem and on the number line separately; `sni_mark` on the logo only (the code beside it is unlabelled). TEMPERLITE left the unlabelled word list.
- `tools/prelabel_marks.py MODEL DIR`: writes a model's boxes as label files for checking in the Train tab. v3's pre-labels on the drawings were too poor to start from, so the 27 were labelled from what is printed (Tesseract word boxes + connected components, two boxes fixed by hand) and every image checked by eye: 254 boxes in `datasets/real_marks/` (gitignored).
- v3 (from v2, 500 + 80 regenerated synthetic, 30 epochs): synthetic val mAP50 0.986, but on the 27 real drawings mAP50 0.657 and the exact set of marks right on 1/27 at conf 0.4 (mostly misses: glass class, date code, AGC, AS, Honda).
- v4 (from v3, synthetic + 20 real drawings x5, val = 7 held out: 3, 8, 13, 16, 22, 24, 26; 30 epochs, batch 4, 25W): held-out real mAP50 0.995, exact set 7/7 at conf 0.4, no miss or false mark in any class; with Required = e_mark, tis_mark, sni_mark, logo_agc the three without SNI judge RED "Missing: sni_mark", the others GREEN. Synthetic val still 0.991. Saved as `weights/standard_marks_v4.pt`.
- Gap: on `mark_sample2.png` (a part number line `5403010-ER51-AA 3.1` / `S27413` and a QR code, not on any drawing) v4 calls `S27413` a `dot` (0.88) and `3.1` a `date_code` (0.69); all eight real marks there are right.
- v5 fixes that gap: the AGC layout now also prints, unlabelled, a part number line (`5403010-ER51-AA 3.1`, 35 %) with a serial under it (`S27413`, 60 % of those) and sometimes a QR code under the TIS line (20 %). Trained from v4 on regenerated synthetic data + the same 20 real x5, 30 epochs. `mark_sample2.png` at conf 0.4: AGC, E-mark, TIS, SNI, TEMPERLITE, date code, nothing else (a `dot` at 0.31 remains below the threshold). Held-out real: mAP50 0.995 (mAP50-95 0.950, was 0.936), exact set 7/7, Missing sni_mark on the same three; synthetic val 0.993. Saved as `weights/standard_marks_v5.pt`.

### Six more marking classes and Mark inspection (layout check)
- `STANDARD_CLASSES` appended (24): `short_name` (58, U8, T1 beside SNI), `type_code` (FI, FY, AC at the bottom), `plant_code` (11T; THA before it is not boxed), `m_code` (M874, M8U74E; also the M-part of the DOT line, which is now split dot / m_code / as_class), `letter_l` / `letter_t` (glass type letter beside the TIS emblem). The generator draws them in the AGC layout; the 27 drawings were relabelled (371 boxes, every sheet checked by eye; IR, AC and one 11T touch the frame line and were boxed by hand).
- v6 (from v5, 40 epochs, same split): held-out real mAP50 0.985, exact set 5/7. Misses: the short name T1 on 16 (called plant_code) and 22; `S27413` on `mark_sample2.png` is called m_code. More T-starting short names / real photos needed. `weights/standard_marks_v6.pt`.
- Mark inspection (`app/marking/layout.py`, `check_layout(detections)`): fixed AGC rules on the boxes. Top to bottom brand logo, AGC AUTOMOTIVE, E-mark, M-code / TIS (centres); 11T right-most (within 1.5 x its height: the AGC AUTOMOTIVE box overshoots the right edge by up to 0.73); SNI centre in the left half of the marking; short name left-most (within 1 x its height, the `//` sits at the same x). A rule is skipped when its marks aren't detected. All 27 hand-labelled drawings and v6's detections on the 7 held-out ones pass; moving 11T left and SNI right on drawing 3 gives both warnings. On `mark_sample2.png` it warns "m_code not below e_mark", which is the model calling S27413 an M-code.
- Settings > Required Marks > "Mark inspection: check layout" (saved as `check_layout`, older jobs off). Warning only: the Monitor shows "Layout: ..." in amber next to "Missing: ..."; the OK / NG verdict is unchanged.

### v7 and v8: T-starting short names and more real drawings in training
- Generator: 35 % of short names start with T (T1, T2, ...) and 30 % have a grey frame line through them, as on the drawings where the dashed border crosses the name. `mark_sample2.png` joined training (`datasets/real_extra/`, its 9 marks; `S27413` left unlabelled).
- v7 (from v6): held-out mAP50 0.985 / mAP50-95 0.937, still 5/7. T1 on 16: short_name 0.24 vs plant_code 0.36; on 22: short_name 0.35. Every real T-name was in the held-out set, so the model had never seen one printed. `weights/standard_marks_v7.pt`.
- v8 (from v7, 80 epochs, batch 4, 25W): drawing 22 moved into training, 23 held out instead (held out: 3, 8, 13, 16, 23, 24, 26). Held-out mAP50 0.983 / mAP50-95 0.934, exact set 7/7 at conf 0.4, no miss or false mark in any class. T1 on 16: short_name 0.95 (v6 0.12, v7 0.24). `mark_sample2.png`: the 9 marks only, no m_code on S27413. Mark inspection: no warning on the 7 held-out drawings or the sample; the moved-11T / SNI test on drawing 3 still gives both. `weights/standard_marks_v8.pt`.

### Judge by: AI Detection | Required Marks, with a confidence per mark
- Settings > AI Detection Filters > **Judge by**: two buttons, saved with the job as `judge_mode`. AI Detection: the Target Classes decide (the Required Marks card is hidden). Required Marks: the ticked marks decide (Target Classes hidden). Older jobs: Required Marks when any mark is ticked, as before.
- The Required Marks card is now a table: tick + **Min conf** (%) per mark, saved as `mark_conf`; a new tick starts at 40 % (the v8 model's checked threshold), older jobs get the slider's confidence. In Required Marks mode a ticked mark's own minimum replaces the slider for that class, both for drawing and for OK / NG (`DetectionFilters.class_threshold`).
- Monitor side panel in Required Marks mode: "Required Marks" table like the Tools tab's, Mark | Conf (best, before filtering, so a weak mark shows its value) | Min | Result: OK green when conf >= Min, NG red when missing or below; "—" while nothing is detected (STANDBY). `DetectionFilters.mark_results`. Mark inspection (layout check) only runs in this mode.

### Kind of inspection shown by badge colour
- New badge `mark` (fuchsia `#d946ef`, a ringed E like the E-mark) for **Mark Inspection**, beside the existing `ai` (purple chip, **AI Detection**) and `vision` (cyan magnifier, **Image Processing**). `jobs.inspection_types(config)`: `mark` when the job has a model and judges by Required Marks with at least one ticked (older jobs: any ticked), `ai` when it has a model otherwise, `vision` when it has image tools; a job can be two kinds.
- Shown on the Monitor info strip (badge + name after Part / Model, follows the Settings switch and tool changes live), on each Job Change tile (one badge per kind, names in the tooltip), on the Judge by buttons, and on the Required Marks headers (Settings card, Monitor panel), which used the AI badge before.

### Mark model loaded automatically, job settings saved as you go, large Part / Model
- `paths.MARK_MODEL_PATH` (`weights/standard_marks_v8.pt`): switching Judge by to Required Marks loads it unless it is already loaded, and a job that judges by Required Marks (older ones: any mark ticked) always opens with it, whatever model it saved. When the loaded model is not the one the job saved (or the job saved no class list), every class is shown.
- Settings > This Job edits (model, confidence, max detections, target classes, classes to show, Judge by, required marks and their Min conf, layout check) are written to the current job 0.5 s after the last change; before, they were only saved when a job was created, so a restart (which reopens the last job) lost them. Applying a job never saves it; a pending save of the previous job is written first.
- Monitor top strip: Part and Model on two rows at 32 pt (2.5x the 13 pt base), the kind-of-inspection badge and the verdict pill beside Part, Missing / Layout warnings beside Model. The two labels take the width their text needs and clip instead of widening the window.

### Settings redesign: live preview, Save / Revert, camera adjustments, relay I/O, EN / TH
Built on ISA-101 / usability basics: gray chrome and colour only for meaning, the effect of a setting visible while setting it, undo and a guard against losing changes, one primary action bottom-right, touch targets >= 44 px.
- **Settings shell**: top strip (section, "Machine: the same for every job" / "Saved with job X"), a nav rail (Machine: Camera, Tower & Buzzer, Relay I/O, Auto Capture; This Job: AI Model, Required Marks), the section's form, and the live camera beside it (opening Settings starts the camera as a setup preview, like Tools). The preview draws what the section changes: the adjusted image, AI boxes with confidence, or the required marks in OK green / NG red.
- **Save / Revert**: every change works at once (on the preview, the tower, the relay, the inspection) but is written only by **Save**: machine values via `AppSettings.commit()` (counters, the buzzer toggle and the language stay written at once, `IMMEDIATE_KEYS`), this job's values (`JOB_KEYS`) merged into its file, keeping tools / ROI / masters. **Revert** goes back to the saved values and widgets. An amber "Unsaved changes (n)" chip counts keys that differ from the saved state (a change and its undo don't count). Leaving Settings, switching job or closing the app with unsaved changes asks Save / Discard / Cancel. Replaces the 0.5 s job auto-save.
- **Camera image** (`app/image_adjust.py`, Settings > Camera > Image): mirror, flip, brightness, contrast, colour, shadows (gamma), local contrast (CLAHE), Reset. Applied by the worker right after each frame is read, before the Top View warp, so the AI, tools, captures and the corner editor see the same image; neutral steps are skipped.
- **Relay I/O** (`app/relay_io.py`, Waveshare Modbus POE ETH Relay 8 DI / 8 DO, as in natn2014/Pokayoke `Relay_IO.py`): Modbus RTU over TCP (CRC16, port 4196, the board's default) or Modbus TCP (MBAP, port 502); functions 0x02 / 0x01 / 0x05; bad CRC / stale replies flushed and retried; a poll thread (50 ms), queued writes, reconnect every 3 s, outputs off on exit. Settings page: board address with IP check and Connect, a role per DI (Not used / Trigger / Confirm NG) and DO (Not used / OK / NG) with live lamps (light blue, not verdict colours), Test pulse per output after a yes / no, OK / NG output held or pulsed.
- **Inspection cycle** (`InspectionCycle`): without a Trigger input as before. With one, the Monitor shows "Waiting for trigger" and judges the next frame per trigger, holding and counting that result once. With a Confirm NG input an NG latches (NG output, red tower, buzzer), triggers are ignored, and a red **Confirm NG** button appears under the pill; the input or the button clears it.
- **Language EN / TH** (`app/i18n.py`, `app/i18n_th.py`): UK / Thai flags (drawn in code) in the tab bar's corner, live switch, remembered. A translator walks the widgets on a switch and as they are shown (texts, tooltips, combo items, headers, tabs, placeholders, suffixes), remembering each English source; changing texts call `tr()`; formatted messages ("Missing: {}", "Unsaved changes ({})") match patterns, with names and part numbers kept as they are. Status messages are translated where shown. A test opens every tab and Settings section in Thai and fails on any English left. Thai wording to be checked by someone on the line.
- Monitor top strip: empty warning lines are hidden so the strip keeps its height.

### Required Marks: short name read by OCR against a preset, NG with zoom
- Settings > Required Marks > **Short name (OCR)**: the value the job's short name must read, an uppercase letter + a digit or two digits (`T1`, `58`; typed lower case is raised; anything else is refused). Saved with the job (`short_name_expected`, part of Save / Revert); empty or older jobs = presence only, as before. Read only while Judge by is Required Marks and `short_name` is ticked.
- Reader (`app/marking/short_name.py`): plain Tesseract on the box misread a third of the drawings (a red frame line through the characters, letters read as other letters). The reader: gray = darkest channel with reddish / saturated pixels (the drawings' red frame) set to paper, the stroke gaps the line leaves bridged across the line only (as wide as the line), long thin lines removed; the box split into its two characters (pieces of a character, such as an arm cut off by the line, joined back); each character read unrestricted first, then limited to its position's kind (letter / digit from the preset's format) when unsure; a lone glyph is also read as a pair ("99"), which Tesseract reads where a single "9" comes back empty; two agreeing unsure reads count. A clear digit where a letter belongs stays a digit (so "20" can't pass as "Q0"); only letter-for-digit look-alikes (O, I, Z, S, G, B) are mapped. ~0.5 s per read.
- Gate on the drawings (v8 boxes): **23 / 23** short names read as printed, **0 / 414** checks against another drawing's name passed; the same with the hand-labelled boxes (22 / 22; drawing 3's label box cuts its 5). Synthetic bold print passes with the red line and blur.
- Runtime: the worker hands the padded crop of the most confident short_name to a reader thread (`ShortNameReader`, newest crop only, at the OCR / code rate) and puts this part's read in that detection (`det["ocr"]`). A part's read is "pending" (STANDBY) until the first clear read and "unreadable" (NG) when none came within 1.5 s; the short name gone for 0.5 s = a new part, so a previous read never passes it. In trigger mode a trigger waits for the read.
- Monitor: NG when the read differs or is unreadable; the short name is shown magnified (at least 3x, ~38 % of the frame wide, red frame, "Read T1 ≠ expected T2" / "Unreadable") in the corner away from it, with a dashed box where it is, kept while the result is NG and burned into the NG image in the Log; "Short name: read T1 ≠ T2" in the top strip; the table row reads `short_name  T1 ≠ T2`. The Settings preview shows the read beside the box.
- Monitor side panel: a full-width **Short name** tile above Required Marks with the preset value in large type, green when the part's short name reads right, red while it is wrong / unreadable (as long as the NG lasts), neutral with no part or while reading; shown only when the job checks the short name.
