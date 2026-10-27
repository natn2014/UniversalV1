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
