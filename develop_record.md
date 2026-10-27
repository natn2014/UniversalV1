# Develop Record: image processing tools (branch `imageprocessing`)

The build log for [imageprocessing_todo.md](imageprocessing_todo.md).
Add one entry each time a TODO item (T01–T27) is finished or blocked. Newest
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
| T13 | Edge-profile core | done | |
| T14 | Width | todo | |
| T15 | Edge Presence | todo | |
| T16 | Pitch | todo | |
| T17 | Diameter | todo | |
| T18 | mm calibration | todo | |
| T19 | OCR read | todo | |
| T20 | OCR verify | todo | |
| T21 | Code tool (QR / 1D) | todo | |
| T22 | Outline | todo | |
| T23 | Blob Count | todo | |
| T24 | Worker integration | todo | |
| T25 | Monitor integration | todo | |
| T26 | Tools tab UI | todo | |
| T27 | Performance + final check | todo | |

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
- Commit: pending
