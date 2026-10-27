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
| T02 | preprocess / shading correction | done | |
| T03 | Position Adjustment | todo | |
| T04 | Total judgement | todo | |
| T05 | Save tools in job | todo | |
| T06 | Color Area | todo | |
| T07 | Color Average (ΔE2000) | todo | |
| T08 | Color Prohibit | todo | |
| T09 | Area (mono) | todo | |
| T10 | Brightness Average | todo | |
| T11 | Edge Pixels | todo | |
| T12 | Shading Uniformity | todo | |
| T13 | Edge-profile core | todo | |
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
- Commit: pending
