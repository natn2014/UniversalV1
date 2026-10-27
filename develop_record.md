# Develop Record: image processing tools (branch `imageprocessing`)

The build log for [imageprocessing_todo.md](imageprocessing_todo.md).
Add one entry each time a TODO item (T01–T27) is finished or blocked. Newest
entries go at the bottom. Do not edit old entries; add a correction entry instead.

**Test command:** `.venv/bin/python -m unittest discover -s tests`
**Baseline before T01:** 86 tests, all passing (2026-10-27).

## Status board

| T | Function | Status | Commit |
|---|---|---|---|
| T01 | vision/base.py framework | todo | |
| T02 | preprocess / shading correction | todo | |
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
