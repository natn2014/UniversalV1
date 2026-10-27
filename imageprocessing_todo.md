# Image Processing TODO (branch `imageprocessing`)

This is the list of functions to build, in order, one block per function.
Background and the IV4 tool map are in [core_function.md](core_function.md).
Progress is logged in [develop_record.md](develop_record.md).

**Rules for every item**
- Code goes in `app/vision/`, tests in `tests/test_vision_<module>.py`
  (unittest, synthetic numpy/cv2 images, the same style as `tests/test_autocapture.py`).
- Run the tests with `.venv/bin/python -m unittest discover -s tests` (pytest is
  not installed in the venv). All tests must pass before the commit; the
  baseline is 86.
- Images are **BGR uint8** (the camera frame after the perspective warp).
  ROIs are fractions `(x1, y1, x2, y2)` in 0..1, as in `app/roi.py`.
- Each tool's `run()` must not raise on an empty or odd ROI. It returns
  `ToolResult(ok=False, ...)` with a `text` saying why.
- Match the code style of `app/roi.py` / `app/judgement.py`: short module
  docstring, type hints, docstrings that say *why*, few inline comments.
- Field template:
  **IV4** (what Keyence does) · **How it works** (algorithm) · **Params**
  (JSON keys = defaults) · **Value / OK** · **Overlay** · **Tests** · **Done when**

---

## Phase 1: Core

### [x] T01 `app/vision/base.py`: tool framework
- **IV4:** Every tool has an ROI, a measured value, and an upper/lower limit that gives OK or NG.
- **How it works:**
  - `Offset(dx: float = 0, dy: float = 0, angle: float = 0, score: float = 1.0, found: bool = True)`. This is the part's movement found by Position Adjustment.
  - `ToolResult(name, kind, value: float, ok: bool, overlay: list, text: str = "", extra: dict = {})`.
  - Overlay items are plain dicts, so the UI layer draws them and `vision` never imports Qt:
    - `{"type": "rect"|"line"|"circle"|"poly"|"text"|"mask_box", "pts": [...], "ok": bool, "label": str}`
    - The coordinates are **frame pixels**.
  - `class Tool` (base class):
    - Fields `name`, `kind` (registry key), `roi: Roi`, `low: Optional[float]`, `high: Optional[float]`, `enabled: bool = True`, `follow_position: bool = True`.
    - `judge(value) -> bool`: true when `low <= value <= high`. A `None` limit means no bound on that side.
    - `crop(frame, offset) -> (patch, (x1, y1))`:
      - Converts the ROI to pixels with `roi_to_pixels` from `app/roi.py`.
      - If `follow_position`, shifts the ROI by `(dx, dy)` and clamps it to the frame.
      - Rotation is applied only by tools that need it; they use `rotate_points` (below). This keeps the crop fast.
    - `teach(frame, offset=Offset())`: the default does nothing.
    - `run(frame, offset) -> ToolResult`: abstract.
    - `params() -> dict` / `set_params(dict)`: the tool-specific keys.
    - `to_dict()` → `{"kind", "name", "roi", "low", "high", "enabled", "follow_position", **params}`. Taught data must be JSON-safe: lists, not ndarrays. A template image is stored as a file path (T05).
    - `Tool.from_dict(d)` uses the registry.
  - A registry: `register(kind)` decorator, `TOOL_KINDS: dict[str, type]`, and `create(kind, **kw)`.
  - Helpers: `rotate_points(pts, center, angle_deg)` and `shift_overlay(items, x0, y0)`. The second turns patch coordinates into frame coordinates.
- **Params:** none of its own.
- **Tests:**
  - `judge` with low only, high only, both, or neither.
  - `crop` clamps when the ROI is shifted past the frame edge.
  - Round trip `to_dict` → `from_dict` with a dummy registered tool.
  - An unknown kind raises `KeyError`.
- **Done when:** the base class and registry are tested, nothing in `vision` imports Qt, and other tools can subclass `Tool`.

### [x] T02 `app/vision/preprocess.py`: pre-filters and shading correction
- **IV4:** Built-in image filters. AI OCR copes with shaded text; we provide explicit shading correction.
- **How it works:** pure functions; every one takes and returns a uint8 image.
  - `to_gray(img)`, `to_hsv(img)`, `to_lab(img)`. Each passes the image through unchanged if it is already in that space.
  - `flat_field(gray, kernel=51)`:
    - `bg = GaussianBlur(gray, (k, k))`
    - `out = gray / max(bg, 1) * mean(bg)`, clipped to 0..255.
    - This removes slow brightness gradients.
  - `top_hat(gray, kernel=15)` brings out small bright defects. `black_hat(gray, kernel=15)` brings out small dark defects. Both use `MORPH_ELLIPSE`.
  - `clahe(gray, clip=2.0, tiles=8)`.
  - `apply_chain(img, steps: list[dict])`:
    - Each step looks like `{"op": "flat_field", "kernel": 51}`.
    - Steps run in order. An unknown op raises `ValueError`.
    - Every tool takes an optional `"pre": []` param that is passed to this function.
- **Tests:**
  - A gray ramp (0→200 across the image) plus a dark square. After `flat_field`, the background std-dev drops by more than 70% and the square still has more than 40 levels of contrast.
  - `top_hat` finds a 3 px bright dot on the ramp.
  - `apply_chain` with 2 steps gives the same result as calling the two functions by hand.
- **Done when:** all ops are tested and `apply_chain` runs on gray and BGR input.

### [x] T03 `app/vision/position.py`: Position Adjustment
- **IV4:** Finds the part and moves all other tool ROIs with it. "Hi-Speed" works along 1 axis or 2 axes.
- **How it works:**
  - `PositionTool(kind="position")`. Its ROI is the **search area**.
  - `teach(frame)`:
    - Stores the template, a gray crop of `template_roi` (a smaller box inside the search area), as a list or a file path. For tests, keep it as an ndarray in memory.
    - Stores the template's taught center, in frame pixels.
  - `run(frame)` in `mode="2d"`:
    - Runs `matchTemplate(search, template, TM_CCOEFF_NORMED)` and takes the best location → `dx, dy` relative to the taught center.
    - When `angle_range > 0`: matches the template rotated at each `angle_step` in `[-range, +range]` (use `warpAffine` on the template, with a border filled by the template's mean).
    - For speed, does a coarse pass on a ½-scale pyramid, then refines at full scale in a ±4 px window.
  - `mode="x"` or `mode="y"` (1-axis hi-speed):
    - Builds a profile of the search area (column means for x) and finds the strongest edge with T13's `find_edges`.
    - The offset is the edge position minus the taught edge position.
  - The result has `value = score × 100`, and `ok = score ≥ min_score`.
  - Also provides `offset() -> Offset`, with `found=False` when the score is too low.
- **Params:** `mode="2d"`, `template_roi=None`, `angle_range=0.0`, `angle_step=1.0`, `min_score=0.6`, `pyramid=True`.
- **Overlay:** the search area, a box around the found template (rotated polygon), and a cross at its center.
- **Tests:**
  - Teach on a synthetic "part" (white L-shape on black). Shift it by (+12, −7) px → offset within ±1 px.
  - Rotate it by 5° with `angle_range=10` → angle within ±1°.
  - A blank frame gives `found=False`.
  - The 1-axis x mode finds a step edge shifted by 9 px.
- **Done when:** the offset is accurate in the tests, and a frame at 1280×720 with a search area of about 400×300 px runs in under 5 ms with `angle_range=0`.

### [x] T04 `app/vision/total.py`: total judgement
- **IV4:** The Total tool combines the tool results with AND/OR.
- **How it works:**
  - `total_state(results: list[ToolResult], yolo_state: Optional[str], gate="AND", position_found=True) -> "GREEN"|"RED"|"YELLOW"`.
    - If `not position_found` → YELLOW (no part).
    - Ignores disabled tools. The existing YOLO state from `DetectionFilters.judge` counts as one more input when it is not None:
      - YOLO GREEN counts as ok.
      - YOLO RED counts as NG.
      - YOLO YELLOW counts as "no part" → the total is YELLOW.
    - AND: all ok → GREEN, otherwise RED. OR: any ok → GREEN, otherwise RED.
    - No inputs at all → YELLOW.
  - Leave `DetectionFilters.judge` unchanged; this function wraps it.
- **Tests:** a truth table that covers AND/OR, YOLO YELLOW, position not found, and an empty list.
- **Done when:** the truth table passes.

### [x] T05 Saving tools in a job (`app/vision/store.py`, plus `app/context.py`)
- **How it works:**
  - In `vision/store.py`:
    - `tools_to_config(tools, part_no) -> list[dict]`.
    - Template or master images go to `part_configs/tools/<part>/<tool_name>.png` (use the paths module `app/paths.py` and add `TOOLS_DIR`).
    - `tools_from_config(entries, part_no) -> list[Tool]` skips broken entries; it logs them and does not raise.
  - In `AppContext`:
    - `self.tools: list = []`, a signal `tools_changed = Signal(list)`, and `set_tools(tools, save_to_job=False)`.
    - `set_tools` follows the `set_masters` / `_save_to_job` pattern. Note that `_save_to_job` turns lists of lists into lists. Write a dedicated save that stores the list of dicts as they are.
  - Load the tools in the code that loads a job (look in `app/tabs/job_change_tab.py`, where it calls `set_masters` from config). A missing `"tools"` key means `[]`, so old jobs keep working.
- **Tests:** in a temporary `PART_CONFIGS_DIR` (patch it the way `tests/test_autocapture.py` patches its paths):
  - Save 2 tools, one of them with a template image, then load them and compare `to_dict()`.
  - A config with no `"tools"` key loads as `[]`.
- **Done when:** the tools round-trip through a job JSON file plus a PNG template.

---

## Phase 2: Color and brightness

### [x] T06 `app/vision/color.py` `ColorAreaTool` (kind `color_area`)
- **IV4:** Color Area: the area of pixels that match a taught color.
- **How it works:**
  1. Crop the ROI and apply `pre`, then convert to HSV.
  2. Make a mask with `inRange(hsv, lo, hi)`.
     - Hue wraps around: if `h_lo > h_hi`, OR two ranges together, `[h_lo, 179]` and `[0, h_hi]`.
  3. Optionally clean the mask with a morphological open (`open_px`).
  4. `value = countNonZero(mask) / mask.size × 100`.
  5. `teach(frame, point=None)`:
     - Takes HSV samples from `teach_roi`, or from a 9×9 box around `point`.
     - For hue, uses the circular mean and spread; for S and V, mean ± `k`·σ, clamped with minimum widths (h ≥ 8, s ≥ 40, v ≥ 40).
- **Params:** `h_lo, h_hi, s_lo, s_hi, v_lo, v_hi` (default: red 170→10, S 80–255, V 50–255), `k=2.5`, `open_px=0`, `pre=[]`.
- **Value / OK:** % of the ROI; OK if it is within the limits.
- **Overlay:** the ROI rectangle plus a `mask_box`, the bounding box of the matching pixels.
- **Tests:**
  - A red patch filling 25% of the ROI gives 25 ± 0.5%.
  - A red patch whose hue straddles 0/179 is still found.
  - Teaching on a blue patch, then running, gives about 100% on the blue patch and about 0% on a red one.
  - With the frame 20% brighter, a taught range still gives ≥ 90% of the original value.

### [x] T07 `ColorAverageTool` (kind `color_average`)
- **IV4:** Color Average: compares the mean color with the master.
- **How it works:**
  - Convert the crop to Lab, as float, scaled to real L*a*b* (OpenCV 8-bit Lab: L × 100/255, a − 128, b − 128).
  - Take the mean, then the **CIEDE2000** ΔE to the taught mean. Implement it in numpy in a `delta_e_2000(lab1, lab2)` helper; check it against the published test pairs from Sharma 2005 (use 3 of them).
  - `teach` stores the master mean, L*a*b* as 3 floats.
- **Params:** `ref_lab=[L, a, b]`, `pre=[]`. The default limit is `high=5.0`.
- **Value / OK:** ΔE; OK if `≤ high`.
- **Tests:**
  - 3 of Sharma's pairs agree within ±0.01.
  - Teach on a color, then run on the same patch → ΔE < 0.5.
  - A noticeably different color → ΔE > 10.

### [x] T08 `ColorProhibitTool` (kind `color_prohibit`)
- **IV4:** Color Prohibit: NG if a forbidden color shows up anywhere in the ROI.
- **How it works:**
  - Make the same HSV mask as T06, then apply a morphological open with `open_px=2` to remove noise.
  - Run `connectedComponentsWithStats`; the value is the area of the largest blob in px (0 if there is none).
  - The default limit is `high = min_blob_px − 1`, so any blob at or above `min_blob_px` is NG.
- **Params:** the same HSV keys as T06, `open_px=2`, `min_blob_px=30`, `pre=[]`.
- **Overlay:** a red box around each blob of at least `min_blob_px`.
- **Tests:**
  - A clean ROI gives 0 → OK.
  - A 10×10 forbidden spot gives about 100 → NG.
  - Scattered single-pixel noise is removed by the open → OK.

### [x] T09 `app/vision/brightness.py` `AreaTool` (kind `area`)
- **IV4:** Area (mono): the area of pixels above or below a gray threshold.
- **How it works:**
  - Convert the crop to gray and apply `pre`.
  - Threshold it: `method="fixed"` uses `threshold`; `method="otsu"` uses `THRESH_OTSU`.
  - `polarity="bright"` counts pixels above the threshold; `"dark"` counts pixels below it.
  - `value = %`.
- **Params:** `method="fixed"`, `threshold=128`, `polarity="bright"`, `pre=[]`.
- **Tests:**
  - A white square covering 30% of a black ROI gives 30 ± 0.5% for both fixed and Otsu.
  - Dark polarity gives 70%.

### [x] T10 `BrightnessAverageTool` (kind `brightness`)
- **IV4:** Brightness Average.
- **How it works:** take the gray mean of the crop. `extra["std"]` holds the std-dev; when `use_std=True`, the value is the std-dev instead of the mean.
- **Params:** `use_std=False`, `pre=[]`.
- **Tests:**
  - A uniform 100 gray gives a mean of 100 and a std of 0.
  - A half-0/half-200 image gives a mean of 100 and a std of 100.

### [x] T11 `EdgePixelsTool` (kind `edge_pixels`)
- **IV4:** Edge Pixels: counts the pixels on strong brightness edges (scratches, burrs, texture).
- **How it works:**
  - Convert the crop to gray, apply `pre`, and blur with `GaussianBlur(3)`.
  - Run `Canny(lo, hi)`; the value is `countNonZero`. `extra["density"]` holds the % of the ROI.
- **Params:** `canny_lo=50`, `canny_hi=150`, `pre=[]`.
- **Tests:**
  - A flat ROI gives about 0.
  - The outline of a 50×50 square gives about 200 ± 20 edge pixels.
  - Adding a scratch line raises the count by about the line's length.

### [x] T12 `ShadingUniformityTool` (kind `shading`)
- **IV4:** none (our own addition). It finds uneven coating, discoloration, or a weak lamp.
- **How it works:**
  - Convert the crop to gray and split it into a `grid_x × grid_y` grid of cells. Use `cv2.resize(INTER_AREA)` down to grid size; that gives the cell means directly.
  - `value = max − min` of the cell means. `extra` holds the cell matrix, the argmin and the argmax.
  - The overlay marks the darkest and brightest cells.
- **Params:** `grid_x=4`, `grid_y=4`, `pre=[]`. The default limit is `high=20`.
- **Tests:**
  - A uniform ROI gives about 0.
  - A ramp from 50 to 150 across 4 columns gives about 75 (the ramp's own `(150 − 50) × 3/4`); accept ±5.
  - After `pre=[flat_field]`, the ramp drops below 10.

---

## Phase 3: Measurement

### [x] T13 `app/vision/edges.py`: edge-profile core
- **IV4:** The shared engine behind Width, Edge Presence and Pitch.
- **How it works:**
  - `profile(gray_patch, direction="x")` gives the mean along the other axis, so a horizontal scan averages the rows into a 1-D array of length w.
  - `gradient(p, smooth=1.0)`: optional Gaussian smoothing (sigma in px), then `np.gradient`.
  - `find_edges(p, threshold=20, polarity="any"|"rising"|"falling", min_distance=3) -> list[(pos: float, strength: float)]`:
    1. Find the local extrema of the gradient whose |g| ≥ threshold and whose sign matches the polarity.
    2. Refine each to sub-pixel with a parabola through the 3 points: `pos = i + 0.5·(g[i−1] − g[i+1]) / (g[i−1] − 2g[i] + g[i+1])`.
    3. Apply non-maximum suppression with `min_distance`.
  - Rising means dark→light in the scan direction.
- **Tests:**
  - A step at x = 40.0 gives an edge at 40 ± 0.2 (a blurred step at 40.5 → 40.5 ± 0.2).
  - Polarity filtering works.
  - Two edges 2 px apart with `min_distance=3` give 1 edge.

### [x] T14 `app/vision/measure.py` `WidthTool` (kind `width`)
- **IV4:** Width: the distance between two edges.
- **How it works:**
  - Apply `pre` to the crop and build the profile along `direction`.
  - Run `find_edges` and choose the edge pair by `edge_pair`:
    - `"outer"`: the first and last edge.
    - `"strongest"`: the two strongest edges.
    - `"first_rising_falling"`: the first rising edge, then the next falling one.
  - `value = (e2 − e1) × mm_per_px`. If `mm_per_px` is not set, the value is in px.
  - No valid pair → `ok=False`, with `text="edge not found"`.
- **Params:** `direction="x"`, `threshold=20`, `polarity_1="any"`, `polarity_2="any"`, `edge_pair="outer"`, `smooth=1.0`, `pre=[]`.
- **Overlay:** two edge lines plus a dimension line, labeled with the value.
- **Tests:**
  - A white bar 40 px wide gives 40 ± 0.5 px.
  - With 1 px of Gaussian noise (σ = 5) it still gives 40 ± 1.
  - With `mm_per_px = 0.1` → 4.0 mm.
  - A blank ROI gives not found.

### [x] T15 `EdgePresenceTool` (kind `edge_presence`)
- **IV4:** Edge Presence: is there an edge in the scan area?
- **How it works:** take the profile and gradient. The value is the max |gradient| that matches the polarity; OK when `≥ low`, with `low=threshold` by default.
- **Params:** `direction="x"`, `polarity="any"`, `smooth=1.0`, `pre=[]`.
- **Tests:** a step edge gives OK; a flat ROI gives NG; the wrong polarity gives NG.

### [x] T16 `PitchTool` (kind `pitch`)
- **IV4:** Pitch: the number and spacing of repeated edges, such as pins or terminals.
- **How it works:**
  - Find the edges with the chosen polarity (default: rising, so each pin is counted once).
  - Compute the edge-to-edge distances.
  - The value depends on `measure`: `"count"` (the default), `"mean"`, `"min"` or `"max"` pitch × `mm_per_px`.
  - All four are stored in `extra`.
- **Params:** `direction="x"`, `polarity="rising"`, `threshold=20`, `measure="count"`, `smooth=1.0`, `pre=[]`.
- **Tests:**
  - 8 bars with a 20 px pitch give count 8 and mean 20 ± 0.3.
  - With one bar removed, count is 7 and max is about 40.

### [x] T17 `DiameterTool` (kind `diameter`)
- **IV4:** Diameter: the diameter of a round feature.
- **How it works:**
  - Convert the crop to gray, apply `pre`, and run Canny to get the edge points (`np.argwhere`).
  - Fit a circle by **algebraic least squares (Kåsa)**: solve `[x y 1]·[a b c]ᵀ = x² + y²` → center `(a/2, b/2)`, `r = sqrt(c + cx² + cy²)`.
  - Then run 1 RANSAC-style refinement: drop points with |dist − r| > 2 px and fit again.
  - `value = 2r × mm_per_px`. `extra` holds the center and the residual RMS.
  - With fewer than 10 points → not found.
- **Params:** `canny_lo=50`, `canny_hi=150`, `pre=[]`.
- **Overlay:** the fitted circle and its center.
- **Tests:**
  - A filled circle with r = 30 gives a diameter of 60 ± 1.
  - Add 10% outlier pixels → still 60 ± 1.5.
  - A blank ROI gives not found.

### [x] T18 mm calibration (`app/vision/calibration.py`)
- **How it works:**
  - The job-level `mm_per_px: Optional[float]` is saved in the part config next to `"tools"` (extend T05).
  - `calibrate_from_width(frame, roi, known_mm, direction)`:
    1. Runs a temporary `WidthTool`.
    2. Returns `known_mm / px`.
  - Measurement tools read `mm_per_px` from the context and the store passes it in at run time. Do not copy it into each tool.
- **Tests:**
  - Calibrating on a bar 50 px wide with known = 5 mm gives 0.1.
  - After calibration, `WidthTool` reports mm.
  - `mm_per_px` survives a save and load.

---

## Phase 4: Text and codes

### [ ] T19 `app/vision/ocr.py`: OCR read
- **IV4:** AI OCR reads printed text, including angled or shaded print.
- **How it works:**
  - Add `pytesseract` as an optional dependency with a guard in `app/optional_deps.py` (`TESSERACT = None` if the import or the `tesseract` binary is missing). **Do not install system packages.** If tesseract is missing, the tool returns `ok=False, text="OCR engine not installed"`.
  - Pre-processing (always on, in this order):
    1. gray
    2. `pre` (default `[flat_field, clahe]`)
    3. scale up so the character height is ≥ 32 px (`char_height_px` param; scale = 32 / char_height)
    4. binarize with Otsu, auto-inverting so the text ends up dark on light
    5. deskew: `minAreaRect` of the dark pixels, rotating by the angle if |angle| < 15°
  - `read_text(patch) -> (text, conf)` uses tesseract with `--psm 7` (single line) or 6 (block), plus a `whitelist` param passed as `tessedit_char_whitelist`.
  - Keep the engine behind a small `OcrEngine` protocol so PaddleOCR can be added later. Leave a TODO for PaddleOCR.
  - `OcrTool(kind="ocr")`: the value is the mean confidence (0–100); `text` holds the result.
- **Params:** `psm=7`, `whitelist=""`, `char_height_px=0` (0 = auto), `pre=[flat_field, clahe]`, `low=60`.
- **Tests:**
  - The pre-processing is testable without tesseract: `putText("LOT 2026")` on a ramp background, tilted by 5° → the deskew output's tilt is under 1°, and the binarized text is dark on light.
  - The read test is `skipUnless(TESSERACT)`: it reads "LOT 2026" exactly.

### [ ] T20 OCR verify
- **IV4:** OCR checks: does the text match the master, is the label the right way up, is the character count right?
- **How it works:** extend `OcrTool` with these checks:
  - `expected` (exact match) or `pattern` (regex `fullmatch`), with `{date}` placeholders → today's date in `date_format` (default `%d/%m/%y`).
  - `char_count` (0 = off): the count of non-space characters.
  - `check_orientation`: read both the patch and the patch rotated 180°. If the rotated read has higher confidence and matches → NG, with `text="upside down"`.
  - OK only if all enabled checks pass and the confidence is ≥ `low`.
- **Tests:**
  - The string checks work without tesseract: test a `verify(text, conf)` function directly, covering regex, `{date}` substitution and char count.
  - The orientation test is `skipUnless(TESSERACT)`.

### [ ] T21 `app/vision/code.py`: code tool (QR / 1D)
- **IV4:** none (Keyence sells separate code readers). UniversalV1 already uses `pyzbar`.
- **How it works:**
  - Convert the crop to gray, apply `pre`, and decode with `zbar_decode` from `app/optional_deps.py` (the pyzbar guard is already there).
  - If that finds nothing and the code type is QR, fall back to `cv2.QRCodeDetector().detectAndDecode`.
  - Then retry once at 2× scale.
  - Clean the text with `decode_scanned_text` from `app/barcode.py`; Code 39 / AIM extended are handled there.
  - The value is 1 if decoded, otherwise 0. `text` holds the decoded string.
  - OK if decoded and it matches `expected` or `pattern` (either may be empty = anything).
  - `use_job_part_no=True`: expected = the current part number, which the store passes in.
- **Params:** `symbology="any"`, `expected=""`, `pattern=""`, `use_job_part_no=False`, `pre=[]`.
- **Overlay:** the code polygon (from pyzbar `polygon` or the cv2 points) and the decoded text.
- **Tests:**
  - Make a QR with `cv2.QRCodeEncoder` (if your OpenCV has it, otherwise skip) → reads "PART-123".
  - Pattern match and mismatch.
  - A blank ROI → value 0 and NG.

---

## Phase 5: Pattern and count

### [ ] T22 `app/vision/pattern.py` `OutlineTool` (kind `outline`)
- **IV4:** Outline: the % match with the master, based on the part's outline.
- **How it works:**
  - `teach` stores the Canny edges of the master crop, dilated by 1 px; this is the edge template.
  - `mode="template"`:
    - Take the Canny edges of the search crop and run `matchTemplate(edges, template, TM_CCORR_NORMED)`.
    - Value = best score × 100.
    - Works with Position Adjustment because the ROI follows the offset.
  - `mode="shape"`:
    - Find the largest external contour in the crop and the master contour.
    - Value = `max(0, 100 − matchShapes(I1) × 100)`. This is rotation- and scale-invariant.
- **Params:** `mode="template"`, `canny_lo=50`, `canny_hi=150`, `low=70`.
- **Tests:**
  - The same part scores ≥ 95.
  - The part with a notch cut out scores lower than the full part by at least 10.
  - A different shape (circle vs square) scores < 60 in shape mode.
  - In shape mode, rotating the part by 30° still scores ≥ 90.

### [ ] T23 `app/vision/count.py` `BlobCountTool` (kind `blob_count`)
- **IV4:** Blob Count: the number of blobs after binarization.
- **How it works:**
  - Make a binary mask: gray threshold like `AreaTool`, or an HSV mask like `ColorAreaTool`, selected by `source="gray"` or `"color"`.
  - Apply a morphological open, then run `connectedComponentsWithStats`.
  - Keep only blobs that pass these filters:
    - `area_min..area_max`
    - aspect ratio (w/h) `aspect_min..aspect_max`
    - circularity `4πA/P²` ≥ `circ_min`. Get the perimeter with `findContours` on each component's mask, or approximate it.
  - The value is the count. `extra` holds the list of blobs.
- **Params:** `source="gray"`, the gray keys from T09, the HSV keys from T06, `open_px=1`, `area_min=20`, `area_max=1e9`, `aspect_min=0`, `aspect_max=1e9`, `circ_min=0`.
- **Overlay:** a numbered box on each kept blob.
- **Tests:**
  - 7 dots give 7.
  - With `area_min` above the size of the small dots, only the big ones are counted.
  - `circ_min=0.8` keeps the circles and drops the bars.

---

## Phase 6: Integration and UI

### [ ] T24 Worker integration (`app/camera.py`)
- **How it works:**
  - `VideoWorker.set_tools(tools)` (thread-safe: replace the list reference, as `set_roi` does) and `set_mm_per_px`.
  - In `run()`, after the warp (the frame is BGR there), run `run_tools(frame, tools)` in `vision/runner.py`:
    1. Position tools first, which gives the Offset.
    2. Then the other enabled tools with that offset (those with `follow_position=False` get a zero offset).
    3. Each tool is wrapped in try/except. An exception becomes `ToolResult(ok=False, text=str(exc))`, so one bad tool cannot stop the stream.
    4. Time each tool and store it in `extra["ms"]`.
  - Change the signal to `frame_ready = Signal(QImage, list, list)`, the third argument being the tool results.
  - Update **every** `frame_ready.connect` and handler (currently `app/tabs/monitor_tab.py:84`, `app/widgets/barcode_scan.py:193`, `tests/test_reconnect.py:177`; grep again before changing).
  - `AppContext.set_tools` also calls `worker.set_tools`.
- **Tests:**
  - `run_tools` on a synthetic frame with position + width: the width ROI follows a shifted part.
  - A tool that raises gives an NG result, not an exception.
  - The existing `tests/test_reconnect.py` and `tests/test_camera_resolution.py` still pass after the signal change.

### [ ] T25 Monitor integration (`app/tabs/monitor_tab.py`)
- **How it works:**
  - `on_frame(image, detections, tool_results)`:
    - YOLO state = `filters.judge(...)` as today when a model is loaded, otherwise None.
    - Position found = whether the position result is ok, or True when there is no position tool.
    - `raw_state = total_state(tool_results, yolo_state, gate=ctx.tools_gate, position_found=...)`, then it goes through the existing `JudgementLatch` unchanged.
    - With **no tools** in the job, the behaviour must stay exactly as it is today (YOLO only).
  - `_render_frame`: draw the overlay dicts with QPainter.
    - OK is green and NG is red; take the colors from `app/theme.py`, do not hard-code them.
    - Draw labels the same way as the YOLO box labels.
  - Side panel: a compact table with one row per tool: name, value (2 decimals, plus the mm unit if calibrated) and an OK/NG pill.
    - Reuse the existing class-table look.
    - Hide it when there are no tools.
- **Tests:**
  - An offscreen Qt smoke test: build `MonitorTab` with a fake context, call `on_frame` with fake results, and check the banner state.
  - A job with no tools gives the same states as before for the same detections.

### [ ] T26 Tools tab (`app/tabs/tools_tab.py`)
- **Style:** read the `industrial-hmi-ui` skill (`.claude/skills/industrial-hmi-ui/`) and `app/theme.py`. Big touch targets (≥ 48 px) and the same card and button styles as the Settings tab.
- **Layout:**
  - On the left, the frozen camera frame (a `VideoView`-style widget) showing all tool ROIs. The selected tool is highlighted.
  - On the right:
    - The tool list (name, kind, live value, OK/NG pill).
    - The buttons **Add Tool**, **Teach**, **Draw ROI**, **Delete**, and **Gate AND/OR**.
- **Flows:**
  - **Add Tool:** a dialog with a grid of touch tiles, one per kind, grouped as Position / Color / Brightness / Measure / Text & Code / Pattern & Count. Then the user draws the ROI on the frozen frame, reusing the drag-to-draw code from the Monitor ROI mode (`app/widgets/video_view.py`).
  - **Teach:**
    - Calls `tool.teach(current_frame)`.
    - Position and Outline also ask for the template box, drawn as a second rectangle.
    - Color tools allow a tap on a point.
  - **Limits:**
    - Spin boxes or sliders for low and high, with the live value shown next to them.
    - A "Set from current ±%" button: low/high = value × (1 ∓ 10%).
  - **Params:** a simple form built from `tool.params()`: int/float → spin box, bool → toggle, str with known choices → combo.
  - **Delete:** use the existing type-to-confirm dialog (`app/widgets/remove_job_dialog.py`).
  - **Calibrate mm:** a button that asks for the known size and draws a width ROI.
  - Every change calls `ctx.set_tools(tools, save_to_job=True)`.
- **Register** the tab in `app/main_window.py`, between Settings and Job Change.
- **Tests:**
  - An offscreen smoke test: build the tab, add a `ColorAreaTool` programmatically through the tab's API, check that the job JSON has 1 tool, delete it, check 0.
  - Do **not** launch the kiosk or touch the running app.

### [ ] T27 Performance and final check
- `tools/bench_vision.py`: on a 1280×720 synthetic frame, run each tool 200 times (with a typical ROI of 300×200), then print a table of mean and p95 ms per tool and the total for one tool of each kind.
- **Target:** all non-OCR tools together < 10 ms/frame on the Orin Nano. Record the table in `develop_record.md`. If a tool is slower, note it and suggest a fix (pyramid, smaller ROI); do not over-optimize.
- Final: run the full test suite, update `core_function.md` §4 with the real status, and tick every box.
