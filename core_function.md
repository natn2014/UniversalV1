# Image Processing Core Functions (branch: `imageprocessing`)

This is the plan for adding rule-based image processing tools to UniversalV1,
next to the existing YOLO detection. The reference is the **Keyence IV4 vision
sensor**. Each tool below says what IV4 does, how we build it with OpenCV on the
Jetson, and how it fits into the current judgement flow.

> **Sources:** Keyence IV4 product and spec pages, the Keyence vision glossary
> and distributor spec sheets (links at the end). Keyence does not publish a
> full public tool manual, so check parameter details against the IV4 user
> manual before copying any exact behaviour.

---

## 1. What IV4 can do: summary

IV4 has **3 operating modes** and **20 tool types**. One program can hold up to
**65 tool instances**. "65" is the number of slots, not the number of
different functions.

| Mode | Purpose |
|---|---|
| **Standard** | Fixed set of tools, each with an OK/NG threshold, combined into a total judgement |
| **Sorting** | Same tools, but the result is "which registered type is this?" (part-type sorting) |
| **AI Through Count** | Counts objects that pass through the field of view (conveyor) |

### The 20 IV4 tool types by category

| Category | IV4 tool | Your question |
|---|---|---|
| **Position** | Position Adjustment, Hi-Speed Position Adjustment (1-axis / 2-axis) | (pre-processing) |
| **Shape / pattern** | Outline, AI Differentiate, AI Identify | presence / wrong part |
| **Color** | Color Area, Color Average, Color Prohibit | **color checking** ✅ |
| **Brightness** | Area (mono), Brightness Average, Edge Pixels | **shading** ✅ (see §4) |
| **Measurement** | Width, Diameter, Edge Presence, Pitch | **measurement** ✅ |
| **Counting** | Blob Count, AI Count | count |
| **Text** | AI OCR (plus rule-based OCR checks: match to master, orientation, character count) | **OCR** ✅ |
| **Logic / trigger** | Total (AND/OR of tools), AI Trigger (auto-trigger when a part appears) | judgement |

**Answers to the questions:**

- **Measurement:** yes. Width, Diameter, Edge Presence and Pitch are 4 tools.
  They measure in pixels and can be scaled to mm.
- **Color checking:** yes. Color Area, Color Average and Color Prohibit are
  3 tools, with color extraction and exclusion and a color histogram.
- **Shading:** IV4 has no tool called "Shading". Brightness Average, Area and
  Edge Pixels handle light/dark and uneven-surface checks. AI OCR is built to
  read *shaded, angled, smudged or broken* text. We add a shading-correction
  pre-filter ourselves (§4).
- **OCR:** yes, AI OCR. It can learn logos and special characters and can
  check text against a master.
- **QR / barcode:** **no.** IV4 has no code-reading tool; Keyence sells
  separate code readers (SR series) for that. **UniversalV1 already reads
  codes** through `pyzbar` ([app/barcode.py](app/barcode.py)), so on this
  point we already do more than IV4.

---

## 2. Design: how the tools fit into UniversalV1

### 2.1 Pipeline

```
camera frame
  └─ PerspectiveWarp (existing, app/perspective.py)
      └─ Position Adjustment  ── gives an offset (dx, dy, dθ) for every tool ROI
          ├─ Tool 1 (e.g. Color Area)    → value, OK/NG
          ├─ Tool 2 (e.g. Width)         → value, OK/NG
          ├─ YOLO detection (existing)   → OK/NG/STANDBY
          └─ ...
              └─ Total judgement (AND / OR) → GREEN / RED / YELLOW
                  └─ JudgementLatch → tower light, counters, NG log (existing)
```

### 2.2 Common tool interface

Every tool follows the same small contract, so tools stay independent and
are easy to test:

```python
@dataclass
class ToolResult:
    value: float          # the measured number (area %, width px/mm, match %, ...)
    ok: bool              # value inside [low, high]
    overlay: list         # shapes to draw on the Monitor view (boxes, lines, text)
    text: str = ""        # OCR / code tools only

class Tool(Protocol):
    name: str
    roi: Roi              # fractions 0..1, same as app/roi.py
    low: float            # OK if low <= value <= high
    high: float
    def teach(self, master_bgr: np.ndarray) -> None: ...   # register from a master image
    def run(self, frame_bgr: np.ndarray, offset: Offset) -> ToolResult: ...
    def to_dict(self) -> dict: ...                         # saved in part_configs/<job>.json
```

- **ROI** is stored as fractions, as in [app/roi.py](app/roi.py), so a job
  still works after the camera resolution changes.
- **Teach** uses the master images that already exist (`part_configs/masters/`).
- **Saving:** tools are stored in each job's JSON under a new `"tools": [...]`
  key, next to the existing detection filters.
- **Judgement:** the Total tool maps to the existing GREEN / RED / YELLOW
  states. YELLOW (STANDBY) is used when Position Adjustment finds no part.

### 2.3 Proposed files

```
app/vision/
    __init__.py
    base.py          # ToolResult, Tool protocol, Offset, registry
    position.py      # Position Adjustment
    color.py         # Color Area, Color Average, Color Prohibit
    brightness.py    # Area (mono), Brightness Average, Edge Pixels, shading correction
    measure.py       # Width, Diameter, Edge Presence, Pitch
    pattern.py       # Outline
    count.py         # Blob Count
    ocr.py           # OCR (read / verify against master)
    code.py          # QR / DataMatrix / 1D (wraps pyzbar, plus cv2 QR fallback)
    total.py         # AND/OR combination
tests/test_vision_*.py
```

---

## 3. Tool specifications

Each tool has: **IV4 behaviour → OpenCV method → output → OK condition.**

### 3.1 Position Adjustment (do this first; every other tool depends on it)

| | |
|---|---|
| IV4 | Finds the part and moves all tool ROIs with it. Hi-Speed has 1-axis (X or Y) and 2-axis variants. |
| Method | Template matching (`cv2.matchTemplate`, TM_CCOEFF_NORMED) on a taught feature. Rotation: match on an angle pyramid or `cv2.findContours` + `minAreaRect`. 1-axis: edge search along one line. |
| Output | `Offset(dx, dy, angle, score)` |
| OK if | `score >= min_score`; otherwise STANDBY (no part) |

### 3.2 Color tools ✅ color checking

| Tool | IV4 behaviour | OpenCV method | Value |
|---|---|---|---|
| **Color Area** | Area of pixels that match the taught color | HSV `cv2.inRange` with hue wrap-around, `countNonZero` | % of ROI |
| **Color Average** | Average color of the ROI compared with the master | Mean in **Lab**, distance ΔE (CIE76 or CIEDE2000) | ΔE (lower = closer) |
| **Color Prohibit** | NG if a forbidden color appears anywhere | `inRange` for the forbidden color, morphological open, largest blob area | px of forbidden color |

- Teach: tap a point or drag a box on the master → store the HSV range (mean ± k·σ)
  and the Lab mean.
- Use **Lab, not RGB** for Color Average, because it stays stable when the
  brightness changes.
- Wire-harness use: wire color order, connector color, missing colored tape.

### 3.3 Brightness and shading tools ✅ shading

| Tool | IV4 behaviour | OpenCV method | Value |
|---|---|---|---|
| **Area (mono)** | Area of pixels above or below a gray threshold | Grayscale, `threshold` (fixed or Otsu), `countNonZero` | % of ROI |
| **Brightness Average** | Average gray level compared with the master | `cv2.mean` on gray; optional std-dev for uniformity | 0–255 |
| **Edge Pixels** | Counts pixels on strong brightness edges (scratches, burrs, texture) | `Canny` or Sobel magnitude > threshold, `countNonZero` | edge px |

**Shading correction (pre-filter, on by choice per tool).** This removes uneven
lighting before the tools above run:

- **Flat-field:** `corrected = frame / blur(frame, large_kernel) * mean`
  (Gaussian blur, or a morphological close for a dark background).
- **Background subtraction:** `cv2.morphologyEx(MORPH_TOPHAT / BLACKHAT)` for
  small bright or dark defects on a large uneven surface.
- **CLAHE** (`cv2.createCLAHE`) to raise local contrast before OCR.

**Shading uniformity check** (our own addition): split the ROI into an N×M
grid, take the mean of each cell, and report `max - min` (or the std-dev).
NG if it is above the limit. This finds uneven coating, discoloration or a
weak lamp.

### 3.4 Measurement tools ✅ measurement

| Tool | IV4 behaviour | OpenCV method | Value |
|---|---|---|---|
| **Width** | Distance between two edges along a scan direction | Average the profile across the ROI, take the 1-D gradient, choose edges by polarity (light→dark / dark→light), then sub-pixel peak (parabola fit) | px / mm |
| **Diameter** | Diameter of a round feature | Edge points (Canny) → least-squares circle fit (`cv2.fitEllipse`, or algebraic circle fit) or `HoughCircles` | px / mm |
| **Edge Presence** | Is there an edge in the scan region? | Same profile; max gradient > threshold | edge strength |
| **Pitch** | Count and spacing of repeated edges (pins, terminals) | Profile → `scipy`-free peak find (local maxima + min distance) | count, mean pitch, min/max pitch |

- **Calibration:** `mm_per_px` per job, taught by measuring a reference of
  known size. This is only correct after the perspective warp
  ([app/perspective.py](app/perspective.py)), so keep that step first.
- Wire-harness use: strip length, terminal crimp width, pin pitch in a connector.

### 3.5 Pattern tool

| Tool | IV4 behaviour | OpenCV method | Value |
|---|---|---|---|
| **Outline** | % match of the part's outline against the master | Canny edges of master and frame → `matchTemplate` on the edge images, or chamfer distance; `cv2.matchShapes` (Hu moments) for rotation-free checks | match % |

AI Differentiate and AI Identify (good/bad and part-type classification) are
already covered by **YOLO**. If a full-image OK/NG classifier is needed, add a
YOLO-cls model instead of building a new tool.

### 3.6 Counting

| Tool | IV4 behaviour | OpenCV method | Value |
|---|---|---|---|
| **Blob Count** | Number of blobs after binarization | Threshold or color mask → `connectedComponentsWithStats`, filter by area, aspect ratio and circularity | count |

AI Count is the existing YOLO detection count.

### 3.7 OCR ✅

| Function | IV4 behaviour | Our method |
|---|---|---|
| **Read** | Read printed text, including date and lot codes | Pre-process (shading correction → CLAHE → binarize → deskew) → OCR engine |
| **Verify** | Text must equal the master or a pattern | Compare with the expected string or a regex (for example a date format `\d{2}/\d{2}/\d{2}`) |
| **Character count / orientation** | Number of characters; label upside down | Count of character boxes; try 0° and 180° and keep the result with higher confidence |

OCR engine options for the Jetson (choose one in phase 3):

1. **PaddleOCR (PP-OCR, ONNX → TensorRT):** best accuracy on angled or
   damaged print, and runs on the GPU. **Recommended.**
2. **Tesseract (`pytesseract`):** easy to install, CPU only, fine for clean
   printed labels.
3. **Our own YOLO character detector:** trained in the existing Train tab.
   Best when only a fixed character set or font is used (like IV4's
   learned characters).

### 3.8 Code reading (QR / barcode) ✅ (IV4 does not have this)

| Code | Library |
|---|---|
| Code 39 / 128, EAN, ITF, QR | `pyzbar` (already used in [app/barcode.py](app/barcode.py)) |
| QR (fallback, handles damage better) | `cv2.QRCodeDetector` / `cv2.QRCodeDetectorAruco` |
| DataMatrix | `pylibdmtx` (needs `libdmtx0b`); add only if needed |

Tool output: the decoded text. OK if it matches the job's expected value or
regex. Optional: a grade for print quality, from the decode rate over N frames.

### 3.9 Total

AND / OR over all tools plus YOLO. This generalizes the current
`target_gate` in [app/judgement.py](app/judgement.py). Any tool NG → RED,
all OK → GREEN, no part → YELLOW.

---

## 4. Plan by phase

| Phase | Scope | Done when |
|---|---|---|
| **1. Core** | `app/vision/base.py`, Position Adjustment, Total, JSON save and load, Monitor overlay | One job runs YOLO + 1 rule-based tool with a combined result |
| **2. Color + brightness** | Color Area, Color Average, Color Prohibit, Area, Brightness Average, Edge Pixels, shading correction | Wire color check works under lamp changes of ±20% |
| **3. Measurement** | Width, Edge Presence, Pitch, Diameter, mm calibration | Repeatability ≤ ±1 px on a fixed part (100 frames) |
| **4. Text + codes** | OCR read/verify, code tool (QR / 1D) inside a job | Reads the date or lot label at ≥ 99% over 200 frames |
| **5. Pattern + count** | Outline, Blob Count | — |
| **6. UI** | "Tools" tab: add a tool, draw the ROI, teach, set the low/high slider with a live value (industrial-hmi-ui style) | An operator can build a job without editing JSON |

Performance target: all rule-based tools together **< 10 ms per frame**
at 1280×720 on the Orin Nano (CPU OpenCV), so YOLO stays the bottleneck.

---

## 5. Testing

- Each tool gets a unit test with **synthetic images** (drawn rectangles,
  circles, color patches, `cv2.putText` text) so the expected value is known
  exactly. This follows the style of `tests/test_autocapture.py`.
- Robustness tests: add brightness change, Gaussian noise, a 1–3 px shift and
  a ±5° rotation, and check that the result stays OK.
- Regression set: real NG images from the Log tab (`ng_log`) with an
  expected result per image.

---

## Sources

- [Keyence IV4 product page](https://www.keyence.com/products/vision/vision-sensor/iv4/)
- [Keyence IV4 specs](https://keyence.com/products/sensor/vision-sensor/iv4/specs)
- [Keyence vision sensor glossary (Outline, Edge Pixels, Position Adjustment)](https://www.keyence.com/ss/products/vision/iv-casestudy/glossary/)
- [Keyence UK: IV4 AI OCR](https://www.keyence.co.uk/landing/lpc/lp_ppc_ai_ocr_vision_sensor.jsp)
- [Keyence UK: IV4 OCR camera](https://www.keyence.co.uk/landing/lpc/lp_ppc_iv4_ocr_camera.jsp)
- [IV4-G120 tool list (distributor spec sheet)](https://plc-direct.com/products/iv4-g120)
- [Vision Systems Design: IV4 launch](https://www.vision-systems.com/cameras-accessories/image-sensors/article/55279655/keyence-launches-new-sensor-series-with-built-in-ai)
