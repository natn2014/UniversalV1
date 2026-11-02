"""Optional runtime dependencies. Each one may be missing (or fail to import on
a given device); the app degrades gracefully instead of refusing to start."""

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from patlite_lr6 import TowerLight, LED_ON, BUZZER_OFF, BUZZER_ON, BUZZER_PATTERN_NAMES
except Exception:
    TowerLight = None
    BUZZER_OFF = 0x0
    BUZZER_ON = 0x1
    BUZZER_PATTERN_NAMES = [
        ("Off", 0x0), ("Continuous", 0x1), ("Pattern 1", 0x2),
        ("Pattern 2", 0x3), ("Pattern 3", 0x4), ("Pattern 4", 0x5),
    ]

try:
    import torch
except Exception:
    torch = None

try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None

# Tesseract inside the process (tesserocr): the model stays loaded, ~10 ms a
# read instead of ~190 ms for pytesseract starting a new tesseract per call.
# Its wheel brings its own Tesseract but no language data: use the system's.
try:
    import glob

    import tesserocr

    _tessdata = sorted(glob.glob("/usr/share/tesseract-ocr/*/tessdata"))
    TESSDATA = _tessdata[-1] if _tessdata else None
    if TESSDATA is None or "eng" not in tesserocr.get_languages(TESSDATA)[1]:
        raise RuntimeError("no English Tesseract data")
    TESSEROCR = tesserocr
except Exception:
    TESSEROCR = TESSDATA = None

try:
    import pytesseract

    pytesseract.get_tesseract_version()  # raises when the `tesseract` binary is missing
    TESSERACT = pytesseract
except Exception:
    TESSERACT = None
