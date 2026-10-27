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

try:
    import pytesseract

    pytesseract.get_tesseract_version()  # raises when the `tesseract` binary is missing
    TESSERACT = pytesseract
except Exception:
    TESSERACT = None
