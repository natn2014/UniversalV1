"""Timing of every vision tool on a 1280x720 synthetic frame.

Runs each tool RUNS times on a 300x200 px ROI (the position tool gets the
same ROI as its search area), prints mean / p95 ms per tool and the total for
one tool of each kind, with and without OCR. Then simulates the camera loop
(VideoWorker._run_vision) at 24 fps with every tool, OCR and code running in
the slow-tool thread, and reports the loop's frame rate and the slow read rate.

    .venv/bin/python tools/bench_vision.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vision import TOOL_KINDS  # noqa: E402
from app.vision.base import Offset, create  # noqa: E402
from app.vision.ocr import get_engine  # noqa: E402

W, H, RUNS = 1280, 720, 200
ROI_W, ROI_H = 300, 200
ROI = (500 / W, 250 / H, (500 + ROI_W) / W, (250 + ROI_H) / H)


def scene() -> np.ndarray:
    """Textured background, bars (width / pitch), a disc, a red patch, QR-ish text and a locator shape."""
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 70, np.uint8)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    for i in range(8):
        cv2.rectangle(img, (520 + i * 34, 270), (540 + i * 34, 330), (210, 210, 210), -1)
    cv2.circle(img, (620, 410), 40, (230, 230, 230), -1)
    cv2.rectangle(img, (700, 380), (780, 430), (0, 0, 220), -1)
    cv2.putText(img, "LOT 2026", (520, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def make_tools(frame: np.ndarray) -> list:
    tools = []
    for kind in sorted(TOOL_KINDS):
        if kind.startswith("_"):
            continue
        kw = {"name": kind, "roi": ROI}
        if kind == "position":
            kw["template_roi"] = (540 / W, 270 / H, 700 / W, 380 / H)
        tool = create(kind, **{k: v for k, v in kw.items() if k != "template_roi"})
        if kind == "position":
            tool.teach(frame, template_roi=kw["template_roi"])
        elif kind in ("color_average", "outline", "color_area"):
            tool.teach(frame)
        tools.append(tool)
    return tools


def bench(tool, frame, runs: int) -> tuple:
    offset = Offset()
    tool.run(frame, offset)  # warm-up: caches, allocations
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tool.run(frame, offset)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.percentile(times, 95))


def main() -> None:
    frame = scene()
    rows = []
    for tool in make_tools(frame):
        runs = 20 if tool.kind == "ocr" else RUNS
        if tool.kind == "ocr" and get_engine() is None:
            continue
        rows.append((tool.kind, *bench(tool, frame, runs)))
    print(f"{W}x{H} frame, ROI {ROI_W}x{ROI_H}, {RUNS} runs per tool (OCR: 20)\n")
    print(f"{'tool':<16}{'mean ms':>10}{'p95 ms':>10}")
    for kind, mean, p95 in rows:
        print(f"{kind:<16}{mean:>10.2f}{p95:>10.2f}")
    non_ocr = [r for r in rows if r[0] != "ocr"]
    print(f"\n{'total, no OCR':<16}{sum(r[1] for r in non_ocr):>10.2f}{sum(r[2] for r in non_ocr):>10.2f}")
    if len(non_ocr) != len(rows):
        print(f"{'total, with OCR':<16}{sum(r[1] for r in rows):>10.2f}{sum(r[2] for r in rows):>10.2f}")

    worker_loop(frame)


def worker_loop(frame: np.ndarray, fps: float = 24.0, seconds: float = 5.0) -> None:
    from app.camera import VideoWorker

    worker = VideoWorker()
    worker.set_slow_hz(2.0)
    worker.set_tools(make_tools(frame))
    period = 1.0 / fps
    frames = reads = 0
    last_age = None
    loop_ms = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        t0 = time.perf_counter()
        results = worker._run_vision(frame, worker._tools)
        loop_ms.append((time.perf_counter() - t0) * 1000.0)
        age = next((r.extra.get("age_ms") for r in results if r.extra.get("slow")), None)
        if age is not None and (last_age is None or age < last_age):
            reads += 1
        last_age = age
        frames += 1
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    worker._slow.stop()
    print(f"\nworker loop, every tool, slow tools at 2 Hz in their own thread, {seconds:.0f} s at {fps:.0f} fps target:")
    print(f"  loop fps {frames / seconds:.1f}   tools per frame mean {np.mean(loop_ms):.2f} ms"
          f" / p95 {np.percentile(loop_ms, 95):.2f} ms   slow reads {reads / seconds:.1f} /s")


if __name__ == "__main__":
    main()
