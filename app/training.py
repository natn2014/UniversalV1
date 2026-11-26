from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from .optional_deps import YOLO
from .paths import ROOT_DIR

# A new pose (orientation) model starts from this, kept in weights/. When it
# is missing, Ultralytics downloads it there once (a full path makes it save
# there, not in the current folder); offline, copy the file in by hand.
POSE_BASE_MODEL = "yolo11n-pose.pt"


# Memory needed to train, by batch size, measured on the Orin (7.5 GB shared
# by CPU and GPU) at 640 px: the process itself takes ~2.5 GB at batch 2-4,
# and validation runs at twice the batch, so batch 8 needs more. Free memory
# (MemAvailable + SwapFree) below these made the kernel's OOM killer end the
# run (2026-11-25, batch 8 at the first validation, with VS Code and Firefox
# open). Each value includes ~0.5 GB of margin.
BATCH_NEEDS_MB = ((8, 4000), (4, 3000), (2, 2600))
MEMINFO = "/proc/meminfo"


def free_memory_mb(meminfo: str = MEMINFO) -> Optional[int]:
    """MemAvailable + SwapFree in MB; None where /proc/meminfo is missing."""
    try:
        values = {}
        with open(meminfo) as f:
            for line in f:
                key, rest = line.split(":", 1)
                values[key] = int(rest.split()[0])
        return (values["MemAvailable"] + values.get("SwapFree", 0)) // 1024
    except (OSError, KeyError, ValueError):
        return None


def batch_for_memory(requested: int, imgsz: int, free_mb: Optional[int]) -> Optional[int]:
    """The largest batch, up to `requested`, that fits in `free_mb`; None
    when even the smallest does not. Needs grow with the image area."""
    if free_mb is None:
        return requested
    area = (imgsz / 640.0) ** 2
    for batch, need in BATCH_NEEDS_MB:
        if batch <= requested and free_mb >= 2000 + (need - 2000) * area:
            return batch
    return None


def pose_base_model() -> str:
    return str(ROOT_DIR / "weights" / POSE_BASE_MODEL)


def model_fits_pose(model, k: int) -> bool:
    """The model is a pose model with k keypoints, so training can carry on from it."""
    if getattr(model, "task", None) != "pose":
        return False
    shape = None
    try:
        shape = model.model.yaml.get("kpt_shape") or getattr(model.model, "kpt_shape", None)
    except AttributeError:
        pass
    return shape is None or int(shape[0]) == int(k)


class TrainingWorker(QThread):
    status_update = Signal(str)
    training_complete = Signal(str)
    training_error = Signal(str)

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    def _load_model(self):
        """The model to train from. Pose: the chosen model only if it is a
        pose model with the dataset's keypoints, else the pose base model."""
        model_path: Optional[str] = self.config.get("model_path")
        if self.config.get("task") != "pose":
            return YOLO(model_path)
        k = int(self.config.get("kpt_count", 2))
        if model_path:
            try:
                model = YOLO(model_path)
                if model_fits_pose(model, k):
                    return model
            except Exception:
                pass
        base = self.config.get("fallback_model") or pose_base_model()
        self.status_update.emit(f"Starting from {Path(base).name} (downloaded once if missing)...")
        try:
            return YOLO(base)
        except Exception as exc:
            raise RuntimeError(
                f"could not load {Path(base).name} ({exc}). Without internet, copy "
                f"{POSE_BASE_MODEL} into the weights folder.") from exc

    def run(self) -> None:
        model = results = None
        try:
            if YOLO is None:
                self.training_error.emit("Ultralytics not available.")
                return

            requested = int(self.config["batch"])
            free_mb = free_memory_mb()
            batch = batch_for_memory(requested, int(self.config["imgsz"]), free_mb)
            if batch is None:
                self.training_error.emit(
                    f"Not enough free memory to train ({free_mb} MB). Close other programs "
                    f"(browser, editor) or stop monitoring, then start again.")
                return
            if batch < requested:
                self.status_update.emit(f"Free memory {free_mb} MB: training with batch {batch} instead of {requested}.")
            self.config["batch"] = batch

            model = self._load_model()

            def _on_epoch_end(trainer):
                try:
                    self.status_update.emit(f"Epoch {trainer.epoch + 1}/{trainer.epochs}")
                except Exception:
                    pass

            model.add_callback("on_train_epoch_end", _on_epoch_end)

            self.status_update.emit("Training started...")
            results = model.train(
                data=self.config["data_yaml"],
                epochs=self.config["epochs"],
                imgsz=self.config["imgsz"],
                batch=self.config["batch"],
                device=self.config["device"],
                project=self.config["project"],
                name=self.config["name"],
                patience=self.config["patience"],
                optimizer=self.config["optimizer"],
                lr0=self.config["lr0"],
                verbose=True,
                plots=False,
                val=True,
                amp=True,
                workers=0,
                # Augmentation preset (app/augmentation.py); always includes
                # fliplr/flipud so Ultralytics' default mirroring can't apply.
                **self.config.get("augment", {}),
            )
            best_path = Path(results.save_dir) / "weights" / "best.pt"
            self.training_complete.emit(str(best_path))
        except Exception as exc:
            self.training_error.emit(f"Training failed: {exc}")
        finally:
            model = results = None  # noqa: F841  (drop the last references before collecting)
            release_training_memory()


def release_training_memory() -> None:
    """Gives back what training held (the model, PyTorch's cached GPU blocks).
    On the Jetson CPU and GPU share one RAM: kept, it starved the next model
    loaded in this process (monitoring, or a second training run)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
