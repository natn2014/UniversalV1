from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .optional_deps import YOLO


class TrainingWorker(QThread):
    status_update = Signal(str)
    training_complete = Signal(str)
    training_error = Signal(str)

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            if YOLO is None:
                self.training_error.emit("Ultralytics not available.")
                return

            model = YOLO(self.config["model_path"])

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
            )
            best_path = Path(results.save_dir) / "weights" / "best.pt"
            self.training_complete.emit(str(best_path))
        except Exception as exc:
            self.training_error.emit(f"Training failed: {exc}")
