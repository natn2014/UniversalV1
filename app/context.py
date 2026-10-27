from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from . import jobs
from .autocapture import load_masters
from .camera import VideoWorker, parse_resolution
from .judgement import DetectionFilters
from .optional_deps import YOLO
from .paths import TOOLS_DIR
from .perspective import normalize_quad
from .roi import normalize_roi
from .settings import AppSettings
from .tower import TowerController
from .vision.store import tools_from_config, tools_to_config


class AppContext(QObject):
    """State shared between the tabs. Tabs read it directly and react to its
    signals instead of reaching into each other."""

    status = Signal(str)
    part_changed = Signal(object)  # Optional[str]
    model_path_changed = Signal(object)  # Optional[Path]
    model_classes_changed = Signal(list)
    roi_changed = Signal(object)  # Optional[Roi]
    perspective_changed = Signal(object)  # Optional[Quad]
    masters_changed = Signal(list)  # [Master]
    tools_changed = Signal(list)  # [vision.base.Tool]
    mm_per_px_changed = Signal(object)  # Optional[float]
    # Requests one tab makes of another, so a missing step can be done for
    # the user instead of telling them to go and do it.
    camera_scan_requested = Signal()  # rescan; handled synchronously
    model_pick_requested = Signal()  # open the .pt picker; synchronous
    stop_stream_requested = Signal()
    show_tab_requested = Signal(str)  # tab title, e.g. "Job Change"

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.settings = AppSettings()
        self.worker = VideoWorker()
        self.worker.set_resolution(parse_resolution(self.settings["camera_resolution"]))
        self.tower = TowerController(self.settings, self)
        self.filters = DetectionFilters()
        self.model_classes: list = []
        self.camera_index: Optional[int] = None
        self.current_part_no: Optional[str] = None
        self.training_active: bool = False
        self.masters: list = load_masters(None)
        self.tools: list = []  # rule-based vision tools (app/vision)
        self.tools_gate: str = "AND"
        self.mm_per_px: Optional[float] = None  # job-level calibration

    @property
    def model_path(self) -> Optional[Path]:
        return self.worker._model_path

    def load_model(self, model_path: Path) -> None:
        self.worker.set_model_path(model_path)
        self.model_path_changed.emit(model_path)
        self.status.emit("Loading model...")

        # Load model to extract class names
        if YOLO is not None:
            try:
                model = YOLO(str(model_path))
                self.model_classes = list(model.names.values())
                self.model_classes_changed.emit(self.model_classes)
                self.status.emit(f"Model loaded: {len(self.model_classes)} classes")
            except Exception as exc:
                self.status.emit(f"Failed to load model: {exc}")
                self.model_classes = []
        else:
            self.status.emit("Ultralytics not available.")
            self.model_classes = []

    def set_perspective(self, quad, save_to_job: bool = False) -> None:
        """Sets the top-view corners (None = camera view as is). A changed
        top view also clears the ROI: its fractions refer to the top view,
        whose shape just changed."""
        quad = normalize_quad(quad)
        changed = quad != self.filters.perspective
        self.filters.perspective = quad
        self.worker.set_perspective(quad)
        self.perspective_changed.emit(quad)
        if changed and save_to_job and self.filters.roi is not None:
            self.set_roi(None)
            self.status.emit("Top view changed: ROI cleared, set it again on the new view.")
        self._save_to_job(save_to_job, perspective=quad, roi=self.filters.roi)

    def _save_to_job(self, save: bool, **values) -> None:
        """Writes the given keys into the current job's saved config."""
        if not save or not self.current_part_no:
            return
        config = jobs.load_part_config(self.current_part_no)
        if config is None:
            return
        for key, value in values.items():
            config[key] = [list(v) if isinstance(v, (list, tuple)) else v for v in value] if value else None
        jobs.save_part_config(self.current_part_no, config)

    def set_roi(self, roi, save_to_job: bool = False) -> None:
        """Sets the detection area (None = whole frame). save_to_job also
        writes it into the current job's saved config right away."""
        roi = normalize_roi(roi)
        self.filters.roi = roi
        self.worker.set_roi(roi)
        self.roi_changed.emit(roi)
        self._save_to_job(save_to_job, roi=roi)

    def set_masters(self, entries, save_to_job: bool = False) -> None:
        """Sets the auto-capture master images from config entries
        ({"class_name", "image"} each)."""
        self.masters = load_masters(entries)
        self.masters_changed.emit(self.masters)
        self._save_to_job(save_to_job, autocapture_masters=self.master_entries())

    def set_tools(self, tools, save_to_job: bool = False) -> None:
        """Sets the vision tools. Saved as a list of dicts (plus template PNGs
        under TOOLS_DIR), not through `_save_to_job`, which flattens lists."""
        self.tools = list(tools)
        self.tools_changed.emit(self.tools)
        if not save_to_job or not self.current_part_no:
            return
        config = jobs.load_part_config(self.current_part_no)
        if config is None:
            return
        config["tools"] = self.tool_entries()
        config["tools_gate"] = self.tools_gate
        config["mm_per_px"] = self.mm_per_px
        jobs.save_part_config(self.current_part_no, config)

    def set_mm_per_px(self, value: Optional[float], save_to_job: bool = False) -> None:
        self.mm_per_px = float(value) if value else None
        self.mm_per_px_changed.emit(self.mm_per_px)
        if save_to_job and self.current_part_no:
            config = jobs.load_part_config(self.current_part_no)
            if config is not None:
                config["mm_per_px"] = self.mm_per_px
                jobs.save_part_config(self.current_part_no, config)

    def tool_entries(self) -> list:
        return tools_to_config(self.tools, self.current_part_no or "unknown_part", TOOLS_DIR)

    def load_tools(self, config: dict) -> None:
        """Applies a saved job's tools; a job without them has none."""
        part_no = config.get("part_no") or self.current_part_no or "unknown_part"
        self.tools_gate = config.get("tools_gate", "AND")
        self.set_mm_per_px(config.get("mm_per_px"))
        self.set_tools(tools_from_config(config.get("tools"), part_no, TOOLS_DIR))

    def master_entries(self) -> list:
        return [m.to_entry() for m in self.masters]

    def set_current_part(self, part_no: Optional[str]) -> None:
        """None drops the part association (and the boot auto-load); the loaded
        model and filters stay active until another job is chosen."""
        self.current_part_no = part_no
        if part_no is None:
            jobs.clear_last_part()
        else:
            jobs.write_last_part(part_no)
        self.part_changed.emit(part_no)
