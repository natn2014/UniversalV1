from pathlib import Path
from typing import Optional

import yaml
from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..autocapture import autocapture_images_dir
from ..context import AppContext
from ..marking.classes import read_classes_file, validate_class_name, write_classes_file
from ..optional_deps import torch
from ..paths import CAPTURES_IMAGES_DIR, IMAGE_EXTENSIONS
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon
from ..training import TrainingWorker
from ..widgets.annotated_image import AnnotatedImageWidget
from ..widgets.layout import bottom_bar, top_bar
from ..yolo_labels import pixel_box_to_yolo, read_yolo_label_file, write_yolo_label_file, yolo_box_to_pixel


class TrainTab(QWidget):
    """Review captured images, draw / delete / correct boxes, and fine-tune the
    model. A dataset with a classes.txt is labelled and trained with those
    classes instead of the loaded model's."""

    def __init__(self, ctx: AppContext, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._train_pairs: list = []
        self._train_current_index: int = -1
        self._train_boxes: list = []
        self._train_selected_box_index: int = -1
        self._dataset_classes: Optional[list] = None
        self._dataset_root: Optional[Path] = None
        self._training_worker: Optional[TrainingWorker] = None

        self.train_model_label = QLabel("No model selected")
        ctx.model_path_changed.connect(self._sync_train_model_label)
        self.train_select_folder_button = QPushButton("Select Folder...")
        self.train_select_folder_button.clicked.connect(self._select_train_folder)
        self.train_folder_label = QLabel("(none selected)")

        self.train_list_widget = QListWidget()
        self.train_list_widget.currentRowChanged.connect(self._load_train_index)
        # Capture names are long; elide the left so the date and time stay visible.
        self.train_list_widget.setTextElideMode(Qt.ElideLeft)
        self.train_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.train_prev_button = QPushButton("◀ Prev")
        self.train_prev_button.clicked.connect(self._train_prev)
        self.train_next_button = QPushButton("Next ▶")
        self.train_next_button.clicked.connect(self._train_next)
        self.train_counter_label = QLabel("Image 0 / 0")

        self.train_image_widget = AnnotatedImageWidget()
        self.train_image_widget.box_clicked.connect(self._on_train_box_clicked)
        self.train_image_widget.box_drawn.connect(self._on_train_box_drawn)

        self.train_selected_label = QLabel("Selected box: none")
        self.train_class_combo = QComboBox()
        ctx.model_classes_changed.connect(self._populate_class_combo)
        self.train_confirm_button = QPushButton("Confirm")
        self.train_confirm_button.clicked.connect(self._on_train_confirm)
        self.train_draw_button = QPushButton("Draw Box")
        self.train_draw_button.setCheckable(True)
        self.train_draw_button.toggled.connect(self.train_image_widget.set_draw_mode)
        self.train_delete_button = QPushButton("Delete Box")
        self.train_delete_button.clicked.connect(self._on_train_delete)
        self.add_class_button = QPushButton("+ Add Class")
        self.add_class_button.clicked.connect(self._on_add_class)
        self.rename_class_button = QPushButton("Rename")
        self.rename_class_button.clicked.connect(self._on_rename_class)

        self.train_start_button = QPushButton("Start Training")
        self.train_start_button.setIcon(make_icon("play"))
        self.train_start_button.setIconSize(QSize(20, 20))
        self.train_start_button.setFixedHeight(44)
        self.train_start_button.setMinimumWidth(190)
        self.train_start_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.train_start_button.clicked.connect(self._start_training)
        self.train_status_label = QLabel("Ready")
        # Long training messages must not widen the bottom bar.
        self.train_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.train_status_label.setStyleSheet("color: #c7cdd4;")

        self._build_layout()

    def _build_layout(self) -> None:
        # Same frame as the Monitor tab: info strip on top, framed image with a
        # light side panel, action bar with the main action bottom right.
        top, top_layout = top_bar()
        model_caption = QLabel("Active Model:")
        model_caption.setObjectName("monitorInfo")
        top_layout.addWidget(model_caption)
        top_layout.addWidget(self.train_model_label)
        top_layout.addSpacing(24)
        dataset_caption = QLabel("Dataset:")
        dataset_caption.setObjectName("monitorInfo")
        top_layout.addWidget(dataset_caption)
        top_layout.addWidget(self.train_folder_label, 1)
        top_layout.addWidget(self.train_select_folder_button)

        image_frame = QFrame()
        image_frame.setObjectName("videoFrame")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(2, 2, 2, 2)
        image_layout.addWidget(self.train_image_widget)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_panel.setFixedWidth(260)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)

        images_header = QLabel("Images")
        images_header.setObjectName("panelHeader")
        side_layout.addWidget(images_header)
        side_layout.addWidget(self.train_list_widget, 1)

        correction_header = QLabel("Box Editing")
        correction_header.setObjectName("panelHeader")
        side_layout.addWidget(correction_header)
        correction_box = QWidget()
        correction_box.setObjectName("counterBox")
        correction_layout = QVBoxLayout(correction_box)
        correction_layout.setContentsMargins(8, 8, 8, 8)
        correction_layout.setSpacing(6)
        self.train_selected_label.setObjectName("panelText")
        self.train_selected_label.setWordWrap(True)
        class_caption = QLabel("Class (new box / change to):")
        class_caption.setObjectName("panelText")
        correction_layout.addWidget(self.train_selected_label)
        correction_layout.addWidget(class_caption)
        correction_layout.addWidget(self.train_class_combo)
        class_row = QHBoxLayout()
        class_row.addWidget(self.add_class_button)
        class_row.addWidget(self.rename_class_button)
        correction_layout.addLayout(class_row)
        classes_hint = QLabel("Classes are saved with this dataset (classes.txt).")
        classes_hint.setObjectName("panelText")
        classes_hint.setWordWrap(True)
        classes_hint.setStyleSheet("font-size: 9pt; color: #4a525c;")
        correction_layout.addWidget(classes_hint)
        correction_layout.addWidget(self.train_confirm_button)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.train_draw_button)
        edit_row.addWidget(self.train_delete_button)
        correction_layout.addLayout(edit_row)
        side_layout.addWidget(correction_box)

        content_row = QHBoxLayout()
        content_row.addWidget(image_frame, 1)
        content_row.addWidget(side_panel)

        bottom, bottom_layout = bottom_bar()
        bottom_layout.addWidget(self.train_prev_button)
        bottom_layout.addWidget(self.train_counter_label)
        bottom_layout.addWidget(self.train_next_button)
        bottom_layout.addSpacing(12)
        bottom_layout.addWidget(self.train_status_label, 1)
        bottom_layout.addSpacing(8)
        bottom_layout.addWidget(self.train_start_button)

        layout = QVBoxLayout(self)
        layout.addWidget(top)
        layout.addLayout(content_row, 1)
        layout.addWidget(bottom)

    def _sync_train_model_label(self, model_path: Optional[Path]) -> None:
        if model_path is not None:
            self.train_model_label.setText(model_path.name)
        else:
            self.train_model_label.setText("No model selected")

    @property
    def _classes(self) -> list:
        """The dataset's classes.txt if it has one, else the model's classes."""
        return self._dataset_classes or self._ctx.model_classes

    @Slot(list)
    def _populate_class_combo(self, model_classes: list) -> None:
        if self._dataset_classes and model_classes is not self._dataset_classes:
            return  # the dataset's classes.txt wins over a model change
        self.train_class_combo.blockSignals(True)
        self.train_class_combo.clear()
        for class_name in sorted(model_classes):
            self.train_class_combo.addItem(class_name)
        self.train_class_combo.blockSignals(False)

    # --- Class list (add / rename) ---------------------------------------

    def _ensure_dataset(self) -> bool:
        """Classes are saved with a dataset: pick one first if none is open."""
        if self._dataset_root is None:
            self.train_status_label.setText("Choose the dataset folder (its images folder).")
            self._select_train_folder()
        return self._dataset_root is not None

    def _ask_class_name(self, title: str, text: str, others: list) -> Optional[str]:
        """Asks until the name is valid against `others`; None if cancelled."""
        while True:
            name, ok = QInputDialog.getText(self, title, "Class name:", text=text)
            if not ok:
                return None
            problem = validate_class_name(name, others)
            if problem is None:
                return name.strip()
            self.train_status_label.setText(problem)
            text = name

    def _write_classes(self, names: list, select: str) -> None:
        """Saves the class list to classes.txt and shows it everywhere. IDs
        are positions, so existing label files keep their meaning."""
        write_classes_file(self._dataset_root, names)
        self._dataset_classes = names
        self._populate_class_combo(names)
        self.train_class_combo.setCurrentText(select)
        for box in self._train_boxes:
            if 0 <= box["cls_id"] < len(names):
                box["class_name"] = names[box["cls_id"]]
        if self._train_current_index >= 0:
            self.train_image_widget.set_image(self.train_image_widget.pixmap(), self._train_boxes)
            self.train_image_widget.set_selected_index(self._train_selected_box_index)
            if self._train_selected_box_index >= 0:
                box = self._train_boxes[self._train_selected_box_index]
                self.train_selected_label.setText(f"Selected box: {box['class_name']}")

    @Slot()
    def _on_add_class(self) -> None:
        if not self._ensure_dataset():
            return
        names = list(self._classes)  # model classes in ID order if no classes.txt yet
        name = self._ask_class_name("Add Class", "", names)
        if name is None:
            return
        self._write_classes(names + [name], select=name)
        self.train_status_label.setText(
            f'Added class "{name}" (id {len(names)}) — saved to classes.txt; retrain to use it.')

    @Slot()
    def _on_rename_class(self) -> None:
        names = list(self._classes)
        old = self.train_class_combo.currentText()
        if old not in names:
            self.train_status_label.setText("Add a class first.")
            self._on_add_class()
            return
        if not self._ensure_dataset():
            return
        index = names.index(old)
        name = self._ask_class_name("Rename Class", old, names[:index] + names[index + 1:])
        if name is None or name == old:
            return
        names[index] = name
        self._write_classes(names, select=name)
        self.train_status_label.setText(f'Renamed "{old}" → "{name}" — labels unchanged.')

    # --- Dataset browsing and box correction ----------------------------

    @Slot()
    def _select_train_folder(self) -> None:
        # The job's auto captures first: they are what is waiting to be labelled.
        auto_dir = autocapture_images_dir(self._ctx.current_part_no) if self._ctx.current_part_no else None
        if auto_dir is not None and auto_dir.is_dir():
            default_dir = str(auto_dir)
        else:
            default_dir = str(CAPTURES_IMAGES_DIR) if CAPTURES_IMAGES_DIR.exists() else ""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", default_dir)
        if folder:
            self.scan_train_folder(Path(folder))

    def scan_train_folder(self, images_dir: Path) -> None:
        labels_dir = images_dir.parent / "labels"
        # Last two path parts fit the top strip; full path in the tooltip.
        self.train_folder_label.setText(str(Path(*images_dir.parts[-2:])))
        self.train_folder_label.setToolTip(str(images_dir))

        self._dataset_root = images_dir.parent
        self._dataset_classes = read_classes_file(images_dir.parent)
        self._populate_class_combo(self._classes)

        # Unlabelled images are listed too; their label file is written on
        # the first box edit.
        pairs = []
        unlabelled = 0
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            pairs.append((image_path, label_path))
            unlabelled += not label_path.exists()

        self._train_pairs = pairs
        self.train_list_widget.blockSignals(True)
        self.train_list_widget.clear()
        for image_path, _ in pairs:
            self.train_list_widget.addItem(image_path.name)
        self.train_list_widget.blockSignals(False)

        classes_source = "classes.txt" if self._dataset_classes else "model classes"
        if pairs:
            self.train_status_label.setText(
                f"Loaded {len(pairs)} images ({unlabelled} unlabelled), {classes_source}"
            )
            self._load_train_index(0)
        else:
            self._train_current_index = -1
            self.train_image_widget.set_image(QPixmap(), [])
            self.train_counter_label.setText("Image 0 / 0")
            self.train_status_label.setText("No images found in this folder")

    def _load_train_index(self, index: int) -> None:
        if index < 0 or index >= len(self._train_pairs):
            return
        self._train_current_index = index
        image_path, label_path = self._train_pairs[index]
        model_classes = self._classes

        pixmap = QPixmap(str(image_path))
        img_w, img_h = pixmap.width(), pixmap.height()

        boxes = []
        for box in read_yolo_label_file(label_path):
            x1, y1, x2, y2 = yolo_box_to_pixel(box, img_w, img_h)
            cls_id = box["cls_id"]
            class_name = (
                model_classes[cls_id]
                if 0 <= cls_id < len(model_classes)
                else str(cls_id)
            )
            boxes.append(
                {
                    "cls_id": cls_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_name": class_name,
                }
            )

        self._train_boxes = boxes
        self._train_selected_box_index = -1
        self.train_image_widget.set_image(pixmap, boxes)
        self.train_counter_label.setText(f"Image {index + 1} / {len(self._train_pairs)}")
        self.train_selected_label.setText("Selected box: none")

        self.train_list_widget.blockSignals(True)
        self.train_list_widget.setCurrentRow(index)
        self.train_list_widget.blockSignals(False)

    @Slot()
    def _train_prev(self) -> None:
        if self._train_current_index > 0:
            self._load_train_index(self._train_current_index - 1)

    @Slot()
    def _train_next(self) -> None:
        if self._train_current_index < len(self._train_pairs) - 1:
            self._load_train_index(self._train_current_index + 1)

    @Slot(int)
    def _on_train_box_clicked(self, index: int) -> None:
        if index < 0 or index >= len(self._train_boxes):
            return
        self._train_selected_box_index = index
        box = self._train_boxes[index]
        self.train_selected_label.setText(f"Selected box: {box['class_name']}")
        combo_index = self.train_class_combo.findText(box["class_name"])
        if combo_index >= 0:
            self.train_class_combo.setCurrentIndex(combo_index)

    def _combo_class(self) -> Optional[tuple]:
        """(cls_id, class_name) chosen in the class combo, or None."""
        class_name = self.train_class_combo.currentText()
        if class_name not in self._classes:
            self.train_status_label.setText("Choose a class first.")
            return None
        return self._classes.index(class_name), class_name

    @Slot()
    def _on_train_confirm(self) -> None:
        if self._train_selected_box_index < 0 or self._train_current_index < 0:
            return
        chosen = self._combo_class()
        if chosen is None:
            return
        box = self._train_boxes[self._train_selected_box_index]
        box["cls_id"], box["class_name"] = chosen
        self._save_train_boxes(self._train_selected_box_index)

    @Slot(int, int, int, int)
    def _on_train_box_drawn(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self._train_current_index < 0:
            return
        chosen = self._combo_class()
        if chosen is None:
            return
        cls_id, class_name = chosen
        self._train_boxes.append(
            {"cls_id": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_name": class_name}
        )
        self._train_selected_box_index = len(self._train_boxes) - 1
        self.train_selected_label.setText(f"Selected box: {class_name}")
        self._save_train_boxes(self._train_selected_box_index)

    @Slot()
    def _on_train_delete(self) -> None:
        if self._train_selected_box_index < 0 or self._train_current_index < 0:
            return
        del self._train_boxes[self._train_selected_box_index]
        self._train_selected_box_index = -1
        self.train_selected_label.setText("Selected box: none")
        self._save_train_boxes(-1)

    def _save_train_boxes(self, selected_index: int) -> None:
        """Redraws the boxes and writes them to the current label file."""
        self.train_image_widget.set_image(self.train_image_widget.pixmap(), self._train_boxes)
        self.train_image_widget.set_selected_index(selected_index)

        _, label_path = self._train_pairs[self._train_current_index]
        label_path.parent.mkdir(parents=True, exist_ok=True)
        pixmap = self.train_image_widget.pixmap()
        img_w, img_h = pixmap.width(), pixmap.height()
        out_boxes = [
            {
                **pixel_box_to_yolo(b["x1"], b["y1"], b["x2"], b["y2"], img_w, img_h),
                "cls_id": b["cls_id"],
            }
            for b in self._train_boxes
        ]
        write_yolo_label_file(label_path, out_boxes)
        self.train_status_label.setText(f"Updated label: {label_path.name}")

    # --- Training -------------------------------------------------------

    @Slot()
    def _start_training(self) -> None:
        ctx = self._ctx
        if self._training_worker is not None and self._training_worker.isRunning():
            self.train_status_label.setText("Training already in progress.")
            return
        # Missing steps are done for the user (pick model / folder, stop the
        # camera) rather than reported, then training carries on.
        if ctx.model_path is None:
            self.train_status_label.setText("Choose the model to train from.")
            ctx.model_pick_requested.emit()
            if ctx.model_path is None:
                self.train_status_label.setText("Training needs a model (.pt) to start from.")
                return
        if not any(label_path.exists() for _, label_path in self._train_pairs):
            self.train_status_label.setText("Choose a folder with labelled images.")
            self._select_train_folder()
            if not any(label_path.exists() for _, label_path in self._train_pairs):
                self.train_status_label.setText(
                    "No labelled images in that folder — draw boxes or pick another folder.")
                return
        if ctx.worker.isRunning():
            answer = QMessageBox.question(
                self, "Start Training",
                "Training needs the GPU, so monitoring will stop.\n\nStop monitoring and start training?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if answer != QMessageBox.Yes:
                return

            def start_when_stopped() -> None:
                ctx.worker.finished.disconnect(start_when_stopped)
                self._start_training()

            ctx.worker.finished.connect(start_when_stopped)
            self.train_status_label.setText("Stopping monitoring...")
            ctx.stop_stream_requested.emit()
            return

        images_dir = self._train_pairs[0][0].parent
        dataset_root = images_dir.parent
        # Train on labelled images only: Ultralytics would take an image with
        # no label file as background and learn to ignore what's on it.
        list_path = dataset_root / "labelled.txt"
        list_path.write_text("".join(
            f"{image_path.resolve()}\n"
            for image_path, label_path in self._train_pairs if label_path.exists()
        ))
        data_yaml_path = dataset_root / "data.yaml"
        with data_yaml_path.open("w") as f:
            yaml.safe_dump(
                {
                    "path": str(dataset_root.resolve()),
                    "train": list_path.name,
                    "val": list_path.name,
                    "nc": len(self._classes),
                    "names": list(self._classes),
                },
                f,
                sort_keys=False,
            )

        # Release the Monitor tab's loaded model before training — otherwise it
        # stays resident in memory (the stream only stops running, it never
        # clears worker._model), and a second full model plus training's
        # own memory needs can exceed this device's RAM. See: OOM kills observed
        # in journalctl during training with both loaded at once.
        ctx.worker._model = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = "0" if (torch is not None and torch.cuda.is_available()) else "cpu"
        config = {
            "model_path": str(ctx.model_path),
            "data_yaml": str(data_yaml_path),
            "epochs": 50,
            "imgsz": 640,
            "batch": 8,
            "patience": 20,
            "optimizer": "auto",
            "lr0": 0.01,
            "device": device,
            "project": str(dataset_root / "runs"),
            "name": "train",
        }

        self._training_worker = TrainingWorker(config)
        self._training_worker.status_update.connect(self.train_status_label.setText)
        self._training_worker.training_complete.connect(self._on_training_complete)
        self._training_worker.training_error.connect(self.train_status_label.setText)
        self._training_worker.finished.connect(self._on_training_thread_finished)
        self.train_start_button.setEnabled(False)
        ctx.training_active = True
        self._training_worker.start()

    @Slot(str)
    def _on_training_complete(self, best_path: str) -> None:
        self.train_status_label.setText(f"Training complete. Best model: {best_path}")

    def _on_training_thread_finished(self) -> None:
        self.train_start_button.setEnabled(True)
        self._ctx.training_active = False
        self._training_worker = None
