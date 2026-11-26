from pathlib import Path
from typing import Optional

import yaml
from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
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
from ..augmentation import build_hyp, describe
from ..marking.classes import read_classes_file, validate_class_name, write_classes_file
from ..optional_deps import torch
from ..paths import CAPTURES_IMAGES_DIR, IMAGE_EXTENSIONS
from ..pose_labels import (
    DEFAULT_KEYPOINTS,
    HIDDEN,
    NOT_LABELLED,
    VISIBLE,
    dataset_yaml,
    kpts_to_normalized,
    kpts_to_pixel,
    label_is_complete,
    parse_keypoint_names,
    read_keypoints_file,
    read_pose_label_file,
    write_keypoints_file,
    write_pose_label_file,
)
from ..theme import RUN_BUTTON_STYLES, action_button_qss, make_icon
from ..training import TrainingWorker, pose_base_model
from ..widgets.annotated_image import AnnotatedImageWidget
from ..widgets.training_options_dialog import TrainingOptionsDialog
from ..widgets.layout import bottom_bar, top_bar
from ..widgets.touch_input import get_text
from ..yolo_labels import pixel_box_to_yolo, read_yolo_label_file, write_yolo_label_file, yolo_box_to_pixel


def load_upright(image_path: Path) -> QPixmap:
    """The image as the trainer sees it. Phone photos are often stored
    sideways with an EXIF rotation flag; OpenCV (and so Ultralytics) applies
    it, plain QPixmap does not, so labels drawn on the sideways picture would
    land in the wrong place when training."""
    reader = QImageReader(str(image_path))
    reader.setAutoTransform(True)
    return QPixmap.fromImage(reader.read())


class TrainTab(QWidget):
    """Review captured images, draw / delete / correct boxes, and fine-tune the
    model. A dataset with a classes.txt is labelled and trained with those
    classes instead of the loaded model's.

    A dataset with a keypoints.txt is a pose dataset (app/pose_labels.py):
    after a box is drawn the tab asks for each keypoint in turn, and training
    makes a pose model (for the Orientation tool) instead of a detector."""

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
        self._images_dir: Optional[Path] = None
        self._pose_names: Optional[list] = None  # keypoint names: a pose dataset
        self._next_kpt: int = -1  # keypoint the next tap places on the selected box
        self._resume_draw = False  # draw mode was on before placing points

        self.train_model_label = QLabel("No model selected")
        ctx.model_path_changed.connect(self._sync_train_model_label)
        self.train_select_folder_button = QPushButton("Select Folder...")
        self.train_select_folder_button.setIcon(make_icon("folder", "#e0e0e0"))
        self.train_select_folder_button.clicked.connect(self._select_train_folder)
        self.train_folder_label = QLabel("(none selected)")
        self.train_mode_label = QLabel("Detect")

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
        self.train_image_widget.point_placed.connect(self._on_point_placed)
        self.train_image_widget.point_moved.connect(self._on_point_moved)

        self.train_selected_label = QLabel("Selected box: none")
        self.train_class_combo = QComboBox()
        ctx.model_classes_changed.connect(self._populate_class_combo)
        self.train_confirm_button = QPushButton("Confirm")
        self.train_confirm_button.setIcon(make_icon("check", "#e0e0e0"))
        self.train_confirm_button.clicked.connect(self._on_train_confirm)
        self.train_draw_button = QPushButton("Draw")
        self.train_draw_button.setIcon(make_icon("roi", "#e0e0e0"))
        self.train_draw_button.setCheckable(True)
        self.train_draw_button.toggled.connect(self._on_draw_toggled)
        self.train_delete_button = QPushButton("Delete")
        self.train_delete_button.setIcon(make_icon("trash", "#e0e0e0"))
        self.train_delete_button.clicked.connect(self._on_train_delete)
        self.add_class_button = QPushButton("Add Class")
        self.add_class_button.setIcon(make_icon("plus", "#e0e0e0"))
        self.add_class_button.clicked.connect(self._on_add_class)
        self.rename_class_button = QPushButton("Rename")
        self.rename_class_button.setIcon(make_icon("pencil", "#e0e0e0"))
        self.rename_class_button.clicked.connect(self._on_rename_class)

        self.make_pose_button = QPushButton("Make Pose Dataset")
        self.make_pose_button.setIcon(make_icon("target", "#e0e0e0"))
        self.make_pose_button.setToolTip(
            "Turn this dataset into a pose dataset: each box gets keypoints (e.g. head, tail) "
            "so a model can learn which way the part faces.")
        self.make_pose_button.clicked.connect(self._on_make_pose_dataset)
        self.kpt_next_label = QLabel("Select a box to place its points.")
        self.kpt_hidden_button = QPushButton("Hidden")
        self.kpt_hidden_button.setCheckable(True)
        self.kpt_hidden_button.setToolTip(
            "The next point is covered: tap where it would be. It is saved as hidden.")
        self.kpt_replace_button = QPushButton("Re-place")
        self.kpt_replace_button.setToolTip("Clear the selected box's points and tap them again.")
        self.kpt_replace_button.clicked.connect(self._on_replace_points)
        self.kpt_names_button = QPushButton("Names")
        self.kpt_names_button.setToolTip("Rename the keypoints (keypoints.txt).")
        self.kpt_names_button.setIcon(make_icon("pencil", "#e0e0e0"))
        self.kpt_names_button.clicked.connect(self._on_edit_keypoint_names)

        self.train_start_button = QPushButton("Start Training")
        self.train_start_button.setIcon(make_icon("play"))
        self.train_start_button.setIconSize(QSize(20, 20))
        self.train_start_button.setFixedHeight(44)
        self.train_start_button.setMinimumWidth(190)
        self.train_start_button.setStyleSheet(action_button_qss(*RUN_BUTTON_STYLES["start"]))
        self.train_start_button.clicked.connect(self._start_training)
        self.train_options_button = QPushButton("Training Options")
        self.train_options_button.setFixedHeight(44)
        self.train_options_button.clicked.connect(self._open_training_options)
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
        mode_caption = QLabel("Type:")
        mode_caption.setObjectName("monitorInfo")
        top_layout.addWidget(mode_caption)
        top_layout.addWidget(self.train_mode_label)
        top_layout.addSpacing(12)
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
        self.classes_hint = QLabel("Classes are saved with this dataset (classes.txt).")
        self.classes_hint.setObjectName("panelText")
        self.classes_hint.setWordWrap(True)
        self.classes_hint.setStyleSheet("font-size: 10pt; color: #2f353c;")
        correction_layout.addWidget(self.classes_hint)
        correction_layout.addWidget(self.train_confirm_button)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.train_draw_button)
        edit_row.addWidget(self.train_delete_button)
        correction_layout.addLayout(edit_row)
        correction_layout.addWidget(self.make_pose_button)
        side_layout.addWidget(correction_box)

        # Pose only, kept to two rows: the panel must still fit a 720 px screen.
        self.kpt_box = QWidget()
        self.kpt_box.setObjectName("counterBox")
        kpt_layout = QVBoxLayout(self.kpt_box)
        kpt_layout.setContentsMargins(0, 2, 0, 0)
        kpt_layout.setSpacing(6)
        self.kpt_next_label.setObjectName("panelText")
        self.kpt_next_label.setWordWrap(True)
        self.kpt_next_label.setStyleSheet("font-weight: bold;")
        kpt_layout.addWidget(self.kpt_next_label)
        kpt_row = QHBoxLayout()
        kpt_row.setSpacing(4)
        for button in (self.kpt_hidden_button, self.kpt_replace_button, self.kpt_names_button):
            button.setStyleSheet("padding: 4px 2px;")  # three across the 260 px panel
        kpt_row.addWidget(self.kpt_hidden_button)
        kpt_row.addWidget(self.kpt_replace_button)
        kpt_row.addWidget(self.kpt_names_button)
        kpt_layout.addLayout(kpt_row)
        correction_layout.addWidget(self.kpt_box)
        self._update_mode_ui()

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
        bottom_layout.addWidget(self.train_options_button)
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
            name, ok = get_text(self, title, "Class name:", text=text)
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
        self._images_dir = images_dir
        self._dataset_classes = read_classes_file(images_dir.parent)
        self._pose_names = read_keypoints_file(images_dir.parent)
        self._end_points()
        self._update_mode_ui()
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
            kind = f"pose ({', '.join(self._pose_names)})" if self._pose_names else classes_source
            self.train_status_label.setText(
                f"Loaded {len(pairs)} images ({unlabelled} unlabelled), {kind}"
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

        pixmap = load_upright(image_path)
        img_w, img_h = pixmap.width(), pixmap.height()

        boxes = []
        k = len(self._pose_names) if self._pose_names else 0
        raw = read_pose_label_file(label_path, k) if k else read_yolo_label_file(label_path)
        for box in raw:
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
            if k:
                boxes[-1]["kpts"] = kpts_to_pixel(box["kpts"], img_w, img_h)

        self._end_points()
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
        if self._pose_names:
            missing = self._first_missing_point(box)
            if missing is None:
                self._end_points()
            else:
                self._begin_points(missing)

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
        box = {"cls_id": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_name": class_name}
        if self._pose_names:
            box["kpts"] = [(0.0, 0.0, NOT_LABELLED)] * len(self._pose_names)
        self._train_boxes.append(box)
        self._train_selected_box_index = len(self._train_boxes) - 1
        self.train_selected_label.setText(f"Selected box: {class_name}")
        self._save_train_boxes(self._train_selected_box_index)
        if self._pose_names:
            self._resume_draw = self.train_draw_button.isChecked()
            self._begin_points(0)

    @Slot()
    def _on_train_delete(self) -> None:
        if self._train_selected_box_index < 0 or self._train_current_index < 0:
            return
        del self._train_boxes[self._train_selected_box_index]
        self._end_points()
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
        if self._pose_names:
            for out, b in zip(out_boxes, self._train_boxes):
                out["kpts"] = kpts_to_normalized(b.get("kpts") or [], img_w, img_h)
            write_pose_label_file(label_path, out_boxes, len(self._pose_names))
        else:
            write_yolo_label_file(label_path, out_boxes)
        self.train_status_label.setText(f"Updated label: {label_path.name}")

    # --- Pose keypoints ---------------------------------------------------

    def _update_mode_ui(self) -> None:
        pose = bool(self._pose_names)
        self.train_mode_label.setText(f"Pose ({', '.join(self._pose_names)})" if pose else "Detect")
        self.make_pose_button.setVisible(not pose)
        self.classes_hint.setVisible(not pose)
        self.kpt_box.setVisible(pose)

    @staticmethod
    def _first_missing_point(box: dict) -> Optional[int]:
        for i, (_, _, v) in enumerate(box.get("kpts") or []):
            if v == NOT_LABELLED:
                return i
        return None

    def _begin_points(self, index: int) -> None:
        """Asks for keypoint `index` of the selected box; the next tap places it."""
        self._next_kpt = index
        if self.train_draw_button.isChecked():
            self.train_draw_button.blockSignals(True)
            self.train_draw_button.setChecked(False)
            self.train_draw_button.blockSignals(False)
        self.train_image_widget.set_point_mode(True)
        name = self._pose_names[index]
        self.kpt_next_label.setText(f"Tap the {name} ({index + 1}/{len(self._pose_names)})")

    def _end_points(self) -> None:
        """Leaves point placing; back to drawing if that was on before."""
        was_placing = self._next_kpt >= 0
        self._next_kpt = -1
        self.kpt_hidden_button.setChecked(False)
        self.train_image_widget.set_point_mode(False)
        self.kpt_next_label.setText("Drag a dot to move it. Select a box to place its points.")
        if was_placing and self._resume_draw:
            self._resume_draw = False
            self.train_draw_button.setChecked(True)

    @Slot(bool)
    def _on_draw_toggled(self, checked: bool) -> None:
        if checked and self._next_kpt >= 0:
            self._resume_draw = False
            self._end_points()
        self.train_image_widget.set_draw_mode(checked)

    @Slot(float, float)
    def _on_point_placed(self, x: float, y: float) -> None:
        index = self._train_selected_box_index
        if self._next_kpt < 0 or not (0 <= index < len(self._train_boxes)):
            return
        box = self._train_boxes[index]
        v = HIDDEN if self.kpt_hidden_button.isChecked() else VISIBLE
        box["kpts"][self._next_kpt] = (x, y, v)
        self.kpt_hidden_button.setChecked(False)
        self._save_train_boxes(index)
        missing = self._first_missing_point(box)
        if missing is None:
            self._end_points()
            self.train_status_label.setText("Points placed. Draw the next box or go to the next image.")
        else:
            self._begin_points(missing)

    @Slot(int, float, float)
    def _on_point_moved(self, _index: int, _x: float, _y: float) -> None:
        if 0 <= self._train_selected_box_index < len(self._train_boxes):
            self._save_train_boxes(self._train_selected_box_index)

    @Slot()
    def _on_replace_points(self) -> None:
        index = self._train_selected_box_index
        if not self._pose_names or not (0 <= index < len(self._train_boxes)):
            self.train_status_label.setText("Select a box first.")
            return
        self._train_boxes[index]["kpts"] = [(0.0, 0.0, NOT_LABELLED)] * len(self._pose_names)
        self._save_train_boxes(index)
        self._begin_points(0)

    def _ask_keypoint_names(self, title: str, current: list) -> Optional[list]:
        text = ", ".join(current)
        while True:
            value, ok = get_text(self, title, "Keypoint names (comma separated, in order):", text=text)
            if not ok:
                return None
            names, problem = parse_keypoint_names(value)
            if names is not None:
                return names
            self.train_status_label.setText(problem)
            text = value

    def _any_points_placed(self) -> bool:
        k = len(self._pose_names or [])
        for _, label_path in self._train_pairs:
            for box in read_pose_label_file(label_path, k):
                if any(v != NOT_LABELLED for _, _, v in box["kpts"]):
                    return True
        return False

    @Slot()
    def _on_make_pose_dataset(self) -> None:
        if not self._ensure_dataset():
            return
        names = self._ask_keypoint_names("Make Pose Dataset", DEFAULT_KEYPOINTS)
        if names is None:
            return
        if not self._dataset_classes:
            # The dataset keeps its own classes: the loaded model's if labels
            # use them, else one class for the part.
            labelled = any(label.exists() for _, label in self._train_pairs)
            write_classes_file(self._dataset_root, list(self._classes) if labelled and self._classes else ["part"])
        write_keypoints_file(self._dataset_root, names)
        index = max(0, self._train_current_index)
        self.scan_train_folder(self._images_dir)
        if 0 <= index < len(self._train_pairs):
            self._load_train_index(index)
        self.train_status_label.setText(
            f"Pose dataset: {', '.join(names)}. Select a box and tap its points, or draw new boxes.")

    @Slot()
    def _on_edit_keypoint_names(self) -> None:
        if not self._pose_names:
            return
        names = self._ask_keypoint_names("Keypoint Names", self._pose_names)
        if names is None or names == self._pose_names:
            return
        if len(names) != len(self._pose_names) and self._any_points_placed():
            self.train_status_label.setText(
                f"Points are already placed: keep {len(self._pose_names)} names (rename only).")
            return
        write_keypoints_file(self._dataset_root, names)
        index = max(0, self._train_current_index)
        self.scan_train_folder(self._images_dir)
        if 0 <= index < len(self._train_pairs):
            self._load_train_index(index)
        self.train_status_label.setText(f"Keypoints: {', '.join(names)}")

    # --- Training -------------------------------------------------------

    @Slot()
    def _start_training(self) -> None:
        ctx = self._ctx
        if self._training_worker is not None and self._training_worker.isRunning():
            self.train_status_label.setText("Training already in progress.")
            return
        # Missing steps are done for the user (pick model / folder, stop the
        # camera) rather than reported, then training carries on.
        pose = bool(self._pose_names)
        # A pose model carries on from the loaded model only if that is a
        # matching pose model; else it starts from the pose base model.
        if ctx.model_path is None and not pose:
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
        if pose:
            complete, waiting = self._pose_label_counts()
            if not complete:
                self.train_status_label.setText(
                    f"No image has all its points yet ({waiting} still need keypoints).")
                return
            if waiting:
                answer = QMessageBox.question(
                    self, "Start Training",
                    f"{waiting} labelled images still need keypoints and will be left out.\n\n"
                    f"Train on the {complete} finished images?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if answer != QMessageBox.Yes:
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

        data_yaml_path = self._write_dataset_files()
        dataset_root = data_yaml_path.parent

        # Release the Monitor tab's loaded model before training — otherwise it
        # stays resident in memory (the stream only stops running, it never
        # clears worker._model), and a second full model plus training's
        # own memory needs can exceed this device's RAM. See: OOM kills observed
        # in journalctl during training with both loaded at once.
        ctx.worker._model = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = "0" if (torch is not None and torch.cuda.is_available()) else "cpu"
        settings = ctx.settings
        config = {
            "model_path": str(ctx.model_path) if ctx.model_path is not None else None,
            "data_yaml": str(data_yaml_path),
            "epochs": settings["train_epochs"],
            "augment": build_hyp(settings["train_augment"], settings["train_mirror"],
                                 settings["train_flip_ud"], settings["train_rotate_any"]),
            "imgsz": 640,
            "batch": 8,
            "patience": 20,
            "optimizer": "auto",
            "lr0": 0.01,
            "device": device,
            "project": str(dataset_root / "runs"),
            "name": "pose" if pose else "train",
            "task": "pose" if pose else "detect",
            "kpt_count": len(self._pose_names or []),
            "fallback_model": pose_base_model(),
        }

        self._training_worker = TrainingWorker(config)
        self._training_worker.status_update.connect(self.train_status_label.setText)
        self.train_status_label.setText(f"Training started ({self._training_options_text()})...")
        self._training_worker.training_complete.connect(self._on_training_complete)
        self._training_worker.training_error.connect(self.train_status_label.setText)
        self._training_worker.finished.connect(self._on_training_thread_finished)
        self.train_start_button.setEnabled(False)
        ctx.training_active = True
        self._training_worker.start()

    def _write_dataset_files(self) -> Optional[Path]:
        """Writes labelled.txt + data.yaml next to the images folder (used by
        training and the augmentation preview); None without labelled images."""
        if self._pose_names:
            # Pose: only images whose every box has all its points.
            k = len(self._pose_names)
            labelled = [image for image, label in self._train_pairs
                        if label.exists() and label_is_complete(read_pose_label_file(label, k))]
        else:
            labelled = [image for image, label in self._train_pairs if label.exists()]
        if not labelled:
            return None
        dataset_root = labelled[0].parent.parent
        # Train on labelled images only: Ultralytics would take an image with
        # no label file as background and learn to ignore what's on it.
        list_path = dataset_root / "labelled.txt"
        list_path.write_text("".join(f"{image.resolve()}\n" for image in labelled))
        data_yaml_path = dataset_root / "data.yaml"
        with data_yaml_path.open("w") as f:
            yaml.safe_dump(dataset_yaml(dataset_root, list_path.name, self._classes, self._pose_names),
                           f, sort_keys=False)
        return data_yaml_path

    def _pose_label_counts(self) -> tuple:
        """(images ready to train, labelled images still missing points)."""
        k = len(self._pose_names or [])
        complete = waiting = 0
        for _, label in self._train_pairs:
            if not label.exists():
                continue
            if label_is_complete(read_pose_label_file(label, k)):
                complete += 1
            else:
                waiting += 1
        return complete, waiting

    def _training_options_text(self) -> str:
        settings = self._ctx.settings
        return (describe(settings["train_augment"], settings["train_mirror"], settings["train_flip_ud"],
                         settings["train_rotate_any"])
                + f", {settings['train_epochs']} epochs")

    @Slot()
    def _open_training_options(self) -> None:
        settings = self._ctx.settings
        data_yaml = self._write_dataset_files() if self._classes else None
        dialog = TrainingOptionsDialog(
            settings["train_augment"], settings["train_mirror"], settings["train_flip_ud"],
            settings["train_epochs"], data_yaml, parent=self, rotate_any=settings["train_rotate_any"])
        if dialog.exec() != TrainingOptionsDialog.Accepted:
            return
        settings.set(train_augment=dialog.preset, train_mirror=dialog.mirror,
                     train_flip_ud=dialog.flip_ud, train_rotate_any=dialog.rotate_any,
                     train_epochs=dialog.epochs)
        self.train_status_label.setText(f"Saved: {self._training_options_text()}")

    @Slot(str)
    def _on_training_complete(self, best_path: str) -> None:
        self.train_status_label.setText(f"Training complete. Best model: {best_path}")

    def _on_training_thread_finished(self) -> None:
        self.train_start_button.setEnabled(True)
        self._ctx.training_active = False
        self._training_worker = None
