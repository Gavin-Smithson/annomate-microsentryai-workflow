"""
MicroSentryAI Window View.

QWidget view — not QMainWindow, designed for tab embedding.
All dialogs here; all computation delegated to InferenceController.
MVC rule: never access .state directly — use model query API only.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Optional, List

from PIL import Image

from PySide6.QtCore import (
    Qt, Signal, QTimer, QModelIndex, QIdentityProxyModel,
)
from PySide6.QtGui import (
    QBrush, QColor, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QSlider, QSpinBox, QDoubleSpinBox,
    QProgressBar, QHeaderView, QAbstractItemView, QTableView,
    QInputDialog, QSplitter, QApplication,
)

from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from controllers.inference_controller import InferenceController
from views.microsentry.canvas import CanvasPair, SegPathItem

logger = logging.getLogger("MicroSentryAI.Window")


# ---------------------------------------------------------------------------
# Proxy — overrides col-1 to show inference status instead of review status
# ---------------------------------------------------------------------------

class _InferenceStatusProxy(QIdentityProxyModel):
    """Thin proxy that replaces column-1 display with inference processing status."""

    def __init__(
        self,
        inference_model: InferenceModel,
        dataset_model: DatasetTableModel,
        parent=None,
    ):
        super().__init__(parent)
        self.setSourceModel(dataset_model)
        self._inference_model = inference_model
        self._dataset_model = dataset_model

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.column() != 1:
            return super().data(index, role)
        row = index.row()
        if row >= self._dataset_model.rowCount():
            return super().data(index, role)
        try:
            path = self._dataset_model.get_image_path(row)
        except Exception:
            return super().data(index, role)
        processed = self._inference_model.is_processed(path)
        if role == Qt.DisplayRole:
            return "Processed" if processed else "Pending"
        if role == Qt.BackgroundRole:
            return (
                QBrush(QColor(210, 245, 210))
                if processed
                else QBrush(QColor(255, 235, 210))
            )
        return super().data(index, role)

    def refresh_status_column(self):
        n = self.rowCount()
        if n > 0:
            self.dataChanged.emit(self.index(0, 1), self.index(n - 1, 1))


# ---------------------------------------------------------------------------
# MicroSentryWindow
# ---------------------------------------------------------------------------

class MicroSentryWindow(QWidget):
    """
    MVC view for the MicroSentryAI pane.

    Constructor args:
        dataset_model        — shared QAbstractTableModel (image list + filenames)
        inference_model      — pure-Python model for heatmap/score-map results
        inference_controller — headless inference + visualisation logic
    """

    polygonsSent = Signal(list, str)           # (polygons in original coords, default class)
    viewChanged  = Signal(float, float, float) # rx, ry, scale — cross-tab pan/zoom sync

    def __init__(
        self,
        dataset_model: DatasetTableModel,
        inference_model: InferenceModel,
        inference_controller: InferenceController,
        parent=None,
    ):
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self.inference_controller = inference_controller

        # Transient view state (not domain data)
        self._current_row: int = -1
        self._current_pil: Optional[Image.Image] = None
        self._last_scale: float = 1.0
        self._last_offset: tuple = (0, 0)
        self._undo_stack: List[list] = []
        self._redo_stack: List[list] = []
        self._block_history: bool = False
        self._worker = None

        self._cached_heatmap_result = None  # (left_pil, right_pil, scale, offset, s, display_w, display_h)
        self._cached_heatmap_params = None  # (alpha, sigma, display_target, heat_min_pct, image_path)
        self._slider_timer = QTimer(self)
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(40)
        self._slider_timer.timeout.connect(self._render_current)

        self._init_ui()
        self._connect_signals()
        self._setup_shortcuts()

        self.dataset_model.modelReset.connect(self._on_model_reset)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        self._proxy = _InferenceStatusProxy(
            self.inference_model, self.dataset_model, self
        )

        splitter = QSplitter(Qt.Horizontal, self)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self._build_toolbar(lv)
        self.canvas_pair = CanvasPair()
        lv.addWidget(self.canvas_pair, stretch=1)
        self._build_bottom_bar(lv)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self._build_sidebar(rv)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter, stretch=1)

        self.status_label = QLabel()
        root.addWidget(self.status_label)

    def _build_toolbar(self, layout: QVBoxLayout):
        r1 = QWidget()
        r1l = QHBoxLayout(r1)
        r1l.setContentsMargins(5, 5, 5, 0)

        self.model_label = QLabel("No Model Loaded")
        self.model_label.setStyleSheet("font-weight: bold; color: #913333;")

        self.display_spin = QSpinBox()
        self.display_spin.setRange(256, 2048)
        self.display_spin.setValue(600)
        self.display_spin.setSuffix(" px")

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.45)

        self.heat_thresh_spin = QSpinBox()
        self.heat_thresh_spin.setRange(0, 100)
        self.heat_thresh_spin.setValue(0)
        self.heat_thresh_spin.setSuffix("%")

        r1l.addStretch()
        r1l.addWidget(QLabel("Model:"))
        r1l.addWidget(self.model_label)
        r1l.addStretch()
        r1l.addWidget(QLabel("Display:"))
        r1l.addWidget(self.display_spin)
        r1l.addSpacing(10)
        r1l.addWidget(QLabel("Heat α:"))
        r1l.addWidget(self.alpha_spin)
        r1l.addSpacing(10)
        r1l.addWidget(QLabel("Heat Min:"))
        r1l.addWidget(self.heat_thresh_spin)

        r2 = QWidget()
        r2l = QHBoxLayout(r2)
        r2l.setContentsMargins(5, 0, 5, 5)

        self.sigma_spin = QSpinBox()
        self.sigma_spin.setRange(0, 16)
        self.sigma_spin.setValue(4)
        self.sigma_label = QLabel("σ: 4")

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.0, 20.0)
        self.eps_spin.setSingleStep(0.5)
        self.eps_spin.setDecimals(1)
        self.eps_spin.setValue(1.5)

        self.btn_simpl_sel  = QPushButton("Simplify Selected")
        self.btn_simpl_all  = QPushButton("Simplify All")
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_images = QPushButton("Load Image Folder")
        self.btn_send_annot = QPushButton("Send to AnnoMate")
        self.btn_send_annot.setStyleSheet(
            "background-color: #d1f7d1; font-weight: bold; padding: 5px;"
        )

        r2l.addWidget(QLabel("Smooth (σ):"))
        r2l.addWidget(self.sigma_spin)
        r2l.addWidget(self.sigma_label)
        r2l.addSpacing(15)
        r2l.addWidget(QLabel("Simplify ε:"))
        r2l.addWidget(self.eps_spin)
        r2l.addWidget(self.btn_simpl_sel)
        r2l.addWidget(self.btn_simpl_all)
        r2l.addStretch()
        r2l.addWidget(self.btn_load_model)
        r2l.addWidget(self.btn_load_images)
        r2l.addWidget(self.btn_send_annot)

        layout.addWidget(r1)
        layout.addWidget(r2)

    def _build_bottom_bar(self, layout: QVBoxLayout):
        self.slider_label = QLabel("Percentile Threshold: 95.0")
        layout.addWidget(self.slider_label)

        bar = QWidget()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(5, 0, 5, 5)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar::chunk { background-color: #00BCD4; }"
        )

        self.btn_prev = QPushButton("< Previous")
        self.btn_next = QPushButton("Next >")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(95)

        bl.addWidget(self.progress_bar)
        bl.addWidget(self.btn_prev)
        bl.addWidget(self.slider)
        bl.addWidget(self.btn_next)

        layout.addWidget(bar)

    def _build_sidebar(self, layout: QVBoxLayout):
        layout.addWidget(QLabel("Dataset"))

        self.table_view = QTableView()
        self.table_view.setModel(self._proxy)
        self.table_view.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.table_view.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)

        layout.addWidget(self.table_view)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.btn_load_model.clicked.connect(self._load_model_clicked)
        self.btn_load_images.clicked.connect(self._load_images_clicked)
        self.btn_send_annot.clicked.connect(self._send_annotations)
        self.btn_prev.clicked.connect(self._prev_image)
        self.btn_next.clicked.connect(self._next_image)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.display_spin.valueChanged.connect(lambda _: self._render_current())
        self.alpha_spin.valueChanged.connect(lambda _: self._render_current())
        self.heat_thresh_spin.valueChanged.connect(lambda _: self._render_current())
        self.sigma_spin.valueChanged.connect(self._on_sigma_change)

        self.btn_simpl_sel.clicked.connect(self._simplify_selected)
        self.btn_simpl_all.clicked.connect(self._simplify_all)

        self.table_view.selectionModel().currentRowChanged.connect(
            self._on_row_changed
        )

        self.canvas_pair.view_left.viewChanged.connect(
            lambda rx, ry, s: self.viewChanged.emit(rx, ry, s)
        )

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo)
        QShortcut(QKeySequence("S"),      self).activated.connect(self._simplify_selected)

    # ------------------------------------------------------------------
    # Row navigation
    # ------------------------------------------------------------------

    def _on_row_changed(self, current: QModelIndex, _previous: QModelIndex):
        row = current.row()
        if row < 0 or row >= self.dataset_model.rowCount():
            return
        if row == self._current_row:
            return
        self._current_row = row
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._load_and_render(row)

    def _prev_image(self):
        if self._current_row > 0:
            self._select_row(self._current_row - 1)

    def _next_image(self):
        if self._current_row < self.dataset_model.rowCount() - 1:
            self._select_row(self._current_row + 1)

    def _select_row(self, row: int):
        self.table_view.setCurrentIndex(self._proxy.index(row, 0))

    # ------------------------------------------------------------------
    # Image loading and rendering
    # ------------------------------------------------------------------

    def _load_and_render(self, row: int):
        path = self.dataset_model.get_image_path(row)
        pil = self.inference_controller.load_image(path)
        if pil is None:
            self.status_label.setText(f"Failed to load: {Path(path).name}")
            return
        self._current_pil = pil
        self._invalidate_heatmap_cache()
        self._render_current()
        n = self.dataset_model.rowCount()
        self.status_label.setText(f"{Path(path).name}  ({row + 1} / {n})")

    def _current_heatmap_params(self) -> tuple:
        path = self.dataset_model.get_image_path(self._current_row)
        return (
            float(self.alpha_spin.value()),
            float(self.sigma_spin.value()),
            int(self.display_spin.value()),
            int(self.heat_thresh_spin.value()),
            path,
        )

    def _is_heatmap_cache_valid(self) -> bool:
        return (
            self._cached_heatmap_result is not None
            and self._cached_heatmap_params == self._current_heatmap_params()
        )

    def _invalidate_heatmap_cache(self):
        self._cached_heatmap_result = None
        self._cached_heatmap_params = None

    def _on_slider_changed(self, _):
        self.slider_label.setText(f"Percentile Threshold: {self.slider.value():.1f}")
        self._slider_timer.start()

    def _render_current(self):
        if self._current_pil is None or self._current_row < 0:
            return

        path = self.dataset_model.get_image_path(self._current_row)
        score_map = self.inference_model.get_score_map(path)

        if self._is_heatmap_cache_valid():
            left_pil, right_pil, scale, offset, s, display_w, display_h = (
                self._cached_heatmap_result
            )
        else:
            left_pil, right_pil, scale, offset, s = (
                self.inference_controller.compute_heatmap(
                    pil_image=self._current_pil,
                    score_map=score_map,
                    alpha=float(self.alpha_spin.value()),
                    sigma=float(self.sigma_spin.value()),
                    display_target=int(self.display_spin.value()),
                    heat_min_pct=int(self.heat_thresh_spin.value()),
                )
            )
            display_w, display_h = left_pil.size
            self._cached_heatmap_result = (
                left_pil, right_pil, scale, offset, s, display_w, display_h
            )
            self._cached_heatmap_params = self._current_heatmap_params()

            contours = self.inference_controller.compute_segmentation(
                smoothed_s=s,
                seg_pct=int(self.slider.value()),
                epsilon=float(self.eps_spin.value()),
                display_w=display_w,
                display_h=display_h,
            )
            self._last_scale = scale
            self._last_offset = offset
            self.slider_label.setText(f"Percentile Threshold: {self.slider.value():.1f}")
            self.canvas_pair.set_images(left_pil, right_pil, self._on_any_edit, contours)
            QTimer.singleShot(50, self.canvas_pair.fit_views)
            return

        # Cache hit — only seg_pct changed; skip full heatmap recompute
        contours = self.inference_controller.compute_segmentation(
            smoothed_s=s,
            seg_pct=int(self.slider.value()),
            epsilon=float(self.eps_spin.value()),
            display_w=display_w,
            display_h=display_h,
        )
        self._last_scale = scale
        self._last_offset = offset
        self.slider_label.setText(f"Percentile Threshold: {self.slider.value():.1f}")
        self.canvas_pair.set_polygons(contours, self._on_any_edit)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Exported Model", "", "PyTorch Models (*.pt *.ckpt)"
        )
        if not path:
            return

        device, ok = QInputDialog.getItem(
            self, "Select Device", "Inference Hardware:",
            ["auto", "cpu", "cuda", "mps"], 0, False,
        )
        if not ok:
            return

        self.status_label.setText(f"Loading {Path(path).name}…")
        QApplication.processEvents()

        try:
            model_name = self.inference_controller.load_model(path, device)
        except RuntimeError as e:
            QMessageBox.critical(self, "Model Load Error", str(e))
            self.status_label.setText("Error loading model.")
            return

        self.model_label.setText(f"Active: {model_name}")
        self.model_label.setStyleSheet("font-weight: bold; color: #538A3F;")
        QMessageBox.information(
            self, "Model Loaded",
            f"Model loaded successfully.\n\nDetails: {model_name}",
        )

        if self.dataset_model.rowCount() > 0:
            self.inference_model.clear()
            self._proxy.refresh_status_column()
            self._start_background_inference()
            if self._current_row >= 0:
                self._render_current()

    # ------------------------------------------------------------------
    # Folder loading (standalone / independent of AnnoMate IO controller)
    # ------------------------------------------------------------------

    def _load_images_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", "")
        if not folder:
            return

        files: List[str] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            files.extend(glob.glob(os.path.join(folder, ext)))
        files = sorted(files)

        if not files:
            QMessageBox.information(self, "No Images", "No compatible images found.")
            return

        self.inference_model.clear()
        self.dataset_model.load_folder(folder, [Path(f).name for f in files])

        self._current_row = -1
        self._current_pil = None
        self._undo_stack.clear()
        self._redo_stack.clear()

        self._select_row(0)

        if self.inference_controller.has_model():
            self._start_background_inference()

    def _on_model_reset(self):
        """React to dataset model reload (e.g. folder loaded from AnnoMate)."""
        self._current_row = -1
        self._current_pil = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.inference_model.clear()
        self._proxy.refresh_status_column()

        if (
            self.dataset_model.rowCount() > 0
            and self.inference_controller.has_model()
        ):
            self._start_background_inference()

    # ------------------------------------------------------------------
    # Background inference
    # ------------------------------------------------------------------

    def _start_background_inference(self):
        n = self.dataset_model.rowCount()
        pending = [
            self.dataset_model.get_image_path(i)
            for i in range(n)
            if not self.inference_model.is_processed(
                self.dataset_model.get_image_path(i)
            )
        ]
        if not pending:
            self.status_label.setText("All images already processed.")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(pending))
        self.progress_bar.setValue(0)

        worker = self.inference_controller.start_batch_inference(pending)
        worker.resultReady.connect(self._on_worker_result)
        worker.progress.connect(self.progress_bar.setValue)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        worker.start()

    def _on_worker_result(self, path: str, score_map):
        self.inference_model.set_score_map(path, score_map)

        for row in range(self.dataset_model.rowCount()):
            if self.dataset_model.get_image_path(row) == path:
                idx = self._proxy.index(row, 1)
                self._proxy.dataChanged.emit(idx, idx)
                break

        if self._current_row >= 0:
            if self.dataset_model.get_image_path(self._current_row) == path:
                self._invalidate_heatmap_cache()
                self._render_current()

    def _on_worker_finished(self):
        self.progress_bar.setVisible(False)
        self.status_label.setText("Batch inference complete.")

    # ------------------------------------------------------------------
    # Render parameter handlers
    # ------------------------------------------------------------------

    def _on_sigma_change(self, v: int):
        self.sigma_label.setText(f"σ: {v}")
        self._render_current()

    # ------------------------------------------------------------------
    # Undo / Redo  (Memento — transient UI state only)
    # ------------------------------------------------------------------

    def _on_any_edit(self, kind: str):
        if kind in ("vertex_drag_begin", "polygon_move"):
            self._push_undo()

    def _push_undo(self):
        if self._block_history:
            return
        self._undo_stack.append(self.canvas_pair.serialize_polygons())
        self._redo_stack.clear()

    def _undo(self):
        if not self._undo_stack:
            return
        self._block_history = True
        try:
            current = self.canvas_pair.serialize_polygons()
            prior = self._undo_stack.pop()
            self._redo_stack.append(current)
            self.canvas_pair.restore_polygons(prior, self._current_pil, self._on_any_edit)
        finally:
            self._block_history = False

    def _redo(self):
        if not self._redo_stack:
            return
        self._block_history = True
        try:
            current = self.canvas_pair.serialize_polygons()
            nxt = self._redo_stack.pop()
            self._undo_stack.append(current)
            self.canvas_pair.restore_polygons(nxt, self._current_pil, self._on_any_edit)
        finally:
            self._block_history = False

    # ------------------------------------------------------------------
    # Key events — Delete, [ = shrink, ] = grow
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._push_undo()
            for item in list(self.canvas_pair.scene_left.selectedItems()):
                self.canvas_pair.scene_left.removeItem(item)
        elif event.key() == Qt.Key_BracketLeft:
            self._push_undo()
            for item in self.canvas_pair.scene_left.selectedItems():
                if isinstance(item, SegPathItem):
                    item.scale_about_center(0.9)
        elif event.key() == Qt.Key_BracketRight:
            self._push_undo()
            for item in self.canvas_pair.scene_left.selectedItems():
                if isinstance(item, SegPathItem):
                    item.scale_about_center(1.1)
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Polygon simplification
    # ------------------------------------------------------------------

    def _simplify_selected(self):
        eps = float(self.eps_spin.value())
        self._push_undo()
        for item in self.canvas_pair.scene_left.selectedItems():
            if isinstance(item, SegPathItem):
                item.simplify(eps)

    def _simplify_all(self):
        eps = float(self.eps_spin.value())
        self._push_undo()
        for item in self.canvas_pair.scene_left.items():
            if isinstance(item, SegPathItem):
                item.simplify(eps)

    # ------------------------------------------------------------------
    # Send polygons to AnnoMate
    # ------------------------------------------------------------------

    def _send_annotations(self):
        selected = [
            i for i in self.canvas_pair.scene_left.selectedItems()
            if isinstance(i, SegPathItem)
        ]
        if selected:
            polys = self.canvas_pair.get_selected_polygons_original_coords(
                self._last_scale, self._last_offset
            )
        else:
            polys = self.canvas_pair.get_polygons_original_coords(
                self._last_scale, self._last_offset
            )

        if not polys:
            QMessageBox.information(self, "Info", "No polygons to send.")
            return

        self.polygonsSent.emit(polys, "Anomaly")

    # ------------------------------------------------------------------
    # External sync API
    # ------------------------------------------------------------------

    def set_view_state(self, rx: float, ry: float, scale: float):
        """Apply pan/zoom from an external source (cross-tab sync)."""
        self.canvas_pair.set_view_state(rx, ry, scale)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(50, self.canvas_pair.fit_views)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvas_pair.fit_views()
