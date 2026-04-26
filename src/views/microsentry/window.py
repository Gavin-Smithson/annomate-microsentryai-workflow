"""
MicroSentryAI Window View.

QWidget view — not QMainWindow, designed for tab embedding.
All dialogs here; all computation delegated to InferenceController.
MVC rule: never access .state directly — use model query API only.
"""

import logging
from pathlib import Path
from typing import Optional, List

from PIL import Image

from PySide6.QtCore import (
    Qt,
    Signal,
    QTimer,
    QModelIndex,
    QIdentityProxyModel,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QHeaderView,
    QAbstractItemView,
    QTableView,
    QInputDialog,
    QSplitter,
    QApplication,
)

from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from controllers.io_controller import IOController
from controllers.inference_controller import InferenceController
from views.microsentry.canvas import CanvasPair, SegPathItem

logger = logging.getLogger("MicroSentryAI.Window")


# ---------------------------------------------------------------------------
# Proxy — overrides col-1 to show inference status instead of review status
# ---------------------------------------------------------------------------


class _InferenceStatusProxy(QIdentityProxyModel):
    """Proxy model that overrides column 1 to show per-image inference status.

    Wraps a :class:`~models.dataset_model.DatasetTableModel` and replaces
    the review-status column with ``"Processed"`` / ``"Pending"`` text and
    matching background colours derived from the inference model.

    Attributes:
        _inference_model (InferenceModel): Source of truth for which images
            have been processed.
        _dataset_model (DatasetTableModel): Underlying dataset model used to
            look up image paths.
    """

    def __init__(
        self,
        inference_model: InferenceModel,
        dataset_model: DatasetTableModel,
        parent=None,
    ) -> None:
        """Initialize the proxy and set *dataset_model* as the source model.

        Args:
            inference_model (InferenceModel): Inference state used to derive
                per-row processing status.
            dataset_model (DatasetTableModel): Source model for all other columns.
            parent: Optional Qt parent object. Defaults to ``None``.
        """
        super().__init__(parent)
        self.setSourceModel(dataset_model)
        self._inference_model = inference_model
        self._dataset_model = dataset_model

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        """Return inference status text or background color for column 1.

        All other columns and roles are forwarded to the source model.

        Args:
            index (QModelIndex): Cell index being queried.
            role (int): Qt item data role. Defaults to ``Qt.DisplayRole``.

        Returns:
            object: ``"Processed"`` / ``"Pending"`` string for
                ``Qt.DisplayRole``; a :class:`~PySide6.QtGui.QBrush` for
                ``Qt.BackgroundRole``; otherwise defers to the source model.
        """
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

    def refresh_status_column(self) -> None:
        """Emit ``dataChanged`` for all rows in column 1 to force a repaint."""
        n = self.rowCount()
        if n > 0:
            self.dataChanged.emit(self.index(0, 1), self.index(n - 1, 1))


# ---------------------------------------------------------------------------
# MicroSentryWindow
# ---------------------------------------------------------------------------


class MicroSentryWindow(QWidget):
    """MVC view for the MicroSentryAI anomaly-detection pane.

    Designed for tab embedding (QWidget, not QMainWindow). All Qt dialogs
    live here; all computation is delegated to the inference controller.
    Never reads ``*.state`` directly — uses model query APIs only.

    Attributes:
        dataset_model (DatasetTableModel): Shared image-list model.
        inference_model (InferenceModel): Pure-Python heatmap/score-map model.
        inference_controller (InferenceController): Headless inference and
            visualisation logic.
        io_controller (IOController): Shared folder-scan I/O controller.
        canvas_pair (CanvasPair): Dual synchronised canvas widget.
        _current_row (int): Zero-based index of the image currently displayed.
        _current_pil (Optional[Image.Image]): PIL image for the current row.
        _last_scale (float): Display scale applied to the last rendered image.
        _last_offset (tuple): Crop offset ``(x, y)`` of the last rendered image.
        _undo_stack (List[list]): Undo history as serialised polygon snapshots.
        _redo_stack (List[list]): Redo history as serialised polygon snapshots.

    Signals:
        polygonsSent (list, str): Emitted when polygons are sent to AnnoMate;
            carries the list of polygon coordinate lists and the default class
            name.
        viewChanged (float, float, float): Emitted with ``(rx, ry, scale)``
            for cross-tab pan/zoom synchronisation.
        row_selection_changed (int): Emitted when the selected row changes,
            for cross-tab sync.
    """

    polygonsSent = Signal(list, str)  # (polygons in original coords, default class)
    viewChanged = Signal(float, float, float)  # rx, ry, scale — cross-tab pan/zoom sync
    row_selection_changed = Signal(int)  # emitted when the current row changes

    def __init__(
        self,
        dataset_model: DatasetTableModel,
        inference_model: InferenceModel,
        inference_controller: InferenceController,
        io_controller: IOController,
        parent=None,
    ) -> None:
        """Initialize MicroSentryWindow, build the UI, and wire controller signals.

        Args:
            dataset_model (DatasetTableModel): Shared image-list model.
            inference_model (InferenceModel): Heatmap/score-map model.
            inference_controller (InferenceController): Inference and
                visualisation logic.
            io_controller (IOController): Shared folder-scan controller.
            parent: Optional Qt parent widget. Defaults to ``None``.
        """
        super().__init__(parent)
        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self.inference_controller = inference_controller
        self.io_controller = io_controller
        self._syncing = False  # guard against recursive cross-tab row sync

        # Transient view state (not domain data)
        self._current_row: int = -1
        self._current_pil: Optional[Image.Image] = None
        self._last_scale: float = 1.0
        self._last_offset: tuple = (0, 0)
        self._undo_stack: List[list] = []
        self._redo_stack: List[list] = []
        self._block_history: bool = False

        self._cached_heatmap_result = (
            None  # (left_pil, right_pil, scale, offset, s, display_w, display_h)
        )
        self._cached_heatmap_params = (
            None  # (alpha, sigma, display_target, heat_min_pct, image_path)
        )
        self._slider_timer = QTimer(self)
        self._slider_timer.setSingleShot(True)
        self._slider_timer.setInterval(40)
        self._slider_timer.timeout.connect(self._render_current)

        self._init_ui()
        self._connect_signals()
        self._setup_shortcuts()

        # Connect to controller proxy signals once — controller owns the worker
        self.inference_controller.result_ready.connect(self._on_worker_result)
        self.inference_controller.progress.connect(self.progress_bar.setValue)
        self.inference_controller.batch_done.connect(self._on_worker_finished)

        self.dataset_model.modelReset.connect(self._on_model_reset)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        """Build the splitter layout with toolbar, canvas, bottom bar, and sidebar."""
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

    def _build_toolbar(self, layout: QVBoxLayout) -> None:
        """Construct the two-row toolbar and append it to *layout*.

        Row 1 contains model status, display size, heatmap alpha, and heat
        minimum threshold controls. Row 2 contains smoothing sigma, simplify
        epsilon, simplify/load/send buttons.

        Args:
            layout (QVBoxLayout): Parent layout to which the toolbar rows are
                appended.
        """
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

        self.btn_simpl_sel = QPushButton("Simplify Selected")
        self.btn_simpl_all = QPushButton("Simplify All")
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

    def _build_bottom_bar(self, layout: QVBoxLayout) -> None:
        """Construct the percentile-threshold slider row and append it to *layout*.

        Args:
            layout (QVBoxLayout): Parent layout to which the bottom bar is
                appended.
        """
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

    def _build_sidebar(self, layout: QVBoxLayout) -> None:
        """Construct the dataset table sidebar and append it to *layout*.

        Args:
            layout (QVBoxLayout): Parent layout to which the sidebar is
                appended.
        """
        layout.addWidget(QLabel("Dataset"))

        self.table_view = QTableView()
        self.table_view.setModel(self._proxy)
        self.table_view.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.table_view.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)

        layout.addWidget(self.table_view)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Wire all button, spin, slider, and table signals to their slot methods."""
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

        self.table_view.selectionModel().currentRowChanged.connect(self._on_row_changed)

        self.canvas_pair.view_left.viewChanged.connect(
            lambda rx, ry, s: self.viewChanged.emit(rx, ry, s)
        )

    def _setup_shortcuts(self) -> None:
        """Register Ctrl+Z (undo), Ctrl+Y (redo), and S (simplify) keyboard shortcuts."""
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo)
        QShortcut(QKeySequence("S"), self).activated.connect(self._simplify_selected)

    # ------------------------------------------------------------------
    # Row navigation
    # ------------------------------------------------------------------

    def _on_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        """Load and render the image for the newly selected dataset row.

        Resets undo/redo history on each row change, then delegates image
        loading and rendering to :meth:`_load_and_render`. Emits
        :attr:`row_selection_changed` unless :attr:`_syncing` is set.

        Args:
            current (QModelIndex): Model index of the newly selected row.
            _previous (QModelIndex): Model index of the previously selected
                row (unused).
        """
        row = current.row()
        if row < 0 or row >= self.dataset_model.rowCount():
            return
        if row == self._current_row:
            return
        self._current_row = row
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._load_and_render(row)
        if not self._syncing:
            self.row_selection_changed.emit(row)

    def _prev_image(self) -> None:
        """Move the selection to the previous row if not already at row 0."""
        if self._current_row > 0:
            self.select_row(self._current_row - 1)

    def _next_image(self) -> None:
        """Move the selection to the next row if not already at the last row."""
        if self._current_row < self.dataset_model.rowCount() - 1:
            self.select_row(self._current_row + 1)

    def select_row(self, row: int) -> None:
        """Select *row* in the dataset table without emitting :attr:`row_selection_changed`.

        Sets :attr:`_syncing` to prevent the resulting ``currentRowChanged``
        signal from triggering a recursive sync emission.

        Args:
            row (int): Zero-based row index to select.
        """
        self._syncing = True
        self.table_view.setCurrentIndex(self._proxy.index(row, 0))
        self._syncing = False

    # ------------------------------------------------------------------
    # Image loading and rendering
    # ------------------------------------------------------------------

    def _load_and_render(self, row: int) -> None:
        """Load the image at *row* and trigger a full render to the canvas.

        Delegates image loading to the inference controller and invalidates
        the heatmap cache so the next render recomputes from scratch.

        Args:
            row (int): Zero-based row index of the image to load and display.
        """
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
        """Return a tuple of the current render-parameter values used for cache keys.

        Returns:
            tuple: ``(alpha, sigma, display_target, heat_min_pct, image_path)``
                reflecting the current UI control values and the active image path.
        """
        path = self.dataset_model.get_image_path(self._current_row)
        return (
            float(self.alpha_spin.value()),
            float(self.sigma_spin.value()),
            int(self.display_spin.value()),
            int(self.heat_thresh_spin.value()),
            path,
        )

    def _is_heatmap_cache_valid(self) -> bool:
        """Return ``True`` when the cached heatmap result matches the current parameters.

        Returns:
            bool: ``True`` if a cached result exists and the render parameters
                have not changed since it was computed.
        """
        return (
            self._cached_heatmap_result is not None
            and self._cached_heatmap_params == self._current_heatmap_params()
        )

    def _invalidate_heatmap_cache(self) -> None:
        """Clear the cached heatmap result and parameter snapshot."""
        self._cached_heatmap_result = None
        self._cached_heatmap_params = None

    def _on_slider_changed(self, _) -> None:
        """Update the threshold label and debounce rendering when the slider moves."""
        self.slider_label.setText(f"Percentile Threshold: {self.slider.value():.1f}")
        self._slider_timer.start()

    def _render_current(self) -> None:
        """Render the current image to the canvas, using the heatmap cache when valid.

        On a cache miss the full heatmap pipeline is run via the inference
        controller, the result is cached, and both background images plus
        polygon overlays are pushed to :attr:`canvas_pair`. On a cache hit
        only the segmentation contours are recomputed and the polygons are
        updated without replacing the background images.
        """
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
                left_pil,
                right_pil,
                scale,
                offset,
                s,
                display_w,
                display_h,
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
            self.slider_label.setText(
                f"Percentile Threshold: {self.slider.value():.1f}"
            )
            self.canvas_pair.set_images(
                left_pil, right_pil, self._on_any_edit, contours
            )
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

    def _load_model_clicked(self) -> None:
        """Open file and device pickers, load the model, and start background inference."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Exported Model", "", "PyTorch Models (*.pt *.ckpt)"
        )
        if not path:
            return

        device, ok = QInputDialog.getItem(
            self,
            "Select Device",
            "Inference Hardware:",
            ["auto", "cpu", "cuda", "mps"],
            0,
            False,
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
            self,
            "Model Loaded",
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

    def _load_images_clicked(self) -> None:
        """Open a folder picker and delegate scanning to the shared I/O controller."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", "")
        if not folder:
            return

        # Delegate scanning to IOController so folder-load behaviour is consistent
        # across both panes.  _on_model_reset fires synchronously via modelReset.
        self.io_controller.load_folder(folder)

        if self.dataset_model.rowCount() == 0:
            QMessageBox.information(self, "No Images", "No compatible images found.")
            return

        self.select_row(0)

    def _on_model_reset(self) -> None:
        """React to a dataset model reset by clearing state and restarting inference.

        Connected to :attr:`~models.dataset_model.DatasetTableModel.modelReset`.
        Clears the current row, PIL cache, undo/redo history, and inference
        state, then launches background inference if a model is loaded.
        """
        self._current_row = -1
        self._current_pil = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.inference_model.clear()
        self._proxy.refresh_status_column()

        if self.dataset_model.rowCount() > 0 and self.inference_controller.has_model():
            self._start_background_inference()

    # ------------------------------------------------------------------
    # Background inference
    # ------------------------------------------------------------------

    def _start_background_inference(self) -> None:
        """Build the list of unprocessed images and start the batch inference worker."""
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

        # Controller owns the worker; signals are wired once in __init__
        self.inference_controller.start_batch_inference(pending)

    def _on_worker_result(self, path: str, score_map) -> None:
        """Store a completed score map and refresh the canvas if this is the current image.

        Args:
            path (str): Absolute path of the processed image.
            score_map: NumPy array score map returned by the inference worker.
        """
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

    def _on_worker_finished(self) -> None:
        """Hide the progress bar and show a completion message in the status label."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Batch inference complete.")

    # ------------------------------------------------------------------
    # Render parameter handlers
    # ------------------------------------------------------------------

    def _on_sigma_change(self, v: int) -> None:
        """Update the sigma label and trigger a full re-render when sigma changes.

        Args:
            v (int): New sigma value from the spin box.
        """
        self.sigma_label.setText(f"σ: {v}")
        self._render_current()

    # ------------------------------------------------------------------
    # Undo / Redo  (Memento — transient UI state only)
    # ------------------------------------------------------------------

    def _on_any_edit(self, kind: str) -> None:
        """Push an undo snapshot for edit kinds that mark the start of a gesture.

        Only ``"vertex_drag_begin"`` and ``"polygon_move"`` push a snapshot;
        intermediate drag events are intentionally skipped.

        Args:
            kind (str): Edit event kind string emitted by :class:`SegPathItem`.
        """
        if kind in ("vertex_drag_begin", "polygon_move"):
            self._push_undo()

    def _push_undo(self) -> None:
        """Serialise the current polygon state onto the undo stack and clear the redo stack."""
        if self._block_history:
            return
        self._undo_stack.append(self.canvas_pair.serialize_polygons())
        self._redo_stack.clear()

    def _undo(self) -> None:
        """Pop the last polygon snapshot from the undo stack and restore it."""
        if not self._undo_stack:
            return
        self._block_history = True
        try:
            current = self.canvas_pair.serialize_polygons()
            prior = self._undo_stack.pop()
            self._redo_stack.append(current)
            self.canvas_pair.restore_polygons(
                prior, self._current_pil, self._on_any_edit
            )
        finally:
            self._block_history = False

    def _redo(self) -> None:
        """Pop the top redo snapshot and restore it, pushing the current state to undo."""
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

    def keyPressEvent(self, event) -> None:
        """Handle Delete/Backspace (remove polygons) and ``[`` / ``]`` (scale polygons).

        Args:
            event: The key press event passed by Qt.
        """
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

    def _simplify_selected(self) -> None:
        """Apply Douglas-Peucker simplification to all selected polygons."""
        eps = float(self.eps_spin.value())
        self._push_undo()
        for item in self.canvas_pair.scene_left.selectedItems():
            if isinstance(item, SegPathItem):
                item.simplify(eps)

    def _simplify_all(self) -> None:
        """Apply Douglas-Peucker simplification to every polygon in the left scene."""
        eps = float(self.eps_spin.value())
        self._push_undo()
        for item in self.canvas_pair.scene_left.items():
            if isinstance(item, SegPathItem):
                item.simplify(eps)

    # ------------------------------------------------------------------
    # Send polygons to AnnoMate
    # ------------------------------------------------------------------

    def _send_annotations(self) -> None:
        """Emit :attr:`polygonsSent` with selected polygons, or all if none are selected.

        Converts display-coordinate polygons to original image coordinates via
        :meth:`~views.microsentry.canvas.CanvasPair.get_selected_polygons_original_coords`
        or :meth:`~views.microsentry.canvas.CanvasPair.get_polygons_original_coords`
        before emitting.
        """
        selected = [
            i
            for i in self.canvas_pair.scene_left.selectedItems()
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

    def set_view_state(self, rx: float, ry: float, scale: float) -> None:
        """Apply a pan/zoom state from an external source for cross-tab sync.

        Args:
            rx (float): Relative horizontal center position (0.0–1.0).
            ry (float): Relative vertical center position (0.0–1.0).
            scale (float): Absolute zoom scale factor.
        """
        self.canvas_pair.set_view_state(rx, ry, scale)

    def showEvent(self, event) -> None:
        """Fit both canvas views to their scenes when the widget is first shown.

        Args:
            event: The show event passed by Qt.
        """
        super().showEvent(event)
        QTimer.singleShot(50, self.canvas_pair.fit_views)

    def resizeEvent(self, event) -> None:
        """Re-fit both canvas views when the widget is resized.

        Args:
            event: The resize event passed by Qt.
        """
        super().resizeEvent(event)
        self.canvas_pair.fit_views()
