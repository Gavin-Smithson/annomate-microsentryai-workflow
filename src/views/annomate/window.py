"""
ImageAnnotator — the AnnoMate View.

Rules enforced here:
  V1  All file/folder dialogs and message boxes live in this file, not in
      controllers.  Controllers receive plain paths and return plain values.
  V2  This file never reads self.model.state directly.  All data access goes
      through the Model's query API (get_annotations, get_class_color, …).
  V3  No disk I/O here.  Image loading is delegated to io_controller and the
      resulting ndarray is handed to the canvas widget.
  V4  Colors are handled as (r, g, b) tuples until the last moment; QColor
      is only constructed right before a draw call.
"""

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit,
    QScrollArea, QTableView, QHeaderView, QAbstractItemView,
    QListWidget, QColorDialog, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from core.utils.geometry import scale_polygon_about_center
from views.annomate.image_label import ImageLabel, POLYGON
from views.annomate.widgets import CustomSplitter


class ImageAnnotator(QMainWindow):
    """Main window for the AnnoMate annotation pane.

    Owns all Qt dialogs (file pickers, message boxes) for the AnnoMate
    workflow. Delegates all file I/O to ``io_controller`` and all data
    mutations to ``model``. Never reads ``model.state`` directly — all data
    access goes through the model's query API.

    Attributes:
        model: Dataset model exposing the query/command API.
        io_controller: I/O controller for folder scanning and export.
        canvas (ImageLabel): Interactive image canvas widget.

    Signals:
        viewChanged (float, float, float): Reserved for future multi-tab
            pan/zoom sync.
        row_selection_changed (int): Emitted when the selected row changes,
            for cross-tab sync.
    """

    viewChanged           = Signal(float, float, float)  # reserved for future multi-tab sync
    row_selection_changed = Signal(int)                  # emitted when the current row changes (for cross-tab sync)

    def __init__(self, model, io_controller) -> None:
        """Initialize ImageAnnotator and build the UI.

        Args:
            model: Dataset model instance exposing query/command API.
            io_controller: I/O controller for folder loading and export.
        """
        super().__init__()
        self.model = model
        self.io_controller = io_controller
        self._syncing = False   # guard against recursive cross-tab row sync

        self._init_ui()
        self._setup_connections()

    # ================================================================== #
    # UI Construction
    # ================================================================== #

    def _init_ui(self) -> None:
        """Build the top-level horizontal splitter with canvas and sidebar."""
        self.main_splitter = CustomSplitter(Qt.Horizontal, self)
        self._setup_canvas()
        self._setup_sidebar()
        self.main_splitter.setSizes([900, 420])
        self.setCentralWidget(self.main_splitter)

    def _setup_canvas(self) -> None:
        """Create the :class:`ImageLabel` canvas and add it to the splitter."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageLabel(self)
        layout.addWidget(self.canvas)
        self.main_splitter.addWidget(container)

    def _setup_sidebar(self) -> None:
        """Build the scrollable sidebar and populate it with control groups."""
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)

        side_widget = QWidget()
        self.side_layout = QVBoxLayout(side_widget)
        self.side_layout.setContentsMargins(10, 10, 10, 10)
        self.side_layout.setSpacing(10)

        self._create_tray_header()
        self._create_nav_controls()
        self._create_zoom_controls()
        self._create_class_controls()
        self._create_meta_inputs()
        self._create_dataset_table()
        self._create_annotation_list()
        self._create_ops_controls()
        self._create_export_controls()

        self.side_layout.addStretch()
        scroll.setWidget(side_widget)
        self.main_splitter.addWidget(scroll)

    def _create_tray_header(self) -> None:
        """Add the folder-open button and current directory label to the sidebar."""
        row = QHBoxLayout()
        self.btn_open_folder = QPushButton("📂 Open Folder")
        self.btn_open_folder.setFixedHeight(35)
        self.lbl_dir = QLabel("—")
        self.lbl_dir.setStyleSheet("font-weight: bold;")
        row.addWidget(self.btn_open_folder)
        row.addWidget(self.lbl_dir, 1)
        self.side_layout.addLayout(row)

    def _create_nav_controls(self) -> None:
        """Add Prev/Next navigation buttons and the image counter label."""
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev (A)")
        self.btn_next = QPushButton("Next (D) ▶")
        self.btn_prev.setShortcut("A")
        self.btn_next.setShortcut("D")
        self.lbl_img = QLabel("0 / 0")
        self.lbl_img.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.lbl_img)
        self.side_layout.addLayout(nav)

    def _create_zoom_controls(self) -> None:
        """Add zoom in/out, reset view, and polygon tool buttons to the sidebar."""
        bar = QHBoxLayout()
        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_zoom_in.clicked.connect(self.canvas.zoom_in)
        self.btn_zoom_out = QPushButton("Zoom -")
        self.btn_zoom_out.clicked.connect(self.canvas.zoom_out)
        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.clicked.connect(self.canvas.reset_view)
        self.btn_poly = QPushButton("Polygon (P)")
        self.btn_poly.setCheckable(True)
        self.btn_poly.setShortcut("P")
        self.btn_poly.toggled.connect(self._on_polygon_tool_toggled)
        for w in (self.btn_zoom_in, self.btn_zoom_out, self.btn_reset_view, self.btn_poly):
            bar.addWidget(w)
        self.side_layout.addLayout(bar)

    def _create_class_controls(self) -> None:
        """Add class combo, name entry, add-class button, and color/delete buttons."""
        row1 = QHBoxLayout()
        self.class_combo = QComboBox()
        # V2: use model query instead of model.state
        self.class_combo.addItems(self.model.get_class_names())
        self.class_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.class_name_edit = QLineEdit()
        self.class_name_edit.setPlaceholderText("Enter class...")
        self.btn_add_class = QPushButton("Add Class")
        row1.addWidget(self.class_combo, 2)
        row1.addWidget(self.class_name_edit, 2)
        row1.addWidget(self.btn_add_class)
        self.side_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_color = QPushButton("Change Color")
        self.btn_del_class = QPushButton("Delete Class")
        row2.addWidget(self.btn_color)
        row2.addWidget(self.btn_del_class)
        self.side_layout.addLayout(row2)

    def _create_meta_inputs(self) -> None:
        """Add inspector name and image-note text inputs to the sidebar."""
        self.side_layout.addWidget(QLabel("Inspector"))
        self.inspector_edit = QLineEdit()
        self.inspector_edit.setPlaceholderText("Inspector name…")
        self.side_layout.addWidget(self.inspector_edit)

        self.side_layout.addWidget(QLabel("Image note"))
        self.note_edit = QTextEdit()
        self.note_edit.setMaximumHeight(80)
        self.side_layout.addWidget(self.note_edit)

    def _create_dataset_table(self) -> None:
        """Add the dataset image table view to the sidebar."""
        self.side_layout.addWidget(QLabel("Dataset Images:"))
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setMinimumHeight(200)
        self.side_layout.addWidget(self.table_view)

    def _create_annotation_list(self) -> None:
        """Add the per-image annotation list widget to the sidebar."""
        self.side_layout.addWidget(QLabel("Annotations in Current Image:"))
        self.ann_list = QListWidget()
        self.ann_list.setMaximumHeight(150)
        self.side_layout.addWidget(self.ann_list)

    def _create_ops_controls(self) -> None:
        """Add Delete Selected and Sort by Area annotation operation buttons."""
        ops = QHBoxLayout()
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_sort = QPushButton("Sort by Area")
        ops.addWidget(self.btn_delete)
        ops.addWidget(self.btn_sort)
        self.side_layout.addLayout(ops)

    def _create_export_controls(self) -> None:
        """Add export polygon/CSV and import data buttons to the sidebar."""
        exp = QHBoxLayout()
        self.btn_export_polys = QPushButton("Export Polygons + Data")
        self.btn_export_csv = QPushButton("Export CSV")
        exp.addWidget(self.btn_export_polys)
        exp.addWidget(self.btn_export_csv)
        self.side_layout.addLayout(exp)

        self.btn_import_data = QPushButton("Import Data JSON")
        self.side_layout.addWidget(self.btn_import_data)

    # ================================================================== #
    # Signal Wiring
    # ================================================================== #

    def _setup_connections(self) -> None:
        """Wire all Qt signals to their corresponding slot methods."""
        self.btn_open_folder.clicked.connect(self.on_open_folder_clicked)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)

        self.table_view.selectionModel().selectionChanged.connect(
            self.on_table_selection_changed
        )
        self.model.modelReset.connect(self.on_model_reset)
        self.model.dataChanged.connect(self.on_model_data_changed)

        # Canvas signals replace the old main_window hasattr callbacks
        self.canvas.polygonFinished.connect(self.finish_polygon)
        self.canvas.polygonEdited.connect(self.update_polygon_points)
        self.canvas.polygonSelected.connect(self.on_polygon_selected)
        self.canvas.toolCanceled.connect(lambda: self.btn_poly.setChecked(False))

        self.class_combo.currentTextChanged.connect(self.update_canvas_active_color)
        self.btn_add_class.clicked.connect(self.add_class_from_edit)
        self.btn_color.clicked.connect(self.change_class_color)
        self.btn_del_class.clicked.connect(self.delete_current_class)

        self.inspector_edit.editingFinished.connect(self._store_inspector)
        self.note_edit.textChanged.connect(self._store_note)

        self.ann_list.itemSelectionChanged.connect(self.on_ann_list_selection)

        self.btn_delete.clicked.connect(self.delete_selected_annotation)
        self.btn_sort.clicked.connect(self.sort_by_area)

        self.btn_export_polys.clicked.connect(self.on_export_polys_clicked)
        self.btn_export_csv.clicked.connect(self.on_export_csv_clicked)
        self.btn_import_data.clicked.connect(self.on_import_data_clicked)

    # ================================================================== #
    # Slots — Navigation
    # ================================================================== #

    def on_open_folder_clicked(self) -> None:
        """Open a folder picker dialog and delegate loading to the I/O controller."""
        # V1: dialog lives in the View
        directory = QFileDialog.getExistingDirectory(
            self, "Open image folder", os.getcwd()
        )
        if directory:
            self.io_controller.load_folder(directory)

    def on_model_reset(self) -> None:
        """React to a full model reset by updating the directory label and selecting row 0."""
        # V2: model query instead of model.state
        image_dir = self.model.get_image_dir()
        self.lbl_dir.setText(Path(image_dir).name if image_dir else "—")
        if self.model.rowCount() > 0:
            self.table_view.selectRow(0)
            self.table_view.setFocus()
        else:
            self.lbl_img.setText("0 / 0")

    def select_row(self, row: int) -> None:
        """Select *row* in the dataset table without emitting :attr:`row_selection_changed`.

        Used by the main window to synchronise the AnnoMate table with the
        MicroSentryAI pane. The :attr:`_syncing` guard prevents the resulting
        ``selectionChanged`` signal from triggering a recursive sync emission.

        Args:
            row (int): Zero-based row index to select.
        """
        self._syncing = True
        self.table_view.selectRow(row)
        self._syncing = False

    def on_table_selection_changed(self, selected, deselected) -> None:
        """Load and display the image for the newly selected dataset row.

        Delegates image loading to the I/O controller and then refreshes
        the canvas overlays, metadata fields, and image counter label.
        Emits :attr:`row_selection_changed` unless :attr:`_syncing` is set.

        Args:
            selected: Qt selection object for newly selected indexes.
            deselected: Qt selection object for deselected indexes.
        """
        indexes = self.table_view.selectionModel().selectedRows()
        if not indexes:
            return
        row = indexes[0].row()

        # V3: disk I/O delegated to controller
        bgr = self.io_controller.load_image_for_display(row)
        if bgr is not None:
            self.canvas.set_image(bgr)      # V3: canvas receives data, not a path
            self.refresh_image_view(row)
            self.refresh_meta_fields(row)

        self._update_image_counter(row)
        if not self._syncing:
            self.row_selection_changed.emit(row)

    def next_image(self) -> None:
        """Advance the selection to the next row, clamped at the last row."""
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        self.table_view.selectRow(min(sel.currentIndex().row() + 1, self.model.rowCount() - 1))

    def prev_image(self) -> None:
        """Move the selection to the previous row, clamped at row 0."""
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        self.table_view.selectRow(max(sel.currentIndex().row() - 1, 0))

    def _update_image_counter(self, row: int) -> None:
        """Update the image counter label to show ``current / total``.

        Args:
            row (int): Zero-based index of the currently selected image.
        """
        total = self.model.rowCount()
        self.lbl_img.setText(f"{row + 1} / {total}" if total > 0 else "0 / 0")

    # ================================================================== #
    # Slots — Model observer
    # ================================================================== #

    def on_model_data_changed(self, top_left, bottom_right, roles) -> None:
        """Refresh the canvas overlays when the model data affecting the current row changes.

        Skips the refresh if a vertex or polygon drag is live to avoid
        discarding the in-progress drag preview from ``_overlays``.

        Args:
            top_left: Top-left model index of the changed region.
            bottom_right: Bottom-right model index of the changed region.
            roles: List of data roles that changed (unused).
        """
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        current_row = sel.currentIndex().row()
        if top_left.row() <= current_row <= bottom_right.row():
            # Do not replace overlays while a vertex/polygon drag is live; the
            # drag preview lives in _overlays and would be silently discarded,
            # causing the commit on mouseRelease to write stale coordinates.
            if not self.canvas.is_dragging():
                self.refresh_image_view(current_row)

    # ================================================================== #
    # Slots — Canvas callbacks (called by ImageLabel)
    # ================================================================== #

    def finish_polygon(self, points: list) -> None:
        """Add a completed polygon from the canvas to the model.

        Called when the canvas emits :attr:`~ImageLabel.polygonFinished`.
        Does nothing if no row is selected or no class is active.

        Args:
            points (list): Sequence of ``(x, y)`` coordinate pairs in
                original-image coordinates.
        """
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        current_class = self.class_combo.currentText()
        if not current_class:
            return
        self.model.add_annotation(sel.currentIndex().row(), current_class, points)

    def update_polygon_points(self, idx: int, points: list) -> None:
        """Commit updated vertex positions for an edited polygon to the model.

        Called when the canvas emits :attr:`~ImageLabel.polygonEdited`.

        Args:
            idx (int): Zero-based annotation index within the current image.
            points (list): New sequence of ``(x, y)`` coordinate pairs.
        """
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        self.model.update_annotation_points(sel.currentIndex().row(), idx, points)

    def on_polygon_selected(self, idx: int) -> None:
        """Sync the annotation list selection when the canvas selects a polygon.

        Called when the canvas emits :attr:`~ImageLabel.polygonSelected`.
        Blocks annotation-list signals to avoid a feedback loop.

        Args:
            idx (int): Zero-based annotation index of the selected polygon.
        """
        self.ann_list.blockSignals(True)
        self.ann_list.setCurrentRow(idx)
        self.ann_list.blockSignals(False)

    # ================================================================== #
    # Slots — Class management
    # ================================================================== #

    def update_canvas_active_color(self, class_name: str) -> None:
        """Push the color for *class_name* to the canvas draw tool.

        Converts the stored ``(r, g, b)`` tuple to a
        :class:`~PySide6.QtGui.QColor` at the last moment (V4 rule).

        Args:
            class_name (str): Class label whose color should become active.
        """
        # V4: tuple → QColor at the last moment, right before the draw call
        rgb = self.model.get_class_color(class_name)
        self.canvas.set_active_color(QColor(*rgb))

    def add_class_from_edit(self) -> None:
        """Register a new class from the name-entry field and activate the polygon tool."""
        name = self.class_name_edit.text().strip()
        if not name:
            return
        color = self._pick_next_unique_color()   # returns a tuple
        if self.model.add_class(name, color):
            self.class_combo.addItem(name)
            self.class_combo.setCurrentText(name)
            self.class_name_edit.clear()
            self.btn_poly.setChecked(True)
            self.update_canvas_active_color(name)

    def change_class_color(self) -> None:
        """Open a color picker and update the active class color in the model."""
        name = self.class_combo.currentText().strip()
        if not name:
            return
        # V2: read via model query; V4: convert tuple → QColor for dialog seed
        current_rgb = self.model.get_class_color(name)
        col = QColorDialog.getColor(QColor(*current_rgb), self)
        if col.isValid():
            # V4: store as tuple, not QColor
            self.model.set_class_color(name, (col.red(), col.green(), col.blue()))
            sel = self.table_view.selectionModel()
            if sel.hasSelection():
                self.refresh_image_view(sel.currentIndex().row())
            self.update_canvas_active_color(name)

    def delete_current_class(self) -> None:
        """Remove the active class from the model and the class combo box."""
        name = self.class_combo.currentText().strip()
        if not name:
            return
        self.model.delete_class(name)
        self.class_combo.removeItem(self.class_combo.currentIndex())
        if self.class_combo.count() > 0:
            self.update_canvas_active_color(self.class_combo.currentText())
        sel = self.table_view.selectionModel()
        if sel.hasSelection():
            self.refresh_image_view(sel.currentIndex().row())

    def _pick_next_unique_color(self) -> tuple:
        """Return the first ``DEFAULT_CLASS_COLOR`` not already registered.

        Falls back to the first default color if all defaults are exhausted.

        Returns:
            tuple: An ``(r, g, b)`` color tuple not present in the current
                class registry, or ``DEFAULT_CLASS_COLORS[0]`` as a fallback.
        """
        from core.utils.constants import DEFAULT_CLASS_COLORS
        # V2: model query; V4: already tuples
        used = set(self.model.get_used_class_colors())
        for cand in DEFAULT_CLASS_COLORS:
            if cand not in used:
                return cand
        return DEFAULT_CLASS_COLORS[0]

    def _on_polygon_tool_toggled(self, checked: bool) -> None:
        """Activate or deactivate the polygon draw tool on the canvas.

        Args:
            checked (bool): ``True`` to activate polygon drawing; ``False``
                to switch back to no-tool mode.
        """
        self.canvas.set_tool(POLYGON if checked else None)

    # ================================================================== #
    # Slots — Metadata
    # ================================================================== #

    def _store_inspector(self) -> None:
        """Write the inspector field value to the model for the current row."""
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        self.model.set_inspector(sel.currentIndex().row(), self.inspector_edit.text().strip())

    def _store_note(self) -> None:
        """Write the note text-area value to the model for the current row."""
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        self.model.set_note(sel.currentIndex().row(), self.note_edit.toPlainText().strip())

    def refresh_meta_fields(self, row: int) -> None:
        """Populate inspector and note fields from the model for *row*.

        Blocks signals on both widgets while updating to prevent feedback
        loops from the ``editingFinished`` and ``textChanged`` slots.

        Args:
            row (int): Zero-based row index of the image to display metadata for.
        """
        # V2: model queries instead of state access
        self.inspector_edit.blockSignals(True)
        self.note_edit.blockSignals(True)
        self.inspector_edit.setText(self.model.get_inspector(row))
        self.note_edit.setPlainText(self.model.get_note(row))
        self.inspector_edit.blockSignals(False)
        self.note_edit.blockSignals(False)

    # ================================================================== #
    # Slots — Annotation list
    # ================================================================== #

    def on_ann_list_selection(self) -> None:
        """Sync the canvas's selected polygon index when the annotation list selection changes."""
        idxs = [i.row() for i in self.ann_list.selectedIndexes()]
        self.canvas.selected_polygon_idx = idxs[0] if idxs else -1
        self.canvas.update()

    def delete_selected_annotation(self) -> None:
        """Delete the annotation selected in the annotation list from the model."""
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        selected_items = self.ann_list.selectedIndexes()
        if selected_items:
            self.model.delete_annotation(sel.currentIndex().row(), selected_items[0].row())

    def sort_by_area(self) -> None:
        """Sort the current image's annotations by polygon area descending."""
        sel = self.table_view.selectionModel()
        if sel.hasSelection():
            self.model.sort_annotations(sel.currentIndex().row())

    # ================================================================== #
    # Slots — Export / Import  (V1: dialogs live here, not in controller)
    # ================================================================== #

    def on_export_polys_clicked(self) -> None:
        """Open a folder picker and export polygons and metadata JSON to the chosen path."""
        out_dir = QFileDialog.getExistingDirectory(
            self, "Choose output folder", os.getcwd()
        )
        if not out_dir:
            return
        try:
            msg = self.io_controller.export_polygons_and_data(out_dir)
            QMessageBox.information(self, "Export", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def on_export_csv_clicked(self) -> None:
        """Open a save-file dialog and export annotation metadata as CSV."""
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "metadata.csv", "CSV (*.csv)"
        )
        if not out_path:
            return
        try:
            msg = self.io_controller.export_csv(out_path)
            QMessageBox.information(self, "Export", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def on_import_data_clicked(self) -> None:
        """Open a file picker and import annotation data from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data JSON", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.io_controller.import_data_json(path)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))
            return

        self.refresh_class_combo()

    def refresh_class_combo(self) -> None:
        """Re-populate the class combo box from the model.

        Called after any external operation that may change the class
        registry (project open, COCO import, etc.).
        """
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(self.model.get_class_names())
        self.class_combo.blockSignals(False)
        if self.class_combo.count() > 0:
            self.update_canvas_active_color(self.class_combo.currentText())

    # ================================================================== #
    # View refresh helpers
    # ================================================================== #

    def refresh_image_view(self, row: int) -> None:
        """Refresh the annotation list and canvas overlays from the model for *row*.

        Reads annotations via the model query API (V2 rule), converts each
        class color tuple to :class:`~PySide6.QtGui.QColor` at the draw
        boundary (V4 rule), and pushes the overlay list to the canvas.

        Args:
            row (int): Zero-based row index of the image to refresh.
        """
        # V2: model query — never touches .state
        annos = self.model.get_annotations(row)

        self.ann_list.blockSignals(True)
        self.ann_list.clear()
        for a in annos:
            self.ann_list.addItem(f"{a['category_name']} — {len(a['polygon'])} pts")
        self.ann_list.blockSignals(False)

        # V4: tuple → QColor right at the draw boundary
        overlays = [
            (a["polygon"], QColor(*self.model.get_class_color(a["category_name"])))
            for a in annos
        ]
        self.canvas.set_overlays(overlays)

    # ================================================================== #
    # Hotkeys
    # ================================================================== #

    def keyPressEvent(self, event) -> None:
        """Scale the selected polygon with ``[`` (shrink) and ``]`` (grow).

        Args:
            event: The key press event passed by Qt.
        """
        if event.key() == Qt.Key_BracketLeft:
            self._scale_selected_polygon(0.9)
        elif event.key() == Qt.Key_BracketRight:
            self._scale_selected_polygon(1.1)
        super().keyPressEvent(event)

    def _scale_selected_polygon(self, factor: float) -> None:
        """Scale the currently selected polygon by *factor* about its centroid.

        Reads the polygon from the model, applies
        :func:`~core.utils.geometry.scale_polygon_about_center`, and writes
        the result back via the model command API. Does nothing when no
        polygon is selected or the annotation list is empty.

        Args:
            factor (float): Scaling multiplier (e.g. ``0.9`` to shrink,
                ``1.1`` to grow).
        """
        idx = self.canvas.selected_polygon_idx
        sel = self.table_view.selectionModel()
        if idx == -1 or not sel.hasSelection():
            return
        row = sel.currentIndex().row()
        # V2: model query
        annos = self.model.get_annotations(row)
        if not (0 <= idx < len(annos)):
            return
        pts = annos[idx]["polygon"]
        if not pts:
            return
        new_pts = scale_polygon_about_center(pts, factor)
        self.model.update_annotation_points(row, idx, new_pts)

    # ================================================================== #
    # Cross-pane API — called by main.py polygon transfer handler
    # ================================================================== #

    def receive_polygons(self, polygons: list, class_name: str) -> None:
        """Add polygons transferred from the MicroSentryAI pane to the current image.

        Called by the main window's polygon-transfer handler. Iterates over
        each polygon in *polygons* and adds it as a new annotation, then
        refreshes the canvas view.

        Args:
            polygons (list): List of polygon point lists, each a sequence of
                ``(x, y)`` coordinate pairs in original-image coordinates.
            class_name (str): Class label to assign to every transferred polygon.
        """
        sel = self.table_view.selectionModel()
        if not sel.hasSelection():
            return
        row = sel.currentIndex().row()
        for pts in polygons:
            self.model.add_annotation(row, class_name, pts)
        self.refresh_image_view(row)
