import os
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QColor, QBrush

from core.dataset_state import DatasetState
from core.utils import polygon_area

logger = logging.getLogger("AnnoMate.DatasetModel")

class DatasetTableModel(QAbstractTableModel):
    """
    Qt Model layer. Owns the QAbstractTableModel interface and exposes a
    typed query/command API so Views never touch DatasetState directly.

    Color rule: colors are stored as (r, g, b) tuples in State and in
    all controller/domain code. QColor is only constructed here, at the
    Qt boundary, for display roles.
    """

    def __init__(self, state: DatasetState, parent=None):
        super().__init__(parent)
        self.state = state
        self.headers = ["Filename", "Status"]

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.state.image_files)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None

        filename = self.state.image_files[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return Path(filename).stem
            elif col == 1:
                return "Reviewed" if self.state.is_reviewed(filename) else "Pending"

        elif role == Qt.BackgroundRole:
            if col == 1:
                if self.state.is_reviewed(filename):
                    return QBrush(QColor(210, 245, 210))
                else:
                    return QBrush(QColor(255, 235, 210))

        return None

    def load_folder(self, directory: str, files: list):
        self.beginResetModel()
        self.state.clear()
        self.state.image_dir = directory
        self.state.image_files = files
        self.endResetModel()

    def add_annotation(self, row: int, category: str, polygon: list):
        if not (0 <= row < self.rowCount()):
            logger.error("Failed to add annotation: Row %d is out of bounds.", row)
            return
        
        filename = self.state.image_files[row]
        logger.debug("Adding '%s' annotation to '%s' (%d points)", category, filename, len(polygon))
        
        self.state.add_annotation(filename, category, polygon)
        self._emit_row(row)

    def delete_annotation(self, row: int, annotation_idx: int):
        if not (0 <= row < self.rowCount()):
            return
        
        filename = self.state.image_files[row]
        logger.debug("Deleted annotation at index %d from '%s'", annotation_idx, filename)
        
        self.state.delete_annotation(self.state.image_files[row], annotation_idx)
        self._emit_row(row)

    def update_annotation_points(self, row: int, annotation_idx: int, points: list):
        if not (0 <= row < self.rowCount()):
            return
        self.state.update_annotation_points(self.state.image_files[row], annotation_idx, points)
        self._emit_row(row)

    def set_inspector(self, row: int, value: str):
        if not (0 <= row < self.rowCount()):
            return
        self.state.set_inspector(self.state.image_files[row], value)
        self._emit_row(row)

    def set_note(self, row: int, value: str):
        if not (0 <= row < self.rowCount()):
            return
        self.state.set_note(self.state.image_files[row], value)
        self._emit_row(row)

    def add_class(self, name: str, color: tuple) -> bool:
        if name in self.state.class_names:
            return False
        self.state.add_class(name, color)
        return True

    def set_class_color(self, name: str, color: tuple):
        """Update an existing class's display color."""
        self.state.class_colors[name] = color

    def delete_class(self, name: str):
        self.state.delete_class(name)
        if self.rowCount() > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
            )

    def sort_annotations(self, row: int):
        if not (0 <= row < self.rowCount()):
            return
        filename = self.state.image_files[row]
        annos = self.state.annotations.get(filename, [])
        annos.sort(key=lambda a: polygon_area(a["polygon"]), reverse=True)
        self._emit_row(row)

    # ------------------------------------------------------------------ #
    # Query API — Views must use these instead of accessing .state
    # ------------------------------------------------------------------ #

    def get_image_dir(self) -> str:
        return self.state.image_dir

    def get_image_path(self, row: int) -> str:
        return os.path.join(self.state.image_dir, self.state.image_files[row])

    def get_annotations(self, row: int) -> list:
        if not (0 <= row < self.rowCount()):
            return []
        return self.state.annotations.get(self.state.image_files[row], [])

    def get_class_names(self) -> list:
        return list(self.state.class_names)

    def get_class_color(self, class_name: str) -> tuple:
        """Returns an (r, g, b) tuple. Callers convert to QColor at draw time."""
        return self.state.class_colors.get(class_name, (255, 255, 255))

    def get_used_class_colors(self) -> list:
        """Returns all currently assigned (r, g, b) tuples."""
        return list(self.state.class_colors.values())

    def get_inspector(self, row: int) -> str:
        if not (0 <= row < self.rowCount()):
            return ""
        return self.state.inspectors.get(self.state.image_files[row], "")

    def get_note(self, row: int) -> str:
        if not (0 <= row < self.rowCount()):
            return ""
        return self.state.notes.get(self.state.image_files[row], "")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _emit_row(self, row: int):
        self.dataChanged.emit(
            self.index(row, 0),
            self.index(row, self.columnCount() - 1),
        )
