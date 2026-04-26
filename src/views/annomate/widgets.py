"""
Custom UI Widgets for AnnoMate.

This module provides specialized Qt widgets used throughout the application,
including a collapsible splitter handle and a table widget with cyclic navigation.
"""

from PySide6.QtWidgets import QSplitter, QSplitterHandle, QTableWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QKeyEvent


class SidebarHandle(QSplitterHandle):
    """Custom splitter handle with hover effects and double-click collapse/expand.

    Replaces the default :class:`~PySide6.QtWidgets.QSplitterHandle` so that
    QSS ``:hover`` selectors work and the sidebar can be toggled by
    double-clicking the divider.
    """

    def __init__(self, orientation: Qt.Orientation, parent: QSplitter) -> None:
        """Initialize the handle and enable hover attribute for QSS styling.

        Args:
            orientation (Qt.Orientation): Orientation of the parent splitter.
            parent (QSplitter): The splitter that owns this handle.
        """
        super().__init__(orientation, parent)
        # Enable hover attribute so that QSS :hover selectors work correctly
        self.setAttribute(Qt.WA_Hover, True)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Toggle the sidebar (second splitter widget) open or closed.

        Collapses the sidebar to zero width on the first double-click, saving
        the current width so it can be restored. Expands it back to the saved
        width (or ``400`` px as a fallback) on the next double-click.

        Args:
            event (QMouseEvent): The mouse double-click event.
        """
        splitter = self.splitter()
        sizes = splitter.sizes()

        # Ensure we have at least two widgets (Main Canvas, Sidebar)
        if len(sizes) < 2:
            return

        main_width, side_width = sizes[0], sizes[1]

        if side_width > 0:
            # COLLAPSE: Save current width and snap to 0
            splitter._last_side_width = side_width
            splitter.setSizes([main_width + side_width, 0])
        else:
            # EXPAND: Restore last known width or default to 400
            last_width = getattr(splitter, "_last_side_width", 400)
            if last_width == 0:
                last_width = 400

            # Calculate new sizes ensuring we don't exceed total width
            total_width = main_width + side_width
            splitter.setSizes([total_width - last_width, last_width])


class CustomSplitter(QSplitter):
    """QSplitter subclass that provides a :class:`SidebarHandle` for each divider."""

    def createHandle(self) -> QSplitterHandle:
        """Create a :class:`SidebarHandle` instead of the default handle.

        Returns:
            QSplitterHandle: A new :class:`SidebarHandle` bound to this
                splitter's orientation.
        """
        return SidebarHandle(self.orientation(), self)


class WrappingTableWidget(QTableWidget):
    """QTableWidget with cyclic keyboard navigation.

    Pressing ``Down`` on the last row wraps to the first row.
    Pressing ``Up`` on the first row wraps to the last row.
    All other keys fall through to the default implementation.
    """

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press events, wrapping Up/Down navigation at the boundaries.

        Args:
            event (QKeyEvent): The key press event to handle.
        """
        if event.key() == Qt.Key_Down:
            # If on the last row, jump to top
            if self.rowCount() > 0 and self.currentRow() == self.rowCount() - 1:
                self.selectRow(0)
                return

        elif event.key() == Qt.Key_Up:
            # If on the first row, jump to bottom
            if self.rowCount() > 0 and self.currentRow() == 0:
                self.selectRow(self.rowCount() - 1)
                return

        # Default behavior for other keys
        super().keyPressEvent(event)
