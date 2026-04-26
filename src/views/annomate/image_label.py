"""
Image Label Widget for AnnoMate.

This module defines the `ImageLabel` class, a custom PySide6 widget that handles:
1.  Displaying images with high-performance zooming and panning.
2.  Interactive polygon annotation (drawing, canceling, finishing).
3.  Rendering overlays (existing annotations) on top of the image.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (
    QPainter,
    QPen,
    QPolygonF,
    QPixmap,
    QImage,
    QBrush,
    QColor,
    QMouseEvent,
    QWheelEvent,
    QKeyEvent,
    QPaintEvent,
)
from PySide6.QtWidgets import QLabel, QSizePolicy

# Tool Constants
POLYGON = "polygon"


class ImageLabel(QLabel):
    """Custom QLabel for image display with zoom, pan, and polygon annotation.

    Handles rendering of a BGR image alongside existing annotation overlays
    and supports interactive polygon drawing, vertex dragging, and whole-polygon
    dragging. All coordinates reported via signals are in the original (unscaled)
    image coordinate system.

    Attributes:
        current_tool (Optional[str]): Active tool name; ``"polygon"`` or ``None``.
        current_polygon_points (List[QPointF]): In-progress polygon vertices in
            display (scaled) coordinates.
        selected_polygon_idx (int): Index of the currently selected overlay polygon,
            or ``-1`` when none is selected.
        editing_polygon_idx (int): Index of the overlay polygon in vertex-edit mode,
            or ``-1`` when not editing.
        dragging_vertex_idx (int): Index of the vertex being dragged, or ``-1``.

    Signals:
        polygonFinished (list): Emitted when the user completes a polygon. Carries
            ``List[Tuple[float, float]]`` in original image coordinates.
        polygonEdited (int, list): Emitted after a vertex or polygon drag completes.
            Carries ``(polygon_idx, List[Tuple[float, float]])`` in original coords.
        polygonSelected (int): Emitted when a polygon is clicked. Carries its index,
            or ``-1`` on deselect.
        toolCanceled (): Emitted when ``Escape`` is pressed while a tool is active.
    """

    polygonFinished = Signal(list)  # pts: List[Tuple[float, float]] in original coords
    polygonEdited = Signal(int, list)  # (polygon_idx, pts in original coords)
    polygonSelected = Signal(int)  # polygon index (-1 for deselect)
    toolCanceled = Signal()  # Escape pressed while polygon tool active

    def __init__(self, parent: object = None) -> None:
        """Initialize ImageLabel with default zoom, pan, and annotation state.

        Args:
            parent (object): Optional Qt parent widget. Defaults to ``None``.
        """
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)  # Allow widget to catch Key_Escape

        self.current_tool: Optional[str] = None
        self._active_color = QColor(0, 200, 0)

        self._orig_image_bgr: Optional[np.ndarray] = None
        self._display_qpix: Optional[QPixmap] = None

        self._base_scale = 1.0
        self._zoom = 1.0
        self._pan = QPointF(0, 0)

        self._panning = False
        self._last_mouse_pos: Optional[QPointF] = None
        self._mouse_pos: Optional[QPointF] = None

        self.current_polygon_points: List[QPointF] = []
        self._overlays: List[Tuple[List[QPointF], QColor]] = []

        # --- UI State Trackers ---
        self.selected_polygon_idx: int = -1
        self.editing_polygon_idx: int = -1
        self.dragging_vertex_idx: int = -1
        self._dragging_polygon: bool = False

    def set_image(self, bgr: np.ndarray, max_display_dim: int = 1200) -> None:
        """Load a BGR ndarray and prepare it for display.

        Computes a ``_base_scale`` that fits the image within *max_display_dim*
        pixels on its longest edge, converts to RGB, and stores a
        :class:`~PySide6.QtGui.QPixmap` for painting. Resets all in-progress
        annotation and view state (zoom, pan, overlays).

        The caller (controller) is responsible for reading the file from disk;
        this widget only handles rendering.

        Args:
            bgr (np.ndarray): Source image in BGR format as returned by
                ``cv2.imread``.
            max_display_dim (int): Maximum pixel length of the longer image
                edge at ``zoom == 1``. Defaults to ``1200``.
        """
        self._orig_image_bgr = bgr
        h, w = bgr.shape[:2]

        self._base_scale = (
            1.0 if max(h, w) <= max_display_dim else max_display_dim / float(max(h, w))
        )

        self._zoom = 1.0
        self._pan = QPointF(0, 0)

        new_w = int(w * self._base_scale)
        new_h = int(h * self._base_scale)

        resized_bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        qimg = QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
        )
        self._display_qpix = QPixmap.fromImage(qimg)

        # --- Reset all tracking states on new image ---
        self.clear_current_polygon()
        self._overlays = []
        self.selected_polygon_idx = -1
        self.editing_polygon_idx = -1
        self.dragging_vertex_idx = -1
        self._dragging_polygon = False

        self.update()

    def set_tool(self, tool_name: Optional[str]) -> None:
        """Set the active interaction tool.

        Args:
            tool_name (Optional[str]): Tool identifier — ``"polygon"`` to
                enable polygon drawing, or ``None`` to deactivate all tools.
        """
        self.current_tool = tool_name

    def set_active_color(self, color: QColor) -> None:
        """Set the stroke color used when drawing a new polygon.

        Args:
            color (QColor): Desired color. Falls back to ``QColor(0, 200, 0)``
                if *color* is not a valid :class:`~PySide6.QtGui.QColor`.
        """
        self._active_color = color if isinstance(color, QColor) else QColor(0, 200, 0)

    def set_overlays(
        self, poly_list: List[Tuple[List[Tuple[float, float]], QColor]]
    ) -> None:
        """Replace all rendered overlay polygons.

        Converts each polygon from original image coordinates to display
        (base-scaled) coordinates and triggers a repaint.

        Args:
            poly_list (List[Tuple[List[Tuple[float, float]], QColor]]): List of
                ``(points, color)`` pairs where *points* is a list of
                ``(x, y)`` tuples in original image coordinates.
        """
        self._overlays = []
        for pts_orig, color in poly_list:
            disp_pts = [
                QPointF(x * self._base_scale, y * self._base_scale)
                for (x, y) in pts_orig
            ]
            self._overlays.append((disp_pts, color))
        self.update()

    def clear_current_polygon(self) -> None:
        """Discard all in-progress polygon vertices and repaint."""
        self.current_polygon_points.clear()
        self.update()

    def is_dragging(self) -> bool:
        """Return whether a vertex or polygon drag is currently in progress.

        Returns:
            bool: ``True`` if a vertex drag or whole-polygon drag is active.
        """
        return self.dragging_vertex_idx != -1 or self._dragging_polygon

    def view_to_display(self, p_view: QPointF) -> QPointF:
        """Convert a point from view (widget) coordinates to display coordinates.

        Display coordinates are the pixel space of the stored
        :class:`~PySide6.QtGui.QPixmap` before zoom is applied.

        Args:
            p_view (QPointF): Point in widget pixel coordinates.

        Returns:
            QPointF: Corresponding point in display (pixmap) coordinates.
        """
        return QPointF(
            (p_view.x() - self._pan.x()) / self._zoom,
            (p_view.y() - self._pan.y()) / self._zoom,
        )

    def display_to_original(self, p_disp: QPointF) -> Tuple[float, float]:
        """Convert a point from display coordinates to original image coordinates.

        Args:
            p_disp (QPointF): Point in display (pixmap) coordinates.

        Returns:
            Tuple[float, float]: ``(x, y)`` in the original unscaled image.
        """
        return (p_disp.x() / self._base_scale, p_disp.y() / self._base_scale)

    def finish_current_polygon(self) -> None:
        """Emit :attr:`polygonFinished` with the current polygon and clear it.

        Converts all in-progress display-coordinate vertices to original image
        coordinates before emitting. Does nothing if no points have been added.
        """
        if self.current_polygon_points:
            pts_orig = [
                self.display_to_original(p) for p in self.current_polygon_points
            ]
            self.polygonFinished.emit(pts_orig)

        self.current_polygon_points.clear()
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard shortcuts for polygon drawing.

        - ``Escape``: cancel the active tool and clear in-progress points.
        - ``Backspace`` (polygon tool): remove the last placed vertex.

        Args:
            event (QKeyEvent): The key press event.
        """
        if event.key() == Qt.Key_Escape:
            self.clear_current_polygon()
            self.set_tool(None)
            self.toolCanceled.emit()
            return

        if self.current_tool == POLYGON and self.current_polygon_points:
            if event.key() == Qt.Key_Backspace:
                self.current_polygon_points.pop()
                self.update()
                return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Finish the current polygon or enter vertex-edit mode on double-click.

        - While drawing (polygon tool active): finishes and emits the polygon.
        - Otherwise: enters edit mode for the top-most polygon under the cursor.

        Args:
            event (QMouseEvent): The mouse double-click event.
        """
        if self.current_tool == POLYGON and self.current_polygon_points:
            self.finish_current_polygon()
            return

        # --- Double Click to Edit ---
        if event.button() == Qt.LeftButton:
            pos_disp = self.view_to_display(QPointF(event.pos()))
            found_idx = -1
            for i, (pts, _) in enumerate(reversed(self._overlays)):
                if QPolygonF(pts).containsPoint(pos_disp, Qt.OddEvenFill):
                    found_idx = len(self._overlays) - 1 - i
                    break

            if found_idx != -1:
                self.editing_polygon_idx = found_idx
                self.selected_polygon_idx = found_idx
                self.polygonSelected.emit(found_idx)
                self.update()
                return

        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle left-click (add vertex / select / drag) and right-click (pan).

        Left button behaviour depends on active mode:

        1. Edit mode — nearest vertex within 10 px begins a vertex drag;
           clicking inside the polygon body begins a polygon drag; clicking
           outside exits edit mode.
        2. Polygon tool — appends a point to the in-progress polygon.
        3. Normal — selects the top-most polygon under the cursor and begins
           a polygon drag if one was found.

        Right button initiates panning.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        self.setFocus()

        if event.button() == Qt.LeftButton:
            pos_view = QPointF(event.pos())
            pos_disp = self.view_to_display(pos_view)

            # --- PRE-CHECK: Identify if we clicked inside any existing polygon ---
            found_idx = -1
            for i, (pts, _) in enumerate(reversed(self._overlays)):
                if QPolygonF(pts).containsPoint(pos_disp, Qt.OddEvenFill):
                    found_idx = len(self._overlays) - 1 - i
                    break

            # 1. Edit Mode: Vertex Dragging & Polygon Dragging
            if self.editing_polygon_idx != -1:
                pts, _ = self._overlays[self.editing_polygon_idx]
                closest_idx = -1
                min_dist = float("inf")

                for i, p_disp in enumerate(pts):
                    p_view = QPointF(
                        p_disp.x() * self._zoom + self._pan.x(),
                        p_disp.y() * self._zoom + self._pan.y(),
                    )
                    dist = (
                        (p_view.x() - pos_view.x()) ** 2
                        + (p_view.y() - pos_view.y()) ** 2
                    ) ** 0.5
                    if dist < 10.0 and dist < min_dist:
                        min_dist = dist
                        closest_idx = i

                if closest_idx != -1:
                    self.dragging_vertex_idx = closest_idx
                    return

                if QPolygonF(pts).containsPoint(pos_disp, Qt.OddEvenFill):
                    self._dragging_polygon = True
                    self._last_mouse_pos = pos_view
                    return
                else:
                    self.editing_polygon_idx = -1
                    self.setCursor(Qt.ArrowCursor)
                    self.update()

            # 2. Normal Polygon Drawing
            if self.current_tool == POLYGON:
                # Visually highlight the polygon we clicked on, even while drawing
                if not self.current_polygon_points:
                    self.selected_polygon_idx = found_idx
                    self.polygonSelected.emit(found_idx)

                # Unconditionally add the point instead of checking for proximity to the first point
                self.current_polygon_points.append(
                    self.view_to_display(QPointF(event.pos()))
                )
                self.update()
                return

            # 3. Normal Selection & Polygon Dragging (When Polygon Tool is OFF)
            self.selected_polygon_idx = found_idx
            self.update()
            self.polygonSelected.emit(found_idx)

            if found_idx != -1:
                self._dragging_polygon = True
                self._last_mouse_pos = pos_view

        elif event.button() == Qt.RightButton:
            self._panning = True
            self._last_mouse_pos = QPointF(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Update vertex drag, polygon drag, pan, or crosshair cursor on move.

        Args:
            event (QMouseEvent): The mouse move event.
        """
        self._mouse_pos = QPointF(event.pos())

        if self.dragging_vertex_idx != -1 and self.editing_polygon_idx != -1:
            new_pos = self.view_to_display(self._mouse_pos)
            pts, _ = self._overlays[self.editing_polygon_idx]
            pts[self.dragging_vertex_idx] = new_pos
            self.update()
            return

        # --- Drag entire polygon ---
        if (
            self._dragging_polygon
            and self.selected_polygon_idx != -1
            and self._last_mouse_pos is not None
        ):
            delta_view = self._mouse_pos - self._last_mouse_pos
            delta_disp = QPointF(
                delta_view.x() / self._zoom, delta_view.y() / self._zoom
            )

            pts, _ = self._overlays[self.selected_polygon_idx]
            for i in range(len(pts)):
                pts[i] = QPointF(
                    pts[i].x() + delta_disp.x(), pts[i].y() + delta_disp.y()
                )

            self._last_mouse_pos = self._mouse_pos
            self.update()
            return

        if self._panning and self._last_mouse_pos is not None:
            delta = self._mouse_pos - self._last_mouse_pos
            self._pan += delta
            self._last_mouse_pos = self._mouse_pos
            self.update()
            return

        if self.current_tool == POLYGON and self.current_polygon_points:
            self.update()
            return

        if (
            self.editing_polygon_idx != -1
            and not self._dragging_polygon
            and not self._panning
        ):
            pts, _ = self._overlays[self.editing_polygon_idx]
            hovering = False
            for p_disp in pts:
                p_view = QPointF(
                    p_disp.x() * self._zoom + self._pan.x(),
                    p_disp.y() * self._zoom + self._pan.y(),
                )
                dist = (
                    (p_view.x() - self._mouse_pos.x()) ** 2
                    + (p_view.y() - self._mouse_pos.y()) ** 2
                ) ** 0.5
                if dist < 10.0:
                    hovering = True
                    break

            # Switch to thin crosshair if hovering, otherwise standard arrow
            self.setCursor(Qt.CrossCursor if hovering else Qt.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Commit a vertex or polygon drag and emit :attr:`polygonEdited`.

        Resets drag flags before emitting so that ``is_dragging()`` returns
        ``False`` by the time connected slots react.

        Args:
            event (QMouseEvent): The mouse release event.
        """
        if event.button() == Qt.LeftButton:
            # Commit Vertex Drag — reset flag BEFORE emitting so on_model_data_changed
            # sees is_dragging() == False and can safely refresh overlays.
            if self.dragging_vertex_idx != -1:
                idx = self.editing_polygon_idx
                pts, _ = self._overlays[idx]
                pts_orig = [self.display_to_original(p) for p in pts]
                self.dragging_vertex_idx = -1
                self.polygonEdited.emit(idx, pts_orig)
                return

            # Commit Polygon Drag
            if self._dragging_polygon:
                idx = self.selected_polygon_idx
                pts, _ = self._overlays[idx]
                pts_orig = [self.display_to_original(p) for p in pts]
                self._dragging_polygon = False
                self._last_mouse_pos = None
                self.polygonEdited.emit(idx, pts_orig)
                return

        if event.button() == Qt.RightButton:
            self._panning = False
            self._last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Perform cursor-anchored zoom in response to the scroll wheel.

        Clamps the scroll delta to ±5 steps and applies a ``1.15``-per-step
        zoom factor, keeping the point under the cursor stationary.

        Args:
            event (QWheelEvent): The wheel event carrying angle delta.
        """
        if self._display_qpix is None:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        steps = delta / 120.0
        steps = max(-5.0, min(5.0, steps))  # Clamp extreme inputs
        factor = 1.15**steps

        cursor_pos = event.position()
        point_in_disp = self.view_to_display(cursor_pos)

        old_zoom = self._zoom
        self._zoom = max(0.2, min(8.0, old_zoom * factor))

        self._pan = QPointF(
            cursor_pos.x() - point_in_disp.x() * self._zoom,
            cursor_pos.y() - point_in_disp.y() * self._zoom,
        )
        self.update()

    def zoom_in(self) -> None:
        """Zoom in by one step (factor ``1.15``), anchored at the widget center."""
        self._apply_zoom(1.15)

    def zoom_out(self) -> None:
        """Zoom out by one step (factor ``1 / 1.15``), anchored at the widget center."""
        self._apply_zoom(1 / 1.15)

    def reset_view(self) -> None:
        """Reset zoom to ``1.0`` and pan to the origin, then repaint."""
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def _apply_zoom(self, factor: float) -> None:
        """Apply a zoom *factor* anchored at the center of the widget.

        Adjusts :attr:`_pan` so the center point stays fixed after the zoom.
        Clamps zoom to the range ``[0.2, 8.0]``.

        Args:
            factor (float): Multiplicative zoom change (e.g. ``1.15`` to
                zoom in, ``1 / 1.15`` to zoom out).
        """
        if self._display_qpix is None:
            return

        center = QPointF(self.width() / 2, self.height() / 2)
        point_in_disp = self.view_to_display(center)

        self._zoom = max(0.2, min(8.0, self._zoom * factor))

        self._pan = QPointF(
            center.x() - point_in_disp.x() * self._zoom,
            center.y() - point_in_disp.y() * self._zoom,
        )
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Paint the image, annotation overlays, and the in-progress polygon.

        Applies pan and zoom transforms before drawing the pixmap, then draws
        each overlay polygon with selection/edit highlighting. Finally draws
        the rubber-band line from the last placed vertex to the cursor.

        Args:
            event (QPaintEvent): The paint event (used by the Qt framework).
        """
        painter = QPainter(self)
        if self._display_qpix is None:
            return

        painter.translate(self._pan)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._display_qpix)

        # Draw Overlays with Selection Highlighting
        for i, (pts, color) in enumerate(self._overlays):
            if len(pts) >= 2:
                is_selected = i == self.selected_polygon_idx
                pen = QPen(color, 4 if is_selected else 2)  # Thicker if selected
                alpha = 150 if is_selected else 60  # Darker fill if selected

                painter.setPen(pen)
                painter.setBrush(
                    QBrush(QColor(color.red(), color.green(), color.blue(), alpha))
                )
                painter.drawPolygon(QPolygonF(pts + [pts[0]]))

                if i == self.editing_polygon_idx:
                    # Scale handles so they don't get tiny when zoomed out
                    r = max(4.0 / self._zoom, 1.0)
                    painter.setBrush(QBrush(QColor("#FFEB3B")))
                    painter.setPen(QPen(QColor(20, 20, 20), 1))
                    for p in pts:
                        painter.drawEllipse(p, r, r)

        if self.current_tool == POLYGON and self.current_polygon_points:
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(self._active_color, 2))
            painter.drawPolyline(QPolygonF(self.current_polygon_points))

            if self._mouse_pos:
                cursor_in_disp = self.view_to_display(self._mouse_pos)
                painter.drawLine(
                    self.current_polygon_points[-1],
                    cursor_in_disp,
                )
