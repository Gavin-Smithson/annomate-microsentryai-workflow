"""
MicroSentryAI Canvas Widgets.

Contains pure Qt graphics primitives and the CanvasPair composite widget.
No domain logic — only rendering and interaction.
"""

import logging
from typing import List, Tuple, Optional

from PIL import Image

from core.utils.geometry import simplify_polygon, scale_polygon_about_center

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QPen,
    QBrush,
    QColor,
    QPainterPath,
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QWidget,
    QGridLayout,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QGraphicsEllipseItem,
)

logger = logging.getLogger("MicroSentryAI.Canvas")

HANDLE_RADIUS = 4.0


# ---------------------------------------------------------------------------
# Graphics primitives
# ---------------------------------------------------------------------------


class VertexHandle(QGraphicsEllipseItem):
    """Draggable ellipse handle for a single polygon vertex in the MicroSentryAI canvas.

    Rendered as a yellow circle that turns cyan on hover. Dragging the handle
    updates the parent :class:`SegPathItem`'s path in real time and calls the
    ``on_any_edit`` callback when the drag begins and ends.

    Attributes:
        parent_item (SegPathItem): The polygon item that owns this handle.
        idx (int): Index of the vertex this handle represents within
            ``parent_item._pts``.
    """

    def __init__(self, parent: "SegPathItem", idx: int, pos: QPointF) -> None:
        """Create a vertex handle at *pos* for vertex *idx* of *parent*.

        Args:
            parent (SegPathItem): The polygon item that owns this handle.
            idx (int): Zero-based index of the vertex within the parent's
                point list.
            pos (QPointF): Initial position in parent-item local coordinates.
        """
        super().__init__(
            -HANDLE_RADIUS, -HANDLE_RADIUS, HANDLE_RADIUS * 2, HANDLE_RADIUS * 2, parent
        )
        self.setAcceptHoverEvents(True)
        self.setBrush(QBrush(QColor("#FFEB3B")))
        self.setPen(QPen(QColor(20, 20, 20), 1))
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setZValue(100)
        self.setCursor(Qt.CrossCursor)

        self.parent_item = parent
        self.idx = idx
        self.setPos(pos)

    def hoverEnterEvent(self, event) -> None:
        """Highlight the handle cyan when the cursor enters.

        Args:
            event: The hover enter event passed by Qt.
        """
        self.setBrush(QBrush(QColor("#00BCD4")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        """Restore the handle to yellow when the cursor leaves.

        Args:
            event: The hover leave event passed by Qt.
        """
        self.setBrush(QBrush(QColor("#FFEB3B")))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        """Lock polygon movement and fire the ``vertex_drag_begin`` callback.

        Args:
            event: The mouse press event passed by Qt.
        """
        self.parent_item.lock_move = True
        self.parent_item.setFlag(QGraphicsPathItem.ItemIsMovable, False)
        if self.parent_item.on_any_edit:
            self.parent_item.on_any_edit("vertex_drag_begin")
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Update the parent polygon path as the handle is dragged.

        Args:
            event: The mouse move event passed by Qt.
        """
        super().mouseMoveEvent(event)
        self.parent_item.update_vertex(self.idx, self.pos())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """Unlock polygon movement and fire the ``vertex_drag_end`` callback.

        Args:
            event: The mouse release event passed by Qt.
        """
        super().mouseReleaseEvent(event)
        self.parent_item.setFlag(QGraphicsPathItem.ItemIsMovable, True)
        self.parent_item.lock_move = False
        if self.parent_item.on_any_edit:
            self.parent_item.on_any_edit("vertex_drag_end")
        event.accept()

    def cleanup(self) -> None:
        """Detach this handle from its parent item and remove it from the scene."""
        self.setParentItem(None)
        if self.scene():
            self.scene().removeItem(self)


class SegPathItem(QGraphicsPathItem):
    """Editable polygon item for the MicroSentryAI segmentation canvas.

    Supports whole-polygon dragging and, after a double-click, per-vertex
    dragging via :class:`VertexHandle` children. Calls an optional
    ``on_any_edit`` callback after each geometry change so the window can
    manage the undo stack.

    Attributes:
        handles (List[VertexHandle]): Active vertex handles; empty when not
            in edit mode.
        lock_move (bool): Set to ``True`` while a vertex drag is in progress
            to block the item's own movement handler.
        is_editing (bool): ``True`` while vertex handles are visible.
        on_any_edit: Optional callable invoked with an event-kind string after
            each geometry mutation.
    """

    def __init__(self, pts: List[QPointF], on_any_edit=None) -> None:
        """Create an editable polygon from a list of scene-coordinate points.

        Args:
            pts (List[QPointF]): Polygon vertices in scene coordinates.
            on_any_edit: Optional callable ``(kind: str) -> None`` invoked
                after geometry changes. *kind* is one of
                ``"vertex_drag_begin"``, ``"vertex_drag"``,
                ``"vertex_drag_end"``, ``"polygon_move"``,
                ``"polygon_simplify"``, or ``"polygon_scale"``.
        """
        super().__init__()
        self.setFlags(
            QGraphicsPathItem.ItemIsMovable | QGraphicsPathItem.ItemIsSelectable
        )
        self.pen_normal = QPen(QColor(0, 255, 0), 2)
        self.pen_selected = QPen(QColor(255, 235, 59), 2)

        self.handles: List[VertexHandle] = []
        self._pts = pts[:]
        self.lock_move = False
        self.is_editing = False
        self.on_any_edit = on_any_edit

        self.setZValue(10)
        self._rebuild_path()

    def _rebuild_path(self) -> None:
        """Reconstruct the :class:`~PySide6.QtGui.QPainterPath` from ``_pts``."""
        path = QPainterPath()
        if self._pts:
            path.moveTo(self._pts[0])
            for p in self._pts[1:]:
                path.lineTo(p)
            path.closeSubpath()
        self.setPath(path)

    def paint(self, painter, option, widget=None) -> None:
        """Paint the polygon, switching pen based on selection state.

        Args:
            painter: Qt painter object.
            option: Style option provided by the graphics framework.
            widget: Optional target widget (unused).
        """
        self.setPen(self.pen_selected if self.isSelected() else self.pen_normal)
        super().paint(painter, option, widget)

    def mouseDoubleClickEvent(self, event) -> None:
        """Toggle vertex-edit mode on double-click.

        Args:
            event: The mouse double-click event passed by Qt.
        """
        self.is_editing = not self.is_editing
        self.update_handles()
        super().mouseDoubleClickEvent(event)

    def update_handles(self) -> None:
        """Show or hide :class:`VertexHandle` children based on :attr:`is_editing`."""
        if self.is_editing and not self.handles:
            for i, p in enumerate(self._pts):
                self.handles.append(VertexHandle(parent=self, idx=i, pos=p))
        elif not self.is_editing and self.handles:
            for h in self.handles:
                h.cleanup()
            self.handles = []

    def itemChange(self, change, value):
        """Block movement during vertex drags and exit edit mode on deselect.

        Args:
            change: The type of item change as a Qt enum value.
            value: The new value associated with the change.

        Returns:
            The processed value, or the current position when movement is
            locked.
        """
        if change == QGraphicsPathItem.ItemPositionChange and self.lock_move:
            return self.pos()
        if change == QGraphicsPathItem.ItemSelectedHasChanged and not value:
            self.is_editing = False
            self.update_handles()
        return super().itemChange(change, value)

    def update_vertex(self, idx: int, newpos: QPointF) -> None:
        """Move vertex *idx* to *newpos* and rebuild the path.

        Args:
            idx (int): Zero-based index of the vertex to move.
            newpos (QPointF): New position in parent-item local coordinates.
        """
        if 0 <= idx < len(self._pts):
            self._pts[idx] = newpos
            self._rebuild_path()
            if self.on_any_edit:
                self.on_any_edit("vertex_drag")

    def simplify(self, epsilon: float) -> None:
        """Simplify the polygon using Douglas-Peucker with tolerance *epsilon*.

        Rebuilds vertex handles if they are currently visible. Calls
        ``on_any_edit("polygon_simplify")`` on success.

        Args:
            epsilon (float): Maximum deviation tolerance in scene pixels.
        """
        pts_xy = [(p.x(), p.y()) for p in self._pts]
        result = simplify_polygon(pts_xy, epsilon)
        if result is pts_xy:
            return
        self._pts = [QPointF(x, y) for (x, y) in result]
        self._rebuild_path()
        if self.handles:
            for h in self.handles:
                h.cleanup()
            self.handles = [
                VertexHandle(parent=self, idx=i, pos=p) for i, p in enumerate(self._pts)
            ]
        if self.on_any_edit:
            self.on_any_edit("polygon_simplify")

    def mousePressEvent(self, event) -> None:
        """Record the starting position for move-distance detection.

        Args:
            event: The mouse press event passed by Qt.
        """
        self._start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Emit ``polygon_move`` callback if the item actually moved.

        Args:
            event: The mouse release event passed by Qt.
        """
        super().mouseReleaseEvent(event)
        if not self.lock_move and self.pos() != getattr(self, "_start_pos", self.pos()):
            if self.on_any_edit:
                self.on_any_edit("polygon_move")

    def scale_about_center(self, factor: float) -> None:
        """Scale all polygon vertices about the polygon centroid by *factor*.

        Updates any visible :class:`VertexHandle` positions and calls
        ``on_any_edit("polygon_scale")``.

        Args:
            factor (float): Scaling multiplier (e.g. ``0.9`` to shrink,
                ``1.1`` to grow).
        """
        if not self._pts:
            return
        pts_xy = [(p.x(), p.y()) for p in self._pts]
        scaled = scale_polygon_about_center(pts_xy, factor)
        self._pts = [QPointF(x, y) for (x, y) in scaled]
        for i, h in enumerate(self.handles):
            h.setPos(self._pts[i])
        self._rebuild_path()
        if self.on_any_edit:
            self.on_any_edit("polygon_scale")


class SyncedGraphicsView(QGraphicsView):
    """Graphics view with cursor-anchored zoom and cross-view pan/zoom sync.

    Emits :attr:`viewChanged` whenever the viewport center or zoom level
    changes so a paired view can mirror the state via :meth:`set_view_state`.

    Attributes:
        _is_syncing (bool): Guard flag set during programmatic view state
            updates to prevent recursive sync emission.

    Signals:
        viewChanged (float, float, float): Emitted with relative center
            ``(rx, ry)`` and absolute scale when the viewport changes.
    """

    viewChanged = Signal(float, float, float)  # rx, ry, scale

    def __init__(self, scene: QGraphicsScene, parent: QWidget = None) -> None:
        """Initialize the view, configure drag/anchor modes, and wire scroll bars.

        Args:
            scene (QGraphicsScene): The scene to display.
            parent (QWidget): Optional parent widget. Defaults to ``None``.
        """
        super().__init__(scene, parent)
        self._is_syncing = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.horizontalScrollBar().valueChanged.connect(self._emit_view)
        self.verticalScrollBar().valueChanged.connect(self._emit_view)

    def _emit_view(self) -> None:
        """Compute the relative viewport center and emit :attr:`viewChanged`.

        Does nothing when :attr:`_is_syncing` is set or the scene has no area.
        """
        if self._is_syncing:
            return
        if self.sceneRect().width() <= 0:
            return
        center = self.mapToScene(self.viewport().rect().center())
        w, h = self.sceneRect().width(), self.sceneRect().height()
        self.viewChanged.emit(center.x() / w, center.y() / h, self.transform().m11())

    def wheelEvent(self, event) -> None:
        """Zoom the view at the cursor position using the mouse wheel.

        Clamps the step count to ±5 scroll-wheel clicks to avoid runaway
        zoom. Emits :attr:`viewChanged` after scaling.

        Args:
            event: The wheel event passed by Qt.
        """
        delta = event.angleDelta().y()
        if delta == 0:
            return
        steps = max(-5.0, min(5.0, delta / 120.0))
        self.scale(1.15**steps, 1.15**steps)
        self._emit_view()
        event.accept()

    def set_view_state(self, rx: float, ry: float, scale: float) -> None:
        """Apply an external pan/zoom state without emitting :attr:`viewChanged`.

        Used by a paired view to mirror this view's viewport. No-ops when the
        scene has no area.

        Args:
            rx (float): Relative horizontal center position (0.0–1.0).
            ry (float): Relative vertical center position (0.0–1.0).
            scale (float): Absolute zoom scale factor.
        """
        if self.sceneRect().width() <= 0:
            return
        self._is_syncing = True
        self.resetTransform()
        self.scale(scale, scale)
        w, h = self.sceneRect().width(), self.sceneRect().height()
        self.centerOn(QPointF(rx * w, ry * h))
        self._is_syncing = False


# ---------------------------------------------------------------------------
# Conversion helper (PIL Image → QPixmap, View boundary)
# ---------------------------------------------------------------------------


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    """Convert a PIL :class:`~PIL.Image.Image` to a :class:`~PySide6.QtGui.QPixmap`.

    Forces the image to RGB before conversion so callers do not need to
    pre-convert RGBA or palette-mode images.

    Args:
        pil_img (Image.Image): Input PIL image in any mode.

    Returns:
        QPixmap: A Qt pixmap ready for display in a :class:`QGraphicsScene`.
    """
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


# ---------------------------------------------------------------------------
# CanvasPair — dual synchronised views
# ---------------------------------------------------------------------------


class CanvasPair(QWidget):
    """Dual synchronised :class:`SyncedGraphicsView` views displayed side-by-side.

    The left view shows the segmentation canvas with interactive
    :class:`SegPathItem` polygon overlays. The right view shows the static
    heatmap image. Both views mirror each other's pan and zoom state.

    PIL → :class:`~PySide6.QtGui.QPixmap` conversion happens at this Qt
    boundary; callers always pass :class:`~PIL.Image.Image` objects.

    Attributes:
        scene_left (QGraphicsScene): Scene backing the segmentation view.
        scene_right (QGraphicsScene): Scene backing the heatmap view.
        view_left (SyncedGraphicsView): Interactive segmentation view.
        view_right (SyncedGraphicsView): Read-only heatmap view.
    """

    def __init__(self, parent: QWidget = None) -> None:
        """Create both scenes, views, and wire cross-view sync signals.

        Args:
            parent (QWidget): Optional parent widget. Defaults to ``None``.
        """
        super().__init__(parent)

        self.scene_left = QGraphicsScene()
        self.scene_right = QGraphicsScene()
        self.view_left = SyncedGraphicsView(self.scene_left)
        self.view_right = SyncedGraphicsView(self.scene_right)

        self.view_left.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.view_right.setBackgroundBrush(QBrush(QColor(0, 0, 0)))

        self.view_left.viewChanged.connect(
            lambda rx, ry, s: self._sync(self.view_right, rx, ry, s)
        )
        self.view_right.viewChanged.connect(
            lambda rx, ry, s: self._sync(self.view_left, rx, ry, s)
        )

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(QLabel("Segmentation"), 0, 0, alignment=Qt.AlignHCenter)
        grid.addWidget(QLabel("Heatmap Overlay"), 0, 1, alignment=Qt.AlignHCenter)
        grid.addWidget(self.view_left, 1, 0)
        grid.addWidget(self.view_right, 1, 1)

    def _sync(
        self, target: SyncedGraphicsView, rx: float, ry: float, scale: float
    ) -> None:
        """Forward a view-state update to *target* without re-emitting.

        Args:
            target (SyncedGraphicsView): The view to synchronise.
            rx (float): Relative horizontal center position (0.0–1.0).
            ry (float): Relative vertical center position (0.0–1.0).
            scale (float): Absolute zoom scale factor.
        """
        target.set_view_state(rx, ry, scale)

    def set_images(
        self,
        left_pil: Image.Image,
        right_pil: Image.Image,
        on_any_edit=None,
        contours: Optional[list] = None,
    ) -> None:
        """Replace both background images and optionally restore polygon overlays.

        Clears both scenes, converts the PIL images to
        :class:`~PySide6.QtGui.QPixmap` at this Qt boundary, and re-adds
        background items. If *contours* is provided, calls
        :meth:`set_polygons` to populate the left scene.

        Args:
            left_pil (Image.Image): Segmentation background image.
            right_pil (Image.Image): Heatmap overlay image.
            on_any_edit: Optional edit callback forwarded to each new
                :class:`SegPathItem`. Defaults to ``None``.
            contours (Optional[list]): List of ``[(x, y), ...]`` contour
                point lists to draw as polygons. Defaults to ``None``.
        """
        self.scene_left.clear()
        self.scene_right.clear()

        left_px = pil_to_qpixmap(left_pil)
        right_px = pil_to_qpixmap(right_pil)

        left_bg = QGraphicsPixmapItem(left_px)
        left_bg.setZValue(-10)
        self.scene_left.addItem(left_bg)

        right_bg = QGraphicsPixmapItem(right_px)
        right_bg.setZValue(-10)
        self.scene_right.addItem(right_bg)

        w, h = left_pil.size
        rect = QRectF(0, 0, w, h)
        self.scene_left.setSceneRect(rect)
        self.scene_right.setSceneRect(rect)
        self.view_left.setSceneRect(rect)
        self.view_right.setSceneRect(rect)

        if contours:
            self.set_polygons(contours, on_any_edit)

    def set_polygons(self, contours: list, on_any_edit=None) -> None:
        """Replace all :class:`SegPathItem` overlays in the left scene.

        Removes existing polygon items before adding new ones so the
        background pixmap is preserved.

        Args:
            contours (list): List of ``[(x, y), ...]`` point lists.
            on_any_edit: Optional edit callback forwarded to each new
                :class:`SegPathItem`. Defaults to ``None``.
        """
        for item in list(self.scene_left.items()):
            if isinstance(item, SegPathItem):
                self.scene_left.removeItem(item)

        for pts_raw in contours:
            pts = [QPointF(x, y) for (x, y) in pts_raw]
            item = SegPathItem(pts, on_any_edit=on_any_edit)
            self.scene_left.addItem(item)

    def serialize_polygons(self) -> list:
        """Return a serialisable snapshot of all current left-scene polygons.

        Each entry captures both the local vertex positions and the item's
        scene offset so the state can be fully restored via
        :meth:`restore_polygons`.

        Returns:
            list: List of dicts with keys ``"pts"`` (list of ``(x, y)``
                tuples in item-local coordinates) and ``"pos"``
                (``(x, y)`` scene offset of the item).
        """
        result = []
        for item in self.scene_left.items():
            if isinstance(item, SegPathItem):
                result.append(
                    {
                        "pts": [(p.x(), p.y()) for p in item._pts],
                        "pos": (item.pos().x(), item.pos().y()),
                    }
                )
        return result

    def restore_polygons(
        self, poly_data: list, left_pil: Image.Image, on_any_edit=None
    ) -> None:
        """Restore polygon overlays from a serialised snapshot.

        Removes all current polygon items, re-adds the background pixmap if
        the scene was cleared, then recreates each :class:`SegPathItem` at
        the stored position.

        Args:
            poly_data (list): Snapshot produced by :meth:`serialize_polygons`.
            left_pil (Image.Image): Background image; used to re-add the
                pixmap if the scene is empty.
            on_any_edit: Optional edit callback forwarded to each new
                :class:`SegPathItem`. Defaults to ``None``.
        """
        for item in list(self.scene_left.items()):
            if isinstance(item, SegPathItem):
                self.scene_left.removeItem(item)

        # Re-add background if scene was cleared
        if not any(isinstance(i, QGraphicsPixmapItem) for i in self.scene_left.items()):
            bg = QGraphicsPixmapItem(pil_to_qpixmap(left_pil))
            bg.setZValue(-10)
            self.scene_left.addItem(bg)

        for poly in poly_data:
            pts = [QPointF(x, y) for (x, y) in poly["pts"]]
            item = SegPathItem(pts, on_any_edit=on_any_edit)
            item.setPos(poly["pos"][0], poly["pos"][1])
            self.scene_left.addItem(item)

    def get_polygons_original_coords(
        self, scale: float, offset: Tuple[int, int]
    ) -> list:
        """Return all polygon vertices converted to original image coordinates.

        Accounts for the item's scene offset and the display-to-original
        scale and crop offset applied when the image was prepared for display.

        Args:
            scale (float): Scale factor applied to the image during display
                (original pixels / display pixels).
            offset (Tuple[int, int]): ``(x, y)`` crop offset in display pixels
                subtracted before scaling.

        Returns:
            list: List of polygon point lists, each a list of
                ``(x_orig, y_orig)`` float tuples in original image coordinates.
        """
        off_x, off_y = offset
        result = []
        for item in self.scene_left.items():
            if isinstance(item, SegPathItem):
                mx, my = item.pos().x(), item.pos().y()
                orig_pts = [
                    ((p.x() + mx + off_x) / scale, (p.y() + my + off_y) / scale)
                    for p in item._pts
                ]
                result.append(orig_pts)
        return result

    def get_selected_polygons_original_coords(
        self, scale: float, offset: Tuple[int, int]
    ) -> list:
        """Return only selected polygon vertices converted to original image coordinates.

        Identical to :meth:`get_polygons_original_coords` but restricted to
        items whose :meth:`~QGraphicsItem.isSelected` flag is set.

        Args:
            scale (float): Scale factor applied to the image during display.
            offset (Tuple[int, int]): ``(x, y)`` crop offset in display pixels.

        Returns:
            list: List of polygon point lists for selected items only.
        """
        off_x, off_y = offset
        result = []
        for item in self.scene_left.selectedItems():
            if isinstance(item, SegPathItem):
                mx, my = item.pos().x(), item.pos().y()
                orig_pts = [
                    ((p.x() + mx + off_x) / scale, (p.y() + my + off_y) / scale)
                    for p in item._pts
                ]
                result.append(orig_pts)
        return result

    def set_view_state(self, rx: float, ry: float, scale: float) -> None:
        """Apply a pan/zoom state to both views simultaneously.

        Used by the main window to synchronise the canvas pair with the
        AnnoMate cross-tab pan/zoom signal.

        Args:
            rx (float): Relative horizontal center position (0.0–1.0).
            ry (float): Relative vertical center position (0.0–1.0).
            scale (float): Absolute zoom scale factor.
        """
        self.view_left.set_view_state(rx, ry, scale)
        self.view_right.set_view_state(rx, ry, scale)

    def fit_views(self) -> None:
        """Reset and fit both views to their scene contents.

        Only fits a view if its scene's bounding rect has a positive width,
        preserving aspect ratio via ``Qt.KeepAspectRatio``.
        """
        for view, scene in (
            (self.view_left, self.scene_left),
            (self.view_right, self.scene_right),
        ):
            if scene.itemsBoundingRect().width() > 0:
                view.resetTransform()
                view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
