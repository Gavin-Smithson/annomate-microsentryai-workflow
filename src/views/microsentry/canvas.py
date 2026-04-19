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
    QPen, QBrush, QColor, QPainterPath,
    QImage, QPixmap,
)
from PySide6.QtWidgets import (
    QWidget, QGridLayout, QLabel,
    QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsEllipseItem,
)

logger = logging.getLogger("MicroSentryAI.Canvas")

HANDLE_RADIUS = 4.0


# ---------------------------------------------------------------------------
# Graphics primitives
# ---------------------------------------------------------------------------

class VertexHandle(QGraphicsEllipseItem):
    """Draggable handle for a single polygon vertex."""

    def __init__(self, parent: "SegPathItem", idx: int, pos: QPointF):
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

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor("#00BCD4")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(QColor("#FFEB3B")))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.parent_item.lock_move = True
        self.parent_item.setFlag(QGraphicsPathItem.ItemIsMovable, False)
        if self.parent_item.on_any_edit:
            self.parent_item.on_any_edit("vertex_drag_begin")
        super().mousePressEvent(event)
        event.accept()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.parent_item.update_vertex(self.idx, self.pos())
        event.accept()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.parent_item.setFlag(QGraphicsPathItem.ItemIsMovable, True)
        self.parent_item.lock_move = False
        if self.parent_item.on_any_edit:
            self.parent_item.on_any_edit("vertex_drag_end")
        event.accept()

    def cleanup(self):
        self.setParentItem(None)
        if self.scene():
            self.scene().removeItem(self)


class SegPathItem(QGraphicsPathItem):
    """Editable polygon. Double-click to show/hide vertex handles."""

    def __init__(self, pts: List[QPointF], on_any_edit=None):
        super().__init__()
        self.setFlags(QGraphicsPathItem.ItemIsMovable | QGraphicsPathItem.ItemIsSelectable)
        self.pen_normal   = QPen(QColor(0, 255, 0), 2)
        self.pen_selected = QPen(QColor(255, 235, 59), 2)

        self.handles: List[VertexHandle] = []
        self._pts = pts[:]
        self.lock_move = False
        self.is_editing = False
        self.on_any_edit = on_any_edit

        self.setZValue(10)
        self._rebuild_path()

    def _rebuild_path(self):
        path = QPainterPath()
        if self._pts:
            path.moveTo(self._pts[0])
            for p in self._pts[1:]:
                path.lineTo(p)
            path.closeSubpath()
        self.setPath(path)

    def paint(self, painter, option, widget=None):
        self.setPen(self.pen_selected if self.isSelected() else self.pen_normal)
        super().paint(painter, option, widget)

    def mouseDoubleClickEvent(self, event):
        self.is_editing = not self.is_editing
        self.update_handles()
        super().mouseDoubleClickEvent(event)

    def update_handles(self):
        if self.is_editing and not self.handles:
            for i, p in enumerate(self._pts):
                self.handles.append(VertexHandle(parent=self, idx=i, pos=p))
        elif not self.is_editing and self.handles:
            for h in self.handles:
                h.cleanup()
            self.handles = []

    def itemChange(self, change, value):
        if change == QGraphicsPathItem.ItemPositionChange and self.lock_move:
            return self.pos()
        if change == QGraphicsPathItem.ItemSelectedHasChanged and not value:
            self.is_editing = False
            self.update_handles()
        return super().itemChange(change, value)

    def update_vertex(self, idx: int, newpos: QPointF):
        if 0 <= idx < len(self._pts):
            self._pts[idx] = newpos
            self._rebuild_path()
            if self.on_any_edit:
                self.on_any_edit("vertex_drag")

    def simplify(self, epsilon: float):
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

    def mousePressEvent(self, event):
        self._start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if not self.lock_move and self.pos() != getattr(self, "_start_pos", self.pos()):
            if self.on_any_edit:
                self.on_any_edit("polygon_move")

    def scale_about_center(self, factor: float):
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
    """Graphics view with cursor-anchored zoom and cross-view sync support."""

    viewChanged = Signal(float, float, float)   # rx, ry, scale

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self._is_syncing = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.horizontalScrollBar().valueChanged.connect(self._emit_view)
        self.verticalScrollBar().valueChanged.connect(self._emit_view)

    def _emit_view(self):
        if self._is_syncing:
            return
        if self.sceneRect().width() <= 0:
            return
        center = self.mapToScene(self.viewport().rect().center())
        w, h = self.sceneRect().width(), self.sceneRect().height()
        self.viewChanged.emit(center.x() / w, center.y() / h, self.transform().m11())

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        steps = max(-5.0, min(5.0, delta / 120.0))
        self.scale(1.15 ** steps, 1.15 ** steps)
        self._emit_view()
        event.accept()

    def set_view_state(self, rx: float, ry: float, scale: float):
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
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


# ---------------------------------------------------------------------------
# CanvasPair — dual synchronised views
# ---------------------------------------------------------------------------

class CanvasPair(QWidget):
    """
    Manages two synchronised QGraphicsViews side-by-side:
      - Left:  segmentation canvas with editable SegPathItem polygons
      - Right: heatmap overlay (static pixmap, no interactive items)

    The numpy → QPixmap conversion happens here, at the Qt boundary.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.scene_left  = QGraphicsScene()
        self.scene_right = QGraphicsScene()
        self.view_left   = SyncedGraphicsView(self.scene_left)
        self.view_right  = SyncedGraphicsView(self.scene_right)

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
        grid.addWidget(QLabel("Segmentation"),   0, 0, alignment=Qt.AlignHCenter)
        grid.addWidget(QLabel("Heatmap Overlay"), 0, 1, alignment=Qt.AlignHCenter)
        grid.addWidget(self.view_left,  1, 0)
        grid.addWidget(self.view_right, 1, 1)

    def _sync(self, target: SyncedGraphicsView, rx: float, ry: float, scale: float):
        target.set_view_state(rx, ry, scale)

    def set_images(
        self,
        left_pil: Image.Image,
        right_pil: Image.Image,
        on_any_edit=None,
        contours: Optional[list] = None,
    ):
        """
        Update both background images and replace polygon items.
        Converts PIL Images to QPixmap at this View boundary.
        """
        self.scene_left.clear()
        self.scene_right.clear()

        left_px  = pil_to_qpixmap(left_pil)
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

    def set_polygons(self, contours: list, on_any_edit=None):
        """Replace all SegPathItems in the left scene from a contour list."""
        for item in list(self.scene_left.items()):
            if isinstance(item, SegPathItem):
                self.scene_left.removeItem(item)

        for pts_raw in contours:
            pts = [QPointF(x, y) for (x, y) in pts_raw]
            item = SegPathItem(pts, on_any_edit=on_any_edit)
            self.scene_left.addItem(item)

    def serialize_polygons(self) -> list:
        """Return serialisable snapshot of current polygon state (for undo/redo)."""
        result = []
        for item in self.scene_left.items():
            if isinstance(item, SegPathItem):
                result.append({
                    "pts": [(p.x(), p.y()) for p in item._pts],
                    "pos": (item.pos().x(), item.pos().y()),
                })
        return result

    def restore_polygons(self, poly_data: list, left_pil: Image.Image, on_any_edit=None):
        """Restore polygons from a serialised snapshot (undo/redo)."""
        for item in list(self.scene_left.items()):
            if isinstance(item, SegPathItem):
                self.scene_left.removeItem(item)

        # Re-add background if scene was cleared
        if not any(
            isinstance(i, QGraphicsPixmapItem) for i in self.scene_left.items()
        ):
            bg = QGraphicsPixmapItem(pil_to_qpixmap(left_pil))
            bg.setZValue(-10)
            self.scene_left.addItem(bg)

        for poly in poly_data:
            pts = [QPointF(x, y) for (x, y) in poly["pts"]]
            item = SegPathItem(pts, on_any_edit=on_any_edit)
            item.setPos(poly["pos"][0], poly["pos"][1])
            self.scene_left.addItem(item)

    def get_polygons_original_coords(self, scale: float, offset: Tuple[int, int]) -> list:
        """Convert all left-scene polygons from display to original image coordinates."""
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

    def get_selected_polygons_original_coords(self, scale: float, offset: Tuple[int, int]) -> list:
        """Same as above, restricted to selected items."""
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

    def set_view_state(self, rx: float, ry: float, scale: float):
        """External sync entry point (e.g. from AnnoMate cross-tab sync)."""
        self.view_left.set_view_state(rx, ry, scale)
        self.view_right.set_view_state(rx, ry, scale)

    def fit_views(self):
        for view, scene in (
            (self.view_left, self.scene_left),
            (self.view_right, self.scene_right),
        ):
            if scene.itemsBoundingRect().width() > 0:
                view.resetTransform()
                view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
