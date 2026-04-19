from typing import List, Tuple
import numpy as np
import cv2


def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Shoelace formula."""
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_bbox(points: List[Tuple[float, float]]) -> List[float]:
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, min_y = float(min(xs)), float(min(ys))
    return [min_x, min_y, float(max(xs) - min_x), float(max(ys) - min_y)]


def simplify_polygon(
    pts: List[Tuple[float, float]], epsilon: float
) -> List[Tuple[float, float]]:
    """Douglas-Peucker polygon simplification via OpenCV.

    Returns the original list object unchanged when simplification cannot
    produce at least 3 points, so callers can use identity check (``is``) to
    detect a no-op.
    """
    if len(pts) < 3:
        return pts
    cnt = np.array([[[p[0], p[1]]] for p in pts], dtype=np.float32)
    approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
    if approx is None or len(approx) < 3:
        return pts
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def scale_polygon_about_center(
    pts: List[Tuple[float, float]], factor: float
) -> List[Tuple[float, float]]:
    """Scale polygon points about their centroid by *factor*."""
    if not pts:
        return pts
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [(cx + (p[0] - cx) * factor, cy + (p[1] - cy) * factor) for p in pts]
