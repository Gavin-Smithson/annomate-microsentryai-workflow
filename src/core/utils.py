from typing import List, Tuple
import numpy as np


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
