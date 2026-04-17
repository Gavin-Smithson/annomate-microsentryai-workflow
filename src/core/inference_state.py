import numpy as np


class InferenceState:
    """
    MicroSentryAI domain state.
    Stores inference results keyed by filename. Zero Qt.
    """

    def __init__(self):
        self.score_maps = {}       # { "img.jpg": np.ndarray }  full heatmap arrays
        self.inference_cache = {}  # { "img.jpg": float }  peak anomaly scores

    def clear(self):
        self.score_maps.clear()
        self.inference_cache.clear()

    def set_score_map(self, filename: str, score_map: np.ndarray):
        self.score_maps[filename] = score_map
        self.inference_cache[filename] = float(score_map.max())

    def get_score_map(self, filename: str) -> np.ndarray | None:
        return self.score_maps.get(filename)

    def is_processed(self, filename: str) -> bool:
        return filename in self.score_maps
