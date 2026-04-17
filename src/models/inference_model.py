import numpy as np

from core.inference_state import InferenceState


class InferenceModel:
    """
    Pure Python model for MicroSentryAI inference results.
    Wraps InferenceState with a clean query/command API.
    No Qt — fully testable without QApplication.
    Views must use this API instead of accessing InferenceState directly.
    """

    def __init__(self, state: InferenceState):
        self.state = state

    def set_score_map(self, filename: str, score_map: np.ndarray):
        self.state.set_score_map(filename, score_map)

    def get_score_map(self, filename: str) -> np.ndarray | None:
        return self.state.get_score_map(filename)

    def is_processed(self, filename: str) -> bool:
        return self.state.is_processed(filename)

    def get_processed_count(self) -> int:
        return len(self.state.score_maps)

    def clear(self):
        self.state.clear()
