"""
Abstract Strategy Interface for anomaly detection models.

Defines the contract all concrete strategy implementations must fulfill,
ensuring a consistent API for model loading and inference.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AnomalyDetectionStrategy(ABC):

    def __init__(self):
        self.model_name = "Unknown"

    @abstractmethod
    def load_from_folder(self, folder_path: str) -> None:
        """
        Load model weights from a directory.
        Raises FileNotFoundError or RuntimeError on failure.
        """

    @abstractmethod
    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        """
        Run inference on a single image.

        Returns:
            (anomaly_score, heatmap) where heatmap is float32 normalised 0–1.
        """
