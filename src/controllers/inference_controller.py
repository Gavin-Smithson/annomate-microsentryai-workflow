"""
InferenceController — headless inference orchestration for MicroSentryAI.

Rules:
  - No Qt GUI types (no QFileDialog, QMessageBox, QColor).
  - All methods accept plain Python values; errors raised as exceptions.
  - QThread is permitted here — it is infrastructure, not UI.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from matplotlib import colormaps as mpl_cmaps

from PySide6.QtCore import QThread, Signal

from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel

logger = logging.getLogger("MicroSentryAI.InferenceController")


class InferenceWorker(QThread):
    """Background thread that runs batch inference without blocking the UI."""

    resultReady = Signal(str, object)   # (absolute_path, score_map: np.ndarray)
    progress    = Signal(int)           # count of images processed so far
    finished    = Signal()

    def __init__(self, strategy, file_list: List[str]):
        super().__init__()
        self.strategy = strategy
        self.file_list = file_list
        self._running = True

    def run(self):
        for i, path in enumerate(self.file_list):
            if not self._running:
                break
            try:
                _, score_map = self.strategy.predict(path)
                self.resultReady.emit(path, score_map)
            except Exception as e:
                logger.error("Inference failed for %s: %s", path, e)
            self.progress.emit(i + 1)
        self.finished.emit()

    def stop(self):
        self._running = False


class InferenceController:
    def __init__(self, dataset_model: DatasetTableModel, inference_model: InferenceModel):
        self.dataset_model = dataset_model
        self.inference_model = inference_model
        self._strategy = None
        self._worker: Optional[InferenceWorker] = None

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #

    def load_model(self, model_path: str, device: str) -> str:
        """Load a .pt or .ckpt model. Returns model_name. Raises RuntimeError on failure."""
        from ai_strategies.anomalib_strategy import AnomalibStrategy
        strategy = AnomalibStrategy()
        strategy.set_device(device.lower())
        strategy.load_from_file(model_path)   # raises on failure
        self._strategy = strategy
        logger.info("Model loaded: %s", strategy.model_name)
        return strategy.model_name

    def get_model_name(self) -> str:
        return self._strategy.model_name if self._strategy else ""

    def has_model(self) -> bool:
        return self._strategy is not None

    # ------------------------------------------------------------------ #
    # Image loading
    # ------------------------------------------------------------------ #

    def load_image(self, path: str) -> Optional[Image.Image]:
        """Load an image from disk. Returns PIL Image (RGB) or None on failure."""
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning("Could not read image %s: %s", path, e)
            return None

    # ------------------------------------------------------------------ #
    # Background inference
    # ------------------------------------------------------------------ #

    def start_batch_inference(self, file_paths: List[str]) -> InferenceWorker:
        """
        Create and return an InferenceWorker for the given paths.
        Caller must connect signals, then call worker.start().
        Any previously running worker is stopped first.
        """
        self._stop_worker()
        self._worker = InferenceWorker(self._strategy, file_paths)
        return self._worker

    def _stop_worker(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()

    # ------------------------------------------------------------------ #
    # Visualisation computation (pure Python, no Qt)
    # ------------------------------------------------------------------ #

    def compute_heatmap(
        self,
        pil_image: Image.Image,
        score_map: np.ndarray,
        alpha: float,
        sigma: float,
        display_target: int,
        heat_min_pct: int,
    ) -> Tuple[Image.Image, Image.Image, float, tuple, Optional[np.ndarray]]:
        """
        Compute resized raw image and heatmap overlay. Returns smoothed score array
        so compute_segmentation() can reuse it without re-running the Gaussian.

        Returns:
            left_image  — PIL Image (raw, resized to display_target)
            right_image — PIL Image (heatmap composited on raw)
            scale       — uniform scale factor applied (display / original)
            offset      — (off_x, off_y) crop offset; always (0, 0) for now
            smoothed_s  — smoothed score map ndarray, or None if score_map is None
        """
        w, h = pil_image.size
        scale = display_target / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        left_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        offset = (0, 0)

        if score_map is None:
            return left_image, left_image.copy(), scale, offset, None

        s = gaussian_filter(score_map, sigma=sigma) if sigma > 0 else score_map.copy()

        v_min_thr = np.percentile(s, heat_min_pct)
        s_clipped = np.clip(s, v_min_thr, s.max())
        mx, mn = s_clipped.max(), s_clipped.min()
        s_norm = (s_clipped - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(s_clipped)

        s_resized = cv2.resize(s_norm, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        heat_arr = (s_resized * 255).astype(np.uint8)
        colored = (mpl_cmaps["jet"](heat_arr / 255.0) * 255).astype(np.uint8)

        overlay_pil = Image.fromarray(colored, mode="RGBA")
        r, g, b, a_ch = overlay_pil.split()
        a_ch = a_ch.point(lambda p: int(p * alpha))
        overlay_pil = Image.merge("RGBA", (r, g, b, a_ch))

        comp = left_image.convert("RGBA")
        right_image = Image.alpha_composite(comp, overlay_pil).convert("RGB")

        return left_image, right_image, scale, offset, s

    def compute_segmentation(
        self,
        smoothed_s: Optional[np.ndarray],
        seg_pct: int,
        epsilon: float,
        display_w: int,
        display_h: int,
    ) -> list:
        """
        Compute polygon contours from a smoothed score map at the given percentile threshold.

        Returns:
            contours — list of [(x, y), ...] point lists in display coordinates
        """
        if smoothed_s is None:
            return []

        seg_thr = np.percentile(smoothed_s, seg_pct)
        mask = (smoothed_s > seg_thr).astype(np.uint8) * 255
        mask = cv2.resize(mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        for cnt in raw_contours:
            if len(cnt) < 3:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
            pts = [
                (float(pt[0][0]), float(pt[0][1]))
                for pt in (approx if approx is not None else cnt)
            ]
            if len(pts) >= 3:
                contours.append(pts)

        return contours

    def compute_visualization(
        self,
        pil_image: Image.Image,
        score_map: np.ndarray,
        alpha: float,
        sigma: float,
        display_target: int,
        heat_min_pct: int,
        seg_pct: int,
        epsilon: float,
    ) -> Tuple[Image.Image, Image.Image, list, float, tuple]:
        """Compatibility shim — delegates to compute_heatmap + compute_segmentation."""
        left_pil, right_pil, scale, offset, s = self.compute_heatmap(
            pil_image, score_map, alpha, sigma, display_target, heat_min_pct
        )
        w, h = left_pil.size
        contours = self.compute_segmentation(s, seg_pct, epsilon, w, h)
        return left_pil, right_pil, contours, scale, offset
