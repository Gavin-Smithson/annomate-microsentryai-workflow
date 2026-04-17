"""
Anomalib Strategy for MicroSentryAI.

Primary path: Anomalib TorchInferencer.
Fallback: raw torch.load with DynamicUnpickler to handle missing checkpoint classes.
No Qt dependencies.
"""

import os
import io
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from anomalib.deploy import TorchInferencer
import cv2

logger = logging.getLogger("MicroSentryAI.AnomalibStrategy")


# ---------------------------------------------------------------------------
# Dynamic Unpickler — bypasses missing classes in torch.load
# ---------------------------------------------------------------------------

class DummyMeta(type):
    def __getattr__(cls, name):
        return DummyClass


class DummyClass(metaclass=DummyMeta):
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return DummyClass()
    def __getattr__(self, name): return DummyClass()
    def __getitem__(self, key): return DummyClass()
    def __setitem__(self, key, value): pass
    def __setstate__(self, state): pass


class DynamicUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            logger.warning("Mocking missing checkpoint class: %s.%s", module, name)
            return DummyClass


class DynamicPickleModule:
    Unpickler = DynamicUnpickler

    @staticmethod
    def load(file, **kwargs):
        return DynamicUnpickler(file).load()

    @staticmethod
    def loads(b, **kwargs):
        return DynamicUnpickler(io.BytesIO(b)).load()


# ---------------------------------------------------------------------------

class AnomalibStrategy:
    """Strategy for PyTorch (.pt, .ckpt) anomaly detection models."""

    def __init__(self):
        self.torch_inferencer: Optional[TorchInferencer] = None
        self.raw_model = None
        self.device = "auto"
        self.model_type = "unknown"
        self.model_name = "Unknown"
        self._device_verified = False

    def set_device(self, device_code: str):
        self.device = device_code.lower()
        logger.info("Target device set to: %s", self.device)

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_from_folder(self, folder_path: str) -> None:
        raise NotImplementedError("Use load_from_file() for PyTorch models.")

    def load_from_file(self, model_path: str):
        """
        Load a .pt or .ckpt model.
        Attempts Anomalib TorchInferencer first; falls back to raw torch.load.
        Raises RuntimeError on failure.
        """
        path = Path(model_path)
        self._device_verified = False
        self.torch_inferencer = None
        self.raw_model = None

        try:
            os.environ["TRUST_REMOTE_CODE"] = "1"

            if path.suffix not in (".pt", ".ckpt"):
                raise ValueError(f"Unsupported file type: {path.suffix}. Expected .pt or .ckpt")

            self.model_type = "torch"
            resolved_device = self._resolve_device()
            final_device = resolved_device.upper()

            import functools
            original_torch_load = torch.load

            try:
                # Attempt 1: Anomalib TorchInferencer (monkey-patch forces CPU deserialisation)
                try:
                    torch.load = functools.partial(original_torch_load, map_location="cpu")

                    if resolved_device == "mps":
                        logger.debug("Applying MPS shim: initialising on CPU first.")
                        self.torch_inferencer = TorchInferencer(path=path, device="cpu")
                        mps_device = torch.device("mps")
                        if hasattr(self.torch_inferencer, "model"):
                            self.torch_inferencer.model = self.torch_inferencer.model.to(mps_device)
                        self.torch_inferencer.device = mps_device
                        final_device = "MPS (Apple Silicon)"
                    else:
                        self.torch_inferencer = TorchInferencer(path=path, device=resolved_device)

                    self.model_name = f"Anomalib (Torch) [{final_device}]"
                    logger.info("Loaded %s via TorchInferencer", self.model_name)

                finally:
                    torch.load = original_torch_load

            except Exception as anomalib_err:
                # Attempt 2: Raw PyTorch fallback with DynamicUnpickler
                logger.warning("TorchInferencer rejected the model (%s). Trying raw fallback.", anomalib_err)
                self.torch_inferencer = None

                device_obj = torch.device(resolved_device if resolved_device != "mps" else "cpu")
                loaded_data = torch.load(path, map_location=device_obj, pickle_module=DynamicPickleModule)

                if isinstance(loaded_data, dict):
                    if "state_dict" in loaded_data and "model" not in loaded_data:
                        raise ValueError(
                            "You loaded a training .ckpt file. "
                            "Please select the exported model.pt from your weights/torch folder."
                        )
                    elif "model" in loaded_data:
                        self.raw_model = loaded_data["model"]
                    else:
                        raise ValueError("Loaded dict does not contain a recognisable model graph.")
                else:
                    self.raw_model = loaded_data

                if hasattr(self.raw_model, "eval"):
                    self.raw_model.eval()

                if resolved_device == "mps":
                    try:
                        self.raw_model = self.raw_model.to(torch.device("mps"))
                        final_device = "MPS (Apple Silicon)"
                    except Exception as mps_err:
                        logger.debug("Failed to push raw model to MPS: %s", mps_err)

                self.model_name = f"Raw PyTorch Model [{final_device}]"
                logger.info("Loaded %s via raw torch.load", self.model_name)

        except Exception as e:
            logger.error("Critical failure loading model: %s", e)
            raise RuntimeError(f"Load Error: {e}")

    def predict(self, image_path: str) -> Tuple[float, np.ndarray]:
        if self.torch_inferencer:
            return self._predict_anomalib(image_path)
        if self.raw_model:
            return self._predict_raw(image_path)
        return 0.0, np.zeros((256, 256), dtype=np.float32)

    def _predict_anomalib(self, image_path: str) -> Tuple[float, np.ndarray]:
        try:
            if not self._device_verified:
                try:
                    p = next(self.torch_inferencer.model.parameters())
                    logger.debug("Tensors are on %s", p.device)
                    self._device_verified = True
                except StopIteration:
                    pass

            result = self.torch_inferencer.predict(image=image_path)

            if self.device == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            score = 0.0
            if hasattr(result, "pred_score") and result.pred_score is not None:
                score = float(
                    result.pred_score.item()
                    if isinstance(result.pred_score, torch.Tensor)
                    else result.pred_score
                )

            heatmap = result.anomaly_map
            if isinstance(heatmap, torch.Tensor):
                heatmap = heatmap.detach().cpu().numpy()
            heatmap = heatmap.squeeze()

            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return score, heatmap.astype(np.float32)

        except Exception as e:
            logger.error("Anomalib inference failed: %s", e)
            return 0.0, np.zeros((256, 256), dtype=np.float32)

    def _predict_raw(self, image_path: str) -> Tuple[float, np.ndarray]:
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))

            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            device_obj = (
                next(self.raw_model.parameters()).device
                if hasattr(self.raw_model, "parameters")
                else torch.device("cpu")
            )
            tensor = tensor.to(device_obj)

            with torch.no_grad():
                output = self.raw_model(tensor)

            score = 0.0
            heatmap = np.zeros((256, 256), dtype=np.float32)

            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        if item.ndim >= 2 and item.numel() > 1 and item.is_floating_point():
                            heatmap = item.squeeze().cpu().numpy()
                        elif item.numel() == 1:
                            score = float(item.cpu().item())
            elif isinstance(output, torch.Tensor):
                heatmap = output.squeeze().cpu().numpy()

            heatmap = heatmap.astype(np.float32)
            if heatmap.max() > heatmap.min():
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            return score, heatmap

        except Exception as e:
            logger.error("Raw PyTorch inference failed: %s", e)
            return 0.0, np.zeros((256, 256), dtype=np.float32)
