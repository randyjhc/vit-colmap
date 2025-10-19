from pathlib import Path
from typing import Optional

import numpy as np

from .base_extractor import BaseExtractor


class ViTExtractor(BaseExtractor):
    def __init__(self, weights_path: str | None = None) -> None:
        # TODO: load ViT/ONNX here
        self.weights_path = weights_path

    def _run_inference(self, image_bgr: "np.ndarray") -> tuple[np.ndarray, np.ndarray]:
        # TODO: real inference; return float32 kpts (N,2) and uint8 desc (N,128) or your own D
        kpts = np.zeros((0, 2), dtype=np.float32)
        desc = np.zeros((0, 128), dtype=np.uint8)
        return kpts, desc

    def extract(
        self,
        image_dir: Path,
        db_path: Path,
        camera_model: str,
        camera_params: Optional[list[float]] = None,
    ) -> None:
        """
        Run ViT feature extraction over an image directory and write COLMAP-compatible outputs.

        Currently uses a placeholder inference implementation that returns empty features.
        """
        raise NotImplementedError
