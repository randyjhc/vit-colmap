import numpy as np
from .base_extractor import BaseExtractor


class ViTExtractor(BaseExtractor):
    def __init__(self, weights_path: str | None = None) -> None:
        # TODO: load ViT/ONNX here
        self.weights_path = weights_path

    def extract(self, image_bgr: "np.ndarray") -> tuple[np.ndarray, np.ndarray]:
        # TODO: real inference; return float32 kpts (N,2) and uint8 desc (N,128) or your own D
        kpts = np.zeros((0, 2), dtype=np.float32)
        desc = np.zeros((0, 128), dtype=np.uint8)
        return kpts, desc
