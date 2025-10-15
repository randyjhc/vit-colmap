from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, image_bgr: "np.ndarray") -> Tuple[np.ndarray, np.ndarray]:
        """Return (keypoints[N,2], descriptors[N,D])."""
        raise NotImplementedError
