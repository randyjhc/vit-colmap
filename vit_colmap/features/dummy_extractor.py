import numpy as np
from .base_extractor import BaseExtractor


class DummyExtractor(BaseExtractor):
    """
    Deterministic grid of keypoints + random 128D uint8 descriptors.
    Descriptors are seeded by keypoint position so they're deterministic and matchable.
    """

    def __init__(self, step: int = 32, seed: int = 42):
        self.step = step
        self.seed = seed

    def extract(self, image_bgr: "np.ndarray") -> tuple[np.ndarray, np.ndarray]:
        h, w = image_bgr.shape[:2]
        ys = np.arange(self.step // 2, h, self.step, dtype=np.float32)
        xs = np.arange(self.step // 2, w, self.step, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        kpts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # (N,2)

        # Create random but deterministic descriptors based on keypoint grid position
        # This ensures similar keypoint locations across images will have similar descriptors
        desc_list = []
        for kpt in kpts:
            # Use grid position (quantized coordinates) as seed for reproducibility
            grid_x = int(kpt[0] / self.step)
            grid_y = int(kpt[1] / self.step)
            local_seed = self.seed + grid_x * 1000 + grid_y

            # Generate random descriptor
            rng = np.random.RandomState(local_seed)
            descriptor = rng.randint(0, 256, size=128, dtype=np.uint8)
            desc_list.append(descriptor)

        desc = np.stack(desc_list, axis=0)  # (N, 128)
        return kpts, desc
