import pycolmap
import numpy as np


class ColmapDatabase:
    def __init__(self, db_path: str) -> None:
        self.db = pycolmap.Database(db_path)

    # --- camera & image bookkeeping ---
    def add_pinhole_camera(
        self, width: int, height: int, fx: float, fy: float, cx: float, cy: float
    ) -> int:
        params = [fx, fy, cx, cy]
        camera = pycolmap.Camera(
            model="PINHOLE", width=width, height=height, params=params
        )
        return self.db.write_camera(camera)

    def add_image(self, name: str, camera_id: int) -> int:
        image = pycolmap.Image(name=name, camera_id=camera_id)
        return self.db.write_image(image)

    # --- features & matches ---
    def add_keypoints(self, image_id: int, kpts: np.ndarray) -> None:
        self.db.write_keypoints(image_id, kpts.astype(np.float32))

    def add_descriptors(self, image_id: int, desc: np.ndarray) -> None:
        self.db.write_descriptors(image_id, desc.astype(np.uint8))

    def add_matches(self, image_id1: int, image_id2: int, pairs: np.ndarray) -> None:
        self.db.write_matches(image_id1, image_id2, pairs.astype(np.uint32))

    def commit(self) -> None:
        # pycolmap Database doesn't need explicit commit
        pass
