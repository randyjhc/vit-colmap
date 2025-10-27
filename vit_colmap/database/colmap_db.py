import pycolmap
import numpy as np
from contextlib import contextmanager


class ColmapDatabase:
    def __init__(self, db_path: str) -> None:
        # pycolmap 3.13+: Database.open(path) is a static method
        # pycolmap 3.12: Database().open(path) is an instance method
        try:
            # Try 3.13+ API first (static method)
            self.db = pycolmap.Database.open(db_path)
        except TypeError:
            # Fall back to 3.12 API (instance method)
            self.db = pycolmap.Database()
            self.db.open(db_path)

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

    @staticmethod
    @contextmanager
    def open_database(db_path: str):
        """Open a COLMAP database with version compatibility.

        pycolmap 3.13+: Database.open(path) is a static method that returns instance
        pycolmap 3.12: Database().open(path) is an instance method
        """
        try:
            # Try 3.13+ API (static method with context manager)
            with pycolmap.Database.open(db_path) as db:
                yield db
        except (TypeError, AttributeError):
            # Fall back to 3.12 API (instance method, no context manager)
            db = pycolmap.Database()
            db.open(db_path)
            try:
                yield db
            finally:
                db.close()

    @staticmethod
    def get_db_count(db, attr_name: str) -> int:
        """Get database count with version compatibility.

        pycolmap 3.13+: num_* are methods
        pycolmap 3.12: num_* are properties
        """
        attr = getattr(db, attr_name)
        return attr() if callable(attr) else attr
