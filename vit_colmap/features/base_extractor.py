from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        image_dir: Path,
        db_path: Path,
        camera_model: str,
        camera_params: Optional[list[float]] = None,
    ) -> None:
        """Process images in `image_dir` and write features into the COLMAP database at `db_path`."""
        raise NotImplementedError
