"""COLMAP SIFT extractor wrapper."""

import pycolmap
from pathlib import Path
from typing import Optional
from .base_extractor import BaseExtractor


class ColmapSiftExtractor(BaseExtractor):
    """
    Wrapper for COLMAP's built-in SIFT extractor.

    This extractor uses pycolmap's SIFT implementation directly,
    providing compatibility with COLMAP's native feature extraction.
    Processes all images in a directory at once.
    """

    def __init__(self):
        """Initialize COLMAP SIFT extractor with default options."""
        pass

    def extract(
        self,
        image_dir: Path,
        db_path: Path,
        camera_model: str,
        camera_params: Optional[list[float]] = None,
    ) -> None:
        """
        Extract SIFT features for all images in directory and write to database.

        Args:
            image_dir: Directory containing images
            db_path: Path to COLMAP database
            camera_model: Camera model string
            camera_params: Optional camera parameters (ignored, uses AUTO mode)
        """
        pycolmap.extract_features(
            database_path=str(db_path),
            image_path=str(image_dir),
            camera_mode=pycolmap.CameraMode.AUTO,
            camera_model=camera_model,
        )
