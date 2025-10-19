import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from .base_extractor import BaseExtractor


class DummyExtractor(BaseExtractor):
    """
    Deterministic grid of keypoints + random 128D uint8 descriptors.
    Descriptors are seeded by keypoint position so they're deterministic and matchable.
    Now supports batch processing of entire directories.
    """

    def __init__(self, step: int = 32, seed: int = 42):
        self.step = step
        self.seed = seed

    def extract(
        self,
        image_dir: Path,
        db_path: Path,
        camera_model: str,
        camera_params: Optional[list[float]] = None,
    ) -> None:
        """
        Extract features for all images in directory and write to database.
        Generates dummy images if directory is empty.

        Args:
            image_dir: Directory containing images (or where to generate them)
            db_path: Path to COLMAP database
            camera_model: Camera model string
            camera_params: Optional camera parameters
        """
        import pycolmap
        from vit_colmap.database.colmap_db import ColmapDatabase

        # Get list of existing image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = sorted(
            [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
        )

        # If no images exist, generate dummy images
        if not image_files:
            print(f"No images found in {image_dir}, generating 10 dummy images...")
            image_dir.mkdir(parents=True, exist_ok=True)

            for i in range(10):
                # Generate a simple random image
                img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                img_path = image_dir / f"dummy_{i:03d}.png"
                cv2.imwrite(str(img_path), img)
                image_files.append(img_path)

        # Initialize database
        db = ColmapDatabase(str(db_path))

        # Read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            return

        height, width = first_img.shape[:2]

        # Set default camera parameters if not provided
        if camera_params is None:
            if camera_model == "SIMPLE_PINHOLE":
                f = max(width, height)
                camera_params = [f, width / 2.0, height / 2.0]
            elif camera_model == "PINHOLE":
                f = max(width, height)
                camera_params = [f, f, width / 2.0, height / 2.0]
            else:
                raise ValueError(f"Unsupported camera model: {camera_model}")

        # Add camera to database
        camera = pycolmap.Camera(
            model=camera_model, width=width, height=height, params=camera_params
        )
        camera_id = db.db.write_camera(camera)

        # Process each image
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            # Add image to database
            image_id = db.add_image(img_file.name, camera_id=camera_id)

            # Extract features for this image
            h, w = img.shape[:2]
            ys = np.arange(self.step // 2, h, self.step, dtype=np.float32)
            xs = np.arange(self.step // 2, w, self.step, dtype=np.float32)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            kpts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

            # Create deterministic descriptors
            desc_list = []
            for kpt in kpts:
                grid_x = int(kpt[0] / self.step)
                grid_y = int(kpt[1] / self.step)
                local_seed = self.seed + grid_x * 1000 + grid_y

                rng = np.random.RandomState(local_seed)
                descriptor = rng.randint(0, 256, size=128, dtype=np.uint8)
                desc_list.append(descriptor)

            desc = np.stack(desc_list, axis=0)

            # Store in database
            db.add_keypoints(image_id, kpts)
            db.add_descriptors(image_id, desc)

        db.commit()
