import argparse
import logging
from pathlib import Path
from typing import Optional
import cv2
import pycolmap

from vit_colmap.features.base_extractor import BaseExtractor
from vit_colmap.features.vit_extractor import ViTExtractor
from vit_colmap.database.colmap_db import ColmapDatabase
from vit_colmap.utils.config import Config

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config if config is not None else Config()

    def run(
        self,
        image_dir: Path,
        output_dir: Path,
        db_path: Path,
        extractor: BaseExtractor,
    ) -> Optional[dict[int, pycolmap.Reconstruction]]:
        """
        Run the full SfM pipeline: feature extraction, matching, and reconstruction.

        Uses the configuration provided during Pipeline initialization.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory for output reconstruction
            db_path: Path to COLMAP database file
            extractor: Feature extractor instance

        Returns:
            Dictionary of reconstructions if do_reconstruction=True, else None
        """
        # Use config for settings
        camera_model = self.config.camera.model
        camera_params = self.config.camera.params
        do_matching = self.config.do_matching
        do_reconstruction = self.config.do_reconstruction

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        db = ColmapDatabase(str(db_path))

        # Get list of image files (common image extensions)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = sorted(
            [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
        )

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        logger.info(f"Found {len(image_files)} images in {image_dir}")

        # Create a shared camera for all images (assuming same camera)
        # Read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            raise ValueError(f"Failed to read image: {image_files[0]}")

        height, width = first_img.shape[:2]

        # Set default camera parameters if not provided
        if camera_params is None:
            if camera_model == "SIMPLE_PINHOLE":
                # f, cx, cy
                f = max(width, height)
                camera_params = [f, width / 2.0, height / 2.0]
            elif camera_model == "PINHOLE":
                # fx, fy, cx, cy
                f = max(width, height)
                camera_params = [f, f, width / 2.0, height / 2.0]
            else:
                raise ValueError(f"Unsupported camera model: {camera_model}")

        # Add camera to database
        camera = pycolmap.Camera(
            model=camera_model, width=width, height=height, params=camera_params
        )
        camera_id = db.db.write_camera(camera)
        logger.info(f"Added camera (model={camera_model}, id={camera_id})")

        # Extract features for each image
        logger.info("Extracting features...")
        image_ids = []
        for img_file in image_files:
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Failed to read {img_file}, skipping")
                continue

            # Add image to database
            image_id = db.add_image(img_file.name, camera_id=camera_id)
            image_ids.append(image_id)

            # Extract features
            keypoints, descriptors = extractor.extract(img)

            # Store in database
            db.add_keypoints(image_id, keypoints)
            db.add_descriptors(image_id, descriptors)

            logger.debug(f"  {img_file.name}: {len(keypoints)} keypoints")

        db.commit()
        logger.info(f"Stored features for {len(image_ids)} images")

        # Feature matching
        if do_matching:
            logger.info("Running feature matching...")
            sift_opts = self.config.matching.to_sift_options()

            pycolmap.match_exhaustive(
                database_path=str(db_path), sift_options=sift_opts
            )

            # Check matches
            db_check = pycolmap.Database(str(db_path))
            logger.info(f"Matched {db_check.num_matched_image_pairs} image pairs")

        # 3D Reconstruction
        if do_reconstruction:
            logger.info("Running 3D reconstruction...")
            sparse_dir = output_dir / "sparse"
            sparse_dir.mkdir(parents=True, exist_ok=True)

            mapper_options = self.config.reconstruction.to_mapper_options()

            reconstructions = pycolmap.incremental_mapping(
                database_path=str(db_path),
                image_path=str(image_dir),
                output_path=str(sparse_dir),
                options=mapper_options,
            )

            if reconstructions:
                logger.info(f"Created {len(reconstructions)} reconstruction(s)")
                for idx, recon in reconstructions.items():
                    logger.info(
                        f"  Reconstruction {idx}: {len(recon.images)} images, {len(recon.points3D)} points"
                    )
            else:
                logger.warning("No reconstructions created")

            return reconstructions

        return None


def main() -> None:
    """Command-line interface for the SfM pipeline."""

    ap = argparse.ArgumentParser(
        description="Run ViT-COLMAP Structure-from-Motion pipeline"
    )
    ap.add_argument(
        "--images", required=True, type=Path, help="Directory containing input images"
    )
    ap.add_argument(
        "--output", required=True, type=Path, help="Output directory for reconstruction"
    )
    ap.add_argument(
        "--db",
        default=Path("data/intermediate/database.db"),
        type=Path,
        help="Path to COLMAP database file",
    )
    ap.add_argument(
        "--model", default=None, type=Path, help="Path to ViT model weights (optional)"
    )
    ap.add_argument(
        "--camera-model",
        default="SIMPLE_PINHOLE",
        type=str,
        help="COLMAP camera model (SIMPLE_PINHOLE or PINHOLE)",
    )
    ap.add_argument(
        "--skip-matching", action="store_true", help="Skip feature matching"
    )
    ap.add_argument(
        "--skip-reconstruction", action="store_true", help="Skip 3D reconstruction"
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )
    args = ap.parse_args()

    # Create configuration from arguments
    config = Config.from_args(args)
    logger.info("Configuration loaded")
    logger.debug(f"\n{config.summary()}")

    # Initialize feature extractor
    extractor = ViTExtractor(weights_path=str(args.model) if args.model else None)

    # Run pipeline
    pipeline = Pipeline(config=config)
    pipeline.run(
        image_dir=args.images,
        output_dir=args.output,
        db_path=args.db,
        extractor=extractor,
    )

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
