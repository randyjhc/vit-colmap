import argparse
import logging
from pathlib import Path
from typing import Optional
import pycolmap

from vit_colmap.features.base_extractor import BaseExtractor
from vit_colmap.features.vit_extractor import ViTExtractor
from vit_colmap.features.colmap_sift_extractor import ColmapSiftExtractor
from vit_colmap.features.dummy_extractor import DummyExtractor
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
    ) -> Optional[dict[int, pycolmap.Reconstruction]]:
        """
        Run the full SfM pipeline: feature extraction, matching, and reconstruction.

        Uses the configuration provided during Pipeline initialization.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory for output reconstruction
            db_path: Path to COLMAP database file

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

        extractor: BaseExtractor
        # Choose extractor and run feature extraction
        if self.config.extractor.extractor_type == "dummy":
            logger.info("Using Dummy extractor")
            extractor = DummyExtractor(step=32)
        elif self.config.extractor.extractor_type == "colmap_sift":
            logger.info("Using COLMAP SIFT extractor")
            extractor = ColmapSiftExtractor()
        else:
            logger.info("Using ViT extractor")
            extractor = ViTExtractor(
                weights_path=self.config.extractor.vit_weights_path
            )

        # Extract features for all images in directory (writes to database)
        logger.info("Extracting features...")
        extractor.extract(image_dir, db_path, camera_model, camera_params)

        # Check how many images were processed
        db_check = pycolmap.Database(str(db_path))
        logger.info(f"Extracted features for {db_check.num_images} images")

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
    ap.add_argument(
        "--use-colmap-sift",
        action="store_true",
        help="Use COLMAP's built-in SIFT instead of ViT extractor",
    )
    args = ap.parse_args()

    # Create configuration from arguments
    config = Config.from_args(args)
    logger.info("Configuration loaded")
    logger.debug(f"\n{config.summary()}")

    # Run pipeline
    pipeline = Pipeline(config=config)
    pipeline.run(
        image_dir=args.images,
        output_dir=args.output,
        db_path=args.db,
    )

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
