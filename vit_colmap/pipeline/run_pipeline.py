import argparse
import logging
from pathlib import Path
from typing import Optional
import pycolmap

from vit_colmap.features.base_extractor import BaseExtractor
from vit_colmap.features.vit_extractor import ViTExtractor

# Note: BEiT is now only for visualization, not COLMAP feature extraction
from vit_colmap.features.colmap_sift_extractor import ColmapSiftExtractor
from vit_colmap.features.dummy_extractor import DummyExtractor
from vit_colmap.utils.config import Config
from vit_colmap.database.colmap_db import ColmapDatabase
from vit_colmap.utils.metrics import MetricsExtractor, MetricsResult
from vit_colmap.utils.export import export_metrics

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config if config is not None else Config()

    def _print_summary(
        self,
        db_path: Path,
        output_dir: Path,
        do_matching: bool,
        do_reconstruction: bool,
        reconstructions: Optional[dict[int, pycolmap.Reconstruction]],
    ) -> None:
        """Print a comprehensive summary of pipeline results.

        Args:
            db_path: Path to COLMAP database
            output_dir: Output directory path
            do_matching: Whether matching was performed
            do_reconstruction: Whether reconstruction was performed
            reconstructions: Dictionary of reconstructions (if created)
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)

        # Gather statistics from database
        with ColmapDatabase.open_database(str(db_path)) as db:
            num_images = ColmapDatabase.get_db_count(db, "num_images")
            num_cameras = ColmapDatabase.get_db_count(db, "num_cameras")

            # Feature extraction summary
            logger.info("")
            logger.info("Feature Extraction:")
            logger.info(f"  - Images processed: {num_images}")
            logger.info(f"  - Cameras: {num_cameras}")

        # Feature matching summary
        logger.info("")
        if do_matching:
            # Extract comprehensive matching statistics using MetricsExtractor
            from vit_colmap.utils.metrics import MetricsExtractor

            extractor = MetricsExtractor(db_path=db_path, output_dir=output_dir)
            min_threshold = self.config.reconstruction.min_num_matches
            matching_metrics = extractor.extract_matching_metrics(
                min_threshold=min_threshold
            )

            total_possible_pairs = num_images * (num_images - 1) // 2
            raw_rate = (
                (matching_metrics.matched_pairs / total_possible_pairs * 100)
                if total_possible_pairs > 0
                else 0
            )

            logger.info("Feature Matching:")

            # Stage 1: Raw Feature Matching
            logger.info("  Stage 1: Raw Feature Matching")
            logger.info(
                f"    - Pairs processed: {matching_metrics.matched_pairs} / {total_possible_pairs} ({raw_rate:.1f}%)"
            )
            if matching_metrics.total_raw_matches > 0:
                logger.info(
                    f"    - Total raw matches: {matching_metrics.total_raw_matches:,}"
                )
                logger.info(
                    f"    - Avg matches/pair: {matching_metrics.avg_raw_matches:.1f} (range: {matching_metrics.min_raw_matches}-{matching_metrics.max_raw_matches})"
                )
            else:
                logger.info("    - No raw matches found")

            # Stage 2: Geometric Verification
            logger.info("")
            logger.info("  Stage 2: Geometric Verification (RANSAC)")
            if matching_metrics.verified_pairs > 0:
                logger.info(
                    f"    - Pairs verified: {matching_metrics.verified_pairs} / {matching_metrics.matched_pairs} ({matching_metrics.verification_rate:.1f}%)"
                )
                logger.info(
                    f"    - Total inliers: {matching_metrics.total_inlier_matches:,}"
                )
                logger.info(
                    f"    - Avg inliers/pair: {matching_metrics.avg_inlier_matches:.1f} (range: {matching_metrics.min_inlier_matches}-{matching_metrics.max_inlier_matches})"
                )
                logger.info(
                    f"    - Inlier ratio: {matching_metrics.inlier_ratio*100:.1f}%"
                )
            else:
                logger.info("    - No pairs verified (geometric verification failed)")

            # Configuration distribution
            if matching_metrics.verified_pairs > 0:
                config_parts = []
                for config_name in ["CALIBRATED", "UNCALIBRATED", "DEGENERATE"]:
                    count = matching_metrics.config_distribution.get(config_name, 0)
                    pct = (
                        (count / matching_metrics.verified_pairs * 100)
                        if matching_metrics.verified_pairs > 0
                        else 0
                    )
                    config_parts.append(f"{config_name}={count} ({pct:.0f}%)")

                # Add other configs if present
                other_count = sum(
                    v
                    for k, v in matching_metrics.config_distribution.items()
                    if k not in ["CALIBRATED", "UNCALIBRATED", "DEGENERATE"]
                )
                if other_count > 0:
                    pct = other_count / matching_metrics.verified_pairs * 100
                    config_parts.append(f"OTHER={other_count} ({pct:.0f}%)")

                logger.info("")
                logger.info("  Configuration: " + " | ".join(config_parts))

            # Quality assessment
            if matching_metrics.verified_pairs > 0:
                usable_pct = (
                    (
                        matching_metrics.pairs_above_threshold
                        / matching_metrics.verified_pairs
                        * 100
                    )
                    if matching_metrics.verified_pairs > 0
                    else 0
                )

                if matching_metrics.pairs_above_threshold > 0:
                    status_icon = "✓"
                elif matching_metrics.pairs_above_threshold == 0:
                    status_icon = "✗"
                else:
                    status_icon = "⚠"

                logger.info("")
                logger.info(
                    f"  Quality: {status_icon} {matching_metrics.pairs_above_threshold} pairs ({usable_pct:.1f}%) meet threshold (≥{min_threshold} inliers)"
                )
        else:
            logger.info("Feature Matching: SKIPPED")

        # 3D Reconstruction summary
        logger.info("")
        if do_reconstruction:
            if reconstructions and len(reconstructions) > 0:
                logger.info("3D Reconstruction:")
                logger.info(f"  - Reconstructions created: {len(reconstructions)}")

                total_reg_images = 0
                total_points = 0
                for idx, recon in reconstructions.items():
                    num_reg_images = len(recon.images)
                    num_points = len(recon.points3D)
                    total_reg_images += num_reg_images
                    total_points += num_points

                    logger.info(f"  - Reconstruction {idx}:")
                    logger.info(f"      Registered images: {num_reg_images}")
                    logger.info(f"      3D points: {num_points}")

                registration_rate = (
                    (total_reg_images / num_images * 100) if num_images > 0 else 0
                )
                logger.info(
                    f"  - Total registered: {total_reg_images} / {num_images} images ({registration_rate:.1f}%)"
                )
                logger.info(f"  - Total 3D points: {total_points}")
            else:
                logger.info("3D Reconstruction: FAILED (no reconstructions created)")
        else:
            logger.info("3D Reconstruction: SKIPPED")

        # Output paths
        logger.info("")
        logger.info("Output:")
        logger.info(f"  - Database: {db_path}")
        if do_reconstruction:
            sparse_dir = output_dir / "sparse"
            logger.info(f"  - Sparse reconstruction: {sparse_dir}")

        logger.info("=" * 60)
        logger.info("")

    def extract_and_export_metrics(
        self,
        db_path: Path,
        output_dir: Path,
        reconstructions: Optional[dict[int, pycolmap.Reconstruction]],
        dataset: str,
        scene: str,
        results_dir: Optional[Path] = None,
    ) -> Optional[MetricsResult]:
        """Extract metrics and optionally export them.

        Args:
            db_path: Path to COLMAP database
            output_dir: Output directory for reconstruction
            reconstructions: Dictionary of reconstructions
            dataset: Dataset name (e.g., "DTU")
            scene: Scene/scan name (e.g., "scan1")
            results_dir: Directory to export results (if None, no export)

        Returns:
            MetricsResult object, or None if extraction fails
        """
        try:
            # Create metrics extractor
            extractor_obj = MetricsExtractor(db_path=db_path, output_dir=output_dir)

            # Extract all metrics
            extractor_type = self.config.extractor.extractor_type
            if extractor_type == "colmap_sift":
                extractor_type = "sift"  # Normalize name
            elif extractor_type == "vit":
                extractor_type = "vit"
            elif extractor_type == "dummy":
                extractor_type = "dummy"

            config_dict = {
                "camera_model": self.config.camera.model,
                "min_num_matches": self.config.reconstruction.min_num_matches,
                "matching_max_ratio": self.config.matching.max_ratio,
                "matching_use_gpu": self.config.matching.use_gpu,
            }

            metrics = extractor_obj.extract_all_metrics(
                dataset=dataset,
                scene=scene,
                extractor_type=extractor_type,
                config=config_dict,
                reconstructions=reconstructions,
            )

            # Export if results directory specified
            if results_dir:
                export_metrics(metrics, results_dir, formats=["json", "csv"])

            return metrics

        except Exception as e:
            logger.error(f"Failed to extract/export metrics: {e}")
            import traceback

            traceback.print_exc()
            return None

    def run(
        self,
        image_dir: Path,
        output_dir: Path,
        db_path: Path,
        dataset: Optional[str] = None,
        scene: Optional[str] = None,
        results_dir: Optional[Path] = None,
    ) -> Optional[dict[int, pycolmap.Reconstruction]]:
        """
        Run the full SfM pipeline: feature extraction, matching, and reconstruction.

        Uses the configuration provided during Pipeline initialization.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory for output reconstruction
            db_path: Path to COLMAP database file
            dataset: Dataset name for metrics (e.g., "DTU")
            scene: Scene/scan name for metrics (e.g., "scan1")
            results_dir: Directory to export metrics (if None, no export)

        Returns:
            Dictionary of reconstructions if do_reconstruction=True, else None
        """
        # Check GPU availability at the start
        if hasattr(pycolmap, "has_cuda"):
            if pycolmap.has_cuda:
                logger.info("COLMAP built with CUDA support")
                if hasattr(pycolmap, "get_num_cuda_devices"):
                    num_devices = pycolmap.get_num_cuda_devices()
                    logger.info(f"CUDA devices available: {num_devices}")
            else:
                logger.info("COLMAP built WITHOUT CUDA support (CPU only)")

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
            # Default to ViT extractor
            logger.info("Using ViT extractor")
            extractor = ViTExtractor(
                weights_path=self.config.extractor.vit_weights_path
            )

        # Extract features for all images in directory (writes to database)
        logger.info("Extracting features...")
        extractor.extract(image_dir, db_path, camera_model, camera_params)

        # Check how many images were processed
        with ColmapDatabase.open_database(str(db_path)) as db_check:
            num_imgs = ColmapDatabase.get_db_count(db_check, "num_images")
            logger.info(f"Extracted features for {num_imgs} images")

        # Feature matching
        if do_matching:
            logger.info("Running feature matching...")
            matching_opts = self.config.matching.to_matching_options()

            # pycolmap 3.13+ uses matching_options, 3.12 uses sift_options
            if hasattr(pycolmap, "FeatureMatchingOptions"):
                pycolmap.match_exhaustive(
                    database_path=str(db_path), matching_options=matching_opts
                )
            else:
                pycolmap.match_exhaustive(
                    database_path=str(db_path), sift_options=matching_opts
                )

            # Check matches
            with ColmapDatabase.open_database(str(db_path)) as db_check:
                num_pairs = ColmapDatabase.get_db_count(
                    db_check, "num_matched_image_pairs"
                )
                logger.info(f"Matched {num_pairs} image pairs")

        # 3D Reconstruction
        reconstructions = None
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

        # Print comprehensive summary
        self._print_summary(
            db_path=db_path,
            output_dir=output_dir,
            do_matching=do_matching,
            do_reconstruction=do_reconstruction,
            reconstructions=reconstructions,
        )

        # Extract and export metrics if requested
        if dataset and scene:
            self.extract_and_export_metrics(
                db_path=db_path,
                output_dir=output_dir,
                reconstructions=reconstructions,
                dataset=dataset,
                scene=scene,
                results_dir=results_dir,
            )

        return reconstructions


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
    ap.add_argument(
        "--num-keypoints",
        type=int,
        default=2048,
        help="Number of keypoints to extract per image (default: 2048)",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name for metrics (e.g., DTU, HPatches)",
    )
    ap.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene/scan name for metrics (e.g., scan1)",
    )
    ap.add_argument(
        "--export-metrics",
        type=Path,
        default=None,
        help="Directory to export metrics (e.g., data/results)",
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
        dataset=args.dataset,
        scene=args.scene,
        results_dir=args.export_metrics,
    )

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
