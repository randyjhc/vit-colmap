"""Metrics extraction and aggregation for ViT-COLMAP pipeline."""

import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pycolmap

from vit_colmap.database.colmap_db import ColmapDatabase

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetrics:
    """Metrics related to feature extraction."""

    total_images: int
    total_keypoints: int
    avg_keypoints_per_image: float
    min_keypoints: int
    max_keypoints: int
    median_keypoints: float


@dataclass
class MatchingMetrics:
    """Metrics related to feature matching."""

    total_image_pairs: int
    matched_pairs: int
    verified_pairs: int
    match_rate: float  # percentage of pairs with matches

    # Raw matches (before geometric verification)
    total_raw_matches: int
    avg_raw_matches: float
    min_raw_matches: int
    max_raw_matches: int
    median_raw_matches: float

    # Inlier matches (after RANSAC)
    total_inlier_matches: int
    avg_inlier_matches: float
    min_inlier_matches: int
    max_inlier_matches: int
    median_inlier_matches: float
    inlier_ratio: float  # inliers / raw_matches

    # Quality metrics
    verification_rate: float = 0.0  # verified_pairs / matched_pairs (percentage)
    pairs_above_threshold: int = 0  # pairs meeting minimum inlier threshold

    # Configuration distribution
    config_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class ReconstructionMetrics:
    """Metrics related to 3D reconstruction."""

    num_reconstructions: int
    registered_images: int
    registration_rate: float  # percentage of images registered
    total_3d_points: int
    avg_track_length: float  # average observations per 3D point
    avg_reprojection_error: float

    # Per-reconstruction breakdown
    reconstructions: list[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MetricsResult:
    """Complete metrics result for a pipeline run."""

    # Metadata
    dataset: str
    scene: str
    extractor_type: str
    timestamp: str

    # Metrics
    features: FeatureMetrics
    matching: MatchingMetrics
    reconstruction: Optional[ReconstructionMetrics] = None

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert nested dataclasses
        if self.reconstruction:
            result["reconstruction"] = asdict(self.reconstruction)
        result["features"] = asdict(self.features)
        result["matching"] = asdict(self.matching)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsResult":
        """Create from dictionary."""
        # Reconstruct nested dataclasses
        if data.get("features"):
            data["features"] = FeatureMetrics(**data["features"])
        if data.get("matching"):
            data["matching"] = MatchingMetrics(**data["matching"])
        if data.get("reconstruction"):
            data["reconstruction"] = ReconstructionMetrics(**data["reconstruction"])
        return cls(**data)


class MetricsExtractor:
    """Extracts metrics from COLMAP database and reconstruction outputs."""

    # Configuration type names for interpretation
    CONFIG_NAMES = {
        0: "UNDEFINED",
        1: "DEGENERATE",
        2: "CALIBRATED",
        3: "UNCALIBRATED",
        4: "PLANAR",
        5: "PANORAMIC",
        6: "PLANAR_OR_PANORAMIC",
        7: "WATERMARK",
        8: "MULTIPLE",
        9: "CALIBRATED_RIG",
    }

    def __init__(self, db_path: Path, output_dir: Path):
        """Initialize metrics extractor.

        Args:
            db_path: Path to COLMAP database
            output_dir: Path to output directory with reconstructions
        """
        self.db_path = db_path
        self.output_dir = output_dir

    def extract_feature_metrics(self) -> FeatureMetrics:
        """Extract feature extraction metrics from database.

        Returns:
            FeatureMetrics object with statistics
        """
        with ColmapDatabase.open_database(str(self.db_path)) as db:
            num_images = ColmapDatabase.get_db_count(db, "num_images")

            # Get keypoint counts per image
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Query keypoints table
            cursor.execute("SELECT image_id, rows, cols FROM keypoints")
            keypoints_data = cursor.fetchall()

            keypoint_counts = [row[1] for row in keypoints_data]  # rows = num_keypoints

            conn.close()

            total_keypoints = sum(keypoint_counts)
            avg_keypoints = (
                total_keypoints / len(keypoint_counts) if keypoint_counts else 0
            )

            return FeatureMetrics(
                total_images=num_images,
                total_keypoints=total_keypoints,
                avg_keypoints_per_image=avg_keypoints,
                min_keypoints=min(keypoint_counts) if keypoint_counts else 0,
                max_keypoints=max(keypoint_counts) if keypoint_counts else 0,
                median_keypoints=float(
                    np.median(keypoint_counts) if keypoint_counts else 0
                ),
            )

    def extract_matching_metrics(
        self, min_threshold: Optional[int] = None
    ) -> MatchingMetrics:
        """Extract feature matching metrics from database.

        Args:
            min_threshold: Minimum number of inliers required for reconstruction.
                          If provided, calculates pairs_above_threshold.

        Returns:
            MatchingMetrics object with statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get total number of images for computing possible pairs
        cursor.execute("SELECT COUNT(*) FROM images")
        num_images = cursor.fetchone()[0]
        total_possible_pairs = num_images * (num_images - 1) // 2

        # Get raw matches (before geometric verification)
        cursor.execute("SELECT pair_id, rows FROM matches")
        matches_data = cursor.fetchall()
        raw_match_counts = [row[1] for row in matches_data]

        # Get two-view geometries (after geometric verification)
        cursor.execute("SELECT pair_id, rows, config FROM two_view_geometries")
        tvg_data = cursor.fetchall()
        inlier_counts = [row[1] for row in tvg_data]

        # Configuration distribution
        config_distribution: Dict[str, int] = {}
        for row in tvg_data:
            config = row[2]
            config_name = self.CONFIG_NAMES.get(config, f"UNKNOWN({config})")
            config_distribution[config_name] = (
                config_distribution.get(config_name, 0) + 1
            )

        conn.close()

        # Calculate statistics
        total_raw_matches = sum(raw_match_counts)
        total_inlier_matches = sum(inlier_counts)

        match_rate = (
            (len(matches_data) / total_possible_pairs * 100)
            if total_possible_pairs > 0
            else 0
        )
        inlier_ratio = (
            (total_inlier_matches / total_raw_matches) if total_raw_matches > 0 else 0
        )
        verification_rate = (
            (len(tvg_data) / len(matches_data) * 100) if len(matches_data) > 0 else 0
        )

        # Calculate pairs above threshold if provided
        pairs_above_threshold = 0
        if min_threshold is not None and inlier_counts:
            pairs_above_threshold = sum(
                1 for count in inlier_counts if count >= min_threshold
            )

        return MatchingMetrics(
            total_image_pairs=total_possible_pairs,
            matched_pairs=len(matches_data),
            verified_pairs=len(tvg_data),
            match_rate=match_rate,
            total_raw_matches=total_raw_matches,
            avg_raw_matches=np.mean(raw_match_counts) if raw_match_counts else 0,
            min_raw_matches=min(raw_match_counts) if raw_match_counts else 0,
            max_raw_matches=max(raw_match_counts) if raw_match_counts else 0,
            median_raw_matches=float(np.median(raw_match_counts))
            if raw_match_counts
            else 0,
            total_inlier_matches=total_inlier_matches,
            avg_inlier_matches=np.mean(inlier_counts) if inlier_counts else 0,
            min_inlier_matches=min(inlier_counts) if inlier_counts else 0,
            max_inlier_matches=max(inlier_counts) if inlier_counts else 0,
            median_inlier_matches=float(np.median(inlier_counts))
            if inlier_counts
            else 0,
            inlier_ratio=inlier_ratio,
            verification_rate=verification_rate,
            pairs_above_threshold=pairs_above_threshold,
            config_distribution=config_distribution,
        )

    def extract_reconstruction_metrics(
        self, reconstructions: Optional[Dict[int, pycolmap.Reconstruction]]
    ) -> Optional[ReconstructionMetrics]:
        """Extract 3D reconstruction metrics.

        Args:
            reconstructions: Dictionary of reconstructions from pycolmap

        Returns:
            ReconstructionMetrics object, or None if no reconstructions
        """
        if not reconstructions or len(reconstructions) == 0:
            return None

        # Get total images from database
        with ColmapDatabase.open_database(str(self.db_path)) as db:
            total_images = ColmapDatabase.get_db_count(db, "num_images")

        total_registered = 0
        total_points = 0
        all_track_lengths = []
        all_errors = []
        recon_details = []

        for idx, recon in reconstructions.items():
            num_reg_images = len(recon.images)
            num_points = len(recon.points3D)

            total_registered += num_reg_images
            total_points += num_points

            # Calculate average track length (observations per point)
            track_lengths = [
                len(point.track.elements) for point in recon.points3D.values()
            ]
            all_track_lengths.extend(track_lengths)

            # Calculate average reprojection error
            errors = [point.error for point in recon.points3D.values()]
            all_errors.extend(errors)

            recon_details.append(
                {
                    "id": idx,
                    "registered_images": num_reg_images,
                    "num_3d_points": num_points,
                    "avg_track_length": float(np.mean(track_lengths))
                    if track_lengths
                    else 0,
                    "avg_reprojection_error": float(np.mean(errors)) if errors else 0,
                }
            )

        registration_rate = (
            (total_registered / total_images * 100) if total_images > 0 else 0
        )

        return ReconstructionMetrics(
            num_reconstructions=len(reconstructions),
            registered_images=total_registered,
            registration_rate=registration_rate,
            total_3d_points=total_points,
            avg_track_length=float(np.mean(all_track_lengths))
            if all_track_lengths
            else 0,
            avg_reprojection_error=float(np.mean(all_errors)) if all_errors else 0,
            reconstructions=recon_details,
        )

    def extract_all_metrics(
        self,
        dataset: str,
        scene: str,
        extractor_type: str,
        config: Optional[Dict[str, Any]] = None,
        reconstructions: Optional[Dict[int, pycolmap.Reconstruction]] = None,
    ) -> MetricsResult:
        """Extract all metrics and create a complete MetricsResult.

        Args:
            dataset: Dataset name (e.g., "DTU", "HPatches")
            scene: Scene/scan name (e.g., "scan1")
            extractor_type: Feature extractor type (e.g., "sift", "vit")
            config: Configuration dictionary
            reconstructions: Reconstruction results from pipeline

        Returns:
            Complete MetricsResult object
        """
        logger.info("Extracting metrics...")

        feature_metrics = self.extract_feature_metrics()
        logger.info(
            f"  Features: {feature_metrics.total_keypoints} keypoints across {feature_metrics.total_images} images"
        )

        matching_metrics = self.extract_matching_metrics()
        logger.info(
            f"  Matching: {matching_metrics.matched_pairs} matched pairs, {matching_metrics.verified_pairs} verified"
        )

        reconstruction_metrics = None
        if reconstructions:
            reconstruction_metrics = self.extract_reconstruction_metrics(
                reconstructions
            )
            if reconstruction_metrics:
                logger.info(
                    f"  Reconstruction: {reconstruction_metrics.registered_images} images, "
                    f"{reconstruction_metrics.total_3d_points} 3D points"
                )

        return MetricsResult(
            dataset=dataset,
            scene=scene,
            extractor_type=extractor_type,
            timestamp=datetime.now().isoformat(),
            features=feature_metrics,
            matching=matching_metrics,
            reconstruction=reconstruction_metrics,
            config=config or {},
        )
