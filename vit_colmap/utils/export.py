"""Export functionality for metrics results."""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional

from vit_colmap.utils.metrics import MetricsResult

logger = logging.getLogger(__name__)


class MetricsExporter:
    """Handles exporting metrics to various formats."""

    @staticmethod
    def export_json(
        metrics: MetricsResult,
        output_path: Path,
        indent: int = 2,
        overwrite: bool = True,
    ) -> None:
        """Export metrics to JSON format.

        Args:
            metrics: MetricsResult object to export
            output_path: Path to output JSON file
            indent: Indentation level for pretty printing
            overwrite: If False, will skip if file already exists
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite and output_path.exists():
            logger.warning(f"File already exists, skipping: {output_path}")
            return

        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=indent)

        logger.info(f"Exported metrics to: {output_path}")

    @staticmethod
    def load_json(input_path: Path) -> MetricsResult:
        """Load metrics from JSON file.

        Args:
            input_path: Path to JSON file

        Returns:
            MetricsResult object
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        return MetricsResult.from_dict(data)

    @staticmethod
    def export_csv_row(
        metrics: MetricsResult,
        output_path: Path,
        append: bool = True,
    ) -> None:
        """Export metrics as a row in a CSV file.

        Args:
            metrics: MetricsResult object to export
            output_path: Path to output CSV file
            append: If True, append to existing file; otherwise overwrite
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV header and extract values
        fieldnames = [
            # Metadata
            "dataset",
            "scene",
            "extractor_type",
            "timestamp",
            # Features
            "total_images",
            "total_keypoints",
            "avg_keypoints_per_image",
            "min_keypoints",
            "max_keypoints",
            "median_keypoints",
            # Matching
            "total_image_pairs",
            "matched_pairs",
            "verified_pairs",
            "match_rate",
            "total_raw_matches",
            "avg_raw_matches",
            "median_raw_matches",
            "total_inlier_matches",
            "avg_inlier_matches",
            "median_inlier_matches",
            "inlier_ratio",
            # Reconstruction
            "num_reconstructions",
            "registered_images",
            "registration_rate",
            "total_3d_points",
            "avg_track_length",
            "avg_reprojection_error",
        ]

        # Flatten metrics to row
        row = {
            # Metadata
            "dataset": metrics.dataset,
            "scene": metrics.scene,
            "extractor_type": metrics.extractor_type,
            "timestamp": metrics.timestamp,
            # Features
            "total_images": metrics.features.total_images,
            "total_keypoints": metrics.features.total_keypoints,
            "avg_keypoints_per_image": f"{metrics.features.avg_keypoints_per_image:.2f}",
            "min_keypoints": metrics.features.min_keypoints,
            "max_keypoints": metrics.features.max_keypoints,
            "median_keypoints": f"{metrics.features.median_keypoints:.2f}",
            # Matching
            "total_image_pairs": metrics.matching.total_image_pairs,
            "matched_pairs": metrics.matching.matched_pairs,
            "verified_pairs": metrics.matching.verified_pairs,
            "match_rate": f"{metrics.matching.match_rate:.2f}",
            "total_raw_matches": metrics.matching.total_raw_matches,
            "avg_raw_matches": f"{metrics.matching.avg_raw_matches:.2f}",
            "median_raw_matches": f"{metrics.matching.median_raw_matches:.2f}",
            "total_inlier_matches": metrics.matching.total_inlier_matches,
            "avg_inlier_matches": f"{metrics.matching.avg_inlier_matches:.2f}",
            "median_inlier_matches": f"{metrics.matching.median_inlier_matches:.2f}",
            "inlier_ratio": f"{metrics.matching.inlier_ratio:.4f}",
        }

        # Add reconstruction metrics if available
        if metrics.reconstruction:
            row.update(
                {
                    "num_reconstructions": metrics.reconstruction.num_reconstructions,
                    "registered_images": metrics.reconstruction.registered_images,
                    "registration_rate": f"{metrics.reconstruction.registration_rate:.2f}",
                    "total_3d_points": metrics.reconstruction.total_3d_points,
                    "avg_track_length": f"{metrics.reconstruction.avg_track_length:.2f}",
                    "avg_reprojection_error": f"{metrics.reconstruction.avg_reprojection_error:.4f}",
                }
            )
        else:
            row.update(
                {
                    "num_reconstructions": 0,
                    "registered_images": 0,
                    "registration_rate": "0.00",
                    "total_3d_points": 0,
                    "avg_track_length": "0.00",
                    "avg_reprojection_error": "0.0000",
                }
            )

        # Determine if we need to write header
        write_header = not output_path.exists() or not append

        mode = "a" if append else "w"
        with open(output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"Exported metrics row to: {output_path}")

    @staticmethod
    def create_comparison_table(
        metrics_list: List[MetricsResult],
        output_path: Path,
    ) -> None:
        """Create a comparison table from multiple metrics results.

        Args:
            metrics_list: List of MetricsResult objects to compare
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not metrics_list:
            logger.warning("No metrics to export")
            return

        # Use export_csv_row for each metric, starting fresh
        for i, metrics in enumerate(metrics_list):
            append = i > 0  # First one creates file, rest append
            MetricsExporter.export_csv_row(metrics, output_path, append=append)

        logger.info(
            f"Created comparison table with {len(metrics_list)} entries: {output_path}"
        )

    @staticmethod
    def load_all_metrics(
        results_dir: Path, pattern: str = "**/*.json"
    ) -> List[MetricsResult]:
        """Load all metrics JSON files from a directory.

        Args:
            results_dir: Directory containing results
            pattern: Glob pattern to match JSON files

        Returns:
            List of MetricsResult objects
        """
        metrics_list = []

        for json_file in results_dir.glob(pattern):
            try:
                metrics = MetricsExporter.load_json(json_file)
                metrics_list.append(metrics)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        logger.info(f"Loaded {len(metrics_list)} metrics from {results_dir}")
        return metrics_list

    @staticmethod
    def aggregate_by_dataset(
        results_dir: Path,
        output_path: Optional[Path] = None,
    ) -> Dict[str, List[MetricsResult]]:
        """Aggregate all metrics by dataset.

        Args:
            results_dir: Directory containing results
            output_path: Optional path to save aggregated CSV summary

        Returns:
            Dictionary mapping dataset name to list of metrics
        """
        all_metrics = MetricsExporter.load_all_metrics(results_dir)

        # Group by dataset
        by_dataset: Dict[str, List[MetricsResult]] = {}
        for metrics in all_metrics:
            dataset = metrics.dataset
            if dataset not in by_dataset:
                by_dataset[dataset] = []
            by_dataset[dataset].append(metrics)

        # Optionally export summary
        if output_path and all_metrics:
            MetricsExporter.create_comparison_table(all_metrics, output_path)

        return by_dataset


def export_metrics(
    metrics: MetricsResult,
    base_dir: Path,
    formats: List[str] = ["json", "csv"],
) -> None:
    """Convenience function to export metrics in multiple formats.

    Organizes outputs as: {base_dir}/{dataset}/{scene}/{extractor_type}.{format}

    Args:
        metrics: MetricsResult to export
        base_dir: Base directory for results (e.g., data/results)
        formats: List of formats to export ("json", "csv")
    """
    scene_dir = base_dir / metrics.dataset / metrics.scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    if "json" in formats:
        json_path = scene_dir / f"{metrics.extractor_type}.json"
        MetricsExporter.export_json(metrics, json_path)

    if "csv" in formats:
        # Append to dataset-level summary CSV
        csv_path = base_dir / metrics.dataset / "summary.csv"
        MetricsExporter.export_csv_row(metrics, csv_path, append=True)

    logger.info(f"Metrics exported to {scene_dir}")
