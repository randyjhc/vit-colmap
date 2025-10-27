#!/usr/bin/env python3
"""Aggregate and analyze metrics from multiple pipeline runs.

This script loads all metrics JSON files from a results directory,
aggregates them into comparison tables, and generates summary statistics.
"""

import argparse
import logging
from pathlib import Path
from typing import List
import pandas as pd

from vit_colmap.utils.metrics import MetricsResult
from vit_colmap.utils.export import MetricsExporter
from vit_colmap.utils.plot_metrics import MetricsPlotter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_all_results(results_dir: Path) -> List[MetricsResult]:
    """Load all metrics results from a directory.

    Args:
        results_dir: Directory containing results

    Returns:
        List of MetricsResult objects
    """
    metrics_list = MetricsExporter.load_all_metrics(results_dir)
    logger.info(f"Loaded {len(metrics_list)} results from {results_dir}")
    return metrics_list


def create_comparison_dataframe(metrics_list: List[MetricsResult]) -> pd.DataFrame:
    """Create a pandas DataFrame for easy comparison.

    Args:
        metrics_list: List of MetricsResult objects

    Returns:
        DataFrame with comparison data
    """
    rows = []

    for metrics in metrics_list:
        row = {
            "dataset": metrics.dataset,
            "scene": metrics.scene,
            "extractor": metrics.extractor_type,
            "timestamp": metrics.timestamp,
            # Features
            "total_images": metrics.features.total_images,
            "total_keypoints": metrics.features.total_keypoints,
            "avg_kpts_per_img": metrics.features.avg_keypoints_per_image,
            # Matching
            "matched_pairs": metrics.matching.matched_pairs,
            "match_rate": metrics.matching.match_rate,
            "avg_raw_matches": metrics.matching.avg_raw_matches,
            "avg_inliers": metrics.matching.avg_inlier_matches,
            "inlier_ratio": metrics.matching.inlier_ratio,
        }

        # Add reconstruction metrics if available
        if metrics.reconstruction:
            row.update(
                {
                    "num_reconstructions": metrics.reconstruction.num_reconstructions,
                    "registered_images": metrics.reconstruction.registered_images,
                    "registration_rate": metrics.reconstruction.registration_rate,
                    "total_3d_points": metrics.reconstruction.total_3d_points,
                    "avg_track_length": metrics.reconstruction.avg_track_length,
                    "avg_reproj_error": metrics.reconstruction.avg_reprojection_error,
                }
            )
        else:
            row.update(
                {
                    "num_reconstructions": 0,
                    "registered_images": 0,
                    "registration_rate": 0.0,
                    "total_3d_points": 0,
                    "avg_track_length": 0.0,
                    "avg_reproj_error": 0.0,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def generate_comparison_report(
    df: pd.DataFrame, output_path: Path, dataset: str
) -> None:
    """Generate a comparison report for a specific dataset.

    Args:
        df: DataFrame with results
        output_path: Path to save report
        dataset: Dataset name
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# Metrics Comparison Report: {dataset}\n\n")

        # Group by scene and extractor
        scenes = df["scene"].unique()

        for scene in sorted(scenes):
            scene_df = df[df["scene"] == scene]

            f.write(f"## Scene: {scene}\n\n")

            # Create comparison table
            comparison = scene_df.pivot_table(
                index="extractor",
                values=[
                    "total_keypoints",
                    "avg_kpts_per_img",
                    "matched_pairs",
                    "avg_inliers",
                    "inlier_ratio",
                    "registered_images",
                    "registration_rate",
                    "total_3d_points",
                ],
                aggfunc="first",
            )

            f.write("```\n")
            f.write(comparison.to_string())
            f.write("\n```\n\n")

            # Calculate relative performance
            if "sift" in comparison.index and "vit" in comparison.index:
                f.write("### Relative Performance (ViT vs SIFT)\n\n")

                sift_row = comparison.loc["sift"]
                vit_row = comparison.loc["vit"]

                metrics_to_compare = [
                    ("total_keypoints", "Total Keypoints"),
                    ("avg_kpts_per_img", "Avg Keypoints/Image"),
                    ("matched_pairs", "Matched Pairs"),
                    ("avg_inliers", "Avg Inliers"),
                    ("inlier_ratio", "Inlier Ratio"),
                    ("registered_images", "Registered Images"),
                    ("registration_rate", "Registration Rate (%)"),
                    ("total_3d_points", "Total 3D Points"),
                ]

                for metric_key, metric_name in metrics_to_compare:
                    if metric_key in sift_row.index and sift_row[metric_key] != 0:
                        ratio = (vit_row[metric_key] / sift_row[metric_key] - 1) * 100
                        sign = "+" if ratio >= 0 else ""
                        f.write(
                            f"- **{metric_name}**: {sign}{ratio:.1f}% (ViT: {vit_row[metric_key]:.2f}, SIFT: {sift_row[metric_key]:.2f})\n"
                        )

                f.write("\n")

        # Summary statistics across all scenes
        f.write("## Summary Statistics Across All Scenes\n\n")

        summary = df.groupby("extractor").agg(
            {
                "total_keypoints": ["mean", "std"],
                "avg_inliers": ["mean", "std"],
                "inlier_ratio": ["mean", "std"],
                "registered_images": ["mean", "std"],
                "registration_rate": ["mean", "std"],
                "total_3d_points": ["mean", "std"],
            }
        )

        f.write("```\n")
        f.write(summary.to_string())
        f.write("\n```\n\n")

    logger.info(f"Generated comparison report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze metrics from multiple pipeline runs"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/results"),
        help="Directory containing results (default: data/results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for aggregated results (default: same as results-dir)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by specific dataset (e.g., DTU)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output if args.output else results_dir

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    # Load all results
    all_metrics = load_all_results(results_dir)

    if not all_metrics:
        logger.warning("No metrics found!")
        return

    # Filter by dataset if specified
    if args.dataset:
        all_metrics = [m for m in all_metrics if m.dataset == args.dataset]
        logger.info(
            f"Filtered to {len(all_metrics)} results for dataset: {args.dataset}"
        )

    # Create DataFrame
    df = create_comparison_dataframe(all_metrics)

    # Group by dataset
    datasets = df["dataset"].unique()

    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset]

        logger.info(f"\nProcessing dataset: {dataset}")
        logger.info(f"  Found {len(dataset_df)} results")

        # Export CSV
        if args.format in ["csv", "both"]:
            csv_path = output_dir / dataset / "aggregated_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_df.to_csv(csv_path, index=False)
            logger.info(f"  Saved CSV: {csv_path}")

        # Generate markdown report
        if args.format in ["markdown", "both"]:
            report_path = output_dir / dataset / "comparison_report.md"
            generate_comparison_report(dataset_df, report_path, dataset)

        # Generate summary plot across scans
        dataset_results_dir = results_dir / dataset
        if dataset_results_dir.exists():
            plot_path = output_dir / dataset / "scans_summary.png"
            plotter = MetricsPlotter(dataset_results_dir, enable_cache=True)
            generated_plot = plotter.plot_multiple_scans(
                scans=sorted(dataset_df["scene"].unique()),
                output_path=plot_path,
            )
            if generated_plot:
                logger.info(f"  Saved scans summary plot: {generated_plot}")
            else:
                logger.warning(
                    "  Could not generate scans summary plot for dataset %s (missing data)",
                    dataset,
                )
        else:
            logger.warning(
                "  Results directory for dataset %s not found at %s",
                dataset,
                dataset_results_dir,
            )

    logger.info("\nAggregation complete!")


if __name__ == "__main__":
    main()
