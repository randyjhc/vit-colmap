#!/usr/bin/env python3
"""Example: Extract and compare metrics from SIFT and ViT runs.

This script demonstrates how to:
1. Load metrics from completed pipeline runs
2. Compare metrics between different extractors
3. Generate custom analysis and visualizations
"""

import argparse
from pathlib import Path
from vit_colmap.utils.export import MetricsExporter
from vit_colmap.utils.plot_metrics import MetricsPlotter

# Centralized filename configuration - single source of truth
SIFT_FILENAME = "sift.json"
VIT_FILENAME = "trainable_vit.json"


def main():
    """Load and compare metrics from SIFT and ViT runs."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compare SIFT and ViT metrics for a DTU scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_metrics.py scan1
  python scripts/compare_metrics.py scan2
  python scripts/compare_metrics.py --dataset DTU --scan scan1
        """,
    )
    parser.add_argument(
        "scan_name",
        nargs="?",
        default="scan1",
        help="Name of the scan to compare (default: scan1)",
    )
    parser.add_argument(
        "--dataset",
        default="DTU",
        help="Dataset name (default: DTU)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory (default: data/results/{dataset})",
    )

    args = parser.parse_args()

    # Define paths
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = Path("data/results") / args.dataset

    scan_name = args.scan_name

    sift_path = results_dir / scan_name / SIFT_FILENAME
    vit_path = results_dir / scan_name / VIT_FILENAME

    # Check if files exist
    if not sift_path.exists():
        print(f"SIFT results not found: {sift_path}")
        print("Please run: ./scripts/run_DTU_sift.sh scan1")
        return

    if not vit_path.exists():
        print(f"ViT results not found: {vit_path}")
        print("Please run: ./scripts/run_DTU_vit.sh scan1")
        return

    # Load metrics
    print(f"Loading metrics for {scan_name}...")
    sift_metrics = MetricsExporter.load_json(sift_path)
    vit_metrics = MetricsExporter.load_json(vit_path)

    # Print comparison
    print("\n" + "=" * 70)
    print(f"METRICS COMPARISON: {scan_name}")
    print("=" * 70)

    # Feature extraction
    print("\n### Feature Extraction")
    print(f"{'Metric':<30} {'SIFT':>15} {'ViT':>15} {'Diff':>10}")
    print("-" * 70)

    metrics_to_compare = [
        ("Total keypoints", "total_keypoints"),
        ("Avg keypoints/image", "avg_keypoints_per_image"),
        ("Min keypoints", "min_keypoints"),
        ("Max keypoints", "max_keypoints"),
    ]

    for label, attr in metrics_to_compare:
        sift_val = getattr(sift_metrics.features, attr)
        vit_val = getattr(vit_metrics.features, attr)
        diff = vit_val - sift_val
        diff_pct = (diff / sift_val * 100) if sift_val != 0 else 0
        print(f"{label:<30} {sift_val:>15.2f} {vit_val:>15.2f} {diff_pct:>9.1f}%")

    # Matching
    print("\n### Feature Matching")
    print(f"{'Metric':<30} {'SIFT':>15} {'ViT':>15} {'Diff':>10}")
    print("-" * 70)

    matching_metrics = [
        ("Processed image pairs", "matched_pairs"),
        ("Total raw matches", "total_raw_matches"),
        ("Total inlier matches", "total_inlier_matches"),
        ("Inlier ratio", "inlier_ratio"),
    ]

    for label, attr in matching_metrics:
        sift_val = getattr(sift_metrics.matching, attr)
        vit_val = getattr(vit_metrics.matching, attr)
        diff = vit_val - sift_val
        diff_pct = (diff / sift_val * 100) if sift_val != 0 else 0
        print(f"{label:<30} {sift_val:>15.2f} {vit_val:>15.2f} {diff_pct:>9.1f}%")

    # Reconstruction
    if sift_metrics.reconstruction and vit_metrics.reconstruction:
        print("\n### 3D Reconstruction")
        print(f"{'Metric':<30} {'SIFT':>15} {'ViT':>15} {'Diff':>10}")
        print("-" * 70)

        recon_metrics = [
            ("Registered images", "registered_images"),
            ("Registration rate (%)", "registration_rate"),
            ("Total 3D points", "total_3d_points"),
            ("Avg track length", "avg_track_length"),
            ("Avg reproj error", "avg_reprojection_error"),
        ]

        for label, attr in recon_metrics:
            sift_val = getattr(sift_metrics.reconstruction, attr)
            vit_val = getattr(vit_metrics.reconstruction, attr)
            diff = vit_val - sift_val
            diff_pct = (diff / sift_val * 100) if sift_val != 0 else 0
            print(f"{label:<30} {sift_val:>15.2f} {vit_val:>15.2f} {diff_pct:>9.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Determine winner for key metrics
    winners = {
        "Features": "ViT"
        if vit_metrics.features.total_keypoints > sift_metrics.features.total_keypoints
        else "SIFT",
        "Inlier ratio": "ViT"
        if vit_metrics.matching.inlier_ratio > sift_metrics.matching.inlier_ratio
        else "SIFT",
    }

    if sift_metrics.reconstruction and vit_metrics.reconstruction:
        winners["3D points"] = (
            "ViT"
            if vit_metrics.reconstruction.total_3d_points
            > sift_metrics.reconstruction.total_3d_points
            else "SIFT"
        )
        winners["Registration rate"] = (
            "ViT"
            if vit_metrics.reconstruction.registration_rate
            > sift_metrics.reconstruction.registration_rate
            else "SIFT"
        )

    for metric, winner in winners.items():
        symbol = "✓" if winner == "ViT" else "○"
        print(f"{metric:<30} Winner: {winner} {symbol}")

    print("\n")

    # Generate comparison plot
    plot_output = results_dir / scan_name / "comparison_plot.png"
    plotter = MetricsPlotter(
        results_dir,
        enable_cache=False,
        sift_filename=SIFT_FILENAME,
        vit_filename=VIT_FILENAME,
    )
    generated = plotter.plot_single_scan(scan_name, output_path=plot_output)
    if generated:
        print(f"Saved comparison plot to: {generated}")
    else:
        print("Comparison plot was not generated (missing data).")


if __name__ == "__main__":
    main()
