"""
Multi-scene layer invariance analysis across all DTU scans.

This script sweeps through all DTU scan directories, randomly samples one "_3" image
from each scan, and runs layer-wise invariance analysis to avoid scene bias.

Usage:
    python experiments/exp_multi_scene_analysis.py \
        --dtu-path data/raw/DTU/Cleaned \
        --num-points 100 \
        --model dinov2_vitb14 \
        --output-dir outputs/multi_scene_dtu_analysis
"""

import argparse
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import random

# Import existing functions from exp_layer_invariance
from exp_layer_invariance import (
    LayerFeatureExtractor,
    test_scale_invariance,
    test_rotation_invariance,
    test_illumination_invariance,
    test_viewpoint_invariance,
)


# ============================================================================
# DTU Dataset Sampling
# ============================================================================


def sample_images_from_dtu(dtu_path: Path, seed: int = 42) -> List[Path]:
    """
    Sample one "_3" image randomly from each DTU scan directory.

    Args:
        dtu_path: Path to DTU Cleaned directory
        seed: Random seed for reproducibility

    Returns:
        List of image paths (one per scan)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find all scan directories
    scan_dirs = sorted(
        [d for d in dtu_path.iterdir() if d.is_dir() and d.name.startswith("scan")]
    )

    print(f"\nFound {len(scan_dirs)} scan directories")

    sampled_images = []

    for scan_dir in scan_dirs:
        # Find all images with "_3" pattern
        images_with_3 = list(scan_dir.glob("*_3_r5000.png"))

        if not images_with_3:
            print(f"Warning: No '_3' images found in {scan_dir.name}, skipping")
            continue

        # Randomly select one
        selected_image = random.choice(images_with_3)
        sampled_images.append(selected_image)

    print(f"Sampled {len(sampled_images)} images (one per scan)\n")

    return sampled_images


def save_sampled_images_list(image_paths: List[Path], output_file: Path):
    """Save list of sampled images to a text file."""
    with open(output_file, "w") as f:
        f.write("Sampled images for multi-scene analysis:\n")
        f.write("=" * 80 + "\n\n")
        for idx, img_path in enumerate(image_paths, start=1):
            f.write(f"{idx:3d}. {img_path}\n")

    print(f"Saved sampled images list to: {output_file}\n")


# ============================================================================
# Per-Image Analysis
# ============================================================================


def run_analysis_on_image(
    extractor: LayerFeatureExtractor, image_path: Path, pixel_coords: np.ndarray
) -> Dict[str, Dict]:
    """
    Run invariance analysis on a single image.

    Args:
        extractor: LayerFeatureExtractor instance
        image_path: Path to the image
        pixel_coords: (N, 2) array of pixel coordinates to test

    Returns:
        Dictionary with results for all four tests
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Run all four tests (reusing existing functions)
    results = {}

    results["scale"] = test_scale_invariance(
        extractor, image, pixel_coords, return_per_pixel=False
    )

    results["rotation"] = test_rotation_invariance(
        extractor, image, pixel_coords, return_per_pixel=False
    )

    results["illumination"] = test_illumination_invariance(
        extractor, image, pixel_coords, return_per_pixel=False
    )

    results["viewpoint"] = test_viewpoint_invariance(
        extractor, image, pixel_coords, return_per_pixel=False
    )

    return results


# ============================================================================
# Results Aggregation
# ============================================================================


def aggregate_results(all_results: List[Dict[str, Dict]]) -> Dict[str, Dict]:
    """
    Aggregate results across all images.

    Args:
        all_results: List of results dicts (one per image)

    Returns:
        Aggregated results with mean and std across images
    """
    # Get layer names from first result
    first_result = all_results[0]
    test_names = list(first_result.keys())
    layer_names = list(first_result[test_names[0]].keys())

    aggregated: Dict[str, Dict[str, Dict]] = {}

    for test_name in test_names:
        aggregated[test_name] = {}

        for layer_name in layer_names:
            # Collect mean_cosine_sim values across all images
            cosine_sims = [
                result[test_name][layer_name]["mean_cosine_sim"]
                for result in all_results
            ]

            # Collect std_cosine_sim values
            std_sims = [
                result[test_name][layer_name]["std_cosine_sim"]
                for result in all_results
            ]

            # Compute statistics across images
            aggregated[test_name][layer_name] = {
                "mean_across_scenes": float(np.mean(cosine_sims)),
                "std_across_scenes": float(np.std(cosine_sims)),
                "min_across_scenes": float(np.min(cosine_sims)),
                "max_across_scenes": float(np.max(cosine_sims)),
                "median_across_scenes": float(np.median(cosine_sims)),
                "mean_of_stds": float(
                    np.mean(std_sims)
                ),  # Average within-image variance
                "all_scene_scores": cosine_sims,  # Keep for detailed analysis
            }

    return aggregated


# ============================================================================
# Visualization
# ============================================================================


def plot_aggregated_results(
    aggregated_results: Dict[str, Dict], output_dir: Path, num_images: int
):
    """
    Create visualization plots for aggregated multi-scene results.

    Args:
        aggregated_results: Aggregated results from aggregate_results()
        output_dir: Directory to save plots
        num_images: Number of images analyzed
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract layer names
    first_test = list(aggregated_results.values())[0]
    layer_names = sorted(first_test.keys(), key=lambda x: int(x.split("_")[1]))
    layer_indices = [int(name.split("_")[1]) for name in layer_names]

    test_names = ["scale", "rotation", "illumination", "viewpoint"]
    test_titles = [
        "Scale Invariance",
        "Rotation Invariance",
        "Illumination Invariance",
        "Viewpoint Invariance",
    ]

    # Plot 1: Summary with error bars showing scene variance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Multi-Scene Layer-wise Invariance Analysis ({num_images} DTU Scenes)",
        fontsize=16,
        fontweight="bold",
    )

    for idx, (test_name, test_title) in enumerate(zip(test_names, test_titles)):
        ax = axes[idx // 2, idx % 2]

        if test_name in aggregated_results:
            results = aggregated_results[test_name]

            # Extract metrics
            mean_sims = [results[layer]["mean_across_scenes"] for layer in layer_names]
            std_sims = [results[layer]["std_across_scenes"] for layer in layer_names]

            # Plot with error bars (std shows variance across scenes)
            ax.errorbar(
                layer_indices,
                mean_sims,
                yerr=std_sims,
                marker="o",
                capsize=5,
                linewidth=2,
                markersize=8,
                label="Mean ± Std (across scenes)",
            )
            ax.set_xlabel("Layer Index", fontsize=12)
            ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
            ax.set_title(test_title, fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

            # Highlight best layer
            best_idx = np.argmax(mean_sims)
            ax.axvline(
                layer_indices[best_idx],
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Best: Layer {layer_indices[best_idx]}",
            )
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "aggregated_summary.png", dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_dir / 'aggregated_summary.png'}")
    plt.close()

    # Plot 2: Heatmap showing variance across scenes
    fig, ax = plt.subplots(figsize=(12, 6))

    variance_data_list: List[List[float]] = []
    for test_name in test_names:
        if test_name in aggregated_results:
            results = aggregated_results[test_name]
            row = [results[layer]["std_across_scenes"] for layer in layer_names]
            variance_data_list.append(row)

    variance_data = np.array(variance_data_list)

    sns.heatmap(
        variance_data,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=layer_indices,
        yticklabels=[
            t.replace("_", " ").title() for t in test_names if t in aggregated_results
        ],
        cbar_kws={"label": "Std Dev Across Scenes"},
        ax=ax,
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Invariance Test", fontsize=12)
    ax.set_title(
        f"Scene Variance Heatmap ({num_images} DTU Scenes)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_dir / "scene_variance_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_dir / 'scene_variance_heatmap.png'}")
    plt.close()


def export_aggregated_results(
    aggregated_results: Dict[str, Dict],
    output_dir: Path,
    num_images: int,
    model_name: str,
):
    """Export aggregated results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    metadata = {
        "num_scenes": num_images,
        "model_name": model_name,
        "description": "Aggregated layer-wise invariance across DTU scenes",
    }

    # Remove 'all_scene_scores' for JSON export (too large)
    json_results: Dict[str, Dict[str, Dict]] = {}
    for test_name, test_results in aggregated_results.items():
        json_results[test_name] = {}
        for layer_name, metrics in test_results.items():
            json_results[test_name][layer_name] = {
                k: v for k, v in metrics.items() if k != "all_scene_scores"
            }

    # Export to JSON
    output_data = {"metadata": metadata, "results": json_results}

    json_path = output_dir / "aggregated_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results: {json_path}")

    # Export to CSV
    import csv

    csv_path = output_dir / "aggregated_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Test",
                "Layer",
                "Mean Across Scenes",
                "Std Across Scenes",
                "Min",
                "Max",
                "Median",
                "Mean of Stds",
            ]
        )

        for test_name, test_results in json_results.items():
            for layer_name, metrics in test_results.items():
                writer.writerow(
                    [
                        test_name,
                        layer_name,
                        f"{metrics['mean_across_scenes']:.4f}",
                        f"{metrics['std_across_scenes']:.4f}",
                        f"{metrics['min_across_scenes']:.4f}",
                        f"{metrics['max_across_scenes']:.4f}",
                        f"{metrics['median_across_scenes']:.4f}",
                        f"{metrics['mean_of_stds']:.4f}",
                    ]
                )

    print(f"Saved results: {csv_path}")


def generate_summary_report(
    aggregated_results: Dict[str, Dict],
    output_dir: Path,
    num_images: int,
    model_name: str,
):
    """Generate a text summary report."""
    report_path = output_dir / "summary_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Multi-Scene Layer-wise Invariance Analysis Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model: {model_name}\n")
        f.write(f"Number of DTU scenes analyzed: {num_images}\n")
        f.write("Images sampled: One '_3' image randomly selected from each scan\n\n")

        f.write("=" * 80 + "\n")
        f.write("Best Layers for Each Invariance Property:\n")
        f.write("=" * 80 + "\n\n")

        for test_name, results in aggregated_results.items():
            layer_scores = [
                (layer, metrics["mean_across_scenes"])
                for layer, metrics in results.items()
            ]
            best_layer, best_score = max(layer_scores, key=lambda x: x[1])
            best_std = results[best_layer]["std_across_scenes"]

            f.write(f"{test_name.capitalize()}:\n")
            f.write(f"  Best Layer: {best_layer}\n")
            f.write(f"  Mean Cosine Similarity: {best_score:.4f}\n")
            f.write(f"  Std Across Scenes: {best_std:.4f}\n")
            f.write(
                f"  Range: [{results[best_layer]['min_across_scenes']:.4f}, "
                f"{results[best_layer]['max_across_scenes']:.4f}]\n\n"
            )

        f.write("=" * 80 + "\n")
        f.write("Interpretation:\n")
        f.write("=" * 80 + "\n\n")
        f.write("- Mean Across Scenes: Average invariance across all DTU scenes\n")
        f.write("- Std Across Scenes: Variance in invariance across different scenes\n")
        f.write("  (Lower std = more consistent performance across scenes)\n")
        f.write(
            "- Mean of Stds: Average within-image variance across transformations\n\n"
        )

    print(f"Saved summary report: {report_path}")


# ============================================================================
# Main Experiment
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scene layer invariance analysis across all DTU scans"
    )
    parser.add_argument(
        "--dtu-path",
        type=str,
        default="../data/raw/DTU/Cleaned",
        help="Path to DTU Cleaned directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov2_vitb14",
        help="Model name (e.g., dinov2_vitb14, dinov2_vitl14)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=100,
        help="Number of random pixel locations to test per image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/multi_scene_dtu_analysis",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("Multi-Scene DTU Layer Invariance Analysis")
    print(f"{'='*80}")
    print(f"DTU Path: {args.dtu_path}")
    print(f"Model: {args.model}")
    print(f"Points per image: {args.num_points}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    # Setup
    dtu_path = Path(args.dtu_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Sample images
    print("Step 1: Sampling images from DTU scans...")
    sampled_images = sample_images_from_dtu(dtu_path, seed=args.seed)

    if not sampled_images:
        print("Error: No images found!")
        return

    save_sampled_images_list(sampled_images, output_dir / "sampled_images.txt")

    # Step 2: Initialize feature extractor (once for all images)
    print("Step 2: Initializing feature extractor...")
    extractor = LayerFeatureExtractor(model_name=args.model, device=args.device)

    # Step 3: Run analysis on all images
    print(f"\nStep 3: Running invariance analysis on {len(sampled_images)} images...")
    all_results = []

    for idx, image_path in enumerate(tqdm(sampled_images, desc="Processing images")):
        try:
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"\nWarning: Failed to load {image_path}, skipping")
                continue

            h, w = image.shape[:2]

            # Generate random pixel coordinates for this image
            np.random.seed(args.seed + idx)  # Different seed per image
            pixel_coords = np.random.rand(args.num_points, 2)
            pixel_coords[:, 0] *= w
            pixel_coords[:, 1] *= h
            pixel_coords = pixel_coords.astype(np.float32)

            # Run analysis
            results = run_analysis_on_image(extractor, image_path, pixel_coords)
            all_results.append(results)

        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            continue

    if not all_results:
        print("Error: No results generated!")
        return

    print(f"\n✓ Successfully analyzed {len(all_results)} images")

    # Step 4: Aggregate results
    print("\nStep 4: Aggregating results across scenes...")
    aggregated_results = aggregate_results(all_results)

    # Step 5: Generate outputs
    print("\nStep 5: Generating outputs...")
    aggregated_dir = output_dir / "aggregated_results"

    plot_aggregated_results(aggregated_results, aggregated_dir, len(all_results))
    export_aggregated_results(
        aggregated_results, aggregated_dir, len(all_results), args.model
    )
    generate_summary_report(
        aggregated_results, output_dir, len(all_results), args.model
    )

    # Print summary
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Number of scenes analyzed: {len(all_results)}")
    print("\nBest layers (averaged across all scenes):")

    for test_name, results in aggregated_results.items():
        layer_scores = [
            (layer, metrics["mean_across_scenes"]) for layer, metrics in results.items()
        ]
        best_layer, best_score = max(layer_scores, key=lambda x: x[1])
        std = results[best_layer]["std_across_scenes"]
        print(
            f"  {test_name.capitalize()}: {best_layer} "
            f"(mean={best_score:.4f}, std={std:.4f})"
        )

    # Cleanup
    extractor.cleanup()


if __name__ == "__main__":
    main()
