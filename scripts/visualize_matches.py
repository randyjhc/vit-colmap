#!/usr/bin/env python
"""
Visualize matched keypoints between two images with color-coded inlier/outlier distinction.

This script:
1. Reads keypoints and matches from a COLMAP database
2. Identifies inliers (geometrically verified) vs outliers
3. Visualizes matches with color-coded lines:
   - Green lines for inlier matches
   - Red lines for outlier matches
4. Displays comprehensive match statistics
5. Optionally shows all keypoints with text labels displaying confidence scores

Usage:
    # Basic match visualization
    python scripts/visualize_matches.py --database database.db --image-dir images/ --image1 IMG_001.jpg --image2 IMG_002.jpg

    # Show all keypoints with score labels (top 100 keypoints per image)
    python scripts/visualize_matches.py --database database.db --image-dir images/ --image1 IMG_001.jpg --image2 IMG_002.jpg --show-all-keypoints --show-scores

    # Show all keypoints with custom number of score labels
    python scripts/visualize_matches.py --database database.db --image-dir images/ --image1 IMG_001.jpg --image2 IMG_002.jpg --show-all-keypoints --show-scores --max-score-labels 50

    # Other options
    python scripts/visualize_matches.py --database database.db --image-dir images/ --image1 0 --image2 1 --max-matches 100
    python scripts/visualize_matches.py --database database.db --image-dir images/ --image1 IMG_001.jpg --image2 IMG_002.jpg --filter inliers
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pycolmap


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize matched keypoints between two images with inlier/outlier distinction"
    )
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="Path to COLMAP database",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=False,
        help="Directory containing images",
    )
    parser.add_argument(
        "--image1",
        type=str,
        required=False,
        help="First image name or index (0-based)",
    )
    parser.add_argument(
        "--image2",
        type=str,
        required=False,
        help="Second image name or index (0-based)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for saving visualization (default: display only)",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=None,
        help="Maximum number of matches to display (default: show all)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["all", "inliers", "outliers"],
        default="all",
        help="Filter matches to display (default: all)",
    )
    parser.add_argument(
        "--show-all-keypoints",
        action="store_true",
        help="Show all keypoints, not just matched ones",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show keypoint confidence scores as text labels (requires --show-all-keypoints)",
    )
    parser.add_argument(
        "--max-score-labels",
        type=int,
        default=100,
        help="Maximum number of score labels to display per image (default: 100, shows top-scoring keypoints)",
    )
    parser.add_argument(
        "--show-orientation",
        action="store_true",
        help="Show keypoint orientations as arrows",
    )
    parser.add_argument(
        "--orientation-scale",
        type=float,
        default=10.0,
        help="Scale factor for orientation arrow length (default: 10.0)",
    )
    parser.add_argument(
        "--inlier-color",
        type=str,
        default="green",
        help="Color for inlier matches (default: green)",
    )
    parser.add_argument(
        "--outlier-color",
        type=str,
        default="red",
        help="Color for outlier matches (default: red)",
    )
    parser.add_argument(
        "--keypoint-size",
        type=int,
        default=3,
        help="Size of keypoint markers (default: 3)",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=0.5,
        help="Width of match lines (default: 0.5)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling matches (default: 42)",
    )
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="List all images in the database and exit",
    )
    parser.add_argument(
        "--list-matches",
        action="store_true",
        help="List all image pairs with matches and exit",
    )

    return parser.parse_args()


def get_image_by_name_or_index(
    db: pycolmap.Database, image_spec: str, image_dir: Path
) -> Tuple[int, str, Path]:
    """
    Get image ID, name, and path from database by name or index.

    Args:
        db: COLMAP database
        image_spec: Image name or index (as string)
        image_dir: Directory containing images

    Returns:
        Tuple of (image_id, image_name, image_path)

    Raises:
        ValueError: If image not found
    """
    # Get all images from database
    images = db.read_all_images()

    if not images:
        raise ValueError("No images found in database")

    # Handle different pycolmap versions - images can be a dict or list
    if isinstance(images, dict):
        # Dictionary format {image_id: Image}
        images_list = list(images.values())
    else:
        # List format [Image, Image, ...]
        images_list = images

    # Try to parse as index first
    is_index = False
    try:
        idx = int(image_spec)
        is_index = True
    except ValueError:
        # Not an integer, treat as name
        is_index = False

    if is_index:
        # User provided an index
        if idx < 0 or idx >= len(images_list):
            raise ValueError(f"Index {idx} out of range [0, {len(images_list)-1}]")

        image = images_list[idx]
        image_id = image.image_id
        image_name = image.name
    else:
        # User provided a name
        image_name = image_spec
        image_id = None

        # Search for matching name
        for img in images_list:
            if img.name == image_name:
                image_id = img.image_id
                image_name = img.name
                break

        if image_id is None:
            raise ValueError(
                f"Image '{image_name}' not found in database. "
                f"Available images: {[img.name for img in images_list[:10]]}..."
            )

    image_path = image_dir / image_name
    if not image_path.exists():
        raise ValueError(f"Image file not found: {image_path}")

    return image_id, image_name, image_path


def read_matches_and_inliers(
    db: pycolmap.Database, image_id1: int, image_id2: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read raw matches and inlier matches from database.

    Args:
        db: COLMAP database
        image_id1: First image ID
        image_id2: Second image ID

    Returns:
        Tuple of (raw_matches, inlier_matches)
        raw_matches: Nx2 array of match indices
        inlier_matches: Mx2 array of inlier match indices (None if not available)
    """
    # Read raw matches
    raw_matches = db.read_matches(image_id1, image_id2)

    if raw_matches is None or len(raw_matches) == 0:
        return np.zeros((0, 2), dtype=np.uint32), None

    # Try to read two-view geometry for inliers
    try:
        two_view_geom = db.read_two_view_geometry(image_id1, image_id2)
        if two_view_geom is not None and hasattr(two_view_geom, "inlier_matches"):
            inlier_matches = two_view_geom.inlier_matches
            return raw_matches, inlier_matches
    except Exception:
        pass

    # No inlier information available
    return raw_matches, None


def classify_matches(
    raw_matches: np.ndarray, inlier_matches: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify matches into inliers and outliers.

    Args:
        raw_matches: Nx2 array of all matches
        inlier_matches: Mx2 array of inlier matches (or None)

    Returns:
        Tuple of (inlier_indices, outlier_indices)
        If no inlier information, all matches are treated as inliers
    """
    if inlier_matches is None or len(inlier_matches) == 0:
        # No geometric verification data, treat all as inliers
        return np.arange(len(raw_matches)), np.array([], dtype=int)

    # Convert matches to set of tuples for efficient lookup
    inlier_set = set(map(tuple, inlier_matches))

    # Classify each match
    inlier_indices = []
    outlier_indices = []

    for idx, match in enumerate(raw_matches):
        if tuple(match) in inlier_set:
            inlier_indices.append(idx)
        else:
            outlier_indices.append(idx)

    return np.array(inlier_indices), np.array(outlier_indices)


def visualize_matches(
    img1_path: Path,
    img2_path: Path,
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    matches: np.ndarray,
    inlier_mask: np.ndarray,
    image1_name: str,
    image2_name: str,
    show_all_keypoints: bool = False,
    show_scores: bool = False,
    max_score_labels: int = 100,
    show_orientation: bool = False,
    orientation_scale: float = 10.0,
    inlier_color: str = "green",
    outlier_color: str = "red",
    keypoint_size: int = 3,
    line_width: float = 0.5,
    output_path: Optional[Path] = None,
    dpi: int = 150,
) -> None:
    """
    Create visualization of matched keypoints with color-coded lines.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        kpts1: Keypoints in first image (Nx6 array: x, y, scale, orientation, score, unused)
        kpts2: Keypoints in second image (Mx6 array: x, y, scale, orientation, score, unused)
        matches: Match indices (Kx2 array)
        inlier_mask: Boolean mask indicating inliers (K-length array)
        image1_name: Name of first image
        image2_name: Name of second image
        show_all_keypoints: Show all keypoints, not just matched ones
        show_scores: When True with show_all_keypoints, show score labels for top keypoints
        max_score_labels: Maximum number of score labels to display per image
        show_orientation: Show keypoint orientations as arrows
        orientation_scale: Scale factor for orientation arrow length
        inlier_color: Color for inlier matches
        outlier_color: Color for outlier matches
        keypoint_size: Size of keypoint markers
        line_width: Width of match lines
        output_path: Path to save figure (None to display only)
        dpi: DPI for saved figure
    """
    # Load images
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))

    if img1 is None:
        raise ValueError(f"Failed to load image: {img1_path}")
    if img2 is None:
        raise ValueError(f"Failed to load image: {img2_path}")

    # Convert BGR to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Extract only x, y coordinates (keypoints may have additional columns like scale, orientation)
    # Ensure we only use the first 2 columns
    kpts1_xy = kpts1[:, :2]
    kpts2_xy = kpts2[:, :2]

    # Create side-by-side image
    h_max = max(h1, h2)
    canvas = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 : w1 + w2] = img2

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)
    ax.axis("off")

    # Draw all keypoints if requested
    if show_all_keypoints:
        # Draw all keypoints as cyan dots with fixed alpha (standard style)
        ax.plot(
            kpts1_xy[:, 0], kpts1_xy[:, 1], "c.", markersize=keypoint_size, alpha=0.3
        )
        ax.plot(
            kpts2_xy[:, 0] + w1,
            kpts2_xy[:, 1],
            "c.",
            markersize=keypoint_size,
            alpha=0.3,
        )

        # Add score labels for top-N keypoints if requested
        if show_scores and kpts1.shape[1] >= 6 and kpts2.shape[1] >= 6:
            # Extract scores (5th column, index 4) from 6-column format
            # Format: (x, y, scale, orientation, score, unused)
            scores1 = kpts1[:, 4]
            scores2 = kpts2[:, 4]

            # Select top-N keypoints by score for labeling
            n_labels1 = min(max_score_labels, len(scores1))
            n_labels2 = min(max_score_labels, len(scores2))

            top_indices1 = np.argsort(scores1)[-n_labels1:]  # Top N highest scores
            top_indices2 = np.argsort(scores2)[-n_labels2:]

            # Add text annotations for top scoring keypoints in image 1
            for idx in top_indices1:
                score_text = f"{scores1[idx]:.2f}"
                ax.text(
                    kpts1_xy[idx, 0] + 3,  # Offset to the right
                    kpts1_xy[idx, 1] - 3,  # Offset upward
                    score_text,
                    fontsize=6,
                    color="white",
                    weight="bold",
                    ha="left",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="black",
                        alpha=0.6,
                        edgecolor="none",
                    ),
                )

            # Add text annotations for top scoring keypoints in image 2
            for idx in top_indices2:
                score_text = f"{scores2[idx]:.2f}"
                ax.text(
                    kpts2_xy[idx, 0]
                    + w1
                    + 3,  # Offset to the right (account for image shift)
                    kpts2_xy[idx, 1] - 3,  # Offset upward
                    score_text,
                    fontsize=6,
                    color="white",
                    weight="bold",
                    ha="left",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="black",
                        alpha=0.6,
                        edgecolor="none",
                    ),
                )

    # Statistics
    num_inliers = inlier_mask.sum()
    num_outliers = (~inlier_mask).sum()
    total_matches = len(matches)

    # Draw matches with color coding
    for idx, (i1, i2) in enumerate(matches):
        pt1 = kpts1_xy[i1]
        pt2 = kpts2_xy[i2]

        color = inlier_color if inlier_mask[idx] else outlier_color
        alpha = 0.6 if inlier_mask[idx] else 0.3

        ax.plot(
            [pt1[0], pt2[0] + w1],
            [pt1[1], pt2[1]],
            color=color,
            linewidth=line_width,
            alpha=alpha,
        )

    # Draw keypoints for matched points
    matched_kpts1_idx = matches[:, 0]
    matched_kpts2_idx = matches[:, 1]

    matched_kpts1 = kpts1_xy[matched_kpts1_idx]
    matched_kpts2 = kpts2_xy[matched_kpts2_idx]

    ax.plot(
        matched_kpts1[:, 0], matched_kpts1[:, 1], "y.", markersize=keypoint_size + 1
    )
    ax.plot(
        matched_kpts2[:, 0] + w1,
        matched_kpts2[:, 1],
        "y.",
        markersize=keypoint_size + 1,
    )

    # Draw orientation arrows if requested
    if show_orientation and kpts1.shape[1] >= 4 and kpts2.shape[1] >= 4:
        # Extract orientations (4th column, index 3) in radians
        # Keypoint format: (x, y, scale, orientation, score, unused)

        def draw_orientation_arrows(
            kpts, x_offset=0, color="yellow", alpha=0.6, subset_indices=None
        ):
            """Draw orientation arrows for keypoints."""
            if subset_indices is None:
                subset_indices = range(len(kpts))

            for idx in subset_indices:
                x, y = kpts[idx, 0], kpts[idx, 1]
                orientation = kpts[idx, 3]  # Orientation in radians

                # Optionally use scale for arrow length (if available)
                if kpts.shape[1] >= 3:
                    scale = kpts[idx, 2]
                    arrow_length = scale * orientation_scale
                else:
                    arrow_length = orientation_scale

                # Calculate arrow endpoint
                dx = arrow_length * np.cos(orientation)
                dy = arrow_length * np.sin(orientation)

                # Draw arrow
                ax.arrow(
                    x + x_offset,
                    y,  # Start point
                    dx,
                    dy,  # Direction vector
                    head_width=2,
                    head_length=3,
                    fc=color,
                    ec=color,
                    alpha=alpha,
                    linewidth=0.8,
                    length_includes_head=True,
                )

        # Draw orientations for all keypoints if show_all_keypoints is True
        if show_all_keypoints:
            draw_orientation_arrows(kpts1, x_offset=0, color="cyan", alpha=0.3)
            draw_orientation_arrows(kpts2, x_offset=w1, color="cyan", alpha=0.3)

        # Always draw orientations for matched keypoints (more prominent)
        draw_orientation_arrows(
            kpts1,
            x_offset=0,
            color="yellow",
            alpha=0.8,
            subset_indices=matched_kpts1_idx,
        )
        draw_orientation_arrows(
            kpts2,
            x_offset=w1,
            color="yellow",
            alpha=0.8,
            subset_indices=matched_kpts2_idx,
        )

    # Create legend
    legend_elements = [
        mpatches.Patch(color=inlier_color, label=f"Inliers: {num_inliers}"),
        mpatches.Patch(color=outlier_color, label=f"Outliers: {num_outliers}"),
    ]

    # Add title and statistics
    inlier_ratio = (num_inliers / total_matches * 100) if total_matches > 0 else 0

    title = (
        f"Match Visualization: {image1_name} <-> {image2_name}\n"
        f"Keypoints: {len(kpts1)} / {len(kpts2)} | "
        f"Total Matches: {total_matches} | "
        f"Inliers: {num_inliers} ({inlier_ratio:.1f}%) | "
        f"Outliers: {num_outliers}"
    )

    ax.set_title(title, fontsize=12, pad=10)
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    plt.show()


def list_images_in_database(db: pycolmap.Database) -> None:
    """List all images in the database with their IDs and keypoint counts."""
    images = db.read_all_images()

    # Handle different pycolmap versions
    if isinstance(images, dict):
        images_list = list(images.values())
    else:
        images_list = images

    if not images_list:
        print("No images found in database")
        return

    print(f"\nFound {len(images_list)} images in database:")
    print("-" * 80)
    print(f"{'Index':<8} {'Image ID':<12} {'Keypoints':<12} {'Name':<40}")
    print("-" * 80)

    for idx, img in enumerate(images_list):
        try:
            kpts = db.read_keypoints(img.image_id)
            num_kpts: int | str = len(kpts) if kpts is not None else 0
        except Exception:
            num_kpts = "N/A"

        print(f"{idx:<8} {img.image_id:<12} {num_kpts:<12} {img.name:<40}")

    print("-" * 80)


def list_matches_in_database(db: pycolmap.Database) -> None:
    """List all image pairs with matches in the database."""
    images = db.read_all_images()

    # Handle different pycolmap versions
    if isinstance(images, dict):
        images_list = list(images.values())
    else:
        images_list = images

    if not images_list:
        print("No images found in database")
        return

    print(f"\nFound {len(images_list)} images in database")
    print("Checking all pairs for matches...\n")

    pairs_with_matches = []

    # Check all pairs
    for i in range(len(images_list)):
        for j in range(i + 1, len(images_list)):
            img1 = images_list[i]
            img2 = images_list[j]

            # Read matches
            raw_matches, inlier_matches = read_matches_and_inliers(
                db, img1.image_id, img2.image_id
            )

            if len(raw_matches) > 0:
                num_inliers = len(inlier_matches) if inlier_matches is not None else 0
                pairs_with_matches.append(
                    {
                        "idx1": i,
                        "idx2": j,
                        "id1": img1.image_id,
                        "id2": img2.image_id,
                        "name1": img1.name,
                        "name2": img2.name,
                        "raw": len(raw_matches),
                        "inliers": num_inliers,
                    }
                )

    if not pairs_with_matches:
        print("No pairs with matches found!")
        return

    # Sort by number of raw matches (descending)
    pairs_with_matches.sort(key=lambda x: x["raw"], reverse=True)

    print(f"Found {len(pairs_with_matches)} pairs with matches:")
    print("-" * 100)
    print(
        f"{'Indices':<12} {'Image IDs':<15} {'Raw':<8} {'Inliers':<10} {'Image Pair':<50}"
    )
    print("-" * 100)

    for pair in pairs_with_matches:
        indices = f"{pair['idx1']},{pair['idx2']}"
        ids = f"{pair['id1']},{pair['id2']}"
        names = f"{pair['name1']} <-> {pair['name2']}"
        print(
            f"{indices:<12} {ids:<15} {pair['raw']:<8} {pair['inliers']:<10} {names:<50}"
        )

    print("-" * 100)
    print("\nTo visualize a pair, use: --image1 <index1> --image2 <index2>")
    print(
        f"Example: --image1 {pairs_with_matches[0]['idx1']} --image2 {pairs_with_matches[0]['idx2']}"
    )


def main():
    """Main function."""
    args = parse_args()

    # Validate inputs
    if not args.database.exists():
        print(f"Error: Database not found: {args.database}")
        sys.exit(1)

    # Set random seed
    np.random.seed(args.seed)

    print(f"Opening database: {args.database}")

    # Open database and read data
    try:
        # Handle different pycolmap versions
        try:
            db = pycolmap.Database.open(str(args.database))
        except TypeError:
            db = pycolmap.Database()
            db.open(str(args.database))

        # List images if requested
        if args.list_images:
            list_images_in_database(db)
            return

        # List matches if requested
        if args.list_matches:
            list_matches_in_database(db)
            return

        # Validate required arguments for visualization
        if not args.image_dir:
            print("Error: --image-dir is required for visualization")
            sys.exit(1)
        if not args.image1:
            print("Error: --image1 is required for visualization")
            sys.exit(1)
        if not args.image2:
            print("Error: --image2 is required for visualization")
            sys.exit(1)

        # Validate image directory
        if not args.image_dir.exists():
            print(f"Error: Image directory not found: {args.image_dir}")
            sys.exit(1)

        # Get image information
        print("\nResolving images...")
        print(f"  Requested: image1='{args.image1}', image2='{args.image2}'")

        image_id1, image_name1, image_path1 = get_image_by_name_or_index(
            db, args.image1, args.image_dir
        )
        image_id2, image_name2, image_path2 = get_image_by_name_or_index(
            db, args.image2, args.image_dir
        )

        print(f"  Image 1: {image_name1} (ID: {image_id1})")
        print(f"  Image 2: {image_name2} (ID: {image_id2})")

        if image_id1 == image_id2:
            print(
                "\n⚠️  WARNING: Both images have the same ID! This will show self-matches."
            )

        # Read keypoints
        print("\nReading keypoints...")
        kpts1 = db.read_keypoints(image_id1)
        kpts2 = db.read_keypoints(image_id2)

        print(f"  Image 1: {len(kpts1)} keypoints")
        print(f"  Image 2: {len(kpts2)} keypoints")

        # Print orientation statistics if available
        if kpts1.shape[1] >= 4 and kpts2.shape[1] >= 4:
            orientations1 = kpts1[:, 3]
            orientations2 = kpts2[:, 3]
            print("\nOrientation statistics:")
            print(
                f"  Image 1: min={np.min(orientations1):.4f}, max={np.max(orientations1):.4f}, mean={np.mean(orientations1):.4f}, std={np.std(orientations1):.4f}"
            )
            print(
                f"  Image 2: min={np.min(orientations2):.4f}, max={np.max(orientations2):.4f}, mean={np.mean(orientations2):.4f}, std={np.std(orientations2):.4f}"
            )

            # Check if all orientations are the same
            if np.std(orientations1) < 1e-6 and np.std(orientations2) < 1e-6:
                print(
                    f"  ⚠️  WARNING: All orientations appear to be constant ({np.mean(orientations1):.4f})"
                )
                print(
                    "  This suggests the model was not trained with orientation supervision."
                )
                print(
                    "  Consider training with --lambda-rot > 0 and rotation augmentation."
                )

        # Read matches
        print("\nReading matches...")
        raw_matches, inlier_matches = read_matches_and_inliers(db, image_id1, image_id2)

        if len(raw_matches) == 0:
            print(f"Error: No matches found between {image_name1} and {image_name2}")
            sys.exit(1)

        print(f"  Total raw matches: {len(raw_matches)}")
        if inlier_matches is not None:
            print(f"  Inlier matches: {len(inlier_matches)}")
        else:
            print(
                "  No geometric verification data available (all matches shown as inliers)"
            )

        # Classify matches
        inlier_indices, outlier_indices = classify_matches(raw_matches, inlier_matches)

        # Apply filter
        if args.filter == "inliers":
            selected_indices = inlier_indices
            print(
                f"\nFiltering: showing only inliers ({len(selected_indices)} matches)"
            )
        elif args.filter == "outliers":
            selected_indices = outlier_indices
            print(
                f"\nFiltering: showing only outliers ({len(selected_indices)} matches)"
            )
        else:
            selected_indices = np.arange(len(raw_matches))
            print(f"\nShowing all matches ({len(selected_indices)} matches)")

        # Sample matches if max_matches is specified
        if args.max_matches is not None and len(selected_indices) > args.max_matches:
            print(
                f"Sampling {args.max_matches} random matches from {len(selected_indices)} total..."
            )
            selected_indices = np.random.choice(
                selected_indices, args.max_matches, replace=False
            )

        # Get final matches and inlier mask
        display_matches = raw_matches[selected_indices]
        inlier_mask = np.isin(selected_indices, inlier_indices)

        print(f"\nDisplaying {len(display_matches)} matches:")
        print(f"  Inliers: {inlier_mask.sum()}")
        print(f"  Outliers: {(~inlier_mask).sum()}")

        # Create visualization
        print("\nCreating visualization...")
        visualize_matches(
            image_path1,
            image_path2,
            kpts1,
            kpts2,
            display_matches,
            inlier_mask,
            image_name1,
            image_name2,
            show_all_keypoints=args.show_all_keypoints,
            show_scores=args.show_scores,
            max_score_labels=args.max_score_labels,
            show_orientation=args.show_orientation,
            orientation_scale=args.orientation_scale,
            inlier_color=args.inlier_color,
            outlier_color=args.outlier_color,
            keypoint_size=args.keypoint_size,
            line_width=args.line_width,
            output_path=args.output,
            dpi=args.dpi,
        )

        print("\nDone!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close database if needed (for older pycolmap versions)
        if hasattr(db, "close"):
            db.close()


if __name__ == "__main__":
    main()
