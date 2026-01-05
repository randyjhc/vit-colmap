#!/usr/bin/env python
"""
Visualize selected invariant points on HPatches image pairs.

This script:
1. Loads an HPatches image pair with ground truth homography
2. Extracts ViT backbone features (DINOv2)
3. Selects invariant points using cosine similarity after warping
4. Visualizes points with similarity scores labeled directly beside them

Usage:
    # Basic visualization
    python scripts/visualize_invariant_points.py --data-root data/raw/HPatches --sequence i_ajuntament --pair-idx 0

    # Show only top 100 points
    python scripts/visualize_invariant_points.py --data-root data/raw/HPatches --sequence v_adam --pair-idx 1 --max-points 100

    # Save to file
    python scripts/visualize_invariant_points.py --data-root data/raw/HPatches --sequence i_ajuntament --pair-idx 0 --output viz.png
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vit_colmap.dataloader.hpatches_dataset import HPatchesDataset
from vit_colmap.dataloader.training_sampler import TrainingSampler
from vit_colmap.model.vit_feature_model import ViTFeatureModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize selected invariant points on HPatches image pairs"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to HPatches dataset root directory",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence name (e.g., i_ajuntament, v_adam). If not specified, uses first sequence.",
    )
    parser.add_argument(
        "--pair-idx",
        type=int,
        default=0,
        help="Image pair index within sequence (0-based, default: 0)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum number of invariant points to display (default: show all selected points)",
    )
    parser.add_argument(
        "--top-k-invariant",
        type=int,
        default=512,
        help="Number of invariant points to select (default: 512)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for saving visualization (default: display only)",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=3,
        help="Size of point markers (default: 3)",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=0.3,
        help="Width of correspondence lines (default: 0.3)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=6,
        help="Font size for similarity score labels (default: 6)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (default: cuda if available)",
    )
    parser.add_argument(
        "--show-all-labels",
        action="store_true",
        help="Show similarity labels for all points (default: only show for max-points if specified)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Minimum similarity threshold - only display points with similarity >= this value (e.g., 0.95)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable similarity score labels on all points",
    )

    return parser.parse_args()


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor back to displayable image.

    Args:
        img_tensor: (3, H, W) normalized image tensor

    Returns:
        img: (H, W, 3) numpy array in [0, 1] range
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def visualize_invariant_points(
    img1: torch.Tensor,
    img2: torch.Tensor,
    invariant_coords: torch.Tensor,
    similarity_scores: torch.Tensor,
    H: torch.Tensor,
    patch_size: int,
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    sequence_name: str,
    max_points: int = None,
    show_all_labels: bool = False,
    min_similarity: float = None,
    show_labels: bool = True,
    point_size: int = 3,
    line_width: float = 0.3,
    font_size: int = 6,
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Create visualization of invariant points with similarity scores.

    Args:
        img1: (3, H, W) first image tensor (normalized)
        img2: (3, H, W) second image tensor (normalized)
        invariant_coords: (K, 2) invariant point coordinates in feature space (x, y)
        similarity_scores: (K,) cosine similarity scores for each point
        H: (3, 3) homography matrix (img1 -> img2)
        patch_size: ViT patch size (e.g., 14)
        image_size: (H, W) original image size
        feature_size: (H_p, W_p) feature map size
        sequence_name: Name of the sequence
        max_points: Maximum number of points to display (with labels)
        show_all_labels: Show labels for all points
        min_similarity: Only display points with similarity >= this value
        show_labels: Whether to show similarity score labels
        point_size: Size of point markers
        line_width: Width of correspondence lines
        font_size: Font size for labels
        output_path: Path to save figure (None to display only)
        dpi: DPI for saved figure
    """
    # Convert images to numpy
    img1_np = denormalize_image(img1)
    img2_np = denormalize_image(img2)

    h1, w1 = img1_np.shape[:2]
    h2, w2 = img2_np.shape[:2]

    # Convert feature-space coordinates to image-space coordinates
    # Feature map center = (coord + 0.5) * patch_size
    coords_img2 = invariant_coords.clone()
    coords_img2[:, 0] = (coords_img2[:, 0] + 0.5) * patch_size
    coords_img2[:, 1] = (coords_img2[:, 1] + 0.5) * patch_size

    # Transform to image1 space using inverse homography
    H_inv = torch.linalg.inv(H)
    ones = torch.ones(len(coords_img2), 1, device=coords_img2.device)
    homogeneous = torch.cat([coords_img2, ones], dim=-1)  # (K, 3)

    # Apply homography
    transformed = (H_inv @ homogeneous.T).T  # (K, 3)
    w_coord = transformed[:, 2:3].clamp(min=1e-8)
    coords_img1 = transformed[:, :2] / w_coord  # (K, 2)

    # Convert to numpy
    coords_img1 = coords_img1.cpu().numpy()
    coords_img2 = coords_img2.cpu().numpy()
    similarity_scores = similarity_scores.cpu().numpy()

    # Filter by minimum similarity threshold if specified
    if min_similarity is not None:
        valid_mask = similarity_scores >= min_similarity
        valid_indices = np.where(valid_mask)[0]

        # Apply filter
        coords_img1 = coords_img1[valid_indices]
        coords_img2 = coords_img2[valid_indices]
        similarity_scores = similarity_scores[valid_indices]

        if len(valid_indices) == 0:
            print(f"Warning: No points found with similarity >= {min_similarity}")
            print(
                f"  Available range: [{similarity_scores.min():.3f}, {similarity_scores.max():.3f}]"
            )

    # Sort by similarity score (highest first)
    sorted_indices = np.argsort(-similarity_scores)

    # Determine which points to display and label
    if max_points is not None:
        display_indices = sorted_indices[:max_points]
    else:
        display_indices = sorted_indices

    # Determine which points to show labels for
    if show_labels:
        if show_all_labels:
            label_indices = display_indices
        else:
            # Show labels for top 100 by default
            label_indices = sorted_indices[: min(100, len(sorted_indices))]
    else:
        # No labels
        label_indices = np.array([], dtype=int)

    # Create side-by-side image
    h_max = max(h1, h2)
    canvas = np.ones((h_max, w1 + w2, 3), dtype=np.float32)
    canvas[:h1, :w1] = img1_np
    canvas[:h2, w1 : w1 + w2] = img2_np

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(canvas)
    ax.axis("off")

    # Draw correspondence lines for displayed points
    for idx in display_indices:
        pt1 = coords_img1[idx]
        pt2 = coords_img2[idx]
        score = similarity_scores[idx]

        # Color based on similarity (green = high, yellow = medium, red = low)
        # Map [0.5, 1.0] to color gradient
        normalized_score = (score - 0.5) / 0.5  # Map to [0, 1] range
        color = plt.cm.RdYlGn(normalized_score)

        # Draw line
        ax.plot(
            [pt1[0], pt2[0] + w1],
            [pt1[1], pt2[1]],
            color=color,
            linewidth=line_width,
            alpha=0.5,
        )

    # Draw points
    for idx in display_indices:
        pt1 = coords_img1[idx]
        pt2 = coords_img2[idx]
        score = similarity_scores[idx]
        normalized_score = (score - 0.5) / 0.5
        color = plt.cm.RdYlGn(normalized_score)

        # Draw points
        ax.plot(
            pt1[0],
            pt1[1],
            "o",
            color=color,
            markersize=point_size,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        ax.plot(
            pt2[0] + w1,
            pt2[1],
            "o",
            color=color,
            markersize=point_size,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    # Add similarity score labels for selected points
    for idx in label_indices:
        pt1 = coords_img1[idx]
        pt2 = coords_img2[idx]
        score = similarity_scores[idx]
        score_text = f"{score:.3f}"

        # Add label for image 1
        ax.text(
            pt1[0] + 3,  # Offset to the right
            pt1[1] - 3,  # Offset upward
            score_text,
            fontsize=font_size,
            color="white",
            weight="bold",
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black",
                alpha=0.7,
                edgecolor="none",
            ),
        )

        # Add label for image 2
        ax.text(
            pt2[0] + w1 + 3,  # Offset to the right (account for image shift)
            pt2[1] - 3,  # Offset upward
            score_text,
            fontsize=font_size,
            color="white",
            weight="bold",
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black",
                alpha=0.7,
                edgecolor="none",
            ),
        )

    # Calculate statistics
    displayed_scores = similarity_scores[display_indices]
    mean_sim = displayed_scores.mean()
    min_sim = displayed_scores.min()
    max_sim = displayed_scores.max()
    total_points_after_filter = len(similarity_scores)  # After min_similarity filter

    # Create legend and title
    title_parts = [f"Invariant Points: {sequence_name}"]

    if min_similarity is not None:
        title_parts.append(f"Filter: similarity >= {min_similarity:.3f}")

    title_parts.append(
        f"Displaying: {len(display_indices)}/{total_points_after_filter} points | "
        f"Similarity: mean={mean_sim:.3f}, min={min_sim:.3f}, max={max_sim:.3f}"
    )

    title = "\n".join(title_parts)

    ax.set_title(title, fontsize=14, pad=10)

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0.5, vmax=1.0)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")

    plt.show()


def main():
    """Main function."""
    args = parse_args()

    # Validate inputs
    if not args.data_root.exists():
        print(f"Error: HPatches directory not found: {args.data_root}")
        sys.exit(1)

    print(f"Loading HPatches dataset from: {args.data_root}")
    print(f"Device: {args.device}")

    # Create dataset
    try:
        dataset = HPatchesDataset(
            root_dir=args.data_root,
            split="all",
            target_size=(1200, 1600),
            patch_size=14,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if len(dataset) == 0:
        print("Error: No image pairs found in dataset")
        sys.exit(1)

    # Find the requested sequence
    if args.sequence is not None:
        # Find pair index for the requested sequence
        pair_idx = None
        for idx, (seq_path, _) in enumerate(dataset.pairs):
            if seq_path.name == args.sequence:
                pair_idx = idx
                break

        if pair_idx is None:
            print(f"Error: Sequence '{args.sequence}' not found")
            available_seqs = sorted(set(seq.name for seq, _ in dataset.pairs))
            print(f"Available sequences: {available_seqs[:20]}")
            sys.exit(1)

        # Offset by pair_idx within the sequence
        target_pair = args.pair_idx
        found_idx = None
        count = 0
        for idx, (seq_path, _) in enumerate(dataset.pairs):
            if seq_path.name == args.sequence:
                if count == target_pair:
                    found_idx = idx
                    break
                count += 1

        if found_idx is None:
            print(
                f"Error: Pair index {args.pair_idx} not found in sequence '{args.sequence}'"
            )
            print(f"Sequence has {count} pairs (indices 0-{count-1})")
            sys.exit(1)

        sample_idx = found_idx
    else:
        sample_idx = min(args.pair_idx, len(dataset) - 1)

    # Load sample
    print(f"\nLoading image pair {sample_idx}...")
    sample = dataset[sample_idx]

    img1 = sample["img1"]  # (3, H, W)
    img2 = sample["img2"]  # (3, H, W)
    H = sample["H"]  # (3, 3)
    seq_name = sample["seq_name"]

    print(f"  Sequence: {seq_name}")
    print(f"  Image 1 shape: {img1.shape}")
    print(f"  Image 2 shape: {img2.shape}")

    # Add batch dimension
    img1 = img1.unsqueeze(0).to(args.device)  # (1, 3, H, W)
    img2 = img2.unsqueeze(0).to(args.device)
    H = H.unsqueeze(0).to(args.device)  # (1, 3, 3)

    # Load ViT model
    print("\nLoading ViT model...")
    model = ViTFeatureModel(
        backbone_name="dinov2_vitb14",
        freeze_backbone=True,
    ).to(args.device)
    model.eval()

    # Extract features
    print("Extracting features...")
    with torch.inference_mode():
        features1 = model._extract_backbone_features(img1)  # (1, C, H_p, W_p)
        features2 = model._extract_backbone_features(img2)

    _, C, H_p, W_p = features1.shape
    feature_size = (H_p, W_p)
    image_size = (img1.shape[2], img1.shape[3])

    print(f"  Feature shape: {features1.shape}")
    print(f"  Feature size: {H_p}x{W_p}")

    # Create sampler and select invariant points
    print(f"\nSelecting top-{args.top_k_invariant} invariant points...")
    sampler = TrainingSampler(
        top_k_invariant=args.top_k_invariant,
        patch_size=14,
    )

    invariant_coords = sampler.select_invariant_points(
        features1, features2, H, image_size
    )  # (1, K, 2)

    # Compute similarity scores at invariant points
    from vit_colmap.dataloader.homography_utils import warp_patch_tokens
    import torch.nn.functional as F

    warped_features1 = warp_patch_tokens(
        features1, H, patch_size=14, input_size=image_size
    )

    # Normalize features
    warped_norm = F.normalize(warped_features1, p=2, dim=1)  # (1, C, H_p, W_p)
    features2_norm = F.normalize(features2, p=2, dim=1)

    # Compute similarity at each location
    similarity_map = (warped_norm * features2_norm).sum(dim=1).squeeze(0)  # (H_p, W_p)

    # Get similarity scores at invariant coordinates
    coords = invariant_coords[0].long()  # (K, 2)
    coords[:, 0] = coords[:, 0].clamp(0, W_p - 1)
    coords[:, 1] = coords[:, 1].clamp(0, H_p - 1)
    similarity_scores = similarity_map[coords[:, 1], coords[:, 0]]  # (K,)

    print(f"  Selected {len(invariant_coords[0])} points")
    print(
        f"  Similarity range: [{similarity_scores.min():.3f}, {similarity_scores.max():.3f}]"
    )
    print(f"  Mean similarity: {similarity_scores.mean():.3f}")

    # Create visualization
    print("\nCreating visualization...")
    if args.min_similarity is not None:
        print(f"  Filtering points with similarity >= {args.min_similarity}")
    if args.no_labels:
        print("  Labels disabled")

    visualize_invariant_points(
        img1[0],
        img2[0],
        invariant_coords[0],
        similarity_scores,
        H[0],
        patch_size=14,
        image_size=image_size,
        feature_size=feature_size,
        sequence_name=seq_name,
        max_points=args.max_points,
        show_all_labels=args.show_all_labels,
        min_similarity=args.min_similarity,
        show_labels=not args.no_labels,
        point_size=args.point_size,
        line_width=args.line_width,
        font_size=args.font_size,
        output_path=args.output,
        dpi=args.dpi,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
