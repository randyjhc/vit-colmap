#!/usr/bin/env python
"""
Visualize the complete training sampling pipeline.

This script demonstrates the full training process:
1. Generate synthetic image pair with known homography
2. Extract DINOv2 backbone features
3. Select invariant points using TrainingSampler
4. Generate positive and negative pairs
5. Visualize all components: invariant points, correspondences, negatives

Usage:
    python scripts/visualize_training_sampling.py --image path/to/image.jpg
    python scripts/visualize_training_sampling.py --image path/to/image.jpg --selection-mode threshold
    python scripts/visualize_training_sampling.py --image path/to/image.jpg --top-k 256
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import cv2


def denormalize_image(img_tensor):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def draw_keypoints(ax, img, coords, color="lime", size=20, alpha=0.7, label=None):
    """Draw keypoints on image."""
    ax.imshow(img)
    if len(coords) > 0:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color,
            s=size,
            alpha=alpha,
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            label=label,
        )
    ax.axis("off")


def draw_correspondences(ax, img1, img2, coords1, coords2, num_lines=50):
    """Draw correspondence lines between two images (side-by-side)."""
    # Create side-by-side visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)

    # Create combined canvas
    canvas = np.ones((h, w1 + w2, 3))
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    ax.imshow(canvas)

    # Draw lines (sample if too many)
    n_points = len(coords1)
    if n_points > num_lines:
        indices = np.random.choice(n_points, num_lines, replace=False)
    else:
        indices = np.arange(n_points)

    for idx in indices:
        x1, y1 = coords1[idx]
        x2, y2 = coords2[idx]

        # Random color for each correspondence
        color = plt.cm.hsv(np.random.rand())
        ax.plot([x1, x2 + w1], [y1, y2], "-", color=color, linewidth=0.5, alpha=0.6)
        ax.scatter(
            [x1, x2 + w1], [y1, y2], c=[color], s=30, edgecolors="white", linewidths=0.5
        )

    ax.set_xlim(0, w1 + w2)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.set_title(f"Positive Correspondences ({n_points} pairs, showing {len(indices)})")


def visualize_negative_samples(
    ax, img, positive_coords, negative_coords, negative_types, patch_size=14
):
    """Visualize different types of negative samples with color-coded grouping."""
    ax.imshow(img)

    # Draw negatives for first few positive points (to avoid clutter)
    num_pos_to_show = min(5, len(positive_coords))

    # Use a colormap to assign unique colors to each positive point
    colormap = plt.cm.tab10 if num_pos_to_show <= 10 else plt.cm.hsv
    colors = [colormap(i / max(num_pos_to_show, 1)) for i in range(num_pos_to_show)]

    # Draw each positive point with its negatives in the same color
    for i in range(num_pos_to_show):
        pos_coord = positive_coords[i]
        color = colors[i]

        # Draw the positive point
        ax.scatter(
            [pos_coord[0]],
            [pos_coord[1]],
            c=[color],
            s=80,
            alpha=0.9,
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label=f"Positive #{i+1}" if i < 3 else "",
            zorder=3,
        )

        # Draw circle showing negative radius
        circle = patches.Circle(
            (pos_coord[0], pos_coord[1]),
            patch_size * 16,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.4,
        )
        ax.add_patch(circle)

        # Draw ALL negatives for this positive point in the same color
        all_neg_coords = []
        for neg_type, neg_coords in negative_coords[i].items():
            if len(neg_coords) > 0:
                # Scale negative coords from feature space to image space
                neg_coords_img = neg_coords * patch_size + patch_size / 2
                all_neg_coords.append(neg_coords_img)

        # Combine all negative types and draw together
        if all_neg_coords:
            all_neg_coords = np.vstack(all_neg_coords)
            ax.scatter(
                all_neg_coords[:, 0],
                all_neg_coords[:, 1],
                c=[color] * len(all_neg_coords),
                s=30,
                alpha=0.7,
                marker="x",
                linewidths=1.5,
                zorder=2,
            )

    # Add legend entries for marker types
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Positive Anchor",
            markeredgecolor="white",
            markeredgewidth=1,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markerfacecolor="gray",
            markersize=8,
            label="Negatives",
            markeredgewidth=1.5,
        ),
    ]

    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.axis("off")
    ax.set_title(
        f"Negative Sampling: Color-Coded Groups (showing {num_pos_to_show} anchor points)"
    )


def visualize_in_image_negatives(
    ax, img, anchor_coord, neg_coords, patch_size, anchor_idx
):
    """Visualize in-image negatives for a single anchor point."""
    ax.imshow(img)

    # Convert anchor from feature space to image space
    anchor_img = anchor_coord * patch_size + patch_size / 2

    # Draw anchor point
    color = "dodgerblue"
    ax.scatter(
        [anchor_img[0]],
        [anchor_img[1]],
        c=[color],
        s=120,
        alpha=0.9,
        marker="o",
        edgecolors="white",
        linewidths=2,
        label="Anchor",
        zorder=3,
    )

    # Draw exclusion radius circle
    circle = patches.Circle(
        (anchor_img[0], anchor_img[1]),
        patch_size * 16,
        fill=False,
        edgecolor=color,
        linewidth=4,
        linestyle="--",
        alpha=0.5,
    )
    ax.add_patch(circle)

    # Draw in-image negatives
    if len(neg_coords) > 0:
        neg_coords_img = neg_coords * patch_size + patch_size / 2
        ax.scatter(
            neg_coords_img[:, 0],
            neg_coords_img[:, 1],
            c=[color] * len(neg_coords_img),
            s=40,
            alpha=0.8,
            marker="x",
            linewidths=2,
            label=f"In-Image Negatives ({len(neg_coords)})",
            zorder=2,
        )

    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")
    ax.set_title(
        f"In-Image Negatives for Anchor #{anchor_idx}\n(Image 2: same image, distance ≥ radius)",
        fontsize=13,
        fontweight="bold",
    )


def visualize_hard_negatives(ax, img, anchor_coord, neg_coords, patch_size, anchor_idx):
    """Visualize hard negatives for a single anchor point."""
    ax.imshow(img)

    # Convert anchor from feature space to image space
    anchor_img = anchor_coord * patch_size + patch_size / 2

    # Draw anchor point
    color = "orangered"
    ax.scatter(
        [anchor_img[0]],
        [anchor_img[1]],
        c=[color],
        s=120,
        alpha=0.9,
        marker="o",
        edgecolors="white",
        linewidths=2,
        label="Anchor",
        zorder=3,
    )

    # Draw exclusion radius circle
    circle = patches.Circle(
        (anchor_img[0], anchor_img[1]),
        patch_size * 16,
        fill=False,
        edgecolor=color,
        linewidth=4,
        linestyle="--",
        alpha=0.5,
    )
    ax.add_patch(circle)

    # Draw hard negatives
    if len(neg_coords) > 0:
        neg_coords_img = neg_coords * patch_size + patch_size / 2
        ax.scatter(
            neg_coords_img[:, 0],
            neg_coords_img[:, 1],
            c=[color] * len(neg_coords_img),
            s=40,
            alpha=0.8,
            marker="x",
            linewidths=2,
            label=f"Hard Negatives ({len(neg_coords)})",
            zorder=2,
        )

    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")
    ax.set_title(
        f"Hard Negatives for Anchor #{anchor_idx}\n(Image 2: high similarity, far distance)",
        fontsize=13,
        fontweight="bold",
    )


def visualize_cross_image_negatives(
    ax, img, anchor_coord, neg_coords, patch_size, anchor_idx
):
    """Visualize cross-image negatives sampled from the other image."""
    ax.imshow(img)

    # Convert anchor from feature space to image space
    anchor_img = anchor_coord * patch_size + patch_size / 2

    # Draw anchor's corresponding position in this image (for reference)
    color = "mediumseagreen"
    ax.scatter(
        [anchor_img[0]],
        [anchor_img[1]],
        c=["gold"],
        s=120,
        alpha=0.7,
        marker="s",
        edgecolors="white",
        linewidths=2,
        label="Anchor (in Image 2)",
        zorder=3,
    )

    # Draw cross-image negatives
    if len(neg_coords) > 0:
        neg_coords_img = neg_coords * patch_size + patch_size / 2
        ax.scatter(
            neg_coords_img[:, 0],
            neg_coords_img[:, 1],
            c=[color] * len(neg_coords_img),
            s=40,
            alpha=0.8,
            marker="x",
            linewidths=2,
            label=f"Cross-Image Negatives ({len(neg_coords)})",
            zorder=2,
        )

    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")
    ax.set_title(
        f"Cross-Image Negatives for Anchor #{anchor_idx}\n(sampled from Image 1 - different view)",
        fontsize=13,
        fontweight="bold",
    )


def create_comprehensive_visualization(
    img1_np,
    img2_np,
    invariant_coords_img1,
    invariant_coords_img2,
    negative_coords_dict,
    similarity_scores,
    descriptors_z1,
    descriptors_z2,
    descriptors_neg,
    patch_size,
    selected_anchor_idx,
    save_path=None,
):
    """Create comprehensive visualization of the training sampling pipeline."""

    # Convert coordinates from feature space to image space for visualization
    # invariant_coords are in feature map space, multiply by patch_size to get image coords
    scale = patch_size
    coords1_img = invariant_coords_img1 * scale + scale / 2
    coords2_img = invariant_coords_img2 * scale + scale / 2

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(28, 12))
    gs = GridSpec(2, 12, figure=fig, hspace=0.3, wspace=0.35)

    # Row 0: Image 1 | Correspondences | Image 2 (3 equal sections)
    ax1 = fig.add_subplot(gs[0, 0:4])
    draw_keypoints(ax1, img1_np, coords1_img, color="lime", size=30)
    ax1.set_title(
        f"Image 1: Invariant Points ({len(coords1_img)} points)",
        fontsize=14,
        fontweight="bold",
    )

    ax2 = fig.add_subplot(gs[0, 4:8])
    draw_correspondences(ax2, img1_np, img2_np, coords1_img, coords2_img, num_lines=100)

    ax3 = fig.add_subplot(gs[0, 8:12])
    draw_keypoints(ax3, img2_np, coords2_img, color="lime", size=30)
    ax3.set_title(
        f"Image 2: Invariant Points ({len(coords2_img)} points)",
        fontsize=14,
        fontweight="bold",
    )

    # Row 1: Single Anchor Negative Visualizations
    # Get the selected anchor's coordinates and negatives
    # NOTE: Anchor is in Image 2 (matching real training sampler)
    anchor_coord_img2 = invariant_coords_img2[selected_anchor_idx]
    anchor_coord_img1 = invariant_coords_img1[selected_anchor_idx]
    anchor_negatives = negative_coords_dict[selected_anchor_idx]

    # Left: In-image negatives (from Image 2 - same as anchor)
    ax5 = fig.add_subplot(gs[1, 0:4])
    visualize_in_image_negatives(
        ax5,
        img2_np,
        anchor_coord_img2,
        anchor_negatives["in_image"],
        patch_size,
        selected_anchor_idx,
    )

    # Middle: Cross-image negatives (from Image 1 - different view)
    ax6 = fig.add_subplot(gs[1, 4:8])
    visualize_cross_image_negatives(
        ax6,
        img1_np,
        anchor_coord_img1,
        anchor_negatives["cross_image"],
        patch_size,
        selected_anchor_idx,
    )

    # Right: Hard negatives (from Image 2 - same as anchor, high similarity)
    ax7 = fig.add_subplot(gs[1, 8:12])
    visualize_hard_negatives(
        ax7,
        img2_np,
        anchor_coord_img2,
        anchor_negatives["hard"],
        patch_size,
        selected_anchor_idx,
    )

    plt.suptitle("Training Sampling Pipeline Visualization", fontsize=18, y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training sampling pipeline with synthetic pairs"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image for synthetic pair generation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save visualization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detect if not specified)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        help="DINOv2 backbone model name",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="top_k",
        choices=["top_k", "threshold", "hybrid"],
        help="Invariant point selection strategy",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=512,
        help="Number of invariant points to select (for top_k mode)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for threshold-based selection",
    )
    parser.add_argument(
        "--negative-radius",
        type=int,
        default=16,
        help="Minimum distance for negative samples (in feature space)",
    )
    parser.add_argument(
        "--num-in-image-negatives",
        type=int,
        default=10,
        help="Number of in-image negatives per point",
    )
    parser.add_argument(
        "--num-hard-negatives",
        type=int,
        default=5,
        help="Number of hard negatives per point",
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import modules
    from vit_colmap.model import ViTFeatureModel
    from vit_colmap.dataloader import create_synthetic_pair
    from vit_colmap.dataloader.training_sampler import TrainingSampler
    from vit_colmap.dataloader.training_batch import TrainingBatchProcessor
    import torchvision.transforms as transforms

    # Load and prepare image
    print(f"Loading image: {args.image}")
    if not args.image.exists():
        raise ValueError(f"Image file not found: {args.image}")

    img = cv2.imread(str(args.image))
    if img is None:
        raise ValueError(f"Failed to load image: {args.image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize model
    print(f"Loading ViTFeatureModel with backbone: {args.backbone}")
    model = ViTFeatureModel(
        backbone_name=args.backbone, descriptor_dim=128, freeze_backbone=True
    )
    model.eval()
    model.to(device)
    patch_size = model.patch_size

    # Resize to model's expected size
    target_h = (1200 // patch_size) * patch_size
    target_w = (1600 // patch_size) * patch_size

    print(f"Generating synthetic pair with target size: ({target_h}, {target_w})")

    # Create synthetic pair with random homography
    img1_np, img2_np, H = create_synthetic_pair(
        img,
        image_size=(target_h, target_w),
        rotation_range=30.0,  # Maximum rotation in degrees (±range)
        scale_range=(0.8, 1.2),
        perspective_range=0.0002,
        translation_range=(0.1, 0.1),  # Fraction of image size
    )

    # Convert to tensors
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img1_tensor = transform(img1_np).unsqueeze(0).to(device)
    img2_tensor = transform(img2_np).unsqueeze(0).to(device)
    H_tensor = torch.from_numpy(H).float().unsqueeze(0).to(device)

    print(f"Homography matrix:\n{H}")

    # Initialize training sampler
    print(f"\nInitializing TrainingSampler (mode: {args.selection_mode})")
    sampler = TrainingSampler(
        top_k_invariant=args.top_k,
        negative_radius=args.negative_radius,
        num_in_image_negatives=args.num_in_image_negatives,
        num_hard_negatives=args.num_hard_negatives,
        patch_size=patch_size,
        selection_mode=args.selection_mode,
        similarity_threshold=args.similarity_threshold,
    )

    # Initialize batch processor
    processor = TrainingBatchProcessor(model=model, sampler=sampler)

    # Process batch
    print("\nExtracting features and running sampling pipeline...")
    batch = {
        "img1": img1_tensor,
        "img2": img2_tensor,
        "H": H_tensor,
    }

    with torch.no_grad():
        outputs, targets = processor.process_batch(batch)

    # Extract results
    invariant_coords = (
        targets["invariant_coords"].squeeze(0).cpu().numpy()
    )  # (K, 2) in output space
    z1 = outputs["z1"].squeeze(0)  # (K, 128)
    z2 = outputs["z2"].squeeze(0)  # (K, 128)
    negatives = outputs["negatives"].squeeze(0)  # (K, N, 128)

    # Extract backbone features for similarity map
    with torch.no_grad():
        features1 = model._extract_backbone_features(img1_tensor)
        features2 = model._extract_backbone_features(img2_tensor)

    # Compute similarity map for visualization
    from vit_colmap.dataloader import warp_patch_tokens
    import torch.nn.functional as F

    warped_features1 = warp_patch_tokens(
        features1, H_tensor, patch_size=patch_size, input_size=(target_h, target_w)
    )

    warped_norm = F.normalize(warped_features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)
    similarity_map = (warped_norm * features2_norm).sum(dim=1).squeeze(0).cpu().numpy()

    # Convert invariant coords from output space to feature space
    # Output space is 1/4 resolution, feature space is 1/14 resolution
    # scale_factor in training_batch.py: patch_size / 4.0 = 14/4 = 3.5
    scale_factor = patch_size / 4.0
    invariant_coords_feat = invariant_coords / scale_factor  # Convert to feature space

    # Transform coords to img1 space
    H_inv = torch.linalg.inv(H_tensor)
    coords_in_img1_feat = (
        sampler.transform_coords_with_homography(
            torch.from_numpy(invariant_coords_feat).unsqueeze(0).to(device),
            H_inv,
            from_feature_space=True,
            image_size=(target_h, target_w),
            feature_size=(features1.shape[2], features1.shape[3]),
        )
        .squeeze(0)
        .cpu()
        .numpy()
    )

    print("\nResults:")
    print(f"  Invariant points selected: {len(invariant_coords_feat)}")
    print(f"  Descriptor dimension: {z1.shape[-1]}")
    print(f"  Negatives per point: {negatives.shape[1]}")
    print(
        f"  Similarity range: [{similarity_map.min():.3f}, {similarity_map.max():.3f}]"
    )

    # Generate negative coordinates for visualization
    print("\nGenerating negative samples for visualization...")
    invariant_coords_tensor = (
        torch.from_numpy(invariant_coords_feat).unsqueeze(0).to(device)
    )

    # Generate in-image negatives
    sampler.generate_in_image_negatives(features2, invariant_coords_tensor).squeeze(
        0
    )  # (K, N_in_image, C)

    # Generate hard negatives
    f2_full = sampler.sample_features_at_coords(
        features2, invariant_coords_tensor
    ).squeeze(0)
    sampler.generate_hard_negatives(
        features2, f2_full.unsqueeze(0), invariant_coords_tensor
    ).squeeze(0)  # (K, N_hard, C)

    # We need to reverse-engineer the coordinates from the features
    # For visualization purposes, we'll re-generate the negative coordinates
    # by replicating the sampling logic
    negative_coords_dict = []

    B, C, H_p, W_p = features2.shape

    # Create coordinate grid
    y_coords_grid = torch.arange(H_p, device=device, dtype=torch.float32)
    x_coords_grid = torch.arange(W_p, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords_grid, x_coords_grid, indexing="ij")
    all_coords_grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H_p*W_p, 2)

    # Normalize features for hard negative computation
    import torch.nn.functional as F

    features2_flat = features2.view(B, C, -1).permute(0, 2, 1)  # (B, H_p*W_p, C)
    features2_flat_norm = F.normalize(features2_flat, p=2, dim=-1)
    f2_full_norm = F.normalize(f2_full.unsqueeze(0), p=2, dim=-1)  # (1, K, C)

    num_points_to_viz = min(10, len(invariant_coords_feat))  # Visualize up to 10 points

    for k in range(num_points_to_viz):
        pos_coord = invariant_coords_tensor[0, k]  # (2,)

        # In-image negatives: points >= radius away
        distances = torch.norm(all_coords_grid - pos_coord.unsqueeze(0), dim=-1)
        valid_mask = distances >= sampler.negative_radius
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

        if len(valid_indices) >= sampler.num_in_image_negatives:
            # Sample same as in training
            perm = torch.randperm(len(valid_indices), device=device)[
                : sampler.num_in_image_negatives
            ]
            neg_indices_in_image = valid_indices[perm]
        else:
            neg_indices_in_image = valid_indices

        in_image_coords = all_coords_grid[neg_indices_in_image].cpu().numpy()

        # Hard negatives: high similarity but geometrically far
        pos_feat = f2_full_norm[0, k]  # (C,)
        similarities = features2_flat_norm[0] @ pos_feat  # (H_p*W_p,)

        # Mask out close points
        far_mask = distances >= sampler.negative_radius
        similarities_masked = similarities.clone()
        similarities_masked[~far_mask] = -float("inf")

        # Select top-k most similar (but far)
        num_hard = min(sampler.num_hard_negatives, far_mask.sum().item())
        if num_hard > 0:
            _, hard_neg_indices = similarities_masked.topk(num_hard)
            hard_coords = all_coords_grid[hard_neg_indices].cpu().numpy()
        else:
            hard_coords = np.array([]).reshape(0, 2)

        # Cross-image negatives: random samples from Image 1
        # Sample random coordinates from Image 1's feature grid
        num_cross = sampler.num_in_image_negatives  # Same count as in-image
        if num_cross > 0:
            # Create grid for Image 1
            B_1, C_1, H_p_1, W_p_1 = features1.shape
            rand_indices = torch.randint(0, H_p_1 * W_p_1, (num_cross,), device=device)

            # Convert flat indices to 2D coordinates
            cross_y = (rand_indices // W_p_1).float()
            cross_x = (rand_indices % W_p_1).float()
            cross_coords = (
                torch.stack([cross_x, cross_y], dim=-1).cpu().numpy()
            )  # (num_cross, 2)
        else:
            cross_coords = np.array([]).reshape(0, 2)

        negative_coords_dict.append(
            {
                "in_image": in_image_coords,
                "hard": hard_coords,
                "cross_image": cross_coords,
            }
        )

    # Pad with empty dicts for remaining points
    for k in range(num_points_to_viz, len(invariant_coords_feat)):
        negative_coords_dict.append(
            {
                "in_image": np.array([]).reshape(0, 2),
                "hard": np.array([]).reshape(0, 2),
                "cross_image": np.array([]).reshape(0, 2),
            }
        )

    # Randomly select one anchor for detailed visualization
    selected_anchor_idx = np.random.randint(
        0, min(num_points_to_viz, len(invariant_coords_feat))
    )
    print(f"  Randomly selected anchor index: {selected_anchor_idx}")

    # Create visualization
    print("\nCreating visualization...")
    save_path = args.output_dir / f"training_sampling_{args.image.stem}.png"

    create_comprehensive_visualization(
        img1_np,
        img2_np,
        coords_in_img1_feat,
        invariant_coords_feat,
        negative_coords_dict,
        similarity_map,
        z1,
        z2,
        negatives,
        patch_size,
        selected_anchor_idx,
        save_path=save_path,
    )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
