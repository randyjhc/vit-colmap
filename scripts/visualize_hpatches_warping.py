#!/usr/bin/env python
"""
Visualize HPatches image pair warping and ViT patch token alignment.

This script demonstrates:
1. Loading HPatches image pairs with ground truth homographies
2. Extracting DINOv2 backbone features
3. Warping patch tokens using homography
4. Visualizing feature alignment quality

Usage:
    python scripts/visualize_hpatches_warping.py --data-root data/raw/HPatches
    python scripts/visualize_hpatches_warping.py --data-root data/raw/HPatches --sequence i_ajuntament --pair-idx 0
    python scripts/visualize_hpatches_warping.py --image path/to/image.jpg  # Use single image with synthetic homography
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from sklearn.decomposition import PCA


def denormalize_image(img_tensor):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def features_to_rgb(features, method="pca"):
    """
    Convert high-dimensional features to RGB for visualization.

    Args:
        features: (C, H, W) feature tensor
        method: "pca" or "norm"

    Returns:
        rgb: (H, W, 3) RGB image
    """
    C, H, W = features.shape
    features_flat = features.reshape(C, -1).T  # (H*W, C)

    if method == "pca":
        # Use PCA to reduce to 3 components
        pca = PCA(n_components=3)
        rgb_flat = pca.fit_transform(features_flat)  # (H*W, 3)
        # Normalize to [0, 1]
        rgb_flat = (rgb_flat - rgb_flat.min()) / (
            rgb_flat.max() - rgb_flat.min() + 1e-8
        )
    else:
        # Use L2 norm
        norm = np.linalg.norm(features_flat, axis=1)
        rgb_flat = plt.cm.viridis(norm / norm.max())[:, :3]

    rgb = rgb_flat.reshape(H, W, 3)
    return rgb


def create_visualization(
    img1,
    img2,
    features1,
    features2,
    warped_features1,
    similarity_map,
    valid_mask,
    save_path=None,
):
    """Create comprehensive visualization of warping results."""
    # Convert tensors to numpy
    img1_np = denormalize_image(img1.cpu())
    img2_np = denormalize_image(img2.cpu())
    features1_np = features1.cpu().numpy()
    features2_np = features2.cpu().numpy()
    warped_features1_np = warped_features1.cpu().numpy()
    similarity_np = similarity_map.cpu().numpy()
    valid_mask_np = valid_mask.cpu().numpy()

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)

    # Row 1: Original images and their features
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img1_np)
    ax1.set_title("Image 1 (Reference)")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img2_np)
    ax2.set_title("Image 2 (Target)")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    feat1_rgb = features_to_rgb(features1_np, method="norm")
    ax3.imshow(feat1_rgb)
    ax3.set_title("Features 1 (PCA)")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    feat2_rgb = features_to_rgb(features2_np, method="norm")
    ax4.imshow(feat2_rgb)
    ax4.set_title("Features 2 (PCA)")
    ax4.axis("off")

    # Row 2: Warped features and comparison
    ax5 = fig.add_subplot(gs[1, 0])
    warped_rgb = features_to_rgb(warped_features1_np, method="norm")
    ax5.imshow(warped_rgb)
    ax5.set_title("Warped Features 1 (PCA)")
    ax5.axis("off")

    ax6 = fig.add_subplot(gs[1, 1])
    # Overlay warped on target
    alpha = 0.5
    overlay = alpha * warped_rgb + (1 - alpha) * feat2_rgb
    ax6.imshow(overlay)
    ax6.set_title("Overlay: Warped + Target")
    ax6.axis("off")

    ax7 = fig.add_subplot(gs[1, 2])
    # Feature difference
    diff = np.linalg.norm(warped_features1_np - features2_np, axis=0)
    diff_normalized = diff / diff.max()
    im7 = ax7.imshow(diff_normalized, cmap="hot", vmin=0, vmax=1)
    ax7.set_title("Feature Difference (L2 norm)")
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(similarity_np, cmap="viridis", vmin=-1, vmax=1)
    ax8.set_title("Cosine Similarity")
    ax8.axis("off")
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    # Row 3: Valid mask and statistics
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(valid_mask_np, cmap="gray")
    ax9.set_title(f"Valid Mask ({valid_mask_np.sum()}/{valid_mask_np.size} valid)")
    ax9.axis("off")

    ax10 = fig.add_subplot(gs[2, 1])
    # Masked similarity
    masked_similarity = np.where(valid_mask_np, similarity_np, 0)
    im10 = ax10.imshow(masked_similarity, cmap="viridis", vmin=-1, vmax=1)
    ax10.set_title("Masked Cosine Similarity")
    ax10.axis("off")
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)

    ax11 = fig.add_subplot(gs[2, 2:])
    # Statistics
    valid_similarities = similarity_np[valid_mask_np]
    if len(valid_similarities) > 0:
        ax11.hist(valid_similarities, bins=50, edgecolor="black", alpha=0.7)
        ax11.axvline(valid_similarities.mean(), color="r", linestyle="--", linewidth=2)
        ax11.set_title(
            f"Similarity Distribution\n"
            f"Mean: {valid_similarities.mean():.4f}, "
            f"Std: {valid_similarities.std():.4f}, "
            f"Min: {valid_similarities.min():.4f}, "
            f"Max: {valid_similarities.max():.4f}"
        )
        ax11.set_xlabel("Cosine Similarity")
        ax11.set_ylabel("Count")
    else:
        ax11.text(0.5, 0.5, "No valid correspondences", ha="center", va="center")
        ax11.set_title("Similarity Distribution")

    plt.suptitle("HPatches Warping Visualization", fontsize=16, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize HPatches warping")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/HPatches"),
        help="Path to HPatches root directory",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Specific sequence to visualize (e.g., i_ajuntament)",
    )
    parser.add_argument(
        "--pair-idx",
        type=int,
        default=0,
        help="Index of image pair to visualize",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single image to use with synthetic homography (for testing without HPatches)",
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
    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Import after path setup
    from vit_colmap.model import ViTFeatureModel
    from vit_colmap.dataloader import (
        HPatchesDataset,
        warp_patch_tokens,
        compute_valid_mask,
        compute_feature_similarity,
    )
    import torchvision.transforms as transforms

    # Initialize model
    print(f"Loading ViTFeatureModel with backbone: {args.backbone}")
    model = ViTFeatureModel(
        backbone_name=args.backbone, descriptor_dim=128, freeze_backbone=True
    )
    model.eval()
    model.to(device)

    if args.image is not None:
        # Use single image with synthetic homography
        print(f"Loading single image: {args.image}")

        # Load and preprocess image
        img = cv2.imread(str(args.image))
        if img is None:
            raise ValueError(f"Failed to load image: {args.image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to model's expected size
        patch_size = model.patch_size
        target_h = (1200 // patch_size) * patch_size
        target_w = (1600 // patch_size) * patch_size
        img = cv2.resize(img, (target_w, target_h))

        # Create transform
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        img1_tensor = transform(img).unsqueeze(0).to(device)
        img2_tensor = img1_tensor.clone()  # Same image

        # Create synthetic homography (small perspective transform)
        H = torch.tensor(
            [[1.05, 0.02, -20.0], [-0.01, 1.03, -15.0], [0.0001, 0.0001, 1.0]],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        seq_name = "synthetic"
        input_size = (target_h, target_w)

    else:
        # Load from HPatches dataset
        print(f"Loading HPatches dataset from: {args.data_root}")

        if not args.data_root.exists():
            print(f"Error: HPatches directory not found at {args.data_root}")
            print("Please download HPatches dataset and place it in data/raw/HPatches/")
            print(
                "Download from: http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"
            )
            sys.exit(1)

        dataset = HPatchesDataset(
            root_dir=args.data_root,
            split="all",
            target_size=(1200, 1600),
            patch_size=model.patch_size,
        )

        if len(dataset) == 0:
            print("Error: No image pairs found in HPatches dataset")
            sys.exit(1)

        # Get specific sample
        if args.sequence is not None:
            # Find sample from specific sequence
            found = False
            for idx in range(len(dataset)):
                sample = dataset[idx]
                if sample["seq_name"] == args.sequence:
                    found = True
                    break
            if not found:
                print(f"Error: Sequence '{args.sequence}' not found")
                print(
                    f"Available sequences: {[s.name for s in dataset.sequences[:10]]}..."
                )
                sys.exit(1)
        else:
            # Use specified pair index
            idx = min(args.pair_idx, len(dataset) - 1)
            sample = dataset[idx]

        img1_tensor = sample["img1"].unsqueeze(0).to(device)
        img2_tensor = sample["img2"].unsqueeze(0).to(device)
        H = sample["H"].unsqueeze(0).to(device)
        seq_name = sample["seq_name"]
        input_size = (dataset.target_h, dataset.target_w)

        print(f"Loaded pair from sequence: {seq_name}")

    # Extract backbone features
    print("Extracting DINOv2 features...")
    with torch.no_grad():
        features1 = model._extract_backbone_features(img1_tensor)
        features2 = model._extract_backbone_features(img2_tensor)

    print(f"Feature shape: {features1.shape}")  # (B, C, H_p, W_p)

    # Warp features1 using homography
    print("Warping features with homography...")
    warped_features1 = warp_patch_tokens(
        features1, H, patch_size=model.patch_size, input_size=input_size
    )

    # Compute similarity
    print("Computing feature similarity...")
    similarity = compute_feature_similarity(
        features1, features2, H, patch_size=model.patch_size, input_size=input_size
    )

    # Compute valid mask
    feature_size = (features1.shape[2], features1.shape[3])
    valid_mask = compute_valid_mask(
        H, feature_size, input_size, patch_size=model.patch_size
    )

    # Create visualization
    print("Creating visualization...")
    save_path = args.output_dir / f"hpatches_warping_{seq_name}.png"

    create_visualization(
        img1_tensor.squeeze(0),
        img2_tensor.squeeze(0),
        features1.squeeze(0),
        features2.squeeze(0),
        warped_features1.squeeze(0),
        similarity.squeeze(0),
        valid_mask.squeeze(0),
        save_path=save_path,
    )

    # Print summary statistics
    valid_sim = similarity.squeeze(0)[valid_mask.squeeze(0)]
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Sequence: {seq_name}")
    print(f"Input image size: {input_size}")
    print(f"Feature map size: {feature_size}")
    print(f"Valid correspondences: {valid_mask.sum().item()} / {valid_mask.numel()}")
    if len(valid_sim) > 0:
        print(f"Mean cosine similarity: {valid_sim.mean().item():.4f}")
        print(f"Std cosine similarity: {valid_sim.std().item():.4f}")
        print(f"Min cosine similarity: {valid_sim.min().item():.4f}")
        print(f"Max cosine similarity: {valid_sim.max().item():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
