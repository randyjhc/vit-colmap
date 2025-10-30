"""
Visualization tool for ViT feature extraction.
Creates visualizations showing keypoints, feature maps, and comparisons with SIFT.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def visualize_keypoints(
    image_path: Path,
    keypoints: np.ndarray,
    title: str = "Detected Keypoints",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Visualize keypoints overlaid on image.

    Args:
        image_path: Path to image
        keypoints: (N, 2) array of keypoint coordinates
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
    """
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        c="red",
        s=20,
        alpha=0.6,
        marker="x",
        linewidths=2,
    )
    plt.title(f"{title}\n{len(keypoints)} keypoints detected")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_keypoint_density(
    image_path: Path,
    keypoints: np.ndarray,
    grid_size: int = 32,
    save_path: Optional[Path] = None,
):
    """
    Visualize keypoint spatial density as a heatmap.

    Args:
        image_path: Path to image
        keypoints: (N, 2) array of keypoint coordinates
        grid_size: Grid resolution for heatmap
        save_path: Optional path to save figure
    """
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Create density heatmap
    heatmap = np.zeros((grid_size, grid_size))
    for kp in keypoints:
        x_bin = int(kp[0] / w * grid_size)
        y_bin = int(kp[1] / h * grid_size)
        x_bin = min(max(x_bin, 0), grid_size - 1)
        y_bin = min(max(y_bin, 0), grid_size - 1)
        heatmap[y_bin, x_bin] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original image with keypoints
    axes[0].imshow(img_rgb)
    axes[0].scatter(keypoints[:, 0], keypoints[:, 1], c="red", s=5, alpha=0.5)
    axes[0].set_title(f"Keypoints ({len(keypoints)} total)")
    axes[0].axis("off")

    # Density heatmap
    im = axes[1].imshow(heatmap, cmap="hot", interpolation="nearest")
    axes[1].set_title("Keypoint Density Heatmap")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Count")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_feature_maps(
    extractor,
    image_path: Path,
    num_channels: int = 16,
    save_path: Optional[Path] = None,
):
    """
    Visualize ViT feature map activations.

    Args:
        extractor: ViTExtractor instance
        image_path: Path to image
        num_channels: Number of feature channels to visualize
        save_path: Optional path to save figure
    """
    import torch

    # Load and preprocess image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # Resize to valid dimensions
    h_new = (h_orig // extractor.patch_size) * extractor.patch_size
    w_new = (w_orig // extractor.patch_size) * extractor.patch_size
    if h_new != h_orig or w_new != w_orig:
        img_rgb = cv2.resize(img_rgb, (w_new, h_new))

    # Get feature map
    image_tensor = extractor.transform(img_rgb).unsqueeze(0).to(extractor.device)

    with torch.no_grad():
        features = extractor.model.forward_features(image_tensor)

        if isinstance(features, dict):
            patch_features = features["x_norm_patchtokens"]
        else:
            patch_features = features[:, 1:, :]

    # Reshape to spatial
    h_patches = image_tensor.shape[2] // extractor.patch_size
    w_patches = image_tensor.shape[3] // extractor.patch_size
    feature_dim = patch_features.shape[-1]

    feature_map = patch_features.reshape(1, h_patches, w_patches, feature_dim)
    feature_map = feature_map.permute(0, 3, 1, 2).squeeze(0)  # (C, H, W)
    feature_map = feature_map.cpu().numpy()

    # Visualize first N channels
    n_rows = 4
    n_cols = min(num_channels, 16) // n_rows

    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(3 * n_cols, 3 * (n_rows + 1)))

    # Show original image in first row
    for j in range(n_cols):
        if j == 0:
            axes[0, j].imshow(img_rgb)
            axes[0, j].set_title("Original Image")
        axes[0, j].axis("off")

    # Show feature maps
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < min(num_channels, feature_map.shape[0]):
                _ = axes[i + 1, j].imshow(feature_map[idx], cmap="viridis")
                axes[i + 1, j].set_title(f"Channel {idx}")
                axes[i + 1, j].axis("off")
            else:
                axes[i + 1, j].axis("off")

    plt.suptitle(
        f"ViT Feature Maps ({feature_dim}D features, {h_patches}×{w_patches} grid)",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_extractors(
    image_path: Path,
    vit_extractor,
    sift_keypoints: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
):
    """
    Compare ViT and SIFT keypoint distributions.

    Args:
        image_path: Path to image
        vit_extractor: ViTExtractor instance
        sift_keypoints: Optional SIFT keypoints for comparison
        save_path: Optional path to save figure
    """
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extract ViT features
    vit_kp, vit_desc = vit_extractor._run_inference(img)

    # Extract SIFT features if not provided
    if sift_keypoints is None:
        sift = cv2.SIFT_create(nfeatures=len(vit_kp))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        sift_keypoints = np.array([k.pt for k in kp])

    # Create comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ViT keypoints
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].scatter(vit_kp[:, 0], vit_kp[:, 1], c="red", s=10, alpha=0.5)
    axes[0, 0].set_title(f"ViT Keypoints ({len(vit_kp)} points)")
    axes[0, 0].axis("off")

    # SIFT keypoints
    axes[0, 1].imshow(img_rgb)
    axes[0, 1].scatter(
        sift_keypoints[:, 0], sift_keypoints[:, 1], c="blue", s=10, alpha=0.5
    )
    axes[0, 1].set_title(f"SIFT Keypoints ({len(sift_keypoints)} points)")
    axes[0, 1].axis("off")

    # ViT descriptor distribution
    vit_desc_float = vit_desc.astype(np.float32) / 512.0
    vit_desc_float = vit_desc_float / np.linalg.norm(
        vit_desc_float, axis=1, keepdims=True
    )
    vit_norms = np.linalg.norm(vit_desc_float, axis=1)

    # Check if there's variation in norms
    if vit_norms.std() < 1e-6:
        # Perfect normalization - all values are ~1.0
        # Show as a vertical line instead of histogram
        axes[1, 0].axvline(
            vit_norms.mean(),
            color="red",
            linewidth=3,
            label=f"Mean: {vit_norms.mean():.6f}",
        )
        axes[1, 0].set_xlim([0.99, 1.01])
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].text(
            vit_norms.mean(),
            0.5,
            f"All norms ≈ {vit_norms.mean():.3f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
    else:
        # Normal histogram for varying norms
        axes[1, 0].hist(vit_norms, bins=50, color="red", alpha=0.7, edgecolor="black")

    axes[1, 0].axvline(
        1.0, color="black", linestyle="--", linewidth=2, label="Expected (1.0)"
    )
    axes[1, 0].set_xlabel("Descriptor L2 Norm")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(
        f"ViT Descriptor Norms\nMean: {vit_norms.mean():.3f}, Std: {vit_norms.std():.3f}"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Spatial distribution comparison
    h, w = img_rgb.shape[:2]
    grid_size = 20

    # ViT density
    vit_heatmap = np.zeros((grid_size, grid_size))
    for kp in vit_kp:
        x_bin = int(kp[0] / w * grid_size)
        y_bin = int(kp[1] / h * grid_size)
        x_bin = min(max(x_bin, 0), grid_size - 1)
        y_bin = min(max(y_bin, 0), grid_size - 1)
        vit_heatmap[y_bin, x_bin] += 1

    # SIFT density
    sift_heatmap = np.zeros((grid_size, grid_size))
    for kp in sift_keypoints:
        x_bin = int(kp[0] / w * grid_size)
        y_bin = int(kp[1] / h * grid_size)
        x_bin = min(max(x_bin, 0), grid_size - 1)
        y_bin = min(max(y_bin, 0), grid_size - 1)
        sift_heatmap[y_bin, x_bin] += 1

    # Show difference
    diff = vit_heatmap - sift_heatmap
    im = axes[1, 1].imshow(diff, cmap="RdBu_r", interpolation="nearest")
    axes[1, 1].set_title(
        "Density Difference (ViT - SIFT)\nRed: More ViT, Blue: More SIFT"
    )
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_descriptor_quality(
    descriptors: np.ndarray,
    title: str = "Descriptor Analysis",
    save_path: Optional[Path] = None,
):
    """
    Detailed analysis of descriptor quality.

    Args:
        descriptors: (N, D) descriptor array
        title: Plot title
        save_path: Optional path to save figure
    """
    # Convert to float
    desc_float = descriptors.astype(np.float32) / 512.0
    desc_float = desc_float / np.linalg.norm(
        desc_float, axis=1, keepdims=True
    )  # Re-normalize

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. L2 norms
    norms = np.linalg.norm(desc_float, axis=1)

    # Check if there's variation in norms
    if norms.std() < 1e-6:
        # Perfect normalization - show as vertical line
        axes[0, 0].axvline(
            norms.mean(), color="blue", linewidth=3, label=f"Mean: {norms.mean():.6f}"
        )
        axes[0, 0].set_xlim([0.99, 1.01])
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].text(
            norms.mean(),
            0.5,
            f"All norms ≈ {norms.mean():.3f}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
    else:
        # Normal histogram
        axes[0, 0].hist(norms, bins=50, edgecolor="black", alpha=0.7)

    axes[0, 0].axvline(
        1.0, color="red", linestyle="--", linewidth=2, label="Expected (1.0)"
    )
    axes[0, 0].set_xlabel("L2 Norm")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(
        f"Descriptor Norms\nMean: {norms.mean():.3f}, Std: {norms.std():.3f}"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Value distribution
    axes[0, 1].hist(desc_float.flatten(), bins=100, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Descriptor Value")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(
        f"Value Distribution\nMean: {desc_float.mean():.3f}, Std: {desc_float.std():.3f}"
    )
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Dimension-wise statistics
    dim_means = desc_float.mean(axis=0)
    dim_stds = desc_float.std(axis=0)
    dims = np.arange(len(dim_means))

    axes[1, 0].plot(dims, dim_means, label="Mean", alpha=0.7)
    axes[1, 0].fill_between(
        dims, dim_means - dim_stds, dim_means + dim_stds, alpha=0.3, label="±1 Std"
    )
    axes[1, 0].set_xlabel("Dimension")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Per-Dimension Statistics")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Pairwise similarity (sample)
    sample_size = min(100, len(desc_float))
    sample_indices = np.random.choice(len(desc_float), sample_size, replace=False)
    sample_desc = desc_float[sample_indices]

    similarity = sample_desc @ sample_desc.T
    im = axes[1, 1].imshow(similarity, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_title(f"Pairwise Similarity ({sample_size} samples)")
    plt.colorbar(im, ax=axes[1, 1])

    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def create_full_visualization_suite(
    image_path: Path, output_dir: Path = Path("outputs/visualizations")
):
    """
    Create complete visualization suite for an image.

    Args:
        image_path: Path to test image
        output_dir: Directory to save visualizations
    """
    from vit_colmap.features.vit_extractor import ViTExtractor

    print("\n" + "=" * 60)
    print("Creating Visualization Suite")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    print("\n1. Initializing ViT extractor...")
    extractor = ViTExtractor(
        model_name="dinov2_vitb14", num_keypoints=2048, descriptor_dim=128, device="cpu"
    )

    # Load image and extract features
    print("\n2. Extracting features...")
    img = cv2.imread(str(image_path))
    keypoints, descriptors = extractor._run_inference(img)
    print(f"   → Extracted {len(keypoints)} keypoints")

    # 1. Basic keypoints visualization
    print("\n3. Creating keypoint visualization...")
    visualize_keypoints(
        image_path,
        keypoints,
        title="ViT Keypoints",
        save_path=output_dir / "1_keypoints.png",
    )

    # 2. Keypoint density heatmap
    print("\n4. Creating density heatmap...")
    visualize_keypoint_density(
        image_path, keypoints, save_path=output_dir / "2_density_heatmap.png"
    )

    # 3. Feature maps
    print("\n5. Visualizing feature maps...")
    visualize_feature_maps(
        extractor,
        image_path,
        num_channels=16,
        save_path=output_dir / "3_feature_maps.png",
    )

    # 4. Comparison with SIFT
    print("\n6. Comparing with SIFT...")
    compare_extractors(
        image_path, extractor, save_path=output_dir / "4_vit_vs_sift.png"
    )

    # 5. Descriptor analysis
    print("\n7. Analyzing descriptor quality...")
    analyze_descriptor_quality(
        descriptors,
        title="ViT Descriptor Quality Analysis",
        save_path=output_dir / "5_descriptor_analysis.png",
    )

    print("\n" + "=" * 60)
    print("✓ Visualization suite complete!")
    print(f"Check {output_dir} for all visualizations")
    print("=" * 60)


def main():
    """Main visualization demo."""
    print("\n" + "=" * 60)
    print("ViT Feature Visualization Tool")
    print("=" * 60)

    # Create or find a test image
    test_image = Path("data/raw/test_image.png")

    if not test_image.exists():
        print(f"\nTest image not found at {test_image}")
        print("Creating synthetic test image...")
        test_image.parent.mkdir(parents=True, exist_ok=True)

        # Create a more interesting synthetic image
        img = np.zeros((476, 644, 3), dtype=np.uint8)

        # Add some structure
        for i in range(0, 476, 40):
            cv2.line(img, (0, i), (644, i), (100, 100, 100), 2)
        for j in range(0, 644, 40):
            cv2.line(img, (j, 0), (j, 476), (100, 100, 100), 2)

        # Add some circles
        for _ in range(10):
            center = (np.random.randint(50, 594), np.random.randint(50, 426))
            radius = np.random.randint(20, 60)
            color = tuple(np.random.randint(100, 255, 3).tolist())
            cv2.circle(img, center, radius, color, -1)

        cv2.imwrite(str(test_image), img)
        print(f"✓ Created test image at {test_image}")

    # Create visualizations
    create_full_visualization_suite(test_image)

    print("\n✓ Done! Check outputs/visualizations/ for all images")


if __name__ == "__main__":
    main()
