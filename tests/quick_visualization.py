"""
Quick visualization of test results for ViT feature extraction.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def quick_visualize(image_path=None):
    """Quick visualization of ViT feature extraction."""
    from vit_colmap.features.vit_extractor import ViTExtractor

    print("\n" + "=" * 60)
    print("Quick ViT Feature Visualization")
    print("=" * 60)

    # Load image
    if image_path is None:
        # Use default test image if none provided
        print("\n1. Creating test image...")
        img = np.zeros((476, 644, 3), dtype=np.uint8)
        # Grid pattern
        for i in range(0, 476, 50):
            cv2.line(img, (0, i), (644, i), (80, 80, 80), 1)
        for j in range(0, 644, 50):
            cv2.line(img, (j, 0), (j, 476), (80, 80, 80), 1)

        # Colored circles
        centers = [
            (150, 150),
            (400, 150),
            (500, 300),
            (200, 350),
            (450, 400),
            (300, 250),
        ]
        colors = [
            (255, 100, 100),
            (100, 255, 100),
            (100, 100, 255),
            (255, 255, 100),
            (255, 100, 255),
            (100, 255, 255),
        ]

        for center, color in zip(centers, colors):
            cv2.circle(img, center, 40, color, -1)
            cv2.circle(img, center, 40, (255, 255, 255), 2)

        print("   ✓ Synthetic test image created")
    else:
        print(f"   Loading: {image_path}")
        img = cv2.imread(str(image_path))

        if img is None:
            raise ValueError(f"❌ Could not load image from {image_path}")

        print(f"   ✓ Loaded image: {img.shape[1]}x{img.shape[0]} pixels")

        # Resize to be multiple of 14 (ViT patch size)
        h, w = img.shape[:2]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        if new_h != h or new_w != w:
            img = cv2.resize(img, (new_w, new_h))
            print(f"   ✓ Resized to {new_w}x{new_h} (multiple of 14 for ViT)")

    # Extract features
    print("\n2. Extracting ViT features...")
    extractor = ViTExtractor(
        model_name="dinov2_vitb14", num_keypoints=1000, descriptor_dim=128, device="cpu"
    )

    keypoints, descriptors = extractor._run_inference(img)
    print(f"   ✓ Extracted {len(keypoints)} keypoints")
    print(f"   ✓ Descriptor shape: {descriptors.shape}")

    # Create visualization
    print("\n3. Creating visualization...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 10))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Original image
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(img_rgb)
    ax1.set_title("Original Test Image", fontsize=14, fontweight="bold")
    ax1.axis("off")

    # 2. Keypoints overlay
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.imshow(img_rgb)
    ax2.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        c="red",
        s=30,
        alpha=0.6,
        marker="x",
        linewidths=2,
    )
    ax2.set_title(
        f"Detected Keypoints ({len(keypoints)} points)", fontsize=14, fontweight="bold"
    )
    ax2.axis("off")

    # 3. Keypoint density heatmap
    ax3 = fig.add_subplot(gs[2, :2])
    h, w = img_rgb.shape[:2]
    heatmap = np.zeros((20, 30))
    for kp in keypoints:
        x_bin = int(kp[0] / w * 30)
        y_bin = int(kp[1] / h * 20)
        x_bin = min(max(x_bin, 0), 29)
        y_bin = min(max(y_bin, 0), 19)
        heatmap[y_bin, x_bin] += 1

    im = ax3.imshow(heatmap, cmap="hot", interpolation="bilinear", aspect="auto")
    ax3.set_title("Keypoint Density Heatmap", fontsize=14, fontweight="bold")
    ax3.axis("off")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Keypoint statistics
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.axis("off")
    stats_text = f"""
KEYPOINT STATISTICS

Total Keypoints: {len(keypoints)}

Coordinates:
  X range: [{keypoints[:, 0].min():.1f}, {keypoints[:, 0].max():.1f}]
  Y range: [{keypoints[:, 1].min():.1f}, {keypoints[:, 1].max():.1f}]

  X mean: {keypoints[:, 0].mean():.1f}
  Y mean: {keypoints[:, 1].mean():.1f}

Coverage:
  Width: {w}px
  Height: {h}px
  Coverage: {len(keypoints) / (w * h) * 10000:.2f} per 10k px²
    """
    ax4.text(
        0.1,
        0.5,
        stats_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 5. Descriptor L2 norms
    ax5 = fig.add_subplot(gs[1, 2])
    desc_float = descriptors.astype(np.float32) / 512.0
    desc_float = desc_float / np.linalg.norm(desc_float, axis=1, keepdims=True)
    norms = np.linalg.norm(desc_float, axis=1)

    # Handle perfect normalization case
    if norms.std() < 1e-6:
        # All norms are identical - show as vertical line
        ax5.axvline(
            norms.mean(),
            color="steelblue",
            linewidth=3,
            label=f"Mean: {norms.mean():.6f}",
        )
        ax5.set_xlim([0.99, 1.01])
        ax5.set_ylim([0, 1])
        ax5.text(
            norms.mean(),
            0.5,
            f"All norms ≈ {norms.mean():.3f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
    else:
        # Normal histogram
        ax5.hist(norms, bins=30, edgecolor="black", alpha=0.7, color="steelblue")

    ax5.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Expected")
    ax5.set_xlabel("L2 Norm")
    ax5.set_ylabel("Count")
    ax5.set_title("Descriptor Norms", fontsize=12, fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Descriptor quality
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    desc_stats = f"""
DESCRIPTOR STATISTICS

Shape: {descriptors.shape}
Dtype: {descriptors.dtype}

Normalization:
  Mean L2 norm: {norms.mean():.3f}
  Std L2 norm: {norms.std():.3f}

Value Distribution:
  Min: {descriptors.min()}
  Max: {descriptors.max()}
  Mean: {descriptors.mean():.1f}

Quality: {'✓ Good' if 0.9 < norms.mean() < 1.1 else '⚠ Check'}
    """
    ax6.text(
        0.1,
        0.5,
        desc_stats,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.suptitle(
        "ViT Feature Extraction - Quick Visualization",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save
    output_dir = Path("outputs/quick_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quick_visualization.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved to {output_path}")

    # Also show
    plt.show()

    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)

    return keypoints, descriptors


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Use image path from command line
        quick_visualize(sys.argv[1])
    else:
        # Use default synthetic image
        quick_visualize()
