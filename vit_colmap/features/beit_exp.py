from transformers import AutoImageProcessor, BeitModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import cv2
from scipy.stats import pearsonr, kurtosis

IMAGE_PATH = (
    "/home/randyjhc/Github/vit-colmap/data/raw/DTU/Cleaned/scan6/clean_001_3_r5000.png"
)

# 1. Load model and processor
processor = AutoImageProcessor.from_pretrained(
    "microsoft/beit-base-patch16-224-pt22k-ft22k"
)
model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

# 2. Prepare image
img = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=img, return_tensors="pt")

# 3. Forward pass, extract hidden states
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# outputs.hidden_states: list of tensors [layer0 ... layer12]
# Select a mid layer (e.g., 8)
feat = outputs.hidden_states[2]  # shape: (B, 197, 768)
patch_tokens = feat[:, 1:, :]  # drop [CLS]
B, N, D = patch_tokens.shape
H = W = int(N**0.5)
feat_map = patch_tokens.reshape(B, H, W, D)
feat_map = torch.nn.functional.normalize(feat_map, dim=-1)

print(f"feat_map.shape={feat_map.shape}")


# Visualization functions
def visualize_pca_rgb(feature_map, save_path=None):
    """
    Visualize high-dimensional feature map as RGB using PCA.

    Args:
        feature_map: torch.Tensor of shape (B, H, W, D)
        save_path: Optional path to save the visualization
    """
    B, H, W, D = feature_map.shape

    # Reshape to (N, D) for PCA
    feat_2d = feature_map[0].reshape(-1, D).cpu().numpy()

    # Apply PCA to reduce to 3 components (RGB)
    pca = PCA(n_components=3)
    feat_rgb = pca.fit_transform(feat_2d).reshape(H, W, 3)

    # Normalize to [0, 1] for display
    feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min())

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(feat_rgb)
    plt.title(
        f"PCA Feature Map (RGB)\nExplained variance: {pca.explained_variance_ratio_.sum():.2%}"
    )
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved PCA visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_feature_channels(feature_map, num_channels=12, save_path=None):
    """
    Visualize individual feature channels in a grid.

    Args:
        feature_map: torch.Tensor of shape (B, H, W, D)
        num_channels: Number of channels to display (default: 12)
        save_path: Optional path to save the visualization
    """
    B, H, W, D = feature_map.shape
    feat_np = feature_map[0].cpu().numpy()

    # Create grid
    n_rows = 3
    n_cols = 4
    num_channels = min(num_channels, n_rows * n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 9))

    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            ax.imshow(feat_np[:, :, i], cmap="viridis")
            ax.set_title(f"Channel {i}", fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Feature Map Channels (shape: {H}×{W}×{D})", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved channel visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def compute_edge_map(image_path, target_size=(14, 14)):
    """
    Compute edge map from image using Sobel edge detector.

    Args:
        image_path: Path to image file
        target_size: Size to resize edge map to match feature map

    Returns:
        Edge map as numpy array normalized to [0, 1]
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute Sobel edges in X and Y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute edge magnitude
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to [0, 1]
    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (
        edge_magnitude.max() - edge_magnitude.min()
    )

    # Resize to match feature map resolution
    edge_map_resized = cv2.resize(
        edge_magnitude, target_size, interpolation=cv2.INTER_LINEAR
    )

    return edge_map_resized, edge_magnitude


def find_edge_channels(feature_map, image_path, top_k=12):
    """
    Find feature channels that best correlate with edges.

    Args:
        feature_map: torch.Tensor of shape (B, H, W, D)
        image_path: Path to image file
        top_k: Number of top edge-detecting channels to return

    Returns:
        List of tuples: [(channel_idx, correlation_score), ...]
    """
    B, H, W, D = feature_map.shape

    # Compute edge map at feature resolution
    edge_map, edge_map_full = compute_edge_map(image_path, target_size=(W, H))

    # Convert feature map to numpy
    feat_np = feature_map[0].cpu().numpy()

    # Compute correlation for each channel
    correlations = []
    for ch in range(D):
        channel_map = feat_np[:, :, ch].flatten()
        edge_flat = edge_map.flatten()

        # Compute Pearson correlation
        corr, _ = pearsonr(channel_map, edge_flat)
        correlations.append(
            (ch, abs(corr))
        )  # Use absolute value to catch both positive and negative correlations

    # Sort by correlation (descending)
    correlations.sort(key=lambda x: x[1], reverse=True)

    return correlations[:top_k], edge_map, edge_map_full


def visualize_edge_channels(feature_map, image_path, top_k=12, save_path=None):
    """
    Visualize top edge-detecting channels alongside original image and edge map.

    Args:
        feature_map: torch.Tensor of shape (B, H, W, D)
        image_path: Path to image file
        top_k: Number of top channels to visualize
        save_path: Optional path to save visualization
    """
    # Find edge channels
    top_channels, edge_map, edge_map_full = find_edge_channels(
        feature_map, image_path, top_k
    )

    B, H, W, D = feature_map.shape
    feat_np = feature_map[0].cpu().numpy()

    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure
    n_cols = 4
    n_rows = (top_k + 2) // n_cols + 1  # +1 for original and edge map row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = axes.flatten()

    # Row 1: Original image and edge map
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(edge_map_full, cmap="hot")
    axes[1].set_title("Edge Map (Sobel)", fontsize=10)
    axes[1].axis("off")

    # Hide unused slots in first row
    for i in range(2, n_cols):
        axes[i].axis("off")

    # Remaining rows: Top edge-detecting channels
    for i, (ch_idx, corr) in enumerate(top_channels):
        ax_idx = n_cols + i
        if ax_idx < len(axes):
            axes[ax_idx].imshow(feat_np[:, :, ch_idx], cmap="viridis")
            axes[ax_idx].set_title(f"Ch {ch_idx}\nCorr: {corr:.3f}", fontsize=9)
            axes[ax_idx].axis("off")

    # Hide any remaining unused axes
    for i in range(n_cols + top_k, len(axes)):
        axes[i].axis("off")

    plt.suptitle(f"Top {top_k} Edge-Detecting Channels (out of {D} total)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved edge channel visualization to {save_path}")

        # Also print top channels to console
        print("\nTop edge-detecting channels:")
        for i, (ch_idx, corr) in enumerate(top_channels[:5]):
            print(f"  {i+1}. Channel {ch_idx}: correlation = {corr:.4f}")
    else:
        plt.show()

    plt.close()

    return top_channels


def compute_keypoint_metrics(channel_map):
    """
    Compute multiple metrics that indicate keypoint suitability for a single channel.

    Args:
        channel_map: 2D numpy array (H, W)

    Returns:
        Dictionary of metric scores
    """
    # 1. Local variance (Harris-like response)
    # High local variance indicates corners and distinctive features
    dx = cv2.Sobel(channel_map, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(channel_map, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude variance
    grad_mag = np.sqrt(dx**2 + dy**2)
    gradient_variance = np.var(grad_mag)

    # 2. Spatial entropy (information content)
    # Normalize to [0, 255] for histogram
    channel_norm = (
        (channel_map - channel_map.min())
        / (channel_map.max() - channel_map.min() + 1e-8)
        * 255
    ).astype(np.uint8)
    hist, _ = np.histogram(channel_norm, bins=32, range=(0, 256))
    hist = hist / hist.sum()  # Normalize to probability
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))

    # 3. Sparsity (peakedness - high kurtosis means sparse, localized activations)
    kurt = kurtosis(channel_map.flatten())

    # 4. Overall variance (discriminative power)
    variance = np.var(channel_map)

    # 5. Peak prominence (how strong are the local maxima)
    mean_val = channel_map.mean()
    std_val = channel_map.std()
    peaks = channel_map > (mean_val + 2 * std_val)  # Pixels > 2 std above mean
    peak_ratio = peaks.sum() / channel_map.size

    return {
        "gradient_variance": gradient_variance,
        "entropy": entropy,
        "kurtosis": kurt,
        "variance": variance,
        "peak_ratio": peak_ratio,
    }


def find_keypoint_channels(feature_map, top_k=12):
    """
    Find feature channels most suitable for keypoint detection.

    Args:
        feature_map: torch.Tensor of shape (B, H, W, D)
        top_k: Number of top channels to return

    Returns:
        List of tuples: [(channel_idx, score, metrics), ...]
    """
    B, H, W, D = feature_map.shape
    feat_np = feature_map[0].cpu().numpy()

    channel_scores = []

    for ch in range(D):
        channel_map = feat_np[:, :, ch]

        # Compute metrics
        metrics = compute_keypoint_metrics(channel_map)

        # Combined score (weighted combination)
        # Higher weights for features important for keypoints
        score = (
            0.3
            * metrics["gradient_variance"]
            / (1 + metrics["gradient_variance"])  # Normalized gradient variance
            + 0.2 * metrics["entropy"] / 5.0  # Normalized entropy (max ~5 for 32 bins)
            + 0.2
            * (
                metrics["kurtosis"] / (1 + abs(metrics["kurtosis"]))
            )  # Normalized kurtosis
            + 0.2
            * metrics["variance"]
            / (1 + metrics["variance"])  # Normalized variance
            + 0.1 * metrics["peak_ratio"] * 10  # Peak ratio (usually small)
        )

        channel_scores.append((ch, score, metrics))

    # Sort by score (descending)
    channel_scores.sort(key=lambda x: x[1], reverse=True)

    return channel_scores[:top_k]


def visualize_keypoint_channels(feature_map, image_path, top_k=12, save_path=None):
    """
    Visualize top keypoint-suitable channels alongside original image.

    Args:
        feature_map: torch.Tensor of shape (B, H, W, D)
        image_path: Path to image file
        top_k: Number of top channels to visualize
        save_path: Optional path to save visualization
    """
    # Find keypoint channels
    top_channels = find_keypoint_channels(feature_map, top_k)

    B, H, W, D = feature_map.shape
    feat_np = feature_map[0].cpu().numpy()

    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Compute Harris corners for reference
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    harris_normalized = (harris - harris.min()) / (harris.max() - harris.min())
    harris_resized = cv2.resize(
        harris_normalized, (W, H), interpolation=cv2.INTER_LINEAR
    )

    # Create figure
    n_cols = 4
    n_rows = (top_k + 2) // n_cols + 1  # +1 for original and Harris row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = axes.flatten()

    # Row 1: Original image and Harris corners
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(harris_resized, cmap="hot")
    axes[1].set_title("Harris Corners", fontsize=10)
    axes[1].axis("off")

    # Hide unused slots in first row
    for i in range(2, n_cols):
        axes[i].axis("off")

    # Remaining rows: Top keypoint-suitable channels
    for i, (ch_idx, score, metrics) in enumerate(top_channels):
        ax_idx = n_cols + i
        if ax_idx < len(axes):
            axes[ax_idx].imshow(feat_np[:, :, ch_idx], cmap="viridis")
            axes[ax_idx].set_title(
                f'Ch {ch_idx}\nScore: {score:.3f}\nVar: {metrics["variance"]:.2f}',
                fontsize=8,
            )
            axes[ax_idx].axis("off")

    # Hide any remaining unused axes
    for i in range(n_cols + top_k, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        f"Top {top_k} Keypoint-Suitable Channels (out of {D} total)", fontsize=14
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved keypoint channel visualization to {save_path}")

        # Print top channels to console
        print("\nTop keypoint-suitable channels:")
        for i, (ch_idx, score, metrics) in enumerate(top_channels[:5]):
            print(f"  {i+1}. Channel {ch_idx}:")
            print(f"      Score: {score:.4f}")
            print(f"      Gradient Var: {metrics['gradient_variance']:.4f}")
            print(f"      Entropy: {metrics['entropy']:.4f}")
            print(f"      Kurtosis: {metrics['kurtosis']:.4f}")
            print(f"      Variance: {metrics['variance']:.4f}")
    else:
        plt.show()

    plt.close()

    return top_channels


def extract_all_layer_features(model, inputs, img_size=224, patch_size=16):
    """
    Extract feature maps from all layers of BEiT model.

    Args:
        model: BEiT model
        inputs: Preprocessed input tensors
        img_size: Input image size
        patch_size: Patch size

    Returns:
        List of feature maps, one per layer
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Calculate spatial dimensions
    H = W = img_size // patch_size

    layer_features = []
    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        # hidden_state shape: (B, 197, 768) for BEiT-base
        patch_tokens = hidden_state[:, 1:, :]  # Drop [CLS] token
        B, N, D = patch_tokens.shape

        # Reshape to spatial grid
        feat_map = patch_tokens.reshape(B, H, W, D)
        # Normalize
        feat_map = torch.nn.functional.normalize(feat_map, dim=-1)

        layer_features.append(feat_map)

    return layer_features


def visualize_all_layers_pca(layer_features, image_path, save_path=None):
    """
    Visualize PCA for all layers in a grid.

    Args:
        layer_features: List of feature maps from all layers
        image_path: Path to original image
        save_path: Optional path to save visualization
    """
    num_layers = len(layer_features)

    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create grid: 3 rows of layers + 1 row for original
    n_cols = 5
    n_rows = (num_layers + n_cols - 1) // n_cols + 1  # +1 for original image row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    # First subplot: original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Hide unused slots in first row
    for i in range(1, n_cols):
        axes[i].axis("off")

    # Process each layer
    for layer_idx, feat_map in enumerate(layer_features):
        B, H, W, D = feat_map.shape

        # Reshape for PCA
        feat_2d = feat_map[0].reshape(-1, D).cpu().numpy()

        # Apply PCA
        pca = PCA(n_components=3)
        feat_rgb = pca.fit_transform(feat_2d).reshape(H, W, 3)

        # Normalize to [0, 1]
        feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min())

        # Plot
        ax_idx = n_cols + layer_idx
        axes[ax_idx].imshow(feat_rgb)
        explained_var = pca.explained_variance_ratio_.sum()
        axes[ax_idx].set_title(
            f"Layer {layer_idx}\nVar: {explained_var:.1%}", fontsize=10
        )
        axes[ax_idx].axis("off")

    # Hide any remaining unused axes
    for i in range(n_cols + num_layers, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        "PCA Visualization Across All BEiT Layers", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved all-layers PCA visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_layer_quality(layer_features):
    """
    Quantitative analysis of PCA quality for each layer.

    Args:
        layer_features: List of feature maps from all layers

    Returns:
        Dictionary with metrics for each layer
    """
    layer_metrics = []

    for layer_idx, feat_map in enumerate(layer_features):
        B, H, W, D = feat_map.shape
        feat_2d = feat_map[0].reshape(-1, D).cpu().numpy()

        # 1. Explained variance ratio
        pca = PCA(n_components=3)
        feat_rgb = pca.fit_transform(feat_2d).reshape(H, W, 3)
        explained_var = pca.explained_variance_ratio_.sum()

        # Normalize to [0, 1] for metrics
        feat_rgb_norm = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min())

        # 2. Color diversity (variance in RGB space)
        color_diversity = np.mean([np.var(feat_rgb_norm[:, :, c]) for c in range(3)])

        # 3. Spatial coherence (inverse of gradient magnitude - smoother is more coherent)
        grad_x = np.gradient(feat_rgb_norm, axis=0)
        grad_y = np.gradient(feat_rgb_norm, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2).mean()
        spatial_coherence = 1.0 / (1.0 + grad_mag)  # Normalize

        # 4. Feature variance in original space
        feature_variance = np.var(feat_2d)

        layer_metrics.append(
            {
                "layer": layer_idx,
                "explained_variance": explained_var,
                "color_diversity": color_diversity,
                "spatial_coherence": spatial_coherence,
                "feature_variance": feature_variance,
                "pca_components": pca.explained_variance_ratio_,
            }
        )

    return layer_metrics


def visualize_layer_analysis(layer_metrics, save_path=None):
    """
    Visualize quantitative metrics across layers.

    Args:
        layer_metrics: List of metric dictionaries from analyze_layer_quality
        save_path: Optional path to save visualization
    """
    layers = [m["layer"] for m in layer_metrics]
    explained_var = [m["explained_variance"] for m in layer_metrics]
    color_div = [m["color_diversity"] for m in layer_metrics]
    spatial_coh = [m["spatial_coherence"] for m in layer_metrics]
    # feat_var = [m["feature_variance"] for m in layer_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Explained variance
    axes[0, 0].plot(layers, explained_var, "o-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Layer", fontsize=11)
    axes[0, 0].set_ylabel("Explained Variance Ratio", fontsize=11)
    axes[0, 0].set_title(
        "PCA Explained Variance by Layer\n(Higher = More info in 3D)", fontsize=12
    )
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(layers)

    # Mark best layer
    best_idx = np.argmax(explained_var)
    axes[0, 0].scatter(
        layers[best_idx],
        explained_var[best_idx],
        color="red",
        s=200,
        marker="*",
        zorder=5,
        label=f"Best: Layer {layers[best_idx]}",
    )
    axes[0, 0].legend()

    # 2. Color diversity
    axes[0, 1].plot(layers, color_div, "o-", linewidth=2, markersize=8, color="green")
    axes[0, 1].set_xlabel("Layer", fontsize=11)
    axes[0, 1].set_ylabel("Color Diversity", fontsize=11)
    axes[0, 1].set_title(
        "PCA Color Diversity by Layer\n(Higher = More varied colors)", fontsize=12
    )
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(layers)

    best_idx = np.argmax(color_div)
    axes[0, 1].scatter(
        layers[best_idx],
        color_div[best_idx],
        color="red",
        s=200,
        marker="*",
        zorder=5,
        label=f"Best: Layer {layers[best_idx]}",
    )
    axes[0, 1].legend()

    # 3. Spatial coherence
    axes[1, 0].plot(
        layers, spatial_coh, "o-", linewidth=2, markersize=8, color="purple"
    )
    axes[1, 0].set_xlabel("Layer", fontsize=11)
    axes[1, 0].set_ylabel("Spatial Coherence", fontsize=11)
    axes[1, 0].set_title(
        "Spatial Coherence by Layer\n(Higher = Smoother regions)", fontsize=12
    )
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(layers)

    best_idx = np.argmax(spatial_coh)
    axes[1, 0].scatter(
        layers[best_idx],
        spatial_coh[best_idx],
        color="red",
        s=200,
        marker="*",
        zorder=5,
        label=f"Best: Layer {layers[best_idx]}",
    )
    axes[1, 0].legend()

    # 4. Combined ranking
    # Normalize all metrics to [0, 1]
    explained_var_norm = (np.array(explained_var) - np.min(explained_var)) / (
        np.max(explained_var) - np.min(explained_var) + 1e-8
    )
    color_div_norm = (np.array(color_div) - np.min(color_div)) / (
        np.max(color_div) - np.min(color_div) + 1e-8
    )
    spatial_coh_norm = (np.array(spatial_coh) - np.min(spatial_coh)) / (
        np.max(spatial_coh) - np.min(spatial_coh) + 1e-8
    )

    # Combined score (weighted average)
    combined_score = (
        0.5 * explained_var_norm + 0.3 * color_div_norm + 0.2 * spatial_coh_norm
    )

    axes[1, 1].plot(
        layers, combined_score, "o-", linewidth=2, markersize=8, color="orange"
    )
    axes[1, 1].set_xlabel("Layer", fontsize=11)
    axes[1, 1].set_ylabel("Combined Score", fontsize=11)
    axes[1, 1].set_title(
        "Combined Quality Score by Layer\n(50% ExpVar + 30% ColorDiv + 20% SpatCoh)",
        fontsize=12,
    )
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(layers)

    best_idx = np.argmax(combined_score)
    axes[1, 1].scatter(
        layers[best_idx],
        combined_score[best_idx],
        color="red",
        s=200,
        marker="*",
        zorder=5,
        label=f"Best: Layer {layers[best_idx]}",
    )
    axes[1, 1].legend()

    plt.suptitle("Layer-wise PCA Quality Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved layer analysis to {save_path}")
    else:
        plt.show()

    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("LAYER QUALITY SUMMARY")
    print("=" * 60)
    for i, (layer, score) in enumerate(zip(layers, combined_score)):
        star = " ⭐ BEST" if i == np.argmax(combined_score) else ""
        print(
            f"Layer {layer:2d}: Score={score:.3f} | ExpVar={explained_var[i]:.3f} | "
            f"ColorDiv={color_div[i]:.4f} | SpatCoh={spatial_coh[i]:.3f}{star}"
        )
    print("=" * 60)

    return layers[np.argmax(combined_score)]


def visualize_top_layers_detailed(layer_features, image_path, top_k=4, save_path=None):
    """
    Detailed comparison of top-k layers based on quality metrics.

    Args:
        layer_features: List of feature maps from all layers
        image_path: Path to original image
        top_k: Number of top layers to show
        save_path: Optional path to save visualization
    """
    # Analyze all layers
    layer_metrics = analyze_layer_quality(layer_features)

    # Calculate combined scores
    explained_var = np.array([m["explained_variance"] for m in layer_metrics])
    color_div = np.array([m["color_diversity"] for m in layer_metrics])
    spatial_coh = np.array([m["spatial_coherence"] for m in layer_metrics])

    # Normalize and combine
    explained_var_norm = (explained_var - explained_var.min()) / (
        explained_var.max() - explained_var.min() + 1e-8
    )
    color_div_norm = (color_div - color_div.min()) / (
        color_div.max() - color_div.min() + 1e-8
    )
    spatial_coh_norm = (spatial_coh - spatial_coh.min()) / (
        spatial_coh.max() - spatial_coh.min() + 1e-8
    )

    combined_score = (
        0.5 * explained_var_norm + 0.3 * color_div_norm + 0.2 * spatial_coh_norm
    )

    # Get top-k layers
    top_indices = np.argsort(combined_score)[-top_k:][::-1]

    # Load original image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, axes = plt.subplots(1, top_k + 1, figsize=(5 * (top_k + 1), 5))

    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Top layers
    for i, layer_idx in enumerate(top_indices):
        feat_map = layer_features[layer_idx]
        B, H, W, D = feat_map.shape

        # PCA
        feat_2d = feat_map[0].reshape(-1, D).cpu().numpy()
        pca = PCA(n_components=3)
        feat_rgb = pca.fit_transform(feat_2d).reshape(H, W, 3)
        feat_rgb = (feat_rgb - feat_rgb.min()) / (feat_rgb.max() - feat_rgb.min())

        # Plot
        axes[i + 1].imshow(feat_rgb)
        axes[i + 1].set_title(
            f"Layer {layer_idx} (Rank #{i+1})\n"
            f"Score: {combined_score[layer_idx]:.3f}\n"
            f"ExpVar: {explained_var[layer_idx]:.2%}",
            fontsize=11,
        )
        axes[i + 1].axis("off")

    plt.suptitle(
        f"Top {top_k} Layers for PCA Visualization", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved top layers visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# Execute visualizations
if __name__ == "__main__":
    print(f"Feature map shape: {feat_map.shape}")
    print(f"Feature dimension: {D}")
    print(f"Spatial resolution: {H}×{W}")

    # Create output directory
    import os

    output_dir = "outputs/beit_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # === SINGLE LAYER ANALYSIS (current layer) ===
    print("\n" + "=" * 60)
    print("SINGLE LAYER ANALYSIS (Layer 2)")
    print("=" * 60)

    # Visualize using PCA (RGB)
    print("\nCreating PCA RGB visualization...")
    visualize_pca_rgb(feat_map, save_path=f"{output_dir}/pca_rgb.png")

    # Visualize individual channels
    print("Creating channel grid visualization...")
    visualize_feature_channels(
        feat_map, num_channels=12, save_path=f"{output_dir}/channels.png"
    )

    # Find and visualize edge-detecting channels
    print("\nAnalyzing edge-detecting channels...")
    edge_channels = visualize_edge_channels(
        feat_map, IMAGE_PATH, top_k=12, save_path=f"{output_dir}/edge_channels.png"
    )

    # Find and visualize keypoint-suitable channels
    print("\nAnalyzing keypoint-suitable channels...")
    keypoint_channels = visualize_keypoint_channels(
        feat_map, IMAGE_PATH, top_k=12, save_path=f"{output_dir}/keypoint_channels.png"
    )

    # === MULTI-LAYER COMPARISON ===
    print("\n" + "=" * 60)
    print("MULTI-LAYER COMPARISON (All 13 Layers)")
    print("=" * 60)

    print("\nExtracting features from all layers...")
    layer_features = extract_all_layer_features(
        model, inputs, img_size=224, patch_size=16
    )
    print(f"Extracted {len(layer_features)} layers")

    print("\nCreating all-layers PCA grid visualization...")
    visualize_all_layers_pca(
        layer_features, IMAGE_PATH, save_path=f"{output_dir}/all_layers_pca.png"
    )

    print("\nAnalyzing layer quality metrics...")
    layer_metrics = analyze_layer_quality(layer_features)

    print("\nCreating layer quality analysis plots...")
    best_layer = visualize_layer_analysis(
        layer_metrics, save_path=f"{output_dir}/layer_quality_analysis.png"
    )

    print("\nCreating detailed comparison of top 4 layers...")
    visualize_top_layers_detailed(
        layer_features,
        IMAGE_PATH,
        top_k=4,
        save_path=f"{output_dir}/top_layers_comparison.png",
    )

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nSingle-layer analysis:")
    print("  - pca_rgb.png: PCA-reduced feature visualization (Layer 2)")
    print("  - channels.png: First 12 feature channels (Layer 2)")
    print("  - edge_channels.png: Top edge-detecting channels (Layer 2)")
    print("  - keypoint_channels.png: Top keypoint-suitable channels (Layer 2)")
    print("\nMulti-layer comparison:")
    print("  - all_layers_pca.png: PCA visualization for all 13 layers")
    print("  - layer_quality_analysis.png: Quantitative quality metrics across layers")
    print("  - top_layers_comparison.png: Detailed view of top 4 layers")
    print(f"\n⭐ Recommended best layer: Layer {best_layer}")
    print("=" * 60)
