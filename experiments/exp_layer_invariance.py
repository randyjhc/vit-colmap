"""
Experiment to test layer-wise invariance/equivariance properties of Vision Transformers.

Tests which transformer layers produce features that are invariant/equivariant to:
1. Scale changes (zoom in/out)
2. Rotation
3. Illumination variation
4. Small viewpoint shifts

Usage:
    python experiments/exp_layer_invariance.py --image path/to/image.jpg --model dinov2_vitb14
"""

import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

from invariance_utils import (
    apply_scale_transform,
    apply_rotation_transform,
    apply_illumination_transform,
    apply_viewpoint_transform,
    transform_coordinates,
    cosine_similarity,
    normalized_l2_distance,
    extract_features_at_pixels,
)


# ============================================================================
# Feature Extractor with Layer Hooks
# ============================================================================


class LayerFeatureExtractor:
    """Extract features from all intermediate layers of a Vision Transformer."""

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        """
        Initialize the feature extractor.

        Args:
            model_name: Name of the model to load (e.g., 'dinov2_vitb14', 'beit_base_patch16_224')
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

        # Storage for layer outputs
        self.layer_outputs: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Register hooks
        self._register_hooks()

        # Get model properties
        self.patch_size = self._get_patch_size()
        self.num_layers = len(self.layer_outputs)

        print(f"Loaded {model_name} on {self.device}")
        print(f"Number of transformer layers: {self.num_layers}")
        print(f"Patch size: {self.patch_size}")

    def _load_model(self):
        """Load the vision transformer model."""
        if "dinov2" in self.model_name:
            model = torch.hub.load(
                "facebookresearch/dinov2", self.model_name, pretrained=True
            )
        elif "beit" in self.model_name:
            from transformers import BeitModel

            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model

    def _get_patch_size(self) -> int:
        """Get the patch size of the model."""
        if "dinov2" in self.model_name:
            return 14  # DINOv2 uses 14x14 patches
        elif "beit" in self.model_name:
            return 16  # BEiT uses 16x16 patches
        return 14

    def _register_hooks(self):
        """Register forward hooks to capture intermediate layer outputs."""
        self.layer_outputs.clear()
        self.hooks.clear()

        def make_hook(layer_name):
            def hook(module, input, output):
                # Store the output (typically includes CLS token + patch tokens)
                if isinstance(output, tuple):
                    output = output[0]
                self.layer_outputs[layer_name] = output.detach()

            return hook

        # Hook into transformer blocks
        if "dinov2" in self.model_name:
            # DINOv2 architecture
            for idx, block in enumerate(self.model.blocks):
                layer_name = f"layer_{idx}"
                hook = block.register_forward_hook(make_hook(layer_name))
                self.hooks.append(hook)
        elif "beit" in self.model_name:
            # BEiT architecture
            for idx, layer in enumerate(self.model.encoder.layer):
                layer_name = f"layer_{idx}"
                hook = layer.register_forward_hook(make_hook(layer_name))
                self.hooks.append(hook)

    def extract_features(
        self, image: np.ndarray, pixel_coords: np.ndarray = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from all layers for an image.

        Args:
            image: Input image in BGR format (H, W, 3)
            pixel_coords: Optional (N, 2) array of (x, y) coordinates to extract features at

        Returns:
            Dictionary mapping layer names to feature tensors
            - If pixel_coords is None: {layer_name: (num_patches, feature_dim)}
            - If pixel_coords is provided: {layer_name: (N, feature_dim)}
        """
        # Prepare image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        # Resize to multiple of patch size
        h_new = (h_orig // self.patch_size) * self.patch_size
        w_new = (w_orig // self.patch_size) * self.patch_size
        if h_new != h_orig or w_new != w_orig:
            image_rgb = cv2.resize(
                image_rgb, (w_new, h_new), interpolation=cv2.INTER_LINEAR
            )

        # Convert to tensor
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image_tensor = transform(image_rgb).unsqueeze(0).to(self.device)

        # Clear previous outputs
        self.layer_outputs.clear()

        # Forward pass
        with torch.no_grad():
            _ = self.model(image_tensor)

        # Extract features from each layer
        layer_features = {}
        for layer_name, output in self.layer_outputs.items():
            # Remove CLS token (first token)
            patch_tokens = output[:, 1:, :]  # (1, num_patches, feature_dim)

            if pixel_coords is not None:
                # Extract features at specific pixel locations
                features = extract_features_at_pixels(
                    patch_tokens, pixel_coords, (w_new, h_new), self.patch_size
                )
                layer_features[layer_name] = features
            else:
                # Return all patch features
                layer_features[layer_name] = patch_tokens.squeeze(0)

        return layer_features

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.layer_outputs.clear()


# ============================================================================
# Invariance Testing Functions
# ============================================================================


def test_scale_invariance(
    extractor: LayerFeatureExtractor,
    image: np.ndarray,
    pixel_coords: np.ndarray,
    scale_factors: List[float] = None,
    return_per_pixel: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Test scale invariance across different layers.

    Args:
        extractor: LayerFeatureExtractor instance
        image: Input image in BGR format
        pixel_coords: (N, 2) array of pixel coordinates to test
        scale_factors: List of scale factors to test (default: [0.5, 0.75, 1.25, 1.5, 2.0])
        return_per_pixel: If True, include per-pixel similarity scores in results

    Returns:
        Dictionary mapping layer names to metrics:
        {layer_name: {'mean_cosine_sim': float, 'mean_l2_dist': float, ...}}
        If return_per_pixel=True, also includes 'per_pixel_cosine_sims': (N, num_transformations) array
    """
    if scale_factors is None:
        scale_factors = [0.5, 0.75, 1.25, 1.5, 2.0]

    print(f"\n{'='*80}")
    print("Testing Scale Invariance")
    print(f"{'='*80}")
    print(f"Testing {len(pixel_coords)} pixel locations")
    print(f"Scale factors: {scale_factors}")

    # Extract features from original image
    original_features = extractor.extract_features(image, pixel_coords)

    # Store results
    results: Dict[str, Dict[str, List]] = {
        layer: {"cosine_sims": [], "l2_dists": [], "per_pixel_sims": []}
        for layer in original_features.keys()
    }

    # Test each scale factor
    for scale in scale_factors:
        print(f"\n  Testing scale: {scale:.2f}x")

        # Apply transformation
        transformed_img, transform_matrix = apply_scale_transform(image, scale)

        # Transform coordinates
        transformed_coords = transform_coordinates(pixel_coords, transform_matrix)

        # Extract features from transformed image
        transformed_features = extractor.extract_features(
            transformed_img, transformed_coords
        )

        # Compute metrics for each layer
        for layer_name in original_features.keys():
            orig_feat = original_features[layer_name]
            trans_feat = transformed_features[layer_name]

            # Compute per-pixel similarity metrics
            cos_sim_per_pixel = cosine_similarity(orig_feat, trans_feat)  # (N,) tensor
            l2_dist_per_pixel = normalized_l2_distance(
                orig_feat, trans_feat
            )  # (N,) tensor

            # Store per-pixel scores if requested
            if return_per_pixel:
                results[layer_name]["per_pixel_sims"].append(
                    cos_sim_per_pixel.cpu().numpy()
                )

            # Store averaged metrics
            results[layer_name]["cosine_sims"].append(cos_sim_per_pixel.mean().item())
            results[layer_name]["l2_dists"].append(l2_dist_per_pixel.mean().item())

    # Aggregate results
    aggregated_results = {}
    for layer_name, metrics in results.items():
        aggregated_results[layer_name] = {
            "mean_cosine_sim": np.mean(metrics["cosine_sims"]),
            "std_cosine_sim": np.std(metrics["cosine_sims"]),
            "mean_l2_dist": np.mean(metrics["l2_dists"]),
            "std_l2_dist": np.std(metrics["l2_dists"]),
            "all_cosine_sims": metrics["cosine_sims"],
            "all_l2_dists": metrics["l2_dists"],
        }

        # Add per-pixel data if requested
        if return_per_pixel:
            # Shape: (num_transformations, N) -> transpose to (N, num_transformations)
            per_pixel_array = np.array(metrics["per_pixel_sims"]).T
            aggregated_results[layer_name]["per_pixel_cosine_sims"] = per_pixel_array

    print("\nScale Invariance Results (higher cosine sim = more invariant):")
    for layer_name, metrics in aggregated_results.items():
        print(
            f"  {layer_name}: cos_sim={metrics['mean_cosine_sim']:.4f} (+/- {metrics['std_cosine_sim']:.4f})"
        )

    return aggregated_results


def test_rotation_invariance(
    extractor: LayerFeatureExtractor,
    image: np.ndarray,
    pixel_coords: np.ndarray,
    angles: List[float] = None,
    return_per_pixel: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Test rotation invariance across different layers.

    Args:
        extractor: LayerFeatureExtractor instance
        image: Input image in BGR format
        pixel_coords: (N, 2) array of pixel coordinates to test
        angles: List of rotation angles in degrees (default: [-90, -45, -15, 15, 45, 90])
        return_per_pixel: If True, include per-pixel similarity scores in results

    Returns:
        Dictionary mapping layer names to metrics
    """
    if angles is None:
        angles = [-90, -45, -15, 15, 45, 90]

    print(f"\n{'='*80}")
    print("Testing Rotation Invariance")
    print(f"{'='*80}")
    print(f"Testing {len(pixel_coords)} pixel locations")
    print(f"Rotation angles: {angles}")

    # Extract features from original image
    original_features = extractor.extract_features(image, pixel_coords)

    # Store results
    results: Dict[str, Dict[str, List]] = {
        layer: {"cosine_sims": [], "l2_dists": [], "per_pixel_sims": []}
        for layer in original_features.keys()
    }

    # Test each angle
    for angle in angles:
        print(f"\n  Testing rotation: {angle}°")

        # Apply transformation
        transformed_img, transform_matrix = apply_rotation_transform(image, angle)

        # Transform coordinates
        transformed_coords = transform_coordinates(pixel_coords, transform_matrix)

        # Extract features from transformed image
        transformed_features = extractor.extract_features(
            transformed_img, transformed_coords
        )

        # Compute metrics for each layer
        for layer_name in original_features.keys():
            orig_feat = original_features[layer_name]
            trans_feat = transformed_features[layer_name]

            # Compute per-pixel similarity metrics
            cos_sim_per_pixel = cosine_similarity(orig_feat, trans_feat)
            l2_dist_per_pixel = normalized_l2_distance(orig_feat, trans_feat)

            # Store per-pixel scores if requested
            if return_per_pixel:
                results[layer_name]["per_pixel_sims"].append(
                    cos_sim_per_pixel.cpu().numpy()
                )

            # Store averaged metrics
            results[layer_name]["cosine_sims"].append(cos_sim_per_pixel.mean().item())
            results[layer_name]["l2_dists"].append(l2_dist_per_pixel.mean().item())

    # Aggregate results
    aggregated_results = {}
    for layer_name, metrics in results.items():
        aggregated_results[layer_name] = {
            "mean_cosine_sim": np.mean(metrics["cosine_sims"]),
            "std_cosine_sim": np.std(metrics["cosine_sims"]),
            "mean_l2_dist": np.mean(metrics["l2_dists"]),
            "std_l2_dist": np.std(metrics["l2_dists"]),
            "all_cosine_sims": metrics["cosine_sims"],
            "all_l2_dists": metrics["l2_dists"],
        }

        # Add per-pixel data if requested
        if return_per_pixel:
            per_pixel_array = np.array(metrics["per_pixel_sims"]).T
            aggregated_results[layer_name]["per_pixel_cosine_sims"] = per_pixel_array

    print("\nRotation Invariance Results (higher cosine sim = more invariant):")
    for layer_name, metrics in aggregated_results.items():
        print(
            f"  {layer_name}: cos_sim={metrics['mean_cosine_sim']:.4f} (+/- {metrics['std_cosine_sim']:.4f})"
        )

    return aggregated_results


def test_illumination_invariance(
    extractor: LayerFeatureExtractor,
    image: np.ndarray,
    pixel_coords: np.ndarray,
    brightness_factors: List[float] = None,
    contrast_factors: List[float] = None,
    return_per_pixel: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Test illumination invariance across different layers.

    Args:
        extractor: LayerFeatureExtractor instance
        image: Input image in BGR format
        pixel_coords: (N, 2) array of pixel coordinates to test
        brightness_factors: List of brightness multipliers (default: [0.5, 0.75, 1.25, 1.5])
        contrast_factors: List of contrast multipliers (default: [0.5, 0.75, 1.25, 1.5])
        return_per_pixel: If True, include per-pixel similarity scores in results

    Returns:
        Dictionary mapping layer names to metrics
    """
    if brightness_factors is None:
        brightness_factors = [0.5, 0.75, 1.25, 1.5]
    if contrast_factors is None:
        contrast_factors = [0.5, 0.75, 1.25, 1.5]

    print(f"\n{'='*80}")
    print("Testing Illumination Invariance")
    print(f"{'='*80}")
    print(f"Testing {len(pixel_coords)} pixel locations")

    # Extract features from original image
    original_features = extractor.extract_features(image, pixel_coords)

    # Store results
    results: Dict[str, Dict[str, List]] = {
        layer: {"cosine_sims": [], "l2_dists": [], "per_pixel_sims": []}
        for layer in original_features.keys()
    }

    # Test brightness variations
    print("\nTesting brightness variations:")
    for brightness in brightness_factors:
        print(f"  Brightness: {brightness:.2f}x")

        # Apply transformation
        transformed_img, _ = apply_illumination_transform(
            image, brightness_factor=brightness
        )

        # Extract features
        transformed_features = extractor.extract_features(transformed_img, pixel_coords)

        # Compute metrics
        for layer_name in original_features.keys():
            orig_feat = original_features[layer_name]
            trans_feat = transformed_features[layer_name]

            cos_sim_per_pixel = cosine_similarity(orig_feat, trans_feat)
            l2_dist_per_pixel = normalized_l2_distance(orig_feat, trans_feat)

            if return_per_pixel:
                results[layer_name]["per_pixel_sims"].append(
                    cos_sim_per_pixel.cpu().numpy()
                )

            results[layer_name]["cosine_sims"].append(cos_sim_per_pixel.mean().item())
            results[layer_name]["l2_dists"].append(l2_dist_per_pixel.mean().item())

    # Test contrast variations
    print("\nTesting contrast variations:")
    for contrast in contrast_factors:
        print(f"  Contrast: {contrast:.2f}x")

        # Apply transformation
        transformed_img, _ = apply_illumination_transform(
            image, contrast_factor=contrast
        )

        # Extract features
        transformed_features = extractor.extract_features(transformed_img, pixel_coords)

        # Compute metrics
        for layer_name in original_features.keys():
            orig_feat = original_features[layer_name]
            trans_feat = transformed_features[layer_name]

            cos_sim_per_pixel = cosine_similarity(orig_feat, trans_feat)
            l2_dist_per_pixel = normalized_l2_distance(orig_feat, trans_feat)

            if return_per_pixel:
                results[layer_name]["per_pixel_sims"].append(
                    cos_sim_per_pixel.cpu().numpy()
                )

            results[layer_name]["cosine_sims"].append(cos_sim_per_pixel.mean().item())
            results[layer_name]["l2_dists"].append(l2_dist_per_pixel.mean().item())

    # Aggregate results
    aggregated_results = {}
    for layer_name, metrics in results.items():
        aggregated_results[layer_name] = {
            "mean_cosine_sim": np.mean(metrics["cosine_sims"]),
            "std_cosine_sim": np.std(metrics["cosine_sims"]),
            "mean_l2_dist": np.mean(metrics["l2_dists"]),
            "std_l2_dist": np.std(metrics["l2_dists"]),
            "all_cosine_sims": metrics["cosine_sims"],
            "all_l2_dists": metrics["l2_dists"],
        }

        if return_per_pixel:
            per_pixel_array = np.array(metrics["per_pixel_sims"]).T
            aggregated_results[layer_name]["per_pixel_cosine_sims"] = per_pixel_array

    print("\nIllumination Invariance Results (higher cosine sim = more invariant):")
    for layer_name, metrics in aggregated_results.items():
        print(
            f"  {layer_name}: cos_sim={metrics['mean_cosine_sim']:.4f} (+/- {metrics['std_cosine_sim']:.4f})"
        )

    return aggregated_results


def test_viewpoint_invariance(
    extractor: LayerFeatureExtractor,
    image: np.ndarray,
    pixel_coords: np.ndarray,
    transformations: List[Tuple[float, float, float]] = None,
    return_per_pixel: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Test viewpoint invariance across different layers.

    Args:
        extractor: LayerFeatureExtractor instance
        image: Input image in BGR format
        pixel_coords: (N, 2) array of pixel coordinates to test
        transformations: List of (tilt_x, tilt_y, shear_x) tuples
                        (default: small variations around 0)
        return_per_pixel: If True, include per-pixel similarity scores in results

    Returns:
        Dictionary mapping layer names to metrics
    """
    if transformations is None:
        transformations = [
            (-0.1, 0, 0),  # Horizontal tilt left
            (0.1, 0, 0),  # Horizontal tilt right
            (0, -0.1, 0),  # Vertical tilt down
            (0, 0.1, 0),  # Vertical tilt up
            (0, 0, -0.1),  # Shear left
            (0, 0, 0.1),  # Shear right
            (-0.05, -0.05, 0),  # Combined tilt
            (0.05, 0.05, 0),  # Combined tilt
        ]

    print(f"\n{'='*80}")
    print("Testing Viewpoint Invariance")
    print(f"{'='*80}")
    print(f"Testing {len(pixel_coords)} pixel locations")
    print(f"Number of viewpoint variations: {len(transformations)}")

    # Extract features from original image
    original_features = extractor.extract_features(image, pixel_coords)

    # Store results
    results: Dict[str, Dict[str, List]] = {
        layer: {"cosine_sims": [], "l2_dists": [], "per_pixel_sims": []}
        for layer in original_features.keys()
    }

    # Test each transformation
    for idx, (tilt_x, tilt_y, shear_x) in enumerate(transformations):
        print(
            f"\n  Transformation {idx+1}: tilt_x={tilt_x:.2f}, tilt_y={tilt_y:.2f}, shear_x={shear_x:.2f}"
        )

        # Apply transformation
        transformed_img, transform_matrix = apply_viewpoint_transform(
            image, tilt_x, tilt_y, shear_x
        )

        # Transform coordinates
        transformed_coords = transform_coordinates(pixel_coords, transform_matrix)

        # Extract features
        transformed_features = extractor.extract_features(
            transformed_img, transformed_coords
        )

        # Compute metrics
        for layer_name in original_features.keys():
            orig_feat = original_features[layer_name]
            trans_feat = transformed_features[layer_name]

            cos_sim_per_pixel = cosine_similarity(orig_feat, trans_feat)
            l2_dist_per_pixel = normalized_l2_distance(orig_feat, trans_feat)

            if return_per_pixel:
                results[layer_name]["per_pixel_sims"].append(
                    cos_sim_per_pixel.cpu().numpy()
                )

            results[layer_name]["cosine_sims"].append(cos_sim_per_pixel.mean().item())
            results[layer_name]["l2_dists"].append(l2_dist_per_pixel.mean().item())

    # Aggregate results
    aggregated_results = {}
    for layer_name, metrics in results.items():
        aggregated_results[layer_name] = {
            "mean_cosine_sim": np.mean(metrics["cosine_sims"]),
            "std_cosine_sim": np.std(metrics["cosine_sims"]),
            "mean_l2_dist": np.mean(metrics["l2_dists"]),
            "std_l2_dist": np.std(metrics["l2_dists"]),
            "all_cosine_sims": metrics["cosine_sims"],
            "all_l2_dists": metrics["l2_dists"],
        }

        if return_per_pixel:
            per_pixel_array = np.array(metrics["per_pixel_sims"]).T
            aggregated_results[layer_name]["per_pixel_cosine_sims"] = per_pixel_array

    print("\nViewpoint Invariance Results (higher cosine sim = more invariant):")
    for layer_name, metrics in aggregated_results.items():
        print(
            f"  {layer_name}: cos_sim={metrics['mean_cosine_sim']:.4f} (+/- {metrics['std_cosine_sim']:.4f})"
        )

    return aggregated_results


# ============================================================================
# Visualization and Export
# ============================================================================


def visualize_top_invariant_pixels(
    image: np.ndarray,
    pixel_coords: np.ndarray,
    per_pixel_results: Dict[str, Dict[str, np.ndarray]],
    layer_name: str,
    top_n: int,
    output_path: Path,
):
    """
    Visualize the top N most invariant pixels for all transformation types.

    Args:
        image: Original image in BGR format (H, W, 3)
        pixel_coords: (N, 2) array of (x, y) pixel coordinates tested
        per_pixel_results: Dictionary with structure:
                          {test_name: {layer_name: {'per_pixel_cosine_sims': (N, num_transformations)}}}
        layer_name: Name of the layer to visualize (e.g., 'layer_1')
        top_n: Number of top invariant pixels to highlight
        output_path: Path to save the visualization
    """
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(
        f"Top {top_n} Most Invariant Pixels - {layer_name}",
        fontsize=16,
        fontweight="bold",
    )

    test_names = ["scale", "rotation", "illumination", "viewpoint"]
    test_titles = [
        "Scale Invariance",
        "Rotation Invariance",
        "Illumination Invariance",
        "Viewpoint Invariance",
    ]

    for idx, (test_name, test_title) in enumerate(zip(test_names, test_titles)):
        ax = axes[idx // 2, idx % 2]

        if (
            test_name not in per_pixel_results
            or layer_name not in per_pixel_results[test_name]
        ):
            ax.text(
                0.5,
                0.5,
                f"No data for {test_name}",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.set_title(test_title, fontsize=14, fontweight="bold")
            ax.axis("off")
            continue

        # Get per-pixel cosine similarities (N, num_transformations)
        per_pixel_sims = per_pixel_results[test_name][layer_name][
            "per_pixel_cosine_sims"
        ]

        # Compute average similarity per pixel across all transformations
        avg_sims_per_pixel = per_pixel_sims.mean(axis=1)  # (N,)

        # Find top N most invariant pixels (highest average similarity)
        top_indices = np.argsort(avg_sims_per_pixel)[-top_n:]
        top_coords = pixel_coords[top_indices]
        top_scores = avg_sims_per_pixel[top_indices]

        # Display the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)

        # Create a colormap for the scores (green = most invariant, yellow = less invariant)
        norm = plt.Normalize(vmin=top_scores.min(), vmax=top_scores.max())
        cmap = plt.cm.RdYlGn

        # Plot all tested pixels in gray (background)
        ax.scatter(
            pixel_coords[:, 0],
            pixel_coords[:, 1],
            c="gray",
            s=20,
            alpha=0.3,
            marker="o",
            label="All tested pixels",
        )

        # Plot top N pixels with color coding
        scatter = ax.scatter(
            top_coords[:, 0],
            top_coords[:, 1],
            c=top_scores,
            cmap=cmap,
            norm=norm,
            s=200,
            alpha=0.8,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            label=f"Top {top_n} invariant",
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Average Cosine Similarity", fontsize=10)

        # Add title and legend
        ax.set_title(
            f"{test_title}\n(Avg Sim: {top_scores.mean():.3f} ± {top_scores.std():.3f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved pixel visualization: {output_path}")
    plt.close()


def plot_invariance_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path
):
    """
    Create visualization plots for invariance test results.

    Args:
        all_results: Dictionary with structure:
                    {test_name: {layer_name: {metric_name: value}}}
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract layer names (assuming all tests use the same layers)
    first_test = list(all_results.values())[0]
    layer_names = sorted(first_test.keys(), key=lambda x: int(x.split("_")[1]))
    layer_indices = [int(name.split("_")[1]) for name in layer_names]

    # Create summary plot: Cosine similarity across all tests
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Layer-wise Invariance Analysis", fontsize=16, fontweight="bold")

    test_names = ["scale", "rotation", "illumination", "viewpoint"]
    test_titles = [
        "Scale Invariance",
        "Rotation Invariance",
        "Illumination Invariance",
        "Viewpoint Invariance",
    ]

    for idx, (test_name, test_title) in enumerate(zip(test_names, test_titles)):
        ax = axes[idx // 2, idx % 2]

        if test_name in all_results:
            results = all_results[test_name]

            # Extract metrics
            mean_sims = [results[layer]["mean_cosine_sim"] for layer in layer_names]
            std_sims = [results[layer]["std_cosine_sim"] for layer in layer_names]

            # Plot with error bars
            ax.errorbar(
                layer_indices,
                mean_sims,
                yerr=std_sims,
                marker="o",
                capsize=5,
                linewidth=2,
                markersize=8,
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
    plt.savefig(output_dir / "invariance_summary.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved plot: {output_dir / 'invariance_summary.png'}")
    plt.close()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    heatmap_data_list: List[List[float]] = []
    for test_name in test_names:
        if test_name in all_results:
            results = all_results[test_name]
            row = [results[layer]["mean_cosine_sim"] for layer in layer_names]
            heatmap_data_list.append(row)

    heatmap_data = np.array(heatmap_data_list)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=layer_indices,
        yticklabels=[
            t.replace("_", " ").title() for t in test_names if t in all_results
        ],
        cbar_kws={"label": "Cosine Similarity"},
        ax=ax,
    )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Invariance Test", fontsize=12)
    ax.set_title("Layer-wise Invariance Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "invariance_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_dir / 'invariance_heatmap.png'}")
    plt.close()


def export_results(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    image_path: str,
    model_name: str,
):
    """
    Export results to JSON and CSV files.

    Args:
        all_results: Dictionary with test results
        output_dir: Directory to save files
        image_path: Path to the test image
        model_name: Name of the model tested
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    metadata = {
        "image_path": str(image_path),
        "model_name": model_name,
        "timestamp": str(Path(__file__).stat().st_mtime),
    }

    # Convert to JSON-serializable format (remove numpy arrays)
    json_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for test_name, test_results in all_results.items():
        json_results[test_name] = {}
        for layer_name, metrics in test_results.items():
            json_results[test_name][layer_name] = {
                "mean_cosine_sim": float(metrics["mean_cosine_sim"]),
                "std_cosine_sim": float(metrics["std_cosine_sim"]),
                "mean_l2_dist": float(metrics["mean_l2_dist"]),
                "std_l2_dist": float(metrics["std_l2_dist"]),
            }

    # Export to JSON
    output_data = {"metadata": metadata, "results": json_results}

    json_path = output_dir / "invariance_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results: {json_path}")

    # Export to CSV (summary)
    import csv

    csv_path = output_dir / "invariance_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Test",
                "Layer",
                "Mean Cosine Sim",
                "Std Cosine Sim",
                "Mean L2 Dist",
                "Std L2 Dist",
            ]
        )

        for test_name, test_results in json_results.items():
            for layer_name, metrics in test_results.items():
                writer.writerow(
                    [
                        test_name,
                        layer_name,
                        f"{metrics['mean_cosine_sim']:.4f}",
                        f"{metrics['std_cosine_sim']:.4f}",
                        f"{metrics['mean_l2_dist']:.4f}",
                        f"{metrics['std_l2_dist']:.4f}",
                    ]
                )

    print(f"Saved results: {csv_path}")


# ============================================================================
# Main Experiment
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test layer-wise invariance properties of Vision Transformers"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
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
        help="Number of random pixel locations to test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/invariance_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--visualize-pixels",
        action="store_true",
        help="Generate visualization of top N invariant pixels",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top invariant pixels to visualize (default: 50)",
    )
    parser.add_argument(
        "--layer-to-visualize",
        type=str,
        default="layer_1",
        help="Layer to visualize (e.g., layer_1, layer_6, default: layer_1)",
    )

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")

    h, w = image.shape[:2]
    print(f"\n{'='*80}")
    print("Layer Invariance Experiment")
    print(f"{'='*80}")
    print(f"Image: {args.image}")
    print(f"Size: {w}x{h}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    # Generate random pixel coordinates to test
    np.random.seed(42)
    pixel_coords = np.random.rand(args.num_points, 2)
    pixel_coords[:, 0] *= w
    pixel_coords[:, 1] *= h
    pixel_coords = pixel_coords.astype(np.float32)

    # Initialize feature extractor
    extractor = LayerFeatureExtractor(model_name=args.model, device=args.device)

    # Run all tests
    all_results = {}

    # Determine if we need per-pixel data for visualization
    return_per_pixel = args.visualize_pixels

    # Test 1: Scale invariance
    all_results["scale"] = test_scale_invariance(
        extractor, image, pixel_coords, return_per_pixel=return_per_pixel
    )

    # Test 2: Rotation invariance
    all_results["rotation"] = test_rotation_invariance(
        extractor, image, pixel_coords, return_per_pixel=return_per_pixel
    )

    # Test 3: Illumination invariance
    all_results["illumination"] = test_illumination_invariance(
        extractor, image, pixel_coords, return_per_pixel=return_per_pixel
    )

    # Test 4: Viewpoint invariance
    all_results["viewpoint"] = test_viewpoint_invariance(
        extractor, image, pixel_coords, return_per_pixel=return_per_pixel
    )

    # Create visualizations
    output_dir = Path(args.output_dir)
    plot_invariance_results(all_results, output_dir)

    # Visualize top invariant pixels if requested
    if args.visualize_pixels:
        print(f"\n{'='*80}")
        print(f"Generating Pixel Visualization for {args.layer_to_visualize}")
        print(f"{'='*80}")

        # Check if the layer exists
        if args.layer_to_visualize not in all_results["scale"]:
            print(f"Warning: Layer '{args.layer_to_visualize}' not found in results.")
            print(f"Available layers: {list(all_results['scale'].keys())}")
        else:
            pixel_viz_path = (
                output_dir
                / f"top_{args.top_n}_invariant_pixels_{args.layer_to_visualize}.png"
            )
            visualize_top_invariant_pixels(
                image=image,
                pixel_coords=pixel_coords,
                per_pixel_results=all_results,
                layer_name=args.layer_to_visualize,
                top_n=args.top_n,
                output_path=pixel_viz_path,
            )

    # Export results
    export_results(all_results, output_dir, args.image, args.model)

    # Print summary
    print(f"\n{'='*80}")
    print("Experiment Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print("\nBest layers for each invariance property:")

    for test_name, results in all_results.items():
        layer_scores = [
            (layer, metrics["mean_cosine_sim"]) for layer, metrics in results.items()
        ]
        best_layer, best_score = max(layer_scores, key=lambda x: x[1])
        print(
            f"  {test_name.capitalize()}: {best_layer} (cosine sim = {best_score:.4f})"
        )

    # Cleanup
    extractor.cleanup()


if __name__ == "__main__":
    main()
