"""
Simplified BEiT feature extractor for visualization.
Extracts features from specific layers and visualizes them.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
from transformers import BeitImageProcessor, BeitModel
import matplotlib.pyplot as plt


class BEiTExtractor:
    """
    Simplified BEiT feature extractor for layer visualization.

    Supports two visualization modes:
    1. RGB reconstruction: Treat feature channels as RGB
    2. Heatmap: Show feature magnitude as heatmap overlay
    """

    def __init__(
        self,
        model_name: str = "microsoft/beit-base-patch16-224-pt22k-ft22k",
        layer_idx: int = 9,
        device: Optional[str] = None,
    ):
        """
        Initialize BEiT extractor.

        Args:
            model_name: HuggingFace model identifier
            layer_idx: Which BEiT layer to extract features from (0-12 for base model)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.layer_idx = layer_idx

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print("Initializing BEiT extractor:")
        print(f"  Model: {model_name}")
        print(f"  Layer: {layer_idx}")
        print(f"  Device: {self.device}")

        # Load BEiT model and processor
        self.processor = BeitImageProcessor.from_pretrained(model_name)
        self.model = BeitModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        # Get patch size (typically 16 for BEiT)
        self.patch_size = (
            self.processor.size.get("height", 224) // 14
        )  # BEiT uses 14x14 patches for 224x224
        print(f"  Patch size: {self.patch_size}")
        print("✓ BEiT model loaded successfully\n")

    def extract_layer_features(
        self, image_bgr: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Extract features from specified BEiT layer.

        Args:
            image_bgr: Input image in BGR format (H, W, 3) - OpenCV format

        Returns:
            feature_map: Feature map tensor of shape (1, H_patches, W_patches, D)
            processed_image: Resized RGB image (224, 224, 3) that was processed by BEiT
        """
        from PIL import Image

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for processor
        pil_image = Image.fromarray(image_rgb)

        # Preprocess image (this resizes to 224x224)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get the processed image tensor (convert back to numpy for visualization)
        processed_tensor = inputs["pixel_values"][0]  # (3, 224, 224)

        # Denormalize and convert to numpy
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        processed_tensor = processed_tensor * std + mean
        processed_tensor = processed_tensor.clamp(0, 1)

        # Convert to HWC format and scale to 0-255
        processed_image = (
            processed_tensor.permute(1, 2, 0).cpu().numpy() * 255
        ).astype(np.uint8)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract features from specified layer
        hidden_state = outputs.hidden_states[self.layer_idx]  # (B, N_tokens, D)

        # Remove [CLS] token
        patch_tokens = hidden_state[:, 1:, :]  # (B, N_patches, D)

        # Reshape to spatial grid
        B, N, D = patch_tokens.shape
        H = W = int(N**0.5)  # Assuming square grid (14x14 for 224x224 input)

        feature_map = patch_tokens.reshape(B, H, W, D)

        # L2 normalize features
        feature_map = F.normalize(feature_map, p=2, dim=-1)

        return feature_map, processed_image

    def reconstruct_to_original_size(
        self, feature_map: torch.Tensor, target_size: tuple[int, int]
    ) -> np.ndarray:
        """
        Upsample feature map to target size using bilinear interpolation.

        Args:
            feature_map: (1, H_patches, W_patches, D) tensor
            target_size: (height, width) tuple for output size

        Returns:
            upsampled: (H, W, D) numpy array of upsampled features
        """
        # feature_map: (1, H, W, D) -> need to convert to (1, D, H, W) for interpolate
        feature_map_transposed = feature_map.permute(0, 3, 1, 2)  # (1, D, H, W)

        # Upsample using bilinear interpolation
        h_target, w_target = target_size
        upsampled = F.interpolate(
            feature_map_transposed,
            size=(h_target, w_target),
            mode="bilinear",
            align_corners=False,
        )  # (1, D, H_target, W_target)

        # Convert back to (H, W, D) format and to numpy
        upsampled = upsampled.squeeze(0).permute(1, 2, 0)  # (H_target, W_target, D)
        upsampled_np = upsampled.cpu().numpy()

        return upsampled_np

    def visualize_layer_rgb(
        self,
        image_path: Path | str,
        output_path: Path | str,
        channels: list[int] = [0, 1, 2],
    ) -> None:
        """
        Mode 1: Visualize feature channels as RGB image.

        Takes 3 feature channels and treats them as RGB channels,
        reconstructing them to the original image size.

        Args:
            image_path: Path to input image
            output_path: Path to save visualization
            channels: Which 3 feature channels to use as RGB (default: [0,1,2])
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load image
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        print(f"Extracting layer {self.layer_idx} features from {image_path.name}...")

        # Extract features
        feature_map, processed_image = self.extract_layer_features(img_bgr)

        print(f"  Feature map shape: {feature_map.shape}")
        print(f"  Using channels {channels} as RGB")

        # Take specified channels
        if feature_map.shape[-1] < max(channels) + 1:
            raise ValueError(
                f"Requested channel {max(channels)} but feature map only has "
                f"{feature_map.shape[-1]} channels"
            )

        feature_rgb = feature_map[0, :, :, channels]  # (H, W, 3)

        # Reconstruct to original size
        feature_rgb_full = (
            F.interpolate(
                feature_rgb.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H, W)
                size=(h_orig, w_orig),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )  # (H_orig, W_orig, 3)

        feature_rgb_np = feature_rgb_full.cpu().numpy()

        # Normalize to 0-255 range
        feature_rgb_np = (feature_rgb_np - feature_rgb_np.min()) / (
            feature_rgb_np.max() - feature_rgb_np.min() + 1e-8
        )
        feature_rgb_np = (feature_rgb_np * 255).astype(np.uint8)

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original Image\n{w_orig}×{h_orig}", fontsize=12)
        axes[0].axis("off")

        # Processed image (what BEiT sees)
        axes[1].imshow(processed_image)
        axes[1].set_title("BEiT Input\n224×224 (resized)", fontsize=12)
        axes[1].axis("off")

        # Reconstructed RGB from features
        axes[2].imshow(feature_rgb_np)
        axes[2].set_title(
            f"Layer {self.layer_idx} Features (Channels {channels})\n"
            f"Reconstructed to {w_orig}×{h_orig}",
            fontsize=12,
        )
        axes[2].axis("off")

        plt.suptitle(
            f"BEiT Feature Visualization - RGB Reconstruction\n"
            f"Model: {self.model_name.split('/')[-1]}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved RGB visualization to {output_path}")

    def visualize_layer_heatmap(
        self,
        image_path: Path | str,
        output_path: Path | str,
    ) -> None:
        """
        Mode 2: Visualize feature magnitude as heatmap overlay.

        Computes L2 norm across all feature dimensions and shows as heatmap.

        Args:
            image_path: Path to input image
            output_path: Path to save visualization
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load image
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        print(f"Extracting layer {self.layer_idx} features from {image_path.name}...")

        # Extract features
        feature_map, processed_image = self.extract_layer_features(img_bgr)

        print(f"  Feature map shape: {feature_map.shape}")

        # Compute L2 norm across feature dimension
        feature_magnitude = torch.norm(feature_map.squeeze(0), dim=-1)  # (H, W)

        print(
            f"  Feature magnitude range: [{feature_magnitude.min():.3f}, {feature_magnitude.max():.3f}]"
        )

        # Upsample to original size
        heatmap = F.interpolate(
            feature_magnitude.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            size=(h_orig, w_orig),
            mode="bilinear",
            align_corners=False,
        ).squeeze()  # (H_orig, W_orig)

        heatmap_np = heatmap.cpu().numpy()

        # Normalize to 0-1 range
        heatmap_norm = (heatmap_np - heatmap_np.min()) / (
            heatmap_np.max() - heatmap_np.min() + 1e-8
        )

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original Image\n{w_orig}×{h_orig}", fontsize=12)
        axes[0].axis("off")

        # Heatmap only
        im1 = axes[1].imshow(heatmap_norm, cmap="hot")
        axes[1].set_title(
            f"Layer {self.layer_idx} Feature Magnitude\n"
            f"(L2 norm across {feature_map.shape[-1]} channels)",
            fontsize=12,
        )
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Overlay on original image
        axes[2].imshow(img_rgb, alpha=0.6)
        axes[2].imshow(heatmap_norm, cmap="hot", alpha=0.4)
        axes[2].set_title("Heatmap Overlay", fontsize=12)
        axes[2].axis("off")

        plt.suptitle(
            f"BEiT Feature Visualization - Heatmap\n"
            f"Model: {self.model_name.split('/')[-1]}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved heatmap visualization to {output_path}")


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    print("BEiT Feature Extractor - Simplified Version\n")
    print("This extractor supports two visualization modes:")
    print("  1. RGB reconstruction: Visualize feature channels as RGB")
    print("  2. Heatmap: Show feature magnitude as heatmap overlay")
    print()

    # Initialize extractor
    extractor = BEiTExtractor(
        model_name="microsoft/beit-base-patch16-224-pt22k-ft22k",
        layer_idx=9,  # Layer 9 (can be changed to 0-12)
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Test with an image
    test_image = Path("data/raw/test_image.png")

    if test_image.exists():
        print(f"\n{'='*60}")
        print("Example 1: RGB Reconstruction")
        print(f"{'='*60}\n")

        output_rgb = Path("outputs/beit_layer_rgb.png")
        extractor.visualize_layer_rgb(
            test_image,
            output_rgb,
            channels=[0, 1, 2],  # First 3 channels as RGB
        )

        print(f"\n{'='*60}")
        print("Example 2: Heatmap Visualization")
        print(f"{'='*60}\n")

        output_heatmap = Path("outputs/beit_layer_heatmap.png")
        extractor.visualize_layer_heatmap(test_image, output_heatmap)

        print(f"\n{'='*60}")
        print("✓ Examples complete!")
        print(f"{'='*60}")

    else:
        print(f"Test image not found at {test_image}")
        print("Please provide a valid image path.")
