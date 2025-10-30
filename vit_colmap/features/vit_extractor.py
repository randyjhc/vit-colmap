"""
Vision Transformer feature extractor for COLMAP integration.
Implements ViT-based feature extraction following the existing architecture.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import torchvision.transforms as transforms

from .base_extractor import BaseExtractor


class ViTExtractor(BaseExtractor):
    """
    ViT-based feature extractor compatible with COLMAP database.

    Uses pre-trained DINOv2 models to extract dense features,
    then converts them to sparse keypoints with descriptors.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        model_name: str = "dinov2_vitb14",
        num_keypoints: int = 2048,
        descriptor_dim: int = 128,
        device: str | None = None,
    ):
        """
        Initialize ViT extractor.

        Args:
            weights_path: Path to custom weights (optional, uses pretrained if None)
            model_name: Name of ViT model ('dinov2_vitb14', 'dinov2_vitl14', etc.)
            num_keypoints: Number of keypoints to extract per image
            descriptor_dim: Target descriptor dimension (default: 128 like SIFT)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.weights_path = weights_path
        self.model_name = model_name
        self.num_keypoints = num_keypoints
        self.descriptor_dim = descriptor_dim

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing ViT extractor: {model_name} on {self.device}")

        # Load ViT model
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

        # Get patch size from model (typically 14 for DINOv2)
        self.patch_size = 14  # DINOv2 uses 14x14 patches

        # Image preprocessing transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Descriptor projection (initialized on first use)
        self.descriptor_projection = None

        print("✓ ViT model loaded successfully")

    def _load_model(self):
        """Load pre-trained ViT model."""
        if self.weights_path is not None:
            # Load custom weights
            print(f"Loading custom weights from: {self.weights_path}")
            # TODO: Implement custom weight loading
            raise NotImplementedError("Custom weight loading not yet implemented")
        else:
            # Load from torch hub
            if "dinov2" in self.model_name:
                model = torch.hub.load(
                    "facebookresearch/dinov2", self.model_name, pretrained=True
                )
                return model
            else:
                raise ValueError(
                    f"Unsupported model: {self.model_name}. "
                    f"Currently only DINOv2 models are supported."
                )

    def _run_inference(self, image_bgr: np.ndarray):
        """
        Run ViT inference on a single image.

        Args:
            image_bgr: Input image in BGR format (OpenCV format)

        Returns:
            keypoints: (N, 2) float32 array of (x, y) coordinates
            descriptors: (N, 128) uint8 array of descriptors
        """
        # Convert BGR to RGB (OpenCV loads as BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        # Resize image to be multiple of patch_size (required by ViT)
        h_new = (h_orig // self.patch_size) * self.patch_size
        w_new = (w_orig // self.patch_size) * self.patch_size

        # If dimensions changed, resize
        if h_new != h_orig or w_new != w_orig:
            image_rgb = cv2.resize(
                image_rgb, (w_new, h_new), interpolation=cv2.INTER_LINEAR
            )

        # Preprocess image
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Extract features with ViT
        with torch.no_grad():
            # Forward pass through ViT
            features = self.model.forward_features(image_tensor)

            # Handle different DINOv2 output formats
            if isinstance(features, dict):
                # Newer DINOv2 returns a dictionary
                patch_features = features["x_norm_patchtokens"]
            else:
                # Older DINOv2 returns a tensor
                # Extract patch tokens (skip CLS token at position 0)
                patch_features = features[:, 1:, :]

        # Reshape to spatial grid
        # Calculate grid size based on input image and patch size
        h_patches = image_tensor.shape[2] // self.patch_size
        w_patches = image_tensor.shape[3] // self.patch_size
        feature_dim = patch_features.shape[-1]

        # Reshape: (1, h_patches * w_patches, feature_dim) -> (1, feature_dim, h_patches, w_patches)
        feature_map = patch_features.reshape(1, h_patches, w_patches, feature_dim)
        feature_map = feature_map.permute(0, 3, 1, 2)  # (1, C, H, W)

        # Convert dense features to sparse keypoints
        keypoints, descriptors = self._dense_to_sparse(
            feature_map,
            original_size=(w_orig, h_orig),
            resized_size=(w_new, h_new),
            feature_grid_size=(h_patches, w_patches),
        )

        return keypoints, descriptors

    def _dense_to_sparse(
        self,
        feature_map: torch.Tensor,
        original_size: tuple[int, int],
        resized_size: tuple[int, int],
        feature_grid_size: tuple[int, int],
    ):
        """
        Convert dense ViT feature maps to sparse keypoints with descriptors.

        Strategy:
        1. Compute saliency map based on feature magnitude
        2. Select top-K locations as keypoints
        3. Extract and reduce descriptor dimensionality

        Args:
            feature_map: (1, C, H, W) dense feature tensor
            original_size: (width, height) of original image
            resized_size: (width, height) after patch-size alignment
            feature_grid_size: (h, w) number of patches

        Returns:
            keypoints: (N, 2) float32 array of (x, y) coordinates in original image space
            descriptors: (N, descriptor_dim) uint8 array
        """
        feature_map = feature_map.squeeze(0)  # (C, H, W)
        C, H, W = feature_map.shape
        w_orig, h_orig = original_size
        w_resized, h_resized = resized_size

        # Compute saliency: L2 norm of features at each spatial location
        saliency = torch.norm(feature_map, dim=0)  # (H, W)

        # Flatten and select top-K keypoints
        saliency_flat = saliency.flatten()
        k = min(self.num_keypoints, len(saliency_flat))
        top_k_values, top_k_indices = torch.topk(saliency_flat, k)

        # Convert flat indices to 2D grid coordinates
        grid_y = (top_k_indices // W).float()
        grid_x = (top_k_indices % W).float()

        # Map grid coordinates to original image coordinates
        # First scale to resized image, then to original
        scale_x_resized = w_resized / W
        scale_y_resized = h_resized / H

        scale_x_orig = w_orig / w_resized
        scale_y_orig = h_orig / h_resized

        keypoints_x = (grid_x + 0.5) * scale_x_resized * scale_x_orig
        keypoints_y = (grid_y + 0.5) * scale_y_resized * scale_y_orig
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=1)  # (N, 2)

        # Extract descriptors at keypoint locations
        grid_coords_y = grid_y.long().clamp(0, H - 1)
        grid_coords_x = grid_x.long().clamp(0, W - 1)
        descriptors = feature_map[:, grid_coords_y, grid_coords_x].t()  # (N, C)

        # Reduce descriptor dimensionality if needed
        if descriptors.shape[1] > self.descriptor_dim:
            descriptors = self._reduce_descriptor_dim(descriptors)

        # L2 normalize descriptors
        descriptors = F.normalize(descriptors, p=2, dim=1)

        # Convert to numpy
        keypoints_np = keypoints.cpu().numpy().astype(np.float32)
        descriptors_float = descriptors.cpu().numpy()

        # Standard conversion: multiply by 512, round, clip to uint8
        descriptors_uint8 = (descriptors_float * 512.0).clip(0, 255).astype(np.uint8)

        return keypoints_np, descriptors_uint8

    def _reduce_descriptor_dim(self, descriptors: torch.Tensor):
        """
        Reduce descriptor dimensionality using learned projection.

        Uses random projection initialized on first call.
        Could be replaced with PCA for better results.

        Args:
            descriptors: (N, input_dim) tensor

        Returns:
            (N, descriptor_dim) tensor
        """
        if self.descriptor_projection is None:
            # Initialize random projection matrix
            input_dim = descriptors.shape[1]
            # Xavier initialization
            self.descriptor_projection = torch.randn(
                input_dim, self.descriptor_dim, device=self.device, dtype=torch.float32
            )
            self.descriptor_projection /= np.sqrt(input_dim)
            print(
                f"Initialized descriptor projection: {input_dim} -> {self.descriptor_dim}"
            )

        # Project descriptors
        reduced = descriptors @ self.descriptor_projection

        return reduced

    def extract(
        self,
        image_dir: Path,
        db_path: Path,
        camera_model: str,
        camera_params: Optional[list[float]] = None,
    ):
        """
        Extract ViT features for all images in directory and write to database.

        Args:
            image_dir: Directory containing images
            db_path: Path to COLMAP database
            camera_model: Camera model string ('SIMPLE_PINHOLE', 'PINHOLE', etc.)
            camera_params: Optional camera parameters
        """
        import pycolmap
        from vit_colmap.database.colmap_db import ColmapDatabase

        print(f"\n{'='*60}")
        print("ViT Feature Extraction")
        print(f"{'='*60}")
        print(f"Image directory: {image_dir}")
        print(f"Database: {db_path}")
        print(f"Model: {self.model_name}")
        print(f"Target keypoints per image: {self.num_keypoints}")
        print(f"{'='*60}\n")

        # Get list of image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = sorted(
            [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
        )

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(image_files)} images")

        # Initialize database
        db = ColmapDatabase(str(db_path))

        # Read first image to get dimensions
        first_img = cv2.imread(str(image_files[0]))
        if first_img is None:
            raise ValueError(f"Failed to read first image: {image_files[0]}")

        height, width = first_img.shape[:2]
        print(f"Image dimensions: {width}x{height}")

        # Set camera parameters if not provided
        if camera_params is None:
            if camera_model == "SIMPLE_PINHOLE":
                # focal_length, cx, cy
                f = max(width, height)
                camera_params = [f, width / 2.0, height / 2.0]
            elif camera_model == "PINHOLE":
                # fx, fy, cx, cy
                f = max(width, height)
                camera_params = [f, f, width / 2.0, height / 2.0]
            else:
                raise ValueError(f"Unsupported camera model: {camera_model}")

        print(f"Camera model: {camera_model}")
        print(f"Camera params: {camera_params}")

        # Add camera to database
        camera = pycolmap.Camera(
            model=camera_model, width=width, height=height, params=camera_params
        )
        camera_id = db.db.write_camera(camera)
        print(f"Camera ID: {camera_id}\n")

        # Process each image
        for idx, img_file in enumerate(image_files, start=1):
            print(f"[{idx}/{len(image_files)}] Processing: {img_file.name}")

            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                print("  ⚠ Warning: Failed to read image, skipping")
                continue

            # Add image to database
            image_id = db.add_image(img_file.name, camera_id=camera_id)

            # Extract features
            try:
                keypoints, descriptors = self._run_inference(img)

                print(f"  → Extracted {len(keypoints)} keypoints")
                print(f"  → Descriptor shape: {descriptors.shape}")

                # Verify outputs
                if len(keypoints) == 0:
                    print("  ⚠ Warning: No keypoints extracted")
                    continue

                # Write to database
                db.add_keypoints(image_id, keypoints)
                db.add_descriptors(image_id, descriptors)

            except Exception as e:
                print(f"  ✗ Error during feature extraction: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Commit changes
        db.commit()
        print(f"\n{'='*60}")
        print("✓ Feature extraction complete!")
        print(f"{'='*60}\n")
