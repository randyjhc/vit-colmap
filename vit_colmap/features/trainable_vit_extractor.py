"""
Trainable ViT feature extractor for COLMAP integration.
Uses ViTFeatureModel with learned upsampling and detection heads.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import torchvision.transforms as transforms

from .base_extractor import BaseExtractor
from vit_colmap.model import ViTFeatureModel


class TrainableViTExtractor(BaseExtractor):
    """
    Trainable ViT-based feature extractor compatible with COLMAP database.

    Uses ViTFeatureModel with learned upsampling and detection heads
    for keypoint and descriptor extraction.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        model_name: str = "dinov2_vitb14",
        num_keypoints: int = 2048,
        descriptor_dim: int = 128,
        device: Optional[str] = None,
        score_threshold: float = 0.0,
        nms_radius: int = 4,
    ):
        """
        Initialize trainable ViT extractor.

        Args:
            weights_path: Path to trained model weights (optional)
            model_name: DINOv2 backbone name ('dinov2_vitb14', 'dinov2_vitl14', etc.)
            num_keypoints: Maximum number of keypoints to extract per image
            descriptor_dim: Descriptor dimension (default: 128)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            score_threshold: Minimum score for keypoint detection
            nms_radius: Radius for non-maximum suppression
        """
        self.weights_path = weights_path
        self.model_name = model_name
        self.num_keypoints = num_keypoints
        self.descriptor_dim = descriptor_dim
        self.score_threshold = score_threshold
        self.nms_radius = nms_radius

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing Trainable ViT extractor: {model_name} on {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

        # Get patch size
        self.patch_size = self.model.patch_size

        # Image preprocessing transform (ImageNet normalization)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Print model info
        param_counts = self.model.count_parameters()
        print("✓ Model loaded successfully")
        print(f"  Total parameters: {param_counts['total']:,}")
        print(f"  Trainable parameters: {param_counts['trainable']:,}")
        print(f"  Frozen parameters: {param_counts['frozen']:,}")

    def _load_model(self) -> ViTFeatureModel:
        """Load ViTFeatureModel with optional custom weights."""
        model = ViTFeatureModel(
            backbone_name=self.model_name,
            descriptor_dim=self.descriptor_dim,
            freeze_backbone=True,
        )

        if self.weights_path is not None:
            print(f"Loading custom weights from: {self.weights_path}")
            checkpoint = torch.load(self.weights_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            print("✓ Custom weights loaded")

        return model

    def _simple_nms(self, scores: torch.Tensor, radius: int) -> torch.Tensor:
        """
        Apply non-maximum suppression using max pooling.

        Args:
            scores: (H, W) score map
            radius: NMS radius

        Returns:
            (H, W) boolean mask of kept positions
        """
        # Max pool to find local maxima
        kernel_size = 2 * radius + 1
        padding = radius

        scores_3d = scores.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        max_pool = F.max_pool2d(
            scores_3d, kernel_size=kernel_size, stride=1, padding=padding
        )
        max_pool = max_pool.squeeze(0).squeeze(0)  # (H, W)

        # Keep only positions that are local maxima
        keep = scores == max_pool

        return keep

    def _run_inference(self, image_bgr: np.ndarray):
        """
        Run inference on a single image.

        Args:
            image_bgr: Input image in BGR format (OpenCV format)

        Returns:
            keypoints: (N, 6) float32 array of (x, y, scale, orientation, score, unused)
                       Maps to COLMAP format (x, y, a11, a12, a21, a22) where score is stored in a21
            descriptors: (N, 128) uint8 array of descriptors
            scores: (N,) float32 array of confidence scores [0, 1]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        # Resize to be divisible by patch_size
        h_new = (h_orig // self.patch_size) * self.patch_size
        w_new = (w_orig // self.patch_size) * self.patch_size

        if h_new != h_orig or w_new != w_orig:
            image_rgb = cv2.resize(
                image_rgb, (w_new, h_new), interpolation=cv2.INTER_LINEAR
            )

        # Preprocess
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.inference_mode():
            outputs = self.model(image_tensor)

        keypoints_map = outputs["keypoints"].squeeze(0)  # (4, H/4, W/4)
        descriptors_map = outputs["descriptors"].squeeze(0)  # (128, H/4, W/4)

        # Extract keypoints from score map
        scores = keypoints_map[0]  # (H/4, W/4)
        dx = keypoints_map[1]  # (H/4, W/4)
        dy = keypoints_map[2]  # (H/4, W/4)
        orientation = keypoints_map[3]  # (H/4, W/4)

        # Apply sigmoid to scores to get probabilities
        scores = torch.sigmoid(scores)

        # Apply non-maximum suppression
        nms_mask = self._simple_nms(scores, self.nms_radius)

        # Apply score threshold
        valid_mask = (scores > self.score_threshold) & nms_mask

        # Get coordinates of valid keypoints
        valid_coords = torch.nonzero(valid_mask, as_tuple=False)  # (N, 2) - (y, x)

        if len(valid_coords) == 0:
            # No keypoints found, return empty arrays
            return (
                np.zeros((0, 6), dtype=np.float32),
                np.zeros((0, self.descriptor_dim), dtype=np.uint8),
                np.zeros((0,), dtype=np.float32),
            )

        # Select top-K by score
        valid_scores = scores[valid_coords[:, 0], valid_coords[:, 1]]
        k = min(self.num_keypoints, len(valid_scores))
        top_k_indices = torch.topk(valid_scores, k).indices

        selected_coords = valid_coords[top_k_indices]  # (K, 2) - (y, x)
        selected_scores = valid_scores[top_k_indices]  # (K,) - confidence scores

        # Extract sub-pixel offsets for selected keypoints
        selected_y = selected_coords[:, 0]
        selected_x = selected_coords[:, 1]

        selected_dx = dx[selected_y, selected_x]
        selected_dy = dy[selected_y, selected_x]
        selected_orientation = orientation[selected_y, selected_x]

        # Convert to original image coordinates
        # Feature map is at 1/4 resolution (model outputs at H/4, W/4)
        scale_factor = 4.0
        scale_x_orig = w_orig / w_new
        scale_y_orig = h_orig / h_new

        # Apply sub-pixel refinement and scale to original image
        keypoints_x = (
            (selected_x.float() + selected_dx + 0.5) * scale_factor * scale_x_orig
        )
        keypoints_y = (
            (selected_y.float() + selected_dy + 0.5) * scale_factor * scale_y_orig
        )

        # Clamp to image boundaries
        keypoints_x = keypoints_x.clamp(0, w_orig - 1)
        keypoints_y = keypoints_y.clamp(0, h_orig - 1)

        # Create dummy scale (1.0 for all keypoints)
        keypoints_scale = torch.ones_like(keypoints_x)

        # Create unused column (0.0 for all keypoints)
        keypoints_unused = torch.zeros_like(keypoints_x)

        # Stack keypoints in 6-column COLMAP-compatible format: (x, y, a11, a12, a21, a22)
        # We use: x, y, scale, orientation, score, unused
        # Maps to: x, y, a11, a12, a21, a22 where score is stored in a21 column
        keypoints = torch.stack(
            [
                keypoints_x,
                keypoints_y,
                keypoints_scale,
                selected_orientation,
                selected_scores,
                keypoints_unused,
            ],
            dim=1,
        )  # (K, 6)

        # Extract descriptors at keypoint locations
        descriptors = descriptors_map[:, selected_y, selected_x].t()  # (K, 128)

        # Convert to numpy
        keypoints_np = keypoints.cpu().numpy().astype(np.float32)
        descriptors_float = descriptors.cpu().numpy()
        scores_np = selected_scores.cpu().numpy().astype(np.float32)

        # Convert normalized descriptors to uint8 [0, 255]
        # Descriptors are already normalized, map from [-1, 1] to [0, 255]
        descriptors_uint8 = (
            ((descriptors_float + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        )

        return keypoints_np, descriptors_uint8, scores_np

    def extract(
        self,
        image_dir: Path,
        db_path: Path,
        camera_model: str,
        camera_params: Optional[list[float]] = None,
    ):
        """
        Extract features for all images in directory and write to database.

        Args:
            image_dir: Directory containing images
            db_path: Path to COLMAP database
            camera_model: Camera model string ('SIMPLE_PINHOLE', 'PINHOLE', etc.)
            camera_params: Optional camera parameters
        """
        import pycolmap
        from vit_colmap.database.colmap_db import ColmapDatabase

        print(f"\n{'='*60}")
        print("Trainable ViT Feature Extraction")
        print(f"{'='*60}")
        print(f"Image directory: {image_dir}")
        print(f"Database: {db_path}")
        print(f"Model: {self.model_name}")
        print(f"Target keypoints per image: {self.num_keypoints}")
        print(f"Score threshold: {self.score_threshold}")
        print(f"NMS radius: {self.nms_radius}")
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
                f = max(width, height)
                camera_params = [f, width / 2.0, height / 2.0]
            elif camera_model == "PINHOLE":
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
                print("  Warning: Failed to read image, skipping")
                continue

            # Add image to database
            image_id = db.add_image(img_file.name, camera_id=camera_id)

            # Extract features
            try:
                keypoints, descriptors, scores = self._run_inference(img)

                print(f"  Extracted {len(keypoints)} keypoints")
                print(f"  Descriptor shape: {descriptors.shape}")
                print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

                if len(keypoints) == 0:
                    print("  Warning: No keypoints extracted")
                    continue

                # Write to database
                db.add_keypoints(image_id, keypoints)
                db.add_descriptors(image_id, descriptors)

            except Exception as e:
                print(f"  Error during feature extraction: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Commit changes
        db.commit()
        print(f"\n{'='*60}")
        print("Feature extraction complete!")
        print(f"{'='*60}\n")
