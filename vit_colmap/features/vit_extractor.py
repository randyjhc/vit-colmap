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
        detection_method: str = "harris",  # "harris", "dog", or "combined"
    ):
        """
        Initialize ViT extractor.

        Args:
            weights_path: Path to custom weights (optional, uses pretrained if None)
            model_name: Name of ViT model ('dinov2_vitb14', 'dinov2_vitl14', etc.)
            num_keypoints: Number of keypoints to extract per image
            descriptor_dim: Target descriptor dimension (default: 128 like SIFT)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            detection_method: Method for keypoint detection:
                - "harris": Harris corner detector (best for corners)
                - "dog": Difference of Gaussians (SIFT-like, best for blobs)
                - "combined": Combination of both
        """
        self.weights_path = weights_path
        self.model_name = model_name
        self.num_keypoints = num_keypoints
        self.descriptor_dim = descriptor_dim
        self.detection_method = detection_method

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

        Improved Strategy:
        1. Compute feature distinctiveness (variance-based)
        2. Apply spatial binning for even distribution
        3. Use NMS to avoid clustering
        4. Extract interpolated descriptors for better localization

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

        # Compute feature distinctiveness using selected method
        score_map = self._compute_distinctiveness(
            feature_map, method=self.detection_method
        )

        # Apply spatial binning to ensure even distribution
        keypoint_coords, scores = self._spatial_binning_selection(
            score_map,
            target_keypoints=self.num_keypoints,
            bin_size=16,  # Reduced from 32 for finer control
        )

        if len(keypoint_coords) == 0:
            # Fallback to simple top-k if spatial binning fails
            keypoint_coords, scores = self._simple_topk_selection(
                score_map, self.num_keypoints
            )

        # Apply Non-Maximum Suppression to reduce clustering
        # Reduced radius to allow more keypoints on edges
        keypoint_coords, scores = self._apply_nms(
            keypoint_coords,
            scores,
            nms_radius=1.5,  # Reduced from 3 to allow edge clustering
        )

        # Extract descriptors with bilinear interpolation for better precision
        descriptors = self._extract_descriptors_interp(feature_map, keypoint_coords)

        # Map grid coordinates to original image coordinates
        scale_x_resized = w_resized / W
        scale_y_resized = h_resized / H
        scale_x_orig = w_orig / w_resized
        scale_y_orig = h_orig / h_resized

        keypoints_x = (keypoint_coords[:, 1] + 0.5) * scale_x_resized * scale_x_orig
        keypoints_y = (keypoint_coords[:, 0] + 0.5) * scale_y_resized * scale_y_orig
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=1)  # (N, 2)

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

    def _compute_distinctiveness(
        self, feature_map: torch.Tensor, method: str = "harris"
    ):
        """
        Compute keypoint response using various detection methods.

        Args:
            feature_map: (C, H, W) feature tensor
            method: Detection method - "harris", "dog", or "combined"

        Returns:
            score_map: (H, W) response scores
        """
        if method == "harris":
            return self._harris_response(feature_map)
        elif method == "dog":
            return self._dog_response(feature_map)
        elif method == "combined":
            harris = self._harris_response(feature_map)
            dog = self._dog_response(feature_map)
            # Normalize both to [0, 1]
            harris = (harris - harris.min()) / (harris.max() - harris.min() + 1e-8)
            dog = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)
            return 0.5 * harris + 0.5 * dog
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _harris_response(self, feature_map: torch.Tensor):
        """
        Compute Harris corner response.

        Detects corners and edges using the structure tensor.
        Good for geometric features.

        Args:
            feature_map: (C, H, W) feature tensor

        Returns:
            score_map: (H, W) corner/edge response scores
        """
        C, H, W = feature_map.shape

        # Compute gradients in x and y directions for each feature channel
        # Use Sobel-like filters
        grad_x = feature_map[:, :, 1:] - feature_map[:, :, :-1]  # (C, H, W-1)
        grad_y = feature_map[:, 1:, :] - feature_map[:, :-1, :]  # (C, H-1, W)

        # Pad to original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0))  # (C, H, W)
        grad_y = F.pad(grad_y, (0, 0, 0, 1))  # (C, H, W)

        # Compute structure tensor components (like Harris corner detector)
        # Average across feature channels
        Ixx = (grad_x**2).mean(dim=0)  # (H, W)
        Iyy = (grad_y**2).mean(dim=0)  # (H, W)
        Ixy = (grad_x * grad_y).mean(dim=0)  # (H, W)

        # Apply Gaussian smoothing to structure tensor
        kernel_size = 3
        sigma = 1.0
        gaussian_kernel = self._get_gaussian_kernel(kernel_size, sigma).to(
            feature_map.device
        )

        Ixx = F.conv2d(
            Ixx.unsqueeze(0).unsqueeze(0), gaussian_kernel, padding=kernel_size // 2
        ).squeeze()
        Iyy = F.conv2d(
            Iyy.unsqueeze(0).unsqueeze(0), gaussian_kernel, padding=kernel_size // 2
        ).squeeze()
        Ixy = F.conv2d(
            Ixy.unsqueeze(0).unsqueeze(0), gaussian_kernel, padding=kernel_size // 2
        ).squeeze()

        # Compute corner response (Harris corner measure)
        # R = det(M) - k * trace(M)^2
        # where M is the structure tensor
        k = 0.04  # Harris corner detector constant
        det = Ixx * Iyy - Ixy**2
        trace = Ixx + Iyy
        corner_response = det - k * (trace**2)

        # Also compute edge strength (gradient magnitude)
        edge_strength = torch.sqrt(Ixx + Iyy)

        # Combine corner and edge responses
        # Corners are more important than edges for matching
        score_map = 0.7 * corner_response + 0.3 * edge_strength

        # Normalize to [0, 1]
        score_map = score_map - score_map.min()
        if score_map.max() > 0:
            score_map = score_map / score_map.max()

        return score_map

    def _dog_response(self, feature_map: torch.Tensor):
        """
        Compute Difference of Gaussians (DoG) response.

        This is what SIFT uses for keypoint detection.
        Detects blobs and corners at different scales.

        Args:
            feature_map: (C, H, W) feature tensor

        Returns:
            score_map: (H, W) DoG response scores
        """
        # Reduce feature dimensionality first by averaging across channels
        # This is computationally cheaper
        feature_avg = feature_map.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)

        # Apply Gaussian smoothing at two scales
        sigma1 = 1.0
        sigma2 = 1.6  # SIFT uses k=sqrt(2)≈1.4, we use 1.6

        kernel_size1 = int(6 * sigma1 + 1)
        kernel_size1 = kernel_size1 if kernel_size1 % 2 == 1 else kernel_size1 + 1
        kernel_size2 = int(6 * sigma2 + 1)
        kernel_size2 = kernel_size2 if kernel_size2 % 2 == 1 else kernel_size2 + 1

        gaussian1 = self._get_gaussian_kernel(kernel_size1, sigma1).to(
            feature_map.device
        )
        gaussian2 = self._get_gaussian_kernel(kernel_size2, sigma2).to(
            feature_map.device
        )

        smooth1 = F.conv2d(feature_avg, gaussian1, padding=kernel_size1 // 2)
        smooth2 = F.conv2d(feature_avg, gaussian2, padding=kernel_size2 // 2)

        # Difference of Gaussians
        dog = torch.abs(smooth1 - smooth2).squeeze()

        # Normalize
        dog = dog - dog.min()
        if dog.max() > 0:
            dog = dog / dog.max()

        return dog

    def _get_gaussian_kernel(self, kernel_size: int, sigma: float):
        """Create a Gaussian kernel for smoothing."""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def _spatial_binning_selection(
        self, score_map: torch.Tensor, target_keypoints: int, bin_size: int = 32
    ):
        """
        Select keypoints using spatial binning to ensure even distribution.

        Divides image into bins and selects top keypoints from each bin.

        Args:
            score_map: (H, W) score map
            target_keypoints: Target number of keypoints
            bin_size: Size of spatial bins in pixels

        Returns:
            coords: (N, 2) tensor of (y, x) coordinates
            scores: (N,) tensor of scores
        """
        H, W = score_map.shape

        # Calculate number of bins
        n_bins_h = max(1, H // bin_size)
        n_bins_w = max(1, W // bin_size)
        total_bins = n_bins_h * n_bins_w

        # Calculate keypoints per bin
        keypoints_per_bin = max(1, target_keypoints // total_bins)

        all_coords = []
        all_scores = []

        for i in range(n_bins_h):
            for j in range(n_bins_w):
                # Define bin boundaries
                h_start = i * bin_size
                h_end = min((i + 1) * bin_size, H)
                w_start = j * bin_size
                w_end = min((j + 1) * bin_size, W)

                # Extract bin scores
                bin_scores = score_map[h_start:h_end, w_start:w_end]

                if bin_scores.numel() == 0:
                    continue

                # Get top keypoints in this bin
                bin_scores_flat = bin_scores.flatten()
                k = min(keypoints_per_bin, len(bin_scores_flat))

                if k == 0:
                    continue

                top_vals, top_idx = torch.topk(bin_scores_flat, k)

                # Convert to absolute coordinates
                # bin_h = h_end - h_start
                bin_w = w_end - w_start
                local_y = top_idx // bin_w
                local_x = top_idx % bin_w

                abs_y = local_y + h_start
                abs_x = local_x + w_start

                coords = torch.stack([abs_y, abs_x], dim=1)  # (k, 2)

                all_coords.append(coords)
                all_scores.append(top_vals)

        if not all_coords:
            return torch.empty((0, 2), device=score_map.device), torch.empty(
                0, device=score_map.device
            )

        all_coords = torch.cat(all_coords, dim=0)
        all_scores = torch.cat(all_scores, dim=0)

        # If we have more than target, take top scoring ones
        if len(all_coords) > target_keypoints:
            top_vals, top_idx = torch.topk(all_scores, target_keypoints)
            all_coords = all_coords[top_idx]
            all_scores = top_vals

        return all_coords, all_scores

    def _simple_topk_selection(self, score_map: torch.Tensor, k: int):
        """Fallback: simple top-k selection."""
        scores_flat = score_map.flatten()
        k = min(k, len(scores_flat))
        top_vals, top_idx = torch.topk(scores_flat, k)

        W = score_map.shape[1]
        coords_y = top_idx // W
        coords_x = top_idx % W
        coords = torch.stack([coords_y, coords_x], dim=1)

        return coords, top_vals

    def _apply_nms(
        self, coords: torch.Tensor, scores: torch.Tensor, nms_radius: float = 3.0
    ):
        """
        Apply Non-Maximum Suppression to reduce keypoint clustering.

        Args:
            coords: (N, 2) coordinates (y, x)
            scores: (N,) scores
            nms_radius: Suppression radius in pixels

        Returns:
            filtered_coords: (M, 2) coordinates after NMS
            filtered_scores: (M,) scores after NMS
        """
        if len(coords) == 0:
            return coords, scores

        # Sort by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_coords = coords[sorted_indices]
        sorted_scores = scores[sorted_indices]

        # NMS
        keep_mask = torch.ones(
            len(sorted_coords), dtype=torch.bool, device=coords.device
        )

        for i in range(len(sorted_coords)):
            if not keep_mask[i]:
                continue

            # Suppress nearby points
            coord_i = sorted_coords[i]
            distances = torch.sqrt(((sorted_coords - coord_i) ** 2).sum(dim=1))

            # Suppress points within radius (except self)
            suppress_mask = (distances < nms_radius) & (distances > 0)
            keep_mask[suppress_mask] = False

        filtered_coords = sorted_coords[keep_mask]
        filtered_scores = sorted_scores[keep_mask]

        return filtered_coords, filtered_scores

    def _extract_descriptors_interp(
        self, feature_map: torch.Tensor, coords: torch.Tensor
    ):
        """
        Extract descriptors using bilinear interpolation.

        This provides better localization than nearest neighbor.

        Args:
            feature_map: (C, H, W) features
            coords: (N, 2) coordinates (y, x)

        Returns:
            descriptors: (N, C) interpolated descriptors
        """
        if len(coords) == 0:
            return torch.empty((0, feature_map.shape[0]), device=feature_map.device)

        C, H, W = feature_map.shape

        # Normalize coordinates to [-1, 1] for grid_sample
        coords_norm = coords.float()
        coords_norm[:, 0] = 2.0 * coords_norm[:, 0] / (H - 1) - 1.0  # y
        coords_norm[:, 1] = 2.0 * coords_norm[:, 1] / (W - 1) - 1.0  # x

        # grid_sample expects (x, y) order and shape (1, N, 1, 2)
        grid = coords_norm[:, [1, 0]].unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

        # Interpolate
        feature_map_batch = feature_map.unsqueeze(0)  # (1, C, H, W)
        descriptors = F.grid_sample(
            feature_map_batch,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Reshape to (N, C)
        descriptors = descriptors.squeeze(0).squeeze(2).t()  # (N, C)

        return descriptors

    def _reduce_descriptor_dim(self, descriptors: torch.Tensor):
        """
        Reduce descriptor dimensionality using PCA-based projection.

        Uses incremental PCA that adapts over multiple calls.
        This is better than random projection as it preserves variance.

        Args:
            descriptors: (N, input_dim) tensor

        Returns:
            (N, descriptor_dim) tensor
        """
        if self.descriptor_projection is None:
            # Initialize PCA projection matrix
            input_dim = descriptors.shape[1]

            # For first batch, use SVD to get initial projection
            # This gives a better initialization than random
            if descriptors.shape[0] > self.descriptor_dim:
                # Center the data
                desc_centered = descriptors - descriptors.mean(dim=0, keepdim=True)

                # Compute SVD (on CPU if needed for stability)
                # torch.svd returns U, S, V where A = U @ diag(S) @ V.t()
                # V has shape (n_features, n_components) for thin SVD
                U, S, V = torch.svd(desc_centered.cpu())

                # Take top descriptor_dim components (columns of V)
                # V is (input_dim, min(n_samples, input_dim))
                self.descriptor_projection = V[:, : self.descriptor_dim].to(self.device)
                print(
                    f"Initialized PCA projection: {input_dim} -> {self.descriptor_dim}"
                )
                print(
                    f"  Variance explained: {(S[:self.descriptor_dim].sum() / S.sum()).item():.2%}"
                )

                # Verify shape
                assert self.descriptor_projection is not None  # Type assertion for mypy
                assert (
                    self.descriptor_projection.shape == (input_dim, self.descriptor_dim)
                ), f"Projection matrix has wrong shape: {self.descriptor_projection.shape}, expected ({input_dim}, {self.descriptor_dim})"
            else:
                # Fallback to Xavier initialization if not enough samples
                self.descriptor_projection = torch.randn(
                    input_dim,
                    self.descriptor_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.descriptor_projection /= np.sqrt(input_dim)
                print(
                    f"Initialized random projection (insufficient samples): {input_dim} -> {self.descriptor_dim}"
                )

                # Verify shape
                assert self.descriptor_projection is not None  # Type assertion for mypy
                assert (
                    self.descriptor_projection.shape == (input_dim, self.descriptor_dim)
                ), f"Projection matrix has wrong shape: {self.descriptor_projection.shape}, expected ({input_dim}, {self.descriptor_dim})"

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
