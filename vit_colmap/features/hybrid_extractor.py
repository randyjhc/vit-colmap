"""
Hybrid Feature Extractor: Traditional Detection + ViT Descriptors

Uses OpenCV's corner/edge detectors for precise keypoint localization,
then extracts ViT features at those locations for semantic descriptors.

This combines:
- Precise geometric detection (pixel-level accuracy)
- Rich semantic descriptors (ViT's strength)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Literal
import torchvision.transforms as transforms

from .base_extractor import BaseExtractor


class ViTExtractor(BaseExtractor):
    """
    Hybrid extractor: Traditional keypoint detection + ViT descriptors.

    Best of both worlds:
    - Uses SIFT/FAST/GFTT for precise keypoint detection
    - Uses ViT features for semantic descriptors
    """

    def __init__(
        self,
        weights_path: str | None = None,
        model_name: str = "dinov2_vitb14",
        num_keypoints: int = 2048,
        descriptor_dim: int = 256,
        device: str | None = None,
        detector_type: Literal["sift", "fast", "gftt", "orb"] = "sift",
    ):
        """
        Initialize hybrid extractor.

        Args:
            weights_path: Path to custom ViT weights (optional)
            model_name: ViT model name
            num_keypoints: Target number of keypoints
            descriptor_dim: Descriptor dimension (128 like SIFT)
            device: Device to use
            detector_type: Traditional detector to use:
                - "sift": SIFT detector (best quality, slowest)
                - "fast": FAST corner detector (fast, good for corners)
                - "gftt": Good Features To Track (Shi-Tomasi, balanced)
                - "orb": ORB detector (fastest, decent quality)
        """
        self.weights_path = weights_path
        self.model_name = model_name
        self.num_keypoints = num_keypoints
        self.descriptor_dim = descriptor_dim
        self.detector_type = detector_type

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Initializing Hybrid extractor: {model_name} on {self.device}")
        print(f"Keypoint detector: {detector_type.upper()}")

        # Load ViT model for descriptors
        self.model = self._load_vit_model()
        self.model.eval()
        self.model.to(self.device)

        # Initialize traditional keypoint detector
        self.detector = self._create_detector()

        # ViT preprocessing
        self.patch_size = 14
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Descriptor projection
        self.descriptor_projection = None

        print("✓ Hybrid model loaded successfully")

    def _load_vit_model(self):
        """Load ViT model for feature extraction."""
        if self.weights_path is not None:
            raise NotImplementedError("Custom weight loading not yet implemented")

        if "dinov2" in self.model_name:
            model = torch.hub.load(
                "facebookresearch/dinov2", self.model_name, pretrained=True
            )
            return model
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _create_detector(self):
        """Create traditional keypoint detector."""
        if self.detector_type == "sift":
            # SIFT: Best quality, scale-invariant
            return cv2.SIFT_create(nfeatures=self.num_keypoints)

        elif self.detector_type == "fast":
            # FAST: Very fast corner detector
            return cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)

        elif self.detector_type == "gftt":
            # Good Features To Track (Shi-Tomasi): Balanced quality/speed
            return cv2.goodFeaturesToTrack

        elif self.detector_type == "orb":
            # ORB: Fast, rotation invariant
            return cv2.ORB_create(nfeatures=self.num_keypoints)

        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def _detect_keypoints(self, image_bgr: np.ndarray):
        """
        Detect keypoints using traditional detector.

        Returns:
            keypoints: (N, 2) array of (x, y) coordinates
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if self.detector_type == "sift":
            kpts = self.detector.detect(gray, None)
            keypoints = np.array(
                [[kp.pt[0], kp.pt[1]] for kp in kpts], dtype=np.float32
            )

        elif self.detector_type == "fast":
            kpts = self.detector.detect(gray, None)
            keypoints = np.array(
                [[kp.pt[0], kp.pt[1]] for kp in kpts], dtype=np.float32
            )

            # Limit to num_keypoints
            if len(keypoints) > self.num_keypoints:
                # Sort by response strength
                responses = np.array([kp.response for kp in kpts])
                top_indices = np.argsort(responses)[-self.num_keypoints :]
                keypoints = keypoints[top_indices]

        elif self.detector_type == "gftt":
            # goodFeaturesToTrack returns corners directly
            corners = self.detector(
                gray,
                maxCorners=self.num_keypoints,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7,
            )

            if corners is not None:
                keypoints = corners.reshape(-1, 2).astype(np.float32)
            else:
                keypoints = np.zeros((0, 2), dtype=np.float32)

        elif self.detector_type == "orb":
            kpts = self.detector.detect(gray, None)
            keypoints = np.array(
                [[kp.pt[0], kp.pt[1]] for kp in kpts], dtype=np.float32
            )

        return keypoints

    def _extract_vit_features(self, image_bgr: np.ndarray):
        """
        Extract dense ViT features.

        Returns:
            feature_map: (C, H, W) tensor of ViT features
            image_size: (width, height) of processed image
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        # Resize to multiple of patch_size
        h_new = (h_orig // self.patch_size) * self.patch_size
        w_new = (w_orig // self.patch_size) * self.patch_size

        if h_new != h_orig or w_new != w_orig:
            image_rgb = cv2.resize(
                image_rgb, (w_new, h_new), interpolation=cv2.INTER_LINEAR
            )

        # Preprocess
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model.forward_features(image_tensor)

            if isinstance(features, dict):
                patch_features = features["x_norm_patchtokens"]
            else:
                patch_features = features[:, 1:, :]

        # Reshape to spatial grid
        h_patches = image_tensor.shape[2] // self.patch_size
        w_patches = image_tensor.shape[3] // self.patch_size
        feature_dim = patch_features.shape[-1]

        feature_map = patch_features.reshape(1, h_patches, w_patches, feature_dim)
        feature_map = feature_map.permute(0, 3, 1, 2).squeeze(0)  # (C, H, W)

        return feature_map, (w_new, h_new)

    def _extract_descriptors_at_keypoints(
        self,
        feature_map: torch.Tensor,
        keypoints: np.ndarray,
        original_size: tuple[int, int],
        feature_size: tuple[int, int],
    ):
        """
        Extract ViT descriptors at keypoint locations using bilinear interpolation.

        Args:
            feature_map: (C, H, W) ViT features
            keypoints: (N, 2) keypoint coordinates in original image space
            original_size: (width, height) of original image
            feature_size: (width, height) of feature map space

        Returns:
            descriptors: (N, descriptor_dim) uint8 descriptors
        """
        if len(keypoints) == 0:
            return np.zeros((0, self.descriptor_dim), dtype=np.uint8)

        C, H, W = feature_map.shape
        w_orig, h_orig = original_size
        w_feat, h_feat = feature_size

        # Convert keypoints to feature map coordinates
        scale_x = W / w_feat
        scale_y = H / h_feat

        kp_x = keypoints[:, 0] * (w_feat / w_orig) * scale_x
        kp_y = keypoints[:, 1] * (h_feat / h_orig) * scale_y

        # Normalize to [-1, 1] for grid_sample
        grid_x = 2.0 * kp_x / (W - 1) - 1.0
        grid_y = 2.0 * kp_y / (H - 1) - 1.0

        # Convert numpy arrays to torch tensors first
        grid_x = torch.from_numpy(grid_x.astype(np.float32)).to(self.device)
        grid_y = torch.from_numpy(grid_y.astype(np.float32)).to(self.device)

        # Stack into grid
        grid = torch.stack([grid_x, grid_y], dim=1)  # (N, 2)
        grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

        # Interpolate
        feature_map_batch = feature_map.unsqueeze(0)  # (1, C, H, W)
        descriptors = F.grid_sample(
            feature_map_batch,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        descriptors = descriptors.squeeze(0).squeeze(2).t()  # (N, C)

        # Reduce dimensionality if needed
        if descriptors.shape[1] > self.descriptor_dim:
            descriptors = self._reduce_descriptor_dim(descriptors)

        # L1 normalize, then square root, then L2 normalize (RootSIFT)
        descriptors = F.normalize(descriptors, p=1, dim=1)
        descriptors = torch.sqrt(torch.clamp(descriptors, min=1e-8))
        descriptors = F.normalize(descriptors, p=2, dim=1)

        # Convert to uint8
        descriptors_np = descriptors.cpu().numpy()
        descriptors_uint8 = (descriptors_np * 512.0).clip(0, 255).astype(np.uint8)

        return descriptors_uint8

    def _reduce_descriptor_dim(self, descriptors: torch.Tensor):
        """Reduce descriptor dimensionality using PCA."""
        if self.descriptor_projection is None:
            input_dim = descriptors.shape[1]

            if descriptors.shape[0] > self.descriptor_dim:
                desc_centered = descriptors - descriptors.mean(dim=0, keepdim=True)
                U, S, V = torch.svd(desc_centered.cpu())
                self.descriptor_projection = V[:, : self.descriptor_dim].to(self.device)
                print(
                    f"Initialized PCA projection: {input_dim} -> {self.descriptor_dim}"
                )
                print(
                    f"  Variance explained: {(S[:self.descriptor_dim].sum() / S.sum()).item():.2%}"
                )
            else:
                self.descriptor_projection = torch.randn(
                    input_dim,
                    self.descriptor_dim,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.descriptor_projection /= np.sqrt(input_dim)
                print(
                    f"Initialized random projection: {input_dim} -> {self.descriptor_dim}"
                )

        return descriptors @ self.descriptor_projection

    def _run_inference(self, image_bgr: np.ndarray):
        """
        Run hybrid inference: traditional detection + ViT descriptors.

        Args:
            image_bgr: Input image in BGR format

        Returns:
            keypoints: (N, 2) float32 array of (x, y) coordinates
            descriptors: (N, 128) uint8 array of descriptors
        """
        h_orig, w_orig = image_bgr.shape[:2]

        # Step 1: Detect keypoints with traditional detector
        keypoints = self._detect_keypoints(image_bgr)

        if len(keypoints) == 0:
            print("Warning: No keypoints detected")
            return keypoints, np.zeros((0, self.descriptor_dim), dtype=np.uint8)

        # Step 2: Extract ViT features
        feature_map, feature_size = self._extract_vit_features(image_bgr)

        # Step 3: Extract ViT descriptors at keypoint locations
        descriptors = self._extract_descriptors_at_keypoints(
            feature_map, keypoints, (w_orig, h_orig), feature_size
        )

        return keypoints, descriptors

    def extract(
        self,
        image_dir: Path,
        db_path: Path,
        camera_model: str,
        camera_params: Optional[list[float]] = None,
    ):
        """Extract hybrid features for all images and write to database."""
        import pycolmap
        from vit_colmap.database.colmap_db import ColmapDatabase

        print(f"\n{'='*60}")
        print("Hybrid Feature Extraction")
        print(f"{'='*60}")
        print(f"Keypoint detector: {self.detector_type.upper()}")
        print(f"Descriptor: ViT ({self.model_name})")
        print(f"{'='*60}\n")

        # Get image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = sorted(
            [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
        )

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Found {len(image_files)} images")

        # Initialize database
        db = ColmapDatabase(str(db_path))

        # Get image dimensions
        first_img = cv2.imread(str(image_files[0]))
        height, width = first_img.shape[:2]

        # Set camera parameters
        if camera_params is None:
            if camera_model == "SIMPLE_PINHOLE":
                f = max(width, height)
                camera_params = [f, width / 2.0, height / 2.0]
            elif camera_model == "PINHOLE":
                f = max(width, height)
                camera_params = [f, f, width / 2.0, height / 2.0]
            else:
                raise ValueError(f"Unsupported camera model: {camera_model}")

        # Add camera
        camera = pycolmap.Camera(
            model=camera_model, width=width, height=height, params=camera_params
        )
        camera_id = db.db.write_camera(camera)
        print(f"Camera ID: {camera_id}\n")

        # Process images
        for idx, img_file in enumerate(image_files, start=1):
            print(f"[{idx}/{len(image_files)}] Processing: {img_file.name}")

            img = cv2.imread(str(img_file))
            if img is None:
                print("  ⚠ Warning: Failed to read image, skipping")
                continue

            image_id = db.add_image(img_file.name, camera_id=camera_id)

            try:
                keypoints, descriptors = self._run_inference(img)

                print(f"  → Extracted {len(keypoints)} keypoints")
                print(f"  → Descriptor shape: {descriptors.shape}")

                if len(keypoints) == 0:
                    print("  ⚠ Warning: No keypoints extracted")
                    continue

                db.add_keypoints(image_id, keypoints)
                db.add_descriptors(image_id, descriptors)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback

                traceback.print_exc()
                continue

        db.commit()
        print(f"\n{'='*60}")
        print("✓ Feature extraction complete!")
        print(f"{'='*60}\n")
