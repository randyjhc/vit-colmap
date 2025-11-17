"""
Batch processing for training with positive/negative sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .training_sampler import TrainingSampler


class TrainingBatchProcessor:
    """Process batches for training with positive/negative sampling."""

    def __init__(
        self,
        model: nn.Module,
        sampler: Optional[TrainingSampler] = None,
        descriptor_head: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: ViTFeatureModel instance
            sampler: TrainingSampler instance (created if None)
            descriptor_head: Optional separate descriptor head (uses model's head if None)
        """
        self.model = model
        self.sampler = sampler if sampler is not None else TrainingSampler()
        self.descriptor_head = descriptor_head
        self._projection_matrix: Optional[torch.Tensor] = None

    def extract_rotation_from_homography(self, H: torch.Tensor) -> torch.Tensor:
        """
        Extract rotation angle from homography matrix.

        For a homography H = [[a, b, tx], [c, d, ty], [e, f, 1]],
        the rotation component is approximately atan2(c, a) for small perspective.

        Args:
            H: (B, 3, 3) homography matrices

        Returns:
            rotation_angle: (B,) rotation angles in radians
        """
        # Extract rotation from the top-left 2x2 submatrix
        # This is approximate for perspective transforms
        rotation_angle = torch.atan2(H[:, 1, 0], H[:, 0, 0])
        return rotation_angle

    def project_to_128d(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 768D (or other dim) features to 128D using descriptor head.

        Args:
            features: (B, K, C) features (C=768 for DINOv2)

        Returns:
            descriptors: (B, K, 128) L2-normalized descriptors
        """
        B, K, C = features.shape

        # If we have a separate descriptor head, use it
        if self.descriptor_head is not None:
            # Reshape for conv: (B*K, C, 1, 1)
            features_reshaped = features.view(B * K, C, 1, 1)
            descriptors = self.descriptor_head(features_reshaped)
            descriptors = descriptors.view(B, K, -1)
        else:
            # Use model's descriptor head
            # The model's head expects (B, C, H, W), so reshape accordingly
            features_reshaped = features.permute(0, 2, 1).unsqueeze(-1)  # (B, C, K, 1)

            # Pass through model's trunk and descriptor head
            # Since we already have backbone features, we need to project them
            # Use a simple linear projection as fallback
            if hasattr(self.model, "trunk") and hasattr(self.model, "descriptor_head"):
                # This is a simplification - ideally we'd use the full head
                # For now, use a simple learned projection
                descriptors = self._simple_projection(features, target_dim=128)
            else:
                descriptors = self._simple_projection(features, target_dim=128)

        # L2 normalize
        descriptors = F.normalize(descriptors, p=2, dim=-1)
        return descriptors

    def _simple_projection(
        self,
        features: torch.Tensor,
        target_dim: int = 128,
    ) -> torch.Tensor:
        """
        Simple linear projection to target dimension.

        Args:
            features: (B, K, C) input features
            target_dim: Output dimension

        Returns:
            projected: (B, K, target_dim)
        """
        B, K, C = features.shape
        device = features.device

        # Create projection matrix if not exists
        if self._projection_matrix is None or self._projection_matrix.shape[0] != C:
            # Random orthogonal projection (Xavier initialization)
            self._projection_matrix = torch.randn(
                C, target_dim, device=device, dtype=features.dtype
            )
            self._projection_matrix /= np.sqrt(C)

        # Move to correct device if needed
        if self._projection_matrix.device != device:
            self._projection_matrix = self._projection_matrix.to(device)

        # Project: (B, K, C) @ (C, target_dim) -> (B, K, target_dim)
        projected = features @ self._projection_matrix
        return projected

    def create_score_ground_truth(
        self,
        invariant_coords: torch.Tensor,
        feature_size: Tuple[int, int],
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Create ground truth score heatmap from invariant points.

        Args:
            invariant_coords: (B, K, 2) coordinates in feature map space
            feature_size: (H_p, W_p) feature map size
            sigma: Gaussian sigma for heatmap

        Returns:
            score_gt: (B, 1, H_p, W_p) ground truth heatmap
        """
        B, K, _ = invariant_coords.shape
        H_p, W_p = feature_size
        device = invariant_coords.device

        # Create coordinate grids
        y_coords = torch.arange(H_p, device=device, dtype=torch.float32)
        x_coords = torch.arange(W_p, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # (H_p, W_p)

        # Compute Gaussian heatmap
        score_gt = torch.zeros(B, 1, H_p, W_p, device=device)

        for b in range(B):
            for k in range(K):
                cx, cy = invariant_coords[b, k]
                # Gaussian centered at (cx, cy)
                gaussian = torch.exp(
                    -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
                )
                score_gt[b, 0] = torch.maximum(score_gt[b, 0], gaussian)

        return score_gt

    def sample_orientations_at_coords(
        self,
        keypoint_outputs: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample orientations from keypoint head output at specified coordinates.

        Args:
            keypoint_outputs: (B, 4, H, W) - [score, dx, dy, orientation]
            coords: (B, K, 2) coordinates

        Returns:
            orientations: (B, K) sampled orientations
        """
        B, _, H, W = keypoint_outputs.shape

        # Extract orientation channel
        orientations_map = keypoint_outputs[:, 3:4, :, :]  # (B, 1, H, W)

        # Normalize coordinates for grid_sample
        grid = coords.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1
        grid = grid.unsqueeze(2)  # (B, K, 1, 2)

        # Sample
        sampled = F.grid_sample(
            orientations_map, grid, mode="bilinear", align_corners=False
        )
        orientations = sampled.squeeze(1).squeeze(-1)  # (B, K)

        return orientations

    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a batch of image pairs for training.

        Args:
            batch: Dict from HPatchesDataset containing:
                - img1: (B, 3, H, W)
                - img2: (B, 3, H, W)
                - H: (B, 3, 3) homography

        Returns:
            outputs: Dict containing model predictions
            targets: Dict containing ground truth and generated samples
        """
        img1 = batch["img1"]
        img2 = batch["img2"]
        H = batch["H"]

        B, _, img_H, img_W = img1.shape
        image_size = (img_H, img_W)

        # 1. Extract backbone features ONCE (frozen, use inference_mode for better perf)
        with torch.inference_mode():
            features1 = self.model._extract_backbone_features(
                img1
            )  # (B, 768, H_p, W_p)
            features2 = self.model._extract_backbone_features(img2)

        _, C, H_p, W_p = features1.shape
        feature_size = (H_p, W_p)

        # 2. Select invariant points based on cosine similarity
        invariant_coords = self.sampler.select_invariant_points(
            features1, features2, H, image_size
        )  # (B, K, 2) in features2 space

        # 3. Generate positive pairs
        f1, f2 = self.sampler.generate_positive_pairs(
            features1, features2, H, invariant_coords, image_size
        )  # (B, K, 768)

        # 4. Project to 128D descriptors
        z1 = self.project_to_128d(f1)  # (B, K, 128)
        z2 = self.project_to_128d(f2)  # (B, K, 128)

        # 5. Generate negative examples
        negatives = self.sampler.generate_all_negatives(
            features1, features2, f2, invariant_coords
        )  # (B, K, N, 768)

        # Project negatives to 128D
        B_n, K_n, N_n, C_n = negatives.shape
        negatives_flat = negatives.view(B_n, K_n * N_n, C_n)
        negatives_128d_flat = self.project_to_128d(negatives_flat)
        negatives_128d = negatives_128d_flat.view(B_n, K_n, N_n, 128)

        # 6. Forward pass through trainable heads only (REUSE backbone features)
        # This avoids redundant backbone passes - major speedup!
        outputs1_full = self.model.forward_from_backbone_features(features1)
        outputs2_full = self.model.forward_from_backbone_features(features2)

        # 7. Sample orientations at invariant point locations
        # Map invariant coords (in features2 space) to output space (1/4 resolution)
        # Feature space is 1/14, output space is 1/4, so scale by 14/4 = 3.5
        scale_factor = self.sampler.patch_size / 4.0
        output_coords = invariant_coords * scale_factor  # Scale to 1/4 resolution space

        orientations2 = self.sample_orientations_at_coords(
            outputs2_full["keypoints"], output_coords
        )

        # Map to image1's output space
        H_inv = torch.linalg.inv(H)
        coords_in_img1_feature = self.sampler.transform_coords_with_homography(
            invariant_coords,
            H_inv,
            from_feature_space=True,
            image_size=image_size,
            feature_size=feature_size,
        )
        output_coords_img1 = coords_in_img1_feature * scale_factor

        orientations1 = self.sample_orientations_at_coords(
            outputs1_full["keypoints"], output_coords_img1
        )

        # 8. Extract rotation angle from homography
        rotation_angle = self.extract_rotation_from_homography(H)

        # 9. Create ground truth score heatmap
        # Scale invariant coords to output space (1/2 resolution)
        score_gt = self.create_score_ground_truth(
            output_coords,
            (outputs2_full["keypoints"].shape[2], outputs2_full["keypoints"].shape[3]),
        )

        # Assemble outputs and targets
        outputs = {
            "z1": z1,  # (B, K, 128)
            "z2": z2,  # (B, K, 128)
            "negatives": negatives_128d,  # (B, K, N, 128)
            "orientations1": orientations1,  # (B, K)
            "orientations2": orientations2,  # (B, K)
            "score_logits": outputs2_full["keypoints"][:, 0:1, :, :],  # (B, 1, H, W)
            "keypoints1_full": outputs1_full["keypoints"],
            "keypoints2_full": outputs2_full["keypoints"],
            "descriptors1_full": outputs1_full["descriptors"],
            "descriptors2_full": outputs2_full["descriptors"],
        }

        targets = {
            "invariant_coords": output_coords,  # (B, K, 2) in output space
            "rotation_angle": rotation_angle,  # (B,)
            "score_gt": score_gt,  # (B, 1, H, W)
            "H": H,
            "image_size": image_size,
            "feature_size": feature_size,
        }

        return outputs, targets


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch: List of dicts from HPatchesDataset

    Returns:
        Collated dict with batched tensors
    """
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
        elif isinstance(batch[0][key], str):
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = [item[key] for item in batch]
    return collated
