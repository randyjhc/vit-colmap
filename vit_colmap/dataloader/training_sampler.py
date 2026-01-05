"""
Training sampler for generating positive and negative examples.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class TrainingSampler:
    """Generate positive and negative examples for self-supervised learning."""

    def __init__(
        self,
        top_k_invariant: int = 512,
        negative_radius: int = 16,
        num_in_image_negatives: int = 10,
        num_hard_negatives: int = 5,
        patch_size: int = 14,
        selection_mode: str = "top_k",
        similarity_threshold: float = 0.7,
        min_invariant_points: int = 100,
        max_invariant_points: int = 2048,
    ):
        """
        Args:
            top_k_invariant: Number of invariant points to select (for top_k mode)
            negative_radius: Minimum pixel distance for in-image negatives
            num_in_image_negatives: Number of in-image negatives per point
            num_hard_negatives: Number of hard negatives per point
            patch_size: ViT patch size (for coordinate scaling)
            selection_mode: Point selection strategy - "top_k", "threshold", or "hybrid"
            similarity_threshold: Minimum similarity for threshold-based selection
            min_invariant_points: Minimum number of points to select (for threshold mode)
            max_invariant_points: Maximum number of points to prevent OOM
        """
        self.top_k_invariant = top_k_invariant
        self.negative_radius = negative_radius
        self.num_in_image_negatives = num_in_image_negatives
        self.num_hard_negatives = num_hard_negatives
        self.patch_size = patch_size

        # Threshold-based selection parameters
        self.selection_mode = selection_mode
        assert selection_mode in [
            "top_k",
            "threshold",
            "hybrid",
        ], f"Invalid selection_mode: {selection_mode}"

        self.similarity_threshold = similarity_threshold
        self.min_invariant_points = min_invariant_points
        self.max_invariant_points = max_invariant_points

    def select_invariant_points(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        H: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Select invariant points based on cosine similarity after warping.

        Selection modes:
        - "top_k": Select top K points with highest similarity
        - "threshold": Select all points with similarity > threshold
        - "hybrid": Select points above threshold, then take top K from those

        Args:
            features1: (B, C, H_p, W_p) features from image 1
            features2: (B, C, H_p, W_p) features from image 2
            H: (B, 3, 3) homography matrices (image1 -> image2)
            image_size: (H, W) original image size

        Returns:
            invariant_coords: (B, K, 2) coordinates in feature map space (x, y)
        """
        from vit_colmap.dataloader.homography_utils import warp_patch_tokens

        B, C, H_p, W_p = features1.shape

        # Warp features1 to features2 coordinate system
        warped_features1 = warp_patch_tokens(
            features1, H, patch_size=self.patch_size, input_size=image_size
        )

        # Compute cosine similarity at each location
        # Normalize features
        warped_norm = F.normalize(warped_features1, p=2, dim=1)  # (B, C, H_p, W_p)
        features2_norm = F.normalize(features2, p=2, dim=1)  # (B, C, H_p, W_p)

        # Cosine similarity per location
        similarity = (warped_norm * features2_norm).sum(dim=1)  # (B, H_p, W_p)

        # Flatten similarity map
        similarity_flat = similarity.view(B, -1)  # (B, H_p * W_p)

        if self.selection_mode == "top_k":
            # Select top-k points by similarity
            k = min(self.top_k_invariant, similarity_flat.shape[1])
            _, top_k_indices = similarity_flat.topk(k, dim=1)  # (B, K)

        elif self.selection_mode == "threshold":
            # Select all points above threshold (clamped to min/max range)
            invariant_coords_list = []
            for b in range(B):
                # Find points above threshold
                above_threshold = similarity_flat[b] >= self.similarity_threshold
                indices = torch.nonzero(above_threshold, as_tuple=True)[0]

                # Clamp to min/max range
                num_points = len(indices)
                if num_points < self.min_invariant_points:
                    # Not enough points above threshold, take top min_points
                    k = min(self.min_invariant_points, similarity_flat.shape[1])
                    _, indices = similarity_flat[b].topk(k)
                elif num_points > self.max_invariant_points:
                    # Too many points, select top max_points from those above threshold
                    scores = similarity_flat[b, indices]
                    _, top_indices = scores.topk(self.max_invariant_points)
                    indices = indices[top_indices]

                invariant_coords_list.append(indices)

            # Pad to same length for batching
            max_len = max(len(coords) for coords in invariant_coords_list)
            top_k_indices = torch.zeros(
                B, max_len, dtype=torch.long, device=features1.device
            )
            for b, coords in enumerate(invariant_coords_list):
                top_k_indices[b, : len(coords)] = coords
                # Pad with first index if needed
                if len(coords) < max_len:
                    top_k_indices[b, len(coords) :] = coords[0]

        elif self.selection_mode == "hybrid":
            # First filter by threshold, then select top-k from those
            invariant_coords_list = []
            for b in range(B):
                # Find points above threshold
                above_threshold = similarity_flat[b] >= self.similarity_threshold
                indices = torch.nonzero(above_threshold, as_tuple=True)[0]

                if len(indices) == 0:
                    # No points above threshold, fall back to top-k
                    k = min(self.top_k_invariant, similarity_flat.shape[1])
                    _, indices = similarity_flat[b].topk(k)
                elif len(indices) > self.top_k_invariant:
                    # Select top-k from those above threshold
                    scores = similarity_flat[b, indices]
                    _, top_indices = scores.topk(self.top_k_invariant)
                    indices = indices[top_indices]

                invariant_coords_list.append(indices)

            # Pad to same length for batching
            max_len = max(len(coords) for coords in invariant_coords_list)
            top_k_indices = torch.zeros(
                B, max_len, dtype=torch.long, device=features1.device
            )
            for b, coords in enumerate(invariant_coords_list):
                top_k_indices[b, : len(coords)] = coords
                # Pad with first index if needed
                if len(coords) < max_len:
                    top_k_indices[b, len(coords) :] = coords[0]

        # Convert flat indices to 2D coordinates
        # indices = y * W_p + x
        coords_y = top_k_indices // W_p  # (B, K)
        coords_x = top_k_indices % W_p  # (B, K)

        # Stack as (x, y) coordinates
        invariant_coords = torch.stack(
            [coords_x, coords_y], dim=-1
        ).float()  # (B, K, 2)

        return invariant_coords

    def sample_features_at_coords(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear sample features at specified coordinates.

        Args:
            features: (B, C, H, W) feature map
            coords: (B, K, 2) coordinates (x, y) in feature map space

        Returns:
            sampled: (B, K, C) sampled features
        """
        B, C, H, W = features.shape

        # Normalize coordinates to [-1, 1] for grid_sample
        grid = coords.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # x
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # y

        # Reshape for grid_sample: (B, K, 1, 2)
        grid = grid.unsqueeze(2)

        # Sample: output is (B, C, K, 1)
        sampled = F.grid_sample(features, grid, mode="bilinear", align_corners=False)

        # Reshape to (B, K, C)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        return sampled

    def transform_coords_with_homography(
        self,
        coords: torch.Tensor,
        H: torch.Tensor,
        from_feature_space: bool = True,
        image_size: Tuple[int, int] = None,
        feature_size: Tuple[int, int] = None,
    ) -> torch.Tensor:
        """
        Transform coordinates using homography.

        Args:
            coords: (B, K, 2) coordinates (x, y)
            H: (B, 3, 3) homography matrices
            from_feature_space: If True, coords are in feature map space
            image_size: (H, W) original image size
            feature_size: (H_p, W_p) feature map size

        Returns:
            transformed_coords: (B, K, 2) transformed coordinates
        """
        B, K, _ = coords.shape

        # If in feature space, convert to image space first
        if from_feature_space and feature_size is not None:
            # Scale from feature map coords to image coords
            # Feature map center = (x + 0.5) * patch_size
            image_coords = coords.clone()
            image_coords[..., 0] = (coords[..., 0] + 0.5) * self.patch_size
            image_coords[..., 1] = (coords[..., 1] + 0.5) * self.patch_size
        else:
            image_coords = coords

        # Add homogeneous coordinate
        ones = torch.ones(B, K, 1, device=coords.device, dtype=coords.dtype)
        homogeneous = torch.cat([image_coords, ones], dim=-1)  # (B, K, 3)

        # Apply homography: H @ coords^T
        # (B, 3, 3) @ (B, 3, K) -> (B, 3, K)
        transformed = torch.bmm(H, homogeneous.permute(0, 2, 1))  # (B, 3, K)
        transformed = transformed.permute(0, 2, 1)  # (B, K, 3)

        # Convert from homogeneous to Cartesian
        w = transformed[..., 2:3].clamp(min=1e-8)
        transformed_xy = transformed[..., :2] / w  # (B, K, 2)

        # If needed, convert back to feature space
        if from_feature_space and feature_size is not None:
            transformed_xy[..., 0] = transformed_xy[..., 0] / self.patch_size - 0.5
            transformed_xy[..., 1] = transformed_xy[..., 1] / self.patch_size - 0.5

        return transformed_xy

    def generate_positive_pairs(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        H: torch.Tensor,
        invariant_coords: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate positive descriptor pairs.

        For each invariant point p in features2 (image2):
        - Map p to features1 (image1) using inverse homography
        - Bilinear sample features at both locations

        Args:
            features1: (B, C, H_p, W_p) features from image 1
            features2: (B, C, H_p, W_p) features from image 2
            H: (B, 3, 3) homography (image1 -> image2)
            invariant_coords: (B, K, 2) coordinates in features2 space
            image_size: (H, W) original image size

        Returns:
            f1: (B, K, C) features from image 1 at corresponding locations
            f2: (B, K, C) features from image 2 at invariant locations
        """
        B, C, H_p, W_p = features1.shape
        feature_size = (H_p, W_p)

        # Sample features from image2 at invariant locations
        f2 = self.sample_features_at_coords(features2, invariant_coords)  # (B, K, C)

        # Map invariant_coords from image2 space to image1 space
        # Using inverse homography
        H_inv = torch.linalg.inv(H)

        # Transform coordinates
        coords_in_img1 = self.transform_coords_with_homography(
            invariant_coords,
            H_inv,
            from_feature_space=True,
            image_size=image_size,
            feature_size=feature_size,
        )

        # Sample features from image1 at corresponding locations
        f1 = self.sample_features_at_coords(features1, coords_in_img1)  # (B, K, C)

        return f1, f2

    def generate_in_image_negatives(
        self,
        features: torch.Tensor,
        positive_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate in-image negatives: points >= radius away from positives.

        Args:
            features: (B, C, H_p, W_p) feature map
            positive_coords: (B, K, 2) positive point coordinates

        Returns:
            negatives: (B, K, N, C) negative features
        """
        B, C, H_p, W_p = features.shape
        K = positive_coords.shape[1]
        N = self.num_in_image_negatives
        device = features.device

        # Create grid of all coordinates
        y_coords = torch.arange(H_p, device=device, dtype=torch.float32)
        x_coords = torch.arange(W_p, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H_p*W_p, 2)

        negatives_list = []

        for b in range(B):
            batch_negatives = []
            for k in range(K):
                pos_coord = positive_coords[b, k]  # (2,)

                # Compute distances to all points
                distances = torch.norm(all_coords - pos_coord.unsqueeze(0), dim=-1)

                # Find points >= radius away
                valid_mask = distances >= self.negative_radius
                valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

                if len(valid_indices) >= N:
                    # Randomly sample N negatives
                    perm = torch.randperm(len(valid_indices), device=device)[:N]
                    neg_indices = valid_indices[perm]
                else:
                    # Use all available if not enough
                    neg_indices = valid_indices
                    # Pad with random if needed
                    if len(neg_indices) < N:
                        padding = torch.randint(
                            0, H_p * W_p, (N - len(neg_indices),), device=device
                        )
                        neg_indices = torch.cat([neg_indices, padding])

                neg_coords = all_coords[neg_indices]  # (N, 2)
                batch_negatives.append(neg_coords)

            batch_negatives = torch.stack(batch_negatives, dim=0)  # (K, N, 2)
            negatives_list.append(batch_negatives)

        neg_coords_all = torch.stack(negatives_list, dim=0)  # (B, K, N, 2)

        # Sample features at negative locations
        # Reshape for sampling
        neg_coords_flat = neg_coords_all.view(B, K * N, 2)  # (B, K*N, 2)
        neg_features_flat = self.sample_features_at_coords(
            features, neg_coords_flat
        )  # (B, K*N, C)
        neg_features = neg_features_flat.view(B, K, N, C)  # (B, K, N, C)

        return neg_features

    def generate_cross_image_negatives(
        self,
        features_batch: torch.Tensor,
        positive_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate cross-image negatives from other images in batch.

        Args:
            features_batch: (B, C, H_p, W_p) features from all images in batch
            positive_coords: (B, K, 2) positive point coordinates

        Returns:
            negatives: (B, K, N, C) negative features from other images
        """
        B, C, H_p, W_p = features_batch.shape
        K = positive_coords.shape[1]
        device = features_batch.device

        if B < 2:
            # No other images in batch, return zeros
            return torch.zeros(B, K, self.num_in_image_negatives, C, device=device)

        # For each image, sample from other images
        negatives_list = []

        for b in range(B):
            # Get indices of other images
            other_indices = [i for i in range(B) if i != b]

            # Sample random points from other images
            N = self.num_in_image_negatives
            batch_negatives = []

            for k in range(K):
                # Randomly select which other images to sample from
                selected_images = np.random.choice(
                    other_indices, size=min(N, len(other_indices)), replace=True
                )

                neg_features = []
                for img_idx in selected_images:
                    # Random coordinate in that image
                    rand_x = torch.randint(0, W_p, (1,), device=device).float()
                    rand_y = torch.randint(0, H_p, (1,), device=device).float()
                    rand_coord = torch.stack([rand_x, rand_y], dim=-1).unsqueeze(
                        0
                    )  # (1, 1, 2)

                    # Sample feature
                    feat = self.sample_features_at_coords(
                        features_batch[img_idx : img_idx + 1], rand_coord
                    )  # (1, 1, C)
                    neg_features.append(feat.squeeze(0).squeeze(0))  # (C,)

                neg_features = torch.stack(neg_features, dim=0)  # (N, C)
                batch_negatives.append(neg_features)

            batch_negatives = torch.stack(batch_negatives, dim=0)  # (K, N, C)
            negatives_list.append(batch_negatives)

        negatives = torch.stack(negatives_list, dim=0)  # (B, K, N, C)
        return negatives

    def generate_hard_negatives(
        self,
        features: torch.Tensor,
        positive_features: torch.Tensor,
        positive_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate hard negatives: high cosine similarity but geometrically far.

        Args:
            features: (B, C, H_p, W_p) feature map
            positive_features: (B, K, C) features at positive locations
            positive_coords: (B, K, 2) positive coordinates

        Returns:
            hard_negatives: (B, K, N_hard, C) hard negative features
        """
        B, C, H_p, W_p = features.shape
        K = positive_coords.shape[1]
        N_hard = self.num_hard_negatives
        device = features.device

        # Flatten spatial dimensions for similarity computation
        features_flat = features.view(B, C, -1).permute(0, 2, 1)  # (B, H_p*W_p, C)
        features_flat_norm = F.normalize(features_flat, p=2, dim=-1)
        positive_features_norm = F.normalize(
            positive_features, p=2, dim=-1
        )  # (B, K, C)

        # Create coordinate grid
        y_coords = torch.arange(H_p, device=device, dtype=torch.float32)
        x_coords = torch.arange(W_p, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        all_coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H_p*W_p, 2)

        hard_negatives_list = []

        for b in range(B):
            batch_hard_negs = []
            for k in range(K):
                # Compute cosine similarity to all locations
                pos_feat = positive_features_norm[b, k]  # (C,)
                similarities = features_flat_norm[b] @ pos_feat  # (H_p*W_p,)

                # Compute spatial distances
                pos_coord = positive_coords[b, k]  # (2,)
                distances = torch.norm(all_coords - pos_coord.unsqueeze(0), dim=-1)

                # Mask out geometrically close points
                far_mask = distances >= self.negative_radius
                similarities_masked = similarities.clone()
                similarities_masked[~far_mask] = -float("inf")

                # Select top-k most similar (but geometrically far)
                _, hard_neg_indices = similarities_masked.topk(
                    min(N_hard, far_mask.sum().item())
                )

                # Get features at those locations
                hard_neg_coords = all_coords[hard_neg_indices].unsqueeze(
                    0
                )  # (1, N_hard, 2)
                hard_neg_feats = self.sample_features_at_coords(
                    features[b : b + 1], hard_neg_coords
                )  # (1, N_hard, C)
                hard_neg_feats = hard_neg_feats.squeeze(0)  # (N_hard, C)

                # Pad if needed
                if hard_neg_feats.shape[0] < N_hard:
                    padding = torch.zeros(
                        N_hard - hard_neg_feats.shape[0], C, device=device
                    )
                    hard_neg_feats = torch.cat([hard_neg_feats, padding], dim=0)

                batch_hard_negs.append(hard_neg_feats)

            batch_hard_negs = torch.stack(batch_hard_negs, dim=0)  # (K, N_hard, C)
            hard_negatives_list.append(batch_hard_negs)

        hard_negatives = torch.stack(hard_negatives_list, dim=0)  # (B, K, N_hard, C)
        return hard_negatives

    def generate_all_negatives(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        positive_features: torch.Tensor,
        positive_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate all types of negatives and concatenate.

        Args:
            features1: (B, C, H_p, W_p) features from image 1
            features2: (B, C, H_p, W_p) features from image 2
            positive_features: (B, K, C) features at positive locations (from image 2)
            positive_coords: (B, K, 2) positive coordinates in image 2

        Returns:
            all_negatives: (B, K, N_total, C) all negative features
        """
        # In-image negatives (from image 2)
        in_image_negs = self.generate_in_image_negatives(features2, positive_coords)

        # Cross-image negatives (from image 1, which is different view)
        cross_image_negs = self.generate_cross_image_negatives(
            features1, positive_coords
        )

        # Hard negatives (from image 2, high similarity but far)
        hard_negs = self.generate_hard_negatives(
            features2, positive_features, positive_coords
        )

        # Concatenate all negatives
        all_negatives = torch.cat([in_image_negs, cross_image_negs, hard_negs], dim=2)

        return all_negatives
