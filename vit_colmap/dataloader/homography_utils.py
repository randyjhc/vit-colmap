"""
Homography utilities for warping ViT patch tokens.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def warp_patch_tokens(
    patch_tokens: torch.Tensor,
    H: torch.Tensor,
    patch_size: int = 14,
    input_size: Tuple[int, int] = (1200, 1600),
) -> torch.Tensor:
    """
    Warp ViT patch tokens using homography.

    This function warps the source patch tokens to align with the target coordinate system.
    Given H that maps source -> target, we compute the inverse to perform backward warping:
    for each location in the output (target space), we find where to sample from the input (source).

    Args:
        patch_tokens: (B, C, H_patches, W_patches) feature map from ViT backbone
                     e.g., (B, 768, 85, 114) for 1200x1600 input with patch_size=14
        H: (B, 3, 3) homography matrices mapping source to target coordinates
        patch_size: ViT patch size (default 14 for DINOv2)
        input_size: Original image size (H, W)

    Returns:
        warped_tokens: (B, C, H_patches, W_patches) warped feature map aligned to target space
    """
    B, C, H_p, W_p = patch_tokens.shape
    device = patch_tokens.device

    # Compute inverse homography for backward warping
    # H maps source -> target, H_inv maps target -> source
    H_inv = torch.linalg.inv(H)

    # Create grid of OUTPUT (target) patch centers in pixel coordinates
    # Each patch center is at (patch_size * i + patch_size/2, patch_size * j + patch_size/2)
    y_coords = (
        torch.arange(H_p, device=device, dtype=torch.float32) + 0.5
    ) * patch_size
    x_coords = (
        torch.arange(W_p, device=device, dtype=torch.float32) + 0.5
    ) * patch_size

    # Create meshgrid of patch centers (these are target/output locations)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # (H_p, W_p)

    # Flatten and add homogeneous coordinate
    # Shape: (H_p * W_p, 3)
    ones = torch.ones_like(xx)
    coords = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3)  # (H_p*W_p, 3)

    # Batch the coordinates
    coords = coords.unsqueeze(0).expand(B, -1, -1)  # (B, H_p*W_p, 3)

    # Apply INVERSE homography: find source locations for each target location
    # H_inv @ target_coords = source_coords
    coords_t = coords.permute(0, 2, 1)  # (B, 3, H_p*W_p)
    source_coords = torch.bmm(H_inv, coords_t)  # (B, 3, H_p*W_p)

    # Convert from homogeneous to Cartesian coordinates
    source_coords = source_coords.permute(0, 2, 1)  # (B, H_p*W_p, 3)
    w = source_coords[:, :, 2:3].clamp(min=1e-8)  # Avoid division by zero
    source_xy = source_coords[:, :, :2] / w  # (B, H_p*W_p, 2)

    # Reshape back to grid
    source_xy = source_xy.reshape(B, H_p, W_p, 2)  # (B, H_p, W_p, 2)

    # Normalize coordinates to [-1, 1] for grid_sample
    # x: [0, W*patch_size] -> [-1, 1]
    # y: [0, H*patch_size] -> [-1, 1]
    img_h, img_w = input_size
    source_xy[..., 0] = (source_xy[..., 0] / img_w) * 2 - 1  # x
    source_xy[..., 1] = (source_xy[..., 1] / img_h) * 2 - 1  # y

    # Use grid_sample to interpolate features from source
    # For each output location, sample from the computed source location
    warped_tokens = F.grid_sample(
        patch_tokens,
        source_xy,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    return warped_tokens


def create_correspondence_map(
    H: torch.Tensor,
    feature_size: Tuple[int, int],
    image_size: Tuple[int, int],
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Create pixel-wise correspondence map from homography.

    For each position in the TARGET feature map, compute the corresponding
    position in the SOURCE feature map (backward mapping for sampling).

    Args:
        H: (B, 3, 3) homography matrices mapping source to target
        feature_size: (H_patches, W_patches) size of feature maps
        image_size: (H, W) original image size
        patch_size: ViT patch size

    Returns:
        correspondence_map: (B, H_patches, W_patches, 2) - (x, y) patch indices in source
    """
    B = H.shape[0]
    H_p, W_p = feature_size
    img_h, img_w = image_size
    device = H.device

    # Compute inverse homography for backward mapping
    H_inv = torch.linalg.inv(H)

    # Create grid of TARGET patch centers
    y_coords = (
        torch.arange(H_p, device=device, dtype=torch.float32) + 0.5
    ) * patch_size
    x_coords = (
        torch.arange(W_p, device=device, dtype=torch.float32) + 0.5
    ) * patch_size

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    ones = torch.ones_like(xx)
    coords = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3)
    coords = coords.unsqueeze(0).expand(B, -1, -1)

    # Apply inverse homography to find source locations
    coords_t = coords.permute(0, 2, 1)
    source_coords = torch.bmm(H_inv, coords_t)
    source_coords = source_coords.permute(0, 2, 1)

    # Convert to Cartesian
    w = source_coords[:, :, 2:3].clamp(min=1e-8)
    source_xy = source_coords[:, :, :2] / w

    # Convert to patch indices
    source_patch_xy = source_xy.clone()
    source_patch_xy[..., 0] = source_xy[..., 0] / patch_size - 0.5  # x -> patch x index
    source_patch_xy[..., 1] = source_xy[..., 1] / patch_size - 0.5  # y -> patch y index

    # Reshape to (B, H_p, W_p, 2)
    correspondence_map = source_patch_xy.reshape(B, H_p, W_p, 2)

    return correspondence_map


def compute_valid_mask(
    H: torch.Tensor,
    feature_size: Tuple[int, int],
    image_size: Tuple[int, int],
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Compute mask of valid correspondences (within image bounds).

    For each position in the TARGET feature map, check if the corresponding
    SOURCE location is within valid bounds (for backward warping).

    Args:
        H: (B, 3, 3) homography matrices mapping source to target
        feature_size: (H_patches, W_patches) size of feature maps
        image_size: (H, W) original image size
        patch_size: ViT patch size

    Returns:
        valid_mask: (B, H_patches, W_patches) boolean mask
    """
    B = H.shape[0]
    H_p, W_p = feature_size
    img_h, img_w = image_size
    device = H.device

    # Compute inverse homography for backward mapping
    H_inv = torch.linalg.inv(H)

    # Create grid of TARGET patch centers
    y_coords = (
        torch.arange(H_p, device=device, dtype=torch.float32) + 0.5
    ) * patch_size
    x_coords = (
        torch.arange(W_p, device=device, dtype=torch.float32) + 0.5
    ) * patch_size

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    ones = torch.ones_like(xx)
    coords = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3)
    coords = coords.unsqueeze(0).expand(B, -1, -1)

    # Apply inverse homography to find source locations
    coords_t = coords.permute(0, 2, 1)
    source_coords = torch.bmm(H_inv, coords_t)
    source_coords = source_coords.permute(0, 2, 1)

    # Convert to Cartesian
    w = source_coords[:, :, 2:3]
    source_xy = source_coords[:, :, :2] / w.clamp(min=1e-8)

    # Check if source coordinates are within image bounds
    x_valid = (source_xy[..., 0] >= 0) & (source_xy[..., 0] < img_w)
    y_valid = (source_xy[..., 1] >= 0) & (source_xy[..., 1] < img_h)
    w_valid = w.squeeze(-1) > 0  # Valid homogeneous coordinate

    valid = x_valid & y_valid & w_valid

    # Reshape to (B, H_p, W_p)
    valid_mask = valid.reshape(B, H_p, W_p)

    return valid_mask


def compute_feature_similarity(
    features1: torch.Tensor,
    features2: torch.Tensor,
    H: torch.Tensor,
    patch_size: int = 14,
    input_size: Tuple[int, int] = (1200, 1600),
) -> torch.Tensor:
    """
    Compute similarity between warped features1 and features2.

    This is useful for evaluating how well the homography aligns features.

    Args:
        features1: (B, C, H_p, W_p) source features
        features2: (B, C, H_p, W_p) target features
        H: (B, 3, 3) homography from source to target
        patch_size: ViT patch size
        input_size: Original image size (H, W)

    Returns:
        similarity: (B, H_p, W_p) cosine similarity map
    """
    # Warp features1 to target coordinate system
    warped_features1 = warp_patch_tokens(features1, H, patch_size, input_size)

    # Compute cosine similarity
    # Normalize features
    warped_norm = F.normalize(warped_features1, p=2, dim=1)
    target_norm = F.normalize(features2, p=2, dim=1)

    # Cosine similarity at each spatial location
    similarity = (warped_norm * target_norm).sum(dim=1)  # (B, H_p, W_p)

    return similarity


def warp_image_with_homography(
    img: torch.Tensor, H: torch.Tensor, output_size: Tuple[int, int] = None
) -> torch.Tensor:
    """
    Warp entire image using homography (for visualization).

    Args:
        img: (B, C, H, W) input image tensor
        H: (B, 3, 3) homography matrix
        output_size: (H, W) output size (default: same as input)

    Returns:
        warped_img: (B, C, H, W) warped image
    """
    B, C, H_img, W_img = img.shape
    device = img.device

    if output_size is None:
        out_h, out_w = H_img, W_img
    else:
        out_h, out_w = output_size

    # Create output grid
    y_coords = torch.arange(out_h, device=device, dtype=torch.float32)
    x_coords = torch.arange(out_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    ones = torch.ones_like(xx)
    coords = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3)
    coords = coords.unsqueeze(0).expand(B, -1, -1)

    # Apply inverse homography to find source coordinates
    H_inv = torch.linalg.inv(H)
    coords_t = coords.permute(0, 2, 1)
    source_coords = torch.bmm(H_inv, coords_t)
    source_coords = source_coords.permute(0, 2, 1)

    # Convert to Cartesian
    w = source_coords[:, :, 2:3].clamp(min=1e-8)
    source_xy = source_coords[:, :, :2] / w
    source_xy = source_xy.reshape(B, out_h, out_w, 2)

    # Normalize to [-1, 1]
    source_xy[..., 0] = (source_xy[..., 0] / W_img) * 2 - 1
    source_xy[..., 1] = (source_xy[..., 1] / H_img) * 2 - 1

    # Sample from source image
    warped_img = F.grid_sample(
        img, source_xy, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    return warped_img
