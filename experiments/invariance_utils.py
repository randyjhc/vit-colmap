"""
Utility functions for testing layer invariance/equivariance in vision transformers.
Provides image transformations and similarity metrics.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Image Transformation Functions
# ============================================================================


def apply_scale_transform(
    image: np.ndarray, scale_factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply scale transformation (zoom in/out) to an image.

    Args:
        image: Input image (H, W, C) in BGR format
        scale_factor: Scale factor (>1 for zoom in, <1 for zoom out)

    Returns:
        transformed_image: Scaled and cropped/padded image (same size as input)
        transform_matrix: 3x3 transformation matrix for coordinate mapping
    """
    h, w = image.shape[:2]

    # Calculate new dimensions
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # Resize image
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale_factor > 1.0:
        # Zoom in: crop center
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        transformed = scaled[start_y : start_y + h, start_x : start_x + w]
    else:
        # Zoom out: pad with zeros
        transformed = np.zeros_like(image)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        transformed[start_y : start_y + new_h, start_x : start_x + new_w] = scaled

    # Create transformation matrix
    # For center-based scaling
    cx, cy = w / 2, h / 2
    transform_matrix = np.array(
        [
            [scale_factor, 0, cx * (1 - scale_factor)],
            [0, scale_factor, cy * (1 - scale_factor)],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    return transformed, transform_matrix


def apply_rotation_transform(
    image: np.ndarray, angle_degrees: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply rotation transformation to an image.

    Args:
        image: Input image (H, W, C) in BGR format
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)

    Returns:
        transformed_image: Rotated image (same size as input)
        transform_matrix: 2x3 transformation matrix for coordinate mapping
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Get rotation matrix
    transform_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    # Apply rotation
    transformed = cv2.warpAffine(
        image,
        transform_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Convert to 3x3 matrix
    transform_matrix_3x3 = np.vstack([transform_matrix, [0, 0, 1]])

    return transformed, transform_matrix_3x3


def apply_illumination_transform(
    image: np.ndarray, brightness_factor: float = 1.0, contrast_factor: float = 1.0
) -> tuple[np.ndarray, None]:
    """
    Apply illumination changes (brightness and contrast).

    Args:
        image: Input image (H, W, C) in BGR format
        brightness_factor: Brightness multiplier (0.5 = darker, 1.5 = brighter)
        contrast_factor: Contrast multiplier (0.5 = less contrast, 1.5 = more contrast)

    Returns:
        transformed_image: Image with adjusted illumination
        transform_matrix: None (no geometric transformation)
    """
    # Convert to float for processing
    img_float = image.astype(np.float32)

    # Apply contrast (around mean)
    mean = np.mean(img_float, axis=(0, 1), keepdims=True)
    img_float = mean + contrast_factor * (img_float - mean)

    # Apply brightness
    img_float = img_float * brightness_factor

    # Clip and convert back
    transformed = np.clip(img_float, 0, 255).astype(np.uint8)

    return transformed, None


def apply_viewpoint_transform(
    image: np.ndarray, tilt_x: float = 0.0, tilt_y: float = 0.0, shear_x: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply small viewpoint shift using affine transformation.
    Simulates small camera angle changes.

    Args:
        image: Input image (H, W, C) in BGR format
        tilt_x: Horizontal tilt factor (-0.1 to 0.1 typical)
        tilt_y: Vertical tilt factor (-0.1 to 0.1 typical)
        shear_x: Horizontal shear factor (-0.1 to 0.1 typical)

    Returns:
        transformed_image: Transformed image
        transform_matrix: 3x3 transformation matrix
    """
    h, w = image.shape[:2]

    # Create affine transformation matrix
    # Combines small perspective-like changes
    transform_matrix_2x3 = np.array(
        [[1 + tilt_x, shear_x, tilt_x * w / 2], [0, 1 + tilt_y, tilt_y * h / 2]],
        dtype=np.float32,
    )

    # Apply transformation
    transformed = cv2.warpAffine(
        image,
        transform_matrix_2x3,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Convert to 3x3 matrix
    transform_matrix = np.vstack([transform_matrix_2x3, [0, 0, 1]])

    return transformed, transform_matrix


# ============================================================================
# Coordinate Transformation Functions
# ============================================================================


def transform_coordinates(
    coords: np.ndarray, transform_matrix: np.ndarray
) -> np.ndarray:
    """
    Transform pixel coordinates using a transformation matrix.

    Args:
        coords: (N, 2) array of (x, y) coordinates
        transform_matrix: 3x3 or 2x3 transformation matrix

    Returns:
        transformed_coords: (N, 2) array of transformed (x, y) coordinates
    """
    if transform_matrix is None:
        return coords

    # Convert to homogeneous coordinates
    ones = np.ones((coords.shape[0], 1))
    coords_homogeneous = np.hstack([coords, ones])  # (N, 3)

    # Ensure 3x3 matrix
    if transform_matrix.shape[0] == 2:
        transform_matrix = np.vstack([transform_matrix, [0, 0, 1]])

    # Apply transformation
    transformed = (transform_matrix @ coords_homogeneous.T).T  # (N, 3)

    # Convert back to 2D
    transformed_coords = transformed[:, :2] / transformed[:, 2:3]

    return transformed_coords


# ============================================================================
# Feature Similarity Metrics
# ============================================================================


def cosine_similarity(features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between feature vectors.

    Args:
        features1: (N, D) or (D,) tensor
        features2: (N, D) or (D,) tensor

    Returns:
        similarity: Scalar or (N,) tensor of cosine similarities in range [-1, 1]
    """
    # Normalize features
    features1_norm = F.normalize(features1, p=2, dim=-1)
    features2_norm = F.normalize(features2, p=2, dim=-1)

    # Compute cosine similarity
    similarity = (features1_norm * features2_norm).sum(dim=-1)

    return similarity


def l2_distance(features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 (Euclidean) distance between feature vectors.

    Args:
        features1: (N, D) or (D,) tensor
        features2: (N, D) or (D,) tensor

    Returns:
        distance: Scalar or (N,) tensor of L2 distances
    """
    return torch.norm(features1 - features2, p=2, dim=-1)


def normalized_l2_distance(
    features1: torch.Tensor, features2: torch.Tensor
) -> torch.Tensor:
    """
    Compute normalized L2 distance (0 = identical, 1 = orthogonal for unit vectors).

    Args:
        features1: (N, D) or (D,) tensor
        features2: (N, D) or (D,) tensor

    Returns:
        distance: Scalar or (N,) tensor of normalized distances
    """
    # Normalize features first
    features1_norm = F.normalize(features1, p=2, dim=-1)
    features2_norm = F.normalize(features2, p=2, dim=-1)

    # Compute L2 distance between normalized features
    # For unit vectors, this ranges from 0 (identical) to 2 (opposite)
    distance = torch.norm(features1_norm - features2_norm, p=2, dim=-1)

    # Normalize to [0, 1] range
    return distance / 2.0


def compute_correlation(features1: torch.Tensor, features2: torch.Tensor) -> float:
    """
    Compute Pearson correlation coefficient between feature vectors.

    Args:
        features1: (N, D) tensor
        features2: (N, D) tensor

    Returns:
        correlation: Scalar correlation coefficient in range [-1, 1]
    """
    # Flatten if multidimensional
    f1_flat = features1.flatten()
    f2_flat = features2.flatten()

    # Compute correlation
    f1_centered = f1_flat - f1_flat.mean()
    f2_centered = f2_flat - f2_flat.mean()

    correlation = (f1_centered * f2_centered).sum() / (
        torch.sqrt((f1_centered**2).sum() * (f2_centered**2).sum()) + 1e-8
    )

    return correlation.item()


# ============================================================================
# Pixel to Token Mapping
# ============================================================================


def pixel_to_patch_index(
    pixel_coords: np.ndarray,
    image_size: tuple[int, int],
    patch_size: int = 14,
    num_patches: tuple[int, int] = None,
) -> np.ndarray:
    """
    Convert pixel coordinates to patch indices.

    Args:
        pixel_coords: (N, 2) array of (x, y) pixel coordinates
        image_size: (width, height) of the image
        patch_size: Size of each patch (default: 14 for DINOv2)
        num_patches: (n_patches_h, n_patches_w) if provided

    Returns:
        patch_indices: (N,) array of flat patch indices
    """
    w, h = image_size

    if num_patches is None:
        n_patches_h = h // patch_size
        n_patches_w = w // patch_size
    else:
        n_patches_h, n_patches_w = num_patches

    # Convert to patch coordinates
    patch_x = np.floor(pixel_coords[:, 0] / patch_size).astype(int)
    patch_y = np.floor(pixel_coords[:, 1] / patch_size).astype(int)

    # Clamp to valid range
    patch_x = np.clip(patch_x, 0, n_patches_w - 1)
    patch_y = np.clip(patch_y, 0, n_patches_h - 1)

    # Convert to flat index
    patch_indices = patch_y * n_patches_w + patch_x

    return patch_indices


def extract_features_at_pixels(
    feature_map: torch.Tensor,
    pixel_coords: np.ndarray,
    image_size: tuple[int, int],
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Extract features at specific pixel locations from a patch-based feature map.

    Args:
        feature_map: (B, num_patches, feature_dim) or (num_patches, feature_dim) tensor
        pixel_coords: (N, 2) array of (x, y) pixel coordinates
        image_size: (width, height) of the original image
        patch_size: Size of each patch

    Returns:
        features: (N, feature_dim) tensor of features at the specified pixels
    """
    if feature_map.dim() == 3:
        feature_map = feature_map.squeeze(0)  # Remove batch dimension if present

    num_patches_total, feature_dim = feature_map.shape
    w, h = image_size
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    # Get patch indices
    patch_indices = pixel_to_patch_index(
        pixel_coords, image_size, patch_size, (n_patches_h, n_patches_w)
    )

    # Extract features
    features = feature_map[patch_indices]  # (N, feature_dim)

    return features
