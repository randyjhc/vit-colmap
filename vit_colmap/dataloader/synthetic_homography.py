"""
Synthetic homography generation for data augmentation.

This module provides utilities for generating random homographies and applying
them to images to create synthetic training pairs.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional


def generate_random_homography(
    image_size: Tuple[int, int],
    rotation_range: float = 30.0,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    perspective_range: float = 0.0002,
    translation_range: Tuple[float, float] = (0.1, 0.1),
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Generate a random homography matrix for data augmentation.

    Args:
        image_size: (H, W) image dimensions
        rotation_range: Maximum rotation in degrees (±range)
        scale_range: (min_scale, max_scale) for uniform scaling
        perspective_range: Maximum perspective distortion coefficient
        translation_range: (tx_ratio, ty_ratio) as fraction of image size
        random_state: Random state for reproducibility

    Returns:
        H: (3, 3) homography matrix as numpy array
    """
    if random_state is None:
        random_state = np.random.RandomState()

    H_img, W_img = image_size

    # Image center
    cx, cy = W_img / 2, H_img / 2

    # 1. Random rotation
    angle = random_state.uniform(-rotation_range, rotation_range)
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # 2. Random scale
    scale = random_state.uniform(scale_range[0], scale_range[1])

    # 3. Random translation (as fraction of image size)
    tx = random_state.uniform(-translation_range[0], translation_range[0]) * W_img
    ty = random_state.uniform(-translation_range[1], translation_range[1]) * H_img

    # 4. Build affine transformation matrix
    # Translate to origin -> rotate & scale -> translate back -> translate
    T_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)

    R_S = np.array(
        [
            [scale * cos_a, -scale * sin_a, 0],
            [scale * sin_a, scale * cos_a, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    T_back = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float32)

    # Combine transformations
    H_affine = T_back @ R_S @ T_to_origin

    # 5. Add perspective distortion
    if perspective_range > 0:
        p1 = random_state.uniform(-perspective_range, perspective_range)
        p2 = random_state.uniform(-perspective_range, perspective_range)
        H_affine[2, 0] = p1
        H_affine[2, 1] = p2

    return H_affine


def warp_image_cv2(
    image: np.ndarray,
    H: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int = 0,
) -> np.ndarray:
    """
    Warp image using homography with OpenCV.

    Args:
        image: (H, W, C) or (H, W) numpy array
        H: (3, 3) homography matrix
        output_size: (H, W) output size (default: same as input)
        border_mode: OpenCV border mode
        border_value: Border value for BORDER_CONSTANT

    Returns:
        warped: Warped image with same shape as output_size
    """
    if output_size is None:
        output_size = (image.shape[0], image.shape[1])

    # Note: cv2.warpPerspective expects (width, height) for dsize
    warped = cv2.warpPerspective(
        image,
        H,
        dsize=(output_size[1], output_size[0]),  # (W, H)
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )

    return warped


def create_synthetic_pair(
    image: np.ndarray,
    image_size: Tuple[int, int],
    rotation_range: float = 30.0,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    perspective_range: float = 0.0002,
    translation_range: Tuple[float, float] = (0.1, 0.1),
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic image pair by warping the input image.

    Args:
        image: (H, W, C) input image (RGB, uint8 or float32)
        image_size: (H, W) size for output images
        rotation_range: Maximum rotation in degrees
        scale_range: (min_scale, max_scale)
        perspective_range: Perspective distortion range
        translation_range: (tx_ratio, ty_ratio) as fraction of image size
        random_state: Random state for reproducibility

    Returns:
        img1: Original image (resized to image_size if needed)
        img2: Warped synthetic image
        H: (3, 3) homography matrix (img1 -> img2)
    """
    # Resize original image if needed
    if (image.shape[0], image.shape[1]) != image_size:
        img1 = cv2.resize(
            image,
            (image_size[1], image_size[0]),  # (W, H)
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        img1 = image.copy()

    # Generate random homography
    H = generate_random_homography(
        image_size=image_size,
        rotation_range=rotation_range,
        scale_range=scale_range,
        perspective_range=perspective_range,
        translation_range=translation_range,
        random_state=random_state,
    )

    # Warp image
    img2 = warp_image_cv2(img1, H, output_size=image_size)

    return img1, img2, H


class SyntheticHomographyConfig:
    """Configuration for synthetic homography augmentation."""

    def __init__(
        self,
        rotation_range: float = 30.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        perspective_range: float = 0.0002,
        translation_range: Tuple[float, float] = (0.1, 0.1),
    ):
        """
        Initialize configuration.

        Args:
            rotation_range: Maximum rotation in degrees (±range)
            scale_range: (min_scale, max_scale) for scaling
            perspective_range: Maximum perspective distortion
            translation_range: (tx_ratio, ty_ratio) as fraction of image size
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.perspective_range = perspective_range
        self.translation_range = translation_range

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "rotation_range": self.rotation_range,
            "scale_range": self.scale_range,
            "perspective_range": self.perspective_range,
            "translation_range": self.translation_range,
        }

    @classmethod
    def from_dict(cls, config: Dict) -> "SyntheticHomographyConfig":
        """Create from dictionary."""
        return cls(**config)

    @classmethod
    def conservative(cls) -> "SyntheticHomographyConfig":
        """Conservative augmentation (small perturbations)."""
        return cls(
            rotation_range=15.0,
            scale_range=(0.9, 1.1),
            perspective_range=0.0001,
            translation_range=(0.05, 0.05),
        )

    @classmethod
    def moderate(cls) -> "SyntheticHomographyConfig":
        """Moderate augmentation (default)."""
        return cls(
            rotation_range=30.0,
            scale_range=(0.8, 1.2),
            perspective_range=0.0002,
            translation_range=(0.1, 0.1),
        )

    @classmethod
    def aggressive(cls) -> "SyntheticHomographyConfig":
        """Aggressive augmentation (large transformations)."""
        return cls(
            rotation_range=45.0,
            scale_range=(0.7, 1.4),
            perspective_range=0.0005,
            translation_range=(0.15, 0.15),
        )


def adjust_homography_for_resize(
    H: np.ndarray,
    orig_size: Tuple[int, int],
    new_size: Tuple[int, int],
) -> np.ndarray:
    """
    Adjust homography matrix for image resizing.

    When images are resized, the homography needs to be adjusted accordingly:
    H_new = S_new @ H_orig @ S_orig_inv

    Args:
        H: (3, 3) original homography matrix
        orig_size: (H, W) original image size
        new_size: (H, W) new image size

    Returns:
        H_adjusted: (3, 3) adjusted homography matrix
    """
    orig_h, orig_w = orig_size
    new_h, new_w = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    # Scaling matrices
    S = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)

    S_inv = np.array(
        [[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1]], dtype=np.float32
    )

    # Adjust homography
    H_adjusted = S @ H @ S_inv

    return H_adjusted


def compose_homographies(H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
    """
    Compose two homography matrices.

    If H1 maps A -> B and H2 maps B -> C, then H_composed maps A -> C.

    Args:
        H1: (3, 3) first homography
        H2: (3, 3) second homography

    Returns:
        H_composed: (3, 3) composed homography = H2 @ H1
    """
    return H2 @ H1
