"""
Gradient-based orientation computation for keypoints.

Implements SIFT-like orientation assignment based on local image gradients.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def compute_image_gradients(
    image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute image gradients using Sobel-like central differences.

    Args:
        image: (B, C, H, W) input image tensor (grayscale or RGB)
               If RGB, will convert to grayscale first

    Returns:
        gx: (B, 1, H, W) gradient in x direction
        gy: (B, 1, H, W) gradient in y direction
        grad_mag: (B, 1, H, W) gradient magnitude
    """
    B, C, H, W = image.shape

    # Convert to grayscale if needed
    if C == 3:
        # Use standard RGB to grayscale conversion
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image

    # Compute gradients using central differences (like SIFT)
    # gx = 0.5 * (I[x+1] - I[x-1])
    # gy = 0.5 * (I[y+1] - I[y-1])

    # Pad image to handle boundaries
    gray_padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")

    # Gradient in x direction
    gx = 0.5 * (gray_padded[:, :, 1:-1, 2:] - gray_padded[:, :, 1:-1, :-2])

    # Gradient in y direction
    gy = 0.5 * (gray_padded[:, :, 2:, 1:-1] - gray_padded[:, :, :-2, 1:-1])

    # Gradient magnitude
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

    return gx, gy, grad_mag


def compute_gradient_angle(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient angle from gradient components.

    Args:
        gx: (B, 1, H, W) gradient in x direction
        gy: (B, 1, H, W) gradient in y direction

    Returns:
        angle: (B, 1, H, W) gradient angle in radians [-π, π]
    """
    angle = torch.atan2(gy, gx)  # Range: [-π, π]
    return angle


def compute_keypoint_orientations(
    gx: torch.Tensor,
    gy: torch.Tensor,
    grad_mag: torch.Tensor,
    keypoint_coords: torch.Tensor,
    window_size: float = 1.5,
    num_bins: int = 36,
) -> torch.Tensor:
    """
    Compute dominant orientation for each keypoint using gradient histogram.

    Following SIFT's orientation assignment:
    - Build histogram of gradient directions in a Gaussian-weighted window
    - Find the dominant peak in the histogram
    - Use quadratic interpolation for sub-bin accuracy

    Args:
        gx: (B, 1, H, W) gradient in x direction
        gy: (B, 1, H, W) gradient in y direction
        grad_mag: (B, 1, H, W) gradient magnitude
        keypoint_coords: (B, K, 2) keypoint coordinates (x, y) in feature map space
        window_size: Size of Gaussian window as multiple of keypoint scale (default: 1.5)
        num_bins: Number of histogram bins (default: 36, same as SIFT)

    Returns:
        orientations: (B, K) dominant orientation in radians [-π, π]
    """
    B, _, H, W = gx.shape
    K = keypoint_coords.shape[1]
    device = gx.device

    # Compute gradient angles
    grad_angle = torch.atan2(gy, gx).squeeze(1)  # (B, H, W)
    grad_mag = grad_mag.squeeze(1)  # (B, H, W)

    orientations = torch.zeros(B, K, device=device)

    # Window radius (in pixels)
    W_radius = int(3.0 * window_size)
    sigma_w = window_size

    # Pre-compute Gaussian weight kernel (OPTIMIZATION: avoid repeated tensor creation)
    dy_grid, dx_grid = torch.meshgrid(
        torch.arange(-W_radius, W_radius + 1, device=device, dtype=torch.float32),
        torch.arange(-W_radius, W_radius + 1, device=device, dtype=torch.float32),
        indexing="ij",
    )
    r2_grid = dx_grid**2 + dy_grid**2
    gaussian_weights = torch.exp(-r2_grid / (2 * sigma_w**2))
    circular_mask = r2_grid < (W_radius**2 + 0.6)
    weight_kernel = gaussian_weights * circular_mask  # Shape: (2*W+1, 2*W+1)

    # Pre-define smoothing kernel for histogram (OPTIMIZATION: use conv1d)
    smooth_kernel = torch.tensor([1 / 3, 1 / 3, 1 / 3], device=device).view(1, 1, 3)

    for b in range(B):
        for k in range(K):
            cx, cy = keypoint_coords[b, k]

            # Convert to integer pixel coordinates
            xi = int(torch.round(cx).item())
            yi = int(torch.round(cy).item())

            # Check bounds
            if (
                xi < W_radius
                or xi >= W - W_radius
                or yi < W_radius
                or yi >= H - W_radius
            ):
                # Out of bounds - use zero orientation
                orientations[b, k] = 0.0
                continue

            # Create histogram
            hist = torch.zeros(num_bins, device=device)

            # Extract window from gradient maps
            y_start, y_end = yi - W_radius, yi + W_radius + 1
            x_start, x_end = xi - W_radius, xi + W_radius + 1

            mag_window = grad_mag[b, y_start:y_end, x_start:x_end]  # (2*W+1, 2*W+1)
            ang_window = grad_angle[b, y_start:y_end, x_start:x_end]  # (2*W+1, 2*W+1)

            # Apply weights and mask
            weighted_mag = mag_window * weight_kernel  # (2*W+1, 2*W+1)

            # Flatten for iteration (still need to accumulate into histogram bins)
            weighted_mag_flat = weighted_mag[circular_mask]
            ang_flat = ang_window[circular_mask]

            # Convert angles to [0, 2π] range
            ang_normalized = ang_flat % (2 * torch.pi)

            # Compute histogram bins (bilinear interpolation)
            fbin = num_bins * ang_normalized / (2 * torch.pi)

            # Bilinear distribution to adjacent bins
            bin_idx = torch.floor(fbin - 0.5).long()
            rbin = fbin - bin_idx.float() - 0.5

            # Accumulate into histogram using scatter_add (vectorized)
            hist.scatter_add_(0, bin_idx % num_bins, (1 - rbin) * weighted_mag_flat)
            hist.scatter_add_(0, (bin_idx + 1) % num_bins, rbin * weighted_mag_flat)

            # Smooth histogram using circular convolution (OPTIMIZATION: use conv1d)
            hist_1d = hist.unsqueeze(0).unsqueeze(0)  # (1, 1, num_bins)
            for _ in range(6):
                # Circular padding: wrap last element to beginning, first to end
                hist_padded = torch.cat(
                    [hist_1d[:, :, -1:], hist_1d, hist_1d[:, :, :1]], dim=2
                )
                hist_1d = F.conv1d(hist_padded, smooth_kernel)
            hist = hist_1d.squeeze()

            # Find the maximum peak
            max_bin = torch.argmax(hist).item()

            # Quadratic interpolation for sub-bin accuracy
            h0 = hist[max_bin]
            hm = hist[(max_bin - 1) % num_bins]
            hp = hist[(max_bin + 1) % num_bins]

            # Interpolation offset
            denominator = hp + hm - 2 * h0
            if denominator != 0:
                di = -0.5 * (hp - hm) / denominator
            else:
                di = 0.0

            # Compute angle in radians
            theta = 2 * torch.pi * (max_bin + di + 0.5) / num_bins

            # Convert back to [-π, π] range
            if theta > torch.pi:
                theta = theta - 2 * torch.pi

            orientations[b, k] = theta

    return orientations


def compute_keypoint_orientations_simple(
    gx: torch.Tensor,
    gy: torch.Tensor,
    keypoint_coords: torch.Tensor,
    window_size: int = 3,
) -> torch.Tensor:
    """
    Simplified orientation computation using direct gradient direction.

    Much faster than SIFT-style histogram approach (~10-30x speedup).
    Uses grid_sample for vectorized gradient sampling at keypoint locations,
    with optional local window averaging for robustness.

    Args:
        gx: (B, 1, H, W) gradient in x direction
        gy: (B, 1, H, W) gradient in y direction
        keypoint_coords: (B, K, 2) keypoint coordinates (x, y) in pixel space
        window_size: Number of pixels to average around keypoint (0 = single point)

    Returns:
        orientations: (B, K) dominant orientation in radians [-π, π]
    """
    B, _, H, W = gx.shape
    K = keypoint_coords.shape[1]
    device = gx.device

    # Normalize keypoint coords to [-1, 1] for grid_sample
    grid = keypoint_coords.clone()  # (B, K, 2)
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # x
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # y
    grid = grid.unsqueeze(2)  # (B, K, 1, 2) for grid_sample

    if window_size <= 0:
        # Single point sampling (fastest)
        gx_sampled = (
            F.grid_sample(gx, grid, mode="bilinear", align_corners=False)
            .squeeze(-1)
            .squeeze(1)
        )  # (B, K)
        gy_sampled = (
            F.grid_sample(gy, grid, mode="bilinear", align_corners=False)
            .squeeze(-1)
            .squeeze(1)
        )  # (B, K)
    else:
        # Local window averaging for robustness (still much faster than histogram)
        # Sample in 5×5 grid around keypoint
        num_samples = 5
        offsets = torch.linspace(-window_size, window_size, num_samples, device=device)

        gx_sum = torch.zeros(B, K, device=device)
        gy_sum = torch.zeros(B, K, device=device)

        for dy in offsets:
            for dx in offsets:
                # Offset grid in normalized coordinates
                grid_offset = grid.clone()
                grid_offset[..., 0, 0] += (dx / W) * 2  # x offset
                grid_offset[..., 0, 1] += (dy / H) * 2  # y offset

                # Clamp to valid range [-1, 1] to prevent grid_sample from returning NaN
                grid_offset = torch.clamp(grid_offset, min=-1.0, max=1.0)

                gx_sample = (
                    F.grid_sample(gx, grid_offset, mode="bilinear", align_corners=False)
                    .squeeze(-1)
                    .squeeze(1)
                )
                gy_sample = (
                    F.grid_sample(gy, grid_offset, mode="bilinear", align_corners=False)
                    .squeeze(-1)
                    .squeeze(1)
                )

                gx_sum += gx_sample
                gy_sum += gy_sample

        # Average over window
        gx_sampled = gx_sum / (num_samples**2)
        gy_sampled = gy_sum / (num_samples**2)

    # Compute orientation from averaged gradients
    orientations = torch.atan2(gy_sampled, gx_sampled)  # (B, K) in [-π, π]

    # Safety check: replace NaN/Inf with zeros to prevent error propagation
    # This can happen if gradients are zero or coordinates are out of bounds
    orientations = torch.nan_to_num(orientations, nan=0.0, posinf=0.0, neginf=0.0)

    return orientations


def compute_keypoint_orientations_batched(
    image: torch.Tensor,
    keypoint_coords: torch.Tensor,
    window_size: float = 1.5,
    num_bins: int = 36,
) -> torch.Tensor:
    """
    Convenience function to compute keypoint orientations from image.

    Args:
        image: (B, C, H, W) input image
        keypoint_coords: (B, K, 2) keypoint coordinates (x, y)
        window_size: Size of Gaussian window (default: 1.5)
        num_bins: Number of histogram bins (default: 36)

    Returns:
        orientations: (B, K) dominant orientation in radians [-π, π]
    """
    # Compute gradients
    gx, gy, grad_mag = compute_image_gradients(image)

    # Compute orientations
    orientations = compute_keypoint_orientations(
        gx, gy, grad_mag, keypoint_coords, window_size, num_bins
    )

    return orientations


def combine_gradient_and_homography_orientations(
    gradient_orientations1: torch.Tensor,
    gradient_orientations2: torch.Tensor,
    homography_rotation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine gradient-based orientations with homography rotation.

    The final orientation should satisfy:
    orientation2 = orientation1 + homography_rotation (modulo 2π)

    This ensures rotation equivariance while maintaining local gradient structure.

    Args:
        gradient_orientations1: (B, K) gradient-based orientations in image 1
        gradient_orientations2: (B, K) gradient-based orientations in image 2
        homography_rotation: (B,) global rotation angle from homography

    Returns:
        orientations1: (B, K) final orientations for image 1
        orientations2: (B, K) final orientations for image 2
    """
    # Use gradient orientations directly
    # They already encode local structure
    orientations1 = gradient_orientations1

    # For image 2, we expect the gradient direction to have rotated
    # by the homography rotation angle
    # So: orientations2 = gradient_orientations2
    # And the loss will enforce: orientations2 ≈ orientations1 + homography_rotation
    orientations2 = gradient_orientations2

    return orientations1, orientations2
