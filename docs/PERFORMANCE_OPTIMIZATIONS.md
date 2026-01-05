# Performance Optimizations Summary

## ⚠️ Important Note (December 2025)

**This document describes optimizations applied to the SIFT-style histogram orientation implementation**, which is still available in the codebase as `compute_keypoint_orientations()` for reference purposes.

**Current Training Uses**: The simplified `compute_keypoint_orientations_simple()` function (see [SIMPLIFIED_ORIENTATION.md](SIMPLIFIED_ORIENTATION.md)), which bypasses most of these optimizations by using a fundamentally different algorithm (~100x faster).

This document remains valuable for:
- Understanding the evolution from SIFT-style to simplified approach
- Reference if SIFT-style accuracy is needed in the future
- Learning PyTorch optimization techniques

---

## Overview
This document summarizes the performance optimizations applied to the gradient-based orientation computation implementation (SIFT-style histogram version).

## Optimizations Applied (Phase 1 - Quick Wins)

### 1. Pre-computed Gaussian Weight Kernel
**File**: `vit_colmap/utils/orientation.py` (lines 111-120)

**Problem**:
```python
# OLD: Created inside triple-nested loop (B × K × window² times)
weight = torch.exp(torch.tensor(-r2 / (2 * sigma_w**2), device=device))
```

**Solution**:
```python
# NEW: Pre-computed once before loops
dy_grid, dx_grid = torch.meshgrid(...)
r2_grid = dx_grid**2 + dy_grid**2
gaussian_weights = torch.exp(-r2_grid / (2 * sigma_w**2))
circular_mask = r2_grid < (W_radius**2 + 0.6)
weight_kernel = gaussian_weights * circular_mask
```

**Impact**:
- Eliminates B×K×window² tensor allocations
- Removes host/device synchronization overhead
- Estimated speedup: **100-1000x** on weight computation alone

---

### 2. Vectorized Histogram Smoothing
**File**: `vit_colmap/utils/orientation.py` (lines 122-123, 170-176)

**Problem**:
```python
# OLD: Manual Python loops (6 iterations × 36 bins)
for _ in range(6):
    hist_smoothed = torch.zeros(num_bins, device=device)
    for i in range(num_bins):
        hist_smoothed[i] = (hist[(i-1) % num_bins] + hist[i] + hist[(i+1) % num_bins]) / 3.0
    hist = hist_smoothed
```

**Solution**:
```python
# NEW: Circular convolution with conv1d
smooth_kernel = torch.tensor([1/3, 1/3, 1/3], device=device).view(1, 1, 3)
hist_1d = hist.unsqueeze(0).unsqueeze(0)
for _ in range(6):
    hist_padded = torch.cat([hist_1d[:, :, -1:], hist_1d, hist_1d[:, :, :1]], dim=2)
    hist_1d = F.conv1d(hist_padded, smooth_kernel)
hist = hist_1d.squeeze()
```

**Impact**:
- Removes 216 Python iterations per keypoint (6 × 36)
- Uses optimized CUDA kernel for convolution
- Estimated speedup: **5-10x** on histogram smoothing

---

### 3. Vectorized Window Extraction and Histogram Accumulation
**File**: `vit_colmap/utils/orientation.py` (lines 142-168)

**Problem**:
```python
# OLD: Nested loops over window pixels
for dy in range(-W_radius, W_radius + 1):
    for dx in range(-W_radius, W_radius + 1):
        # ... compute weight, sample gradient, accumulate histogram
```

**Solution**:
```python
# NEW: Extract entire window, vectorized accumulation
mag_window = grad_mag[b, y_start:y_end, x_start:x_end]
ang_window = grad_angle[b, y_start:y_end, x_start:x_end]
weighted_mag = mag_window * weight_kernel
# Flatten and use scatter_add
hist.scatter_add_(0, bin_idx % num_bins, (1 - rbin) * weighted_mag_flat)
hist.scatter_add_(0, (bin_idx + 1) % num_bins, rbin * weighted_mag_flat)
```

**Impact**:
- Extracts entire window in single slicing operation
- Uses `scatter_add` for vectorized histogram accumulation
- Estimated speedup: **2-5x** on histogram building

---

### 4. Removed Unused Predicted Orientations
**File**: `vit_colmap/dataloader/training_batch.py` (lines 265-280)

**Problem**:
```python
# OLD: Computed but never used
orientations2_pred = self.sample_orientations_at_coords(...)
orientations1_pred = self.sample_orientations_at_coords(...)
# Later: only orientations1_gradient and orientations2_gradient were used
```

**Solution**:
```python
# NEW: Removed dead code
# Directly compute gradient-based orientations without sampling predictions
```

**Impact**:
- Saves 2 `grid_sample` forward passes per batch
- Estimated speedup: **~2%** of total batch time

---

### 5. Vectorized Score Heatmap Generation
**File**: `vit_colmap/dataloader/training_batch.py` (lines 127-171)

**Problem**:
```python
# OLD: Nested loops, one Gaussian at a time (B × K iterations)
for b in range(B):
    for k in range(K):
        cx, cy = invariant_coords[b, k]
        gaussian = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        score_gt[b, 0] = torch.maximum(score_gt[b, 0], gaussian)
```

**Solution**:
```python
# NEW: Broadcast all K keypoints at once
xx_expanded = xx.unsqueeze(0).unsqueeze(1)  # (1, 1, H_p, W_p)
yy_expanded = yy.unsqueeze(0).unsqueeze(1)
cx = invariant_coords[:, :, 0:1].unsqueeze(3)  # (B, K, 1, 1)
cy = invariant_coords[:, :, 1:2].unsqueeze(3)
dist2 = (xx_expanded - cx)**2 + (yy_expanded - cy)**2  # (B, K, H_p, W_p)
gaussians = torch.exp(-dist2 / (2 * sigma**2))
score_gt = gaussians.max(dim=1, keepdim=True)[0]  # (B, 1, H_p, W_p)
```

**Impact**:
- Removes B×K Python iterations
- Computes all Gaussians in parallel on GPU
- Estimated speedup: **5-20x** (depends on K)

---

## Overall Performance Impact

### Before Optimizations:
- Orientation computation: **CPU-bound** with Python loops
- Score heatmap: O(B × K × H × W) with Python overhead
- Training bottleneck: Gradient orientation computation

### After Phase 1 Optimizations:
- Orientation computation: **10-50x faster** (moves toward GPU-bound)
- Score heatmap: **5-20x faster** (fully GPU-accelerated)
- **Expected training speedup**: 30-70% reduction in per-batch time

### Specific Metrics (Estimated):
| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Gaussian weight computation | O(B×K×W²) tensor allocs | O(1) pre-compute | 100-1000x |
| Histogram smoothing | 216 Python iters/keypoint | 6 conv1d calls | 5-10x |
| Window extraction | Nested loops | Single slice | 2-5x |
| Score heatmap | B×K loops | Vectorized | 5-20x |
| **Total orientation** | - | - | **10-50x** |

---

## Code Quality Improvements

1. **Reduced .item() calls**: Minimized host/device synchronization
2. **Eliminated temporary allocations**: Reuse pre-computed kernels
3. **Better tensor operations**: Leverages PyTorch's optimized CUDA kernels
4. **Cleaner code**: Removed dead code (unused predictions)

---

## Future Optimizations (Phase 2 - If Needed)

If further speedup is required, consider:

### Full Batch-Level Vectorization
- Process all B×K keypoints in parallel (no loops at all)
- Use `torch.nn.functional.unfold` for window extraction
- Batch histogram computation using advanced indexing

**Expected additional speedup**: 5-10x on top of Phase 1

### JIT Compilation
- Use `torch.jit.script` for the orientation computation function
- Pre-compile repeated operations

**Expected additional speedup**: 1.5-2x

### Mixed Precision
- Use `torch.cuda.amp` for gradient computation
- FP16 for intermediate calculations

**Expected additional speedup**: 1.5-2x with memory savings

---

## Testing and Validation

### Correctness Tests:
- ✅ Verified output matches original implementation (within numerical precision)
- ✅ Orientation diversity maintained across keypoints
- ✅ Rotation equivariance preserved

### Performance Tests:
Run these commands to profile:
```python
import time
import torch

# Before optimization (hypothetical)
start = time.time()
orientations = compute_keypoint_orientations_old(...)
old_time = time.time() - start

# After optimization
start = time.time()
orientations = compute_keypoint_orientations(...)
new_time = time.time() - start

print(f"Speedup: {old_time / new_time:.2f}x")
```

---

## Memory Impact

Phase 1 optimizations have **minimal memory overhead**:
- Pre-computed kernels: ~few KB (window_size dependent)
- Vectorized score heatmap: Same memory as before (just different computation order)
- Removed code: **Saves memory** (no unused orientation tensors)

---

## Recommendations

1. **For current training**: Use `compute_keypoint_orientations_simple()` (default, ~100x faster)
2. **For high accuracy needs**: Consider optimized SIFT histogram with Phase 1 optimizations
3. **If using histogram version**: Apply Phase 1 optimizations (low risk, high reward)
4. **Profile regularly**: Use PyTorch profiler to identify bottlenecks
5. **Test on real data**: Verify speedup on actual training workloads

## Current Status

- ✅ **Training implementation**: Uses simplified orientation (`compute_keypoint_orientations_simple`)
- ✅ **SIFT histogram**: Available with Phase 1 optimizations (`compute_keypoint_orientations`)
- ✅ **Phase 2 optimizations**: Not implemented (simplified approach superseded them)

---

## References

- PyTorch Performance Tuning Guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Vectorization Patterns: https://pytorch.org/docs/stable/notes/broadcasting.html
