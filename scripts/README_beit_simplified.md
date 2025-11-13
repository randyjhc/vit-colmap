# Simplified BEiT Extractor

The BEiT extractor has been cleaned up and simplified from **1063 lines to just 394 lines** (~63% reduction).

## What Was Removed

### 1. All Descriptor Projection Classes (~110 lines)
- `DescriptorProjection` abstract class
- `PCAProjection`
- `RandomProjection`
- `LearnedProjection`

### 2. COLMAP Integration (~116 lines)
- `extract()` method for database creation
- All database writing functionality
- Camera parameter handling
- Batch image processing

### 3. Complex Keypoint Selection (~447 lines)
- `_refine_keypoint_positions()` - gradient-based refinement
- `_allocate_keypoint_budget()` - adaptive allocation
- `_extract_multi_keypoints_from_patch()` - NMS-based multi-keypoint extraction
- `_select_keypoints_from_feature_map()` - all keypoint selection strategies
- Support for 'saliency', 'uniform', and 'adaptive' methods

### 4. Descriptor Processing (~84 lines)
- `_reduce_descriptor_dim()` - PCA/random/learned projection
- `_run_inference()` - COLMAP-specific feature extraction
- Conversion to uint8 descriptors

### 5. Old Visualization (~79 lines)
- Previous `visualize_keypoints()` showing keypoints overlay

### 6. Removed Parameters
- `num_keypoints`
- `descriptor_dim`
- `projection_method`
- `keypoint_method`
- `refine_keypoints`
- `max_keypoints_per_patch`
- `min_patch_saliency_ratio`

## What Was Kept

### Core Functionality

1. **Model Loading** (lines 25-62)
   - Simple initialization with just 3 parameters: `model_name`, `layer_idx`, `device`
   - Loads BEiT model from HuggingFace
   - Auto-detects CUDA availability

2. **Feature Extraction** (lines 64-120)
   - `extract_layer_features()` - Extracts features from any specified layer
   - Returns normalized feature map (H_patches, W_patches, D)
   - Also returns the preprocessed image (224x224) for visualization

3. **Reconstruction** (lines 122-151)
   - `reconstruct_to_original_size()` - Upsamples features to original image size
   - Uses bilinear interpolation
   - Converts tensors to numpy arrays

### Two Visualization Modes

#### Mode 1: RGB Reconstruction (lines 153-251)

**Method**: `visualize_layer_rgb()`

Treats 3 feature channels as RGB and reconstructs them to original image size.

```python
extractor = BEiTExtractor(layer_idx=9)

# Visualize first 3 channels as RGB
extractor.visualize_layer_rgb(
    image_path="image.png",
    output_path="layer9_rgb.png",
    channels=[0, 1, 2]  # Which channels to use as RGB
)
```

**Output**: 3-panel visualization
- Original image
- BEiT input (224x224 resized)
- Reconstructed RGB from feature channels

#### Mode 2: Heatmap (lines 253-342)

**Method**: `visualize_layer_heatmap()`

Computes L2 norm across all feature dimensions and displays as heatmap.

```python
extractor = BEiTExtractor(layer_idx=9)

# Visualize feature magnitude as heatmap
extractor.visualize_layer_heatmap(
    image_path="image.png",
    output_path="layer9_heatmap.png"
)
```

**Output**: 3-panel visualization
- Original image
- Heatmap of feature magnitude
- Overlay of heatmap on original image

## Usage Examples

### Basic Usage

```python
from vit_colmap.features.beit_extractor import BEiTExtractor
from pathlib import Path

# Initialize extractor
extractor = BEiTExtractor(
    model_name="microsoft/beit-base-patch16-224-pt22k-ft22k",
    layer_idx=9,  # Layer to extract (0-12 for base model)
    device='cuda'  # or 'cpu' or None for auto
)

# RGB reconstruction
extractor.visualize_layer_rgb(
    Path("image.png"),
    Path("output_rgb.png"),
    channels=[0, 1, 2]
)

# Heatmap visualization
extractor.visualize_layer_heatmap(
    Path("image.png"),
    Path("output_heatmap.png")
)
```

### Extracting Different Layers

```python
# Compare different layers
for layer_idx in [0, 3, 6, 9, 11]:
    extractor = BEiTExtractor(layer_idx=layer_idx)

    extractor.visualize_layer_heatmap(
        "image.png",
        f"layer{layer_idx}_heatmap.png"
    )
```

### Different Channel Combinations

```python
extractor = BEiTExtractor(layer_idx=9)

# First 3 channels
extractor.visualize_layer_rgb("image.png", "rgb_012.png", [0, 1, 2])

# Middle channels
extractor.visualize_layer_rgb("image.png", "rgb_345.png", [3, 4, 5])

# Higher channels
extractor.visualize_layer_rgb("image.png", "rgb_100_101_102.png", [100, 101, 102])
```

### Programmatic Feature Extraction

```python
import cv2
from vit_colmap.features.beit_extractor import BEiTExtractor

# Load extractor
extractor = BEiTExtractor(layer_idx=9)

# Load image
img_bgr = cv2.imread("image.png")

# Extract features
feature_map, processed_image = extractor.extract_layer_features(img_bgr)

print(f"Feature map shape: {feature_map.shape}")  # (1, 14, 14, 768)
print(f"Feature dimension: {feature_map.shape[-1]}")  # 768 for base model

# Reconstruct to original size
h_orig, w_orig = img_bgr.shape[:2]
features_upsampled = extractor.reconstruct_to_original_size(
    feature_map,
    target_size=(h_orig, w_orig)
)

print(f"Upsampled shape: {features_upsampled.shape}")  # (H_orig, W_orig, 768)

# Now you can analyze individual channels, compute statistics, etc.
```

## Benefits of Simplified Version

### 1. **Much Cleaner**
- 394 lines vs 1063 lines (63% reduction)
- Clear, focused purpose
- Easy to understand and modify

### 2. **Faster to Load**
- No sklearn dependency for PCA
- No complex initialization
- Immediate feature extraction

### 3. **More Flexible**
- Can extract features from any layer
- Easy to experiment with different channels
- Straightforward to add new visualization modes

### 4. **Better Documentation**
- Self-contained example in `__main__`
- Clear method signatures
- Well-commented code

### 5. **No COLMAP Dependency**
- Pure visualization tool
- Can be used independently
- Easier to integrate into other projects

## File Size Comparison

| Version | Lines | Size |
|---------|-------|------|
| Original | 1063 | 45 KB |
| Simplified | 394 | 15 KB |
| **Reduction** | **-669 (-63%)** | **-30 KB (-67%)** |

## Layer Information

For `microsoft/beit-base-patch16-224-pt22k-ft22k`:

- **Total layers**: 13 (0-12)
- **Feature dimension**: 768
- **Patch grid**: 14×14 (for 224×224 input)
- **Patch size**: 16×16 pixels

**Layer recommendations**:
- **Early layers (0-3)**: Low-level features (edges, textures)
- **Middle layers (4-8)**: Mid-level features (patterns, parts)
- **Late layers (9-12)**: High-level features (semantic concepts)

## Test Script

Run the test script to verify functionality:

```bash
python scripts/test_beit_simplified.py
```

This will generate 6 visualizations:
- `layer9_rgb_channels_012.png` - RGB from first 3 channels
- `layer9_rgb_channels_345.png` - RGB from channels 3,4,5
- `layer9_heatmap.png` - Heatmap for layer 9
- `layer0_heatmap.png` - Heatmap for layer 0 (early)
- `layer6_heatmap.png` - Heatmap for layer 6 (middle)
- `layer11_heatmap.png` - Heatmap for layer 11 (late)

## Migration Guide

If you were using the old BEiT extractor for COLMAP integration, you should now use it only for visualization purposes. For COLMAP feature extraction, use the `ViTExtractor` or `ColmapSiftExtractor` instead.

**Old usage (COLMAP integration)**:
```python
# ❌ No longer supported
extractor = BEiTExtractor(
    num_keypoints=2048,
    descriptor_dim=128,
    projection_method='pca',
    keypoint_method='adaptive'
)
extractor.extract(image_dir, db_path, camera_model)
```

**New usage (visualization only)**:
```python
# ✓ Simplified for visualization
extractor = BEiTExtractor(layer_idx=9)
extractor.visualize_layer_heatmap(image_path, output_path)
```
