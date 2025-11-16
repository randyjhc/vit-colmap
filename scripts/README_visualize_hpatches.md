# HPatches Warping Visualization

Visualize ViT (DINOv2) feature alignment using homography warping on HPatches dataset.

## Overview

This script demonstrates how well DINOv2 backbone features align under geometric transformations. It:

1. Loads image pairs with ground truth homographies
2. Extracts DINOv2 patch token features
3. Warps source features to target coordinate system
4. Visualizes feature alignment quality

## Prerequisites

### 1. Install Dependencies

```bash
pip install torch torchvision matplotlib scikit-learn opencv-python
```

### 2. Download HPatches Dataset (Optional)

```bash
# Download HPatches sequences
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz

# Extract to data/raw/HPatches/
tar -xzf hpatches-sequences-release.tar.gz -C data/raw/
mv data/raw/hpatches-sequences-release data/raw/HPatches
```

## Usage

### Mode 1: With HPatches Dataset

```bash
# Use first available pair
python scripts/visualize_hpatches_warping.py --data-root data/raw/HPatches

# Specific sequence
python scripts/visualize_hpatches_warping.py --data-root data/raw/HPatches --sequence i_ajuntament

# Specific pair index
python scripts/visualize_hpatches_warping.py --data-root data/raw/HPatches --pair-idx 10
```

### Mode 2: Single Image (Synthetic Homography)

Test without HPatches using any image:

```bash
python scripts/visualize_hpatches_warping.py --image data/raw/test_image.png
```

This applies a synthetic perspective transform for testing.

### Mode 3: Custom Options

```bash
python scripts/visualize_hpatches_warping.py \
    --data-root data/raw/HPatches \
    --sequence v_adam \
    --output-dir outputs/visualizations \
    --device cuda \
    --backbone dinov2_vitl14
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `data/raw/HPatches` | Path to HPatches directory |
| `--sequence` | None | Specific sequence name (e.g., `i_ajuntament`, `v_adam`) |
| `--pair-idx` | 0 | Index of image pair to visualize |
| `--image` | None | Single image path (uses synthetic homography) |
| `--output-dir` | `outputs` | Directory to save visualizations |
| `--device` | auto | Device: `cuda` or `cpu` |
| `--backbone` | `dinov2_vitb14` | DINOv2 model: `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14` |

## Output Visualization

The script generates a comprehensive visualization with 3 rows:

### Row 1: Input Images and Features
- **Image 1 (Reference)**: Original source image
- **Image 2 (Target)**: Target image with geometric/illumination changes
- **Features 1 (PCA)**: DINOv2 features visualized via PCA → RGB
- **Features 2 (PCA)**: Target image features

### Row 2: Warping Analysis
- **Warped Features 1**: Source features transformed by homography
- **Overlay**: Blend of warped and target features (should align if warping is correct)
- **Feature Difference**: L2 distance heatmap (darker = better alignment)
- **Cosine Similarity**: Similarity map (brighter = better match, range [-1, 1])

### Row 3: Statistics
- **Valid Mask**: White = valid correspondences within image bounds
- **Masked Similarity**: Similarity only for valid regions
- **Histogram**: Distribution of cosine similarity values with mean (red dashed line)

## Interpreting Results

### Good Alignment (Expected for DINOv2)
- Mean cosine similarity: **> 0.6**
- Feature difference: Mostly dark (low error)
- Overlay: Smooth blending without ghosting
- Narrow histogram around high similarity values

### Poor Alignment
- Mean cosine similarity: **< 0.3**
- Feature difference: Bright regions (high error)
- Overlay: Visible misalignment/double edges
- Wide or bimodal histogram

### PCA Visualization Colors
- Similar colors indicate similar semantic features
- Object boundaries should be visible
- Same objects should have consistent colors across views

## Example Output

```
Using device: cuda
Loading ViTFeatureModel with backbone: dinov2_vitb14
ViTFeatureModel initialized:
  Backbone: dinov2_vitb14 (768D)
  Descriptor dim: 128
  Backbone frozen: True
Loading HPatches dataset from: data/raw/HPatches
HPatchesDataset initialized:
  Root: data/raw/HPatches
  Split: all
  Sequences: 116
  Image pairs: 580
  Target size: 1190x1596
Loaded pair from sequence: i_ajuntament
Extracting DINOv2 features...
Feature shape: torch.Size([1, 768, 85, 114])
Warping features with homography...
Computing feature similarity...
Creating visualization...
Visualization saved to: outputs/hpatches_warping_i_ajuntament.png

============================================================
Summary Statistics
============================================================
Sequence: i_ajuntament
Input image size: (1190, 1596)
Feature map size: (85, 114)
Valid correspondences: 9690 / 9690
Mean cosine similarity: 0.7234
Std cosine similarity: 0.1456
Min cosine similarity: 0.2341
Max cosine similarity: 0.9876
============================================================
```

## Sequence Naming Convention

HPatches sequences are named:
- `i_*`: **Illumination** changes (lighting variations, same viewpoint)
- `v_*`: **Viewpoint** changes (geometric transformations)

Examples:
- `i_ajuntament`: Illumination sequence
- `v_adam`: Viewpoint sequence

## Troubleshooting

### "HPatches directory not found"
Download the dataset (see Prerequisites) or use `--image` mode with a single image.

### "No image pairs found"
Ensure HPatches directory structure is correct:
```
data/raw/HPatches/
├── i_ajuntament/
│   ├── 1.ppm
│   ├── 2.ppm
│   ├── H_1_2
│   └── ...
└── v_adam/
    └── ...
```

### CUDA out of memory
Use CPU: `--device cpu` or smaller backbone: `--backbone dinov2_vits14`

### Slow first run
DINOv2 model downloads on first use (~350MB for ViT-B). Subsequent runs use cached weights.
