# HPatches Dataset Augmentation Usage Guide

This document describes the three new data augmentation strategies implemented to address overfitting and high loss values in training.

## Overview

Three augmentation strategies have been implemented:

1. **Extended Image Pair Generation** - Generate more pairs (H_2_3, H_3_4, etc.) beyond just H_1_X pairs
2. **Synthetic Homography Augmentation** - Create synthetic image pairs with random transformations
3. **Threshold-Based Positive Selection** - Select positive samples by similarity score threshold instead of fixed top-K

All features are **optional** and **backward compatible** with existing training scripts.

---

## 1. Extended Image Pair Generation

### What it does
Instead of only pairing image 1 with all other images (H_1_2, H_1_3, ...), this generates additional pairs:
- **consecutive**: Adds consecutive pairs (H_2_3, H_3_4, H_4_5, H_5_6)
- **all_pairs**: Generates all possible combinations

### Benefits
- 3-4x more training samples from the same dataset
- Better coverage of different viewpoint transformations

### Usage

```bash
# Use consecutive pairs (recommended)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --pair-mode consecutive

# Use all possible pairs (maximum augmentation)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --pair-mode all_pairs

# Default behavior (reference only)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --pair-mode reference_only
```

### Example Output
```
HPatchesDataset initialized:
  Root: data/raw/HPatches
  Split: all
  Sequences: 116
  Pair mode: consecutive
  Real pairs: 928  # vs 580 with reference_only
  Total pairs: 928
```

---

## 2. Synthetic Homography Augmentation

### What it does
Applies random homographies to original images to create synthetic training pairs with controlled transformations:
- Random rotation (configurable range)
- Random scaling
- Random perspective distortion
- Random translation

### Benefits
- Can increase dataset size by 2-10x
- Adds diversity without collecting more data
- Controllable augmentation strength

### Usage

```bash
# Basic synthetic augmentation (50% synthetic samples)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --use-synthetic-aug \
  --synthetic-ratio 0.5

# Conservative augmentation (small perturbations)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --use-synthetic-aug \
  --synthetic-ratio 0.3 \
  --synthetic-rotation-range 15.0 \
  --synthetic-scale-range 0.9 1.1 \
  --synthetic-perspective-range 0.0001 \
  --synthetic-translation-range 0.05 0.05

# Aggressive augmentation (large transformations)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --use-synthetic-aug \
  --synthetic-ratio 0.8 \
  --synthetic-rotation-range 45.0 \
  --synthetic-scale-range 0.7 1.4 \
  --synthetic-perspective-range 0.0005 \
  --synthetic-translation-range 0.15 0.15
```

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--use-synthetic-aug` | Enable synthetic augmentation | False | flag |
| `--synthetic-ratio` | Ratio of synthetic samples | 0.5 | 0.0-1.0 |
| `--synthetic-rotation-range` | Max rotation in degrees (±) | 30.0 | 0-180 |
| `--synthetic-scale-range` | Scale range [min, max] | [0.8, 1.2] | (0, ∞) |
| `--synthetic-perspective-range` | Perspective distortion | 0.0002 | 0-0.001 |
| `--synthetic-translation-range` | Translation [tx, ty] as ratio | [0.1, 0.1] | 0-1.0 |

### Example Output
```
HPatchesDataset initialized:
  Root: data/raw/HPatches
  Pair mode: reference_only
  Real pairs: 580
  Synthetic pairs: 290
  Synthetic config: {'rotation_range': 30.0, 'scale_range': [0.8, 1.2],
                     'perspective_range': 0.0002, 'translation_range': [0.1, 0.1]}
  Total pairs: 870
```

---

## 3. Threshold-Based Positive Selection

### What it does
Instead of always selecting a fixed number of top-K points, select points based on similarity threshold:
- **top_k**: Original behavior (select top K points)
- **threshold**: Select all points with similarity > threshold (clamped to min/max range)
- **hybrid**: Select points above threshold, then take top K from those

### Benefits
- More adaptive to image difficulty
- Handles varying overlap better
- Can use more points for easy pairs, fewer for hard pairs

### Usage

```bash
# Threshold-based selection (adaptive)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --selection-mode threshold \
  --similarity-threshold 0.7 \
  --min-invariant-points 100 \
  --max-invariant-points 2048

# Hybrid selection (best of both worlds)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --selection-mode hybrid \
  --similarity-threshold 0.65 \
  --top-k 512

# Default behavior (top-k)
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --selection-mode top_k \
  --top-k 512
```

### Parameters

| Parameter | Description | Default | Used in mode |
|-----------|-------------|---------|--------------|
| `--selection-mode` | Selection strategy | top_k | all |
| `--top-k` | Number of points | 512 | top_k, hybrid |
| `--similarity-threshold` | Min similarity | 0.7 | threshold, hybrid |
| `--min-invariant-points` | Min points to select | 100 | threshold |
| `--max-invariant-points` | Max points (OOM prevention) | 2048 | threshold |

### Selection Mode Comparison

| Mode | Behavior | Best for |
|------|----------|----------|
| **top_k** | Always select K points | Consistent batch sizes, baseline |
| **threshold** | Select points > threshold | Adaptive to image difficulty |
| **hybrid** | Threshold filter + top-k | Quality filtering + consistency |

---

## Combined Usage Examples

### Example 1: Maximum Augmentation
```bash
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --pair-mode consecutive \
  --use-synthetic-aug \
  --synthetic-ratio 0.5 \
  --selection-mode hybrid \
  --similarity-threshold 0.65 \
  --top-k 512 \
  --epochs 100 \
  --batch-size 2
```

**Effect**:
- ~928 real pairs (vs 580 baseline)
- +464 synthetic pairs = 1,392 total pairs
- Hybrid selection for quality + consistency
- **~2.4x dataset size increase**

### Example 2: Conservative Augmentation
```bash
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --pair-mode reference_only \
  --use-synthetic-aug \
  --synthetic-ratio 0.3 \
  --synthetic-rotation-range 15.0 \
  --synthetic-scale-range 0.9 1.1 \
  --selection-mode top_k \
  --top-k 512
```

**Effect**:
- 580 real pairs
- +174 synthetic pairs = 754 total pairs
- Conservative synthetic transformations
- Fixed top-k selection (original behavior)
- **~1.3x dataset size increase**

### Example 3: Threshold-Focused (No Synthetic)
```bash
python scripts/train_vit_features.py \
  --data-root data/raw/HPatches \
  --pair-mode consecutive \
  --selection-mode threshold \
  --similarity-threshold 0.7 \
  --min-invariant-points 100 \
  --max-invariant-points 2048
```

**Effect**:
- 928 real pairs (consecutive mode)
- No synthetic augmentation
- Adaptive point selection (100-2048 points per pair)
- Better for handling varying image overlap

---

## Monitoring and Debugging

### Check Dataset Statistics

The dataset initialization prints detailed statistics:

```python
from vit_colmap.dataloader import HPatchesDataset, SyntheticHomographyConfig

dataset = HPatchesDataset(
    root_dir="data/raw/HPatches",
    pair_mode="consecutive",
    use_synthetic_aug=True,
    synthetic_ratio=0.5,
    synthetic_config=SyntheticHomographyConfig.moderate()
)

# Prints:
# HPatchesDataset initialized:
#   Root: data/raw/HPatches
#   Split: all
#   Sequences: 116
#   Pair mode: consecutive
#   Real pairs: 928
#   Synthetic pairs: 464
#   Synthetic config: {...}
#   Total pairs: 1392
#   Target size: 1200x1600
```

### Verify Augmentation in Training

Check the training logs for sampler configuration:

```
TrainingSampler configuration:
  Selection mode: hybrid
  Similarity threshold: 0.65
  Min points: 100
  Max points: 2048
  Top K: 512
```

### Inspect Batch Data

```python
# Check if synthetic samples are present
batch = next(iter(train_loader))
print(f"Batch size: {batch['img1'].shape[0]}")
print(f"Is synthetic: {batch['is_synthetic']}")
print(f"Pair indices: {batch['pair_idx']}")
```

---

## Performance Considerations

### Memory Usage

- **Threshold mode**: Variable number of points per image → variable memory
  - Use `--max-invariant-points` to prevent OOM
  - Consider reducing batch size if using high thresholds

- **Synthetic augmentation**: Minimal overhead (on-the-fly generation)
  - No extra disk space required
  - Slight CPU overhead during data loading

### Training Speed

- **Extended pairs**: No speed impact (just more samples)
- **Synthetic aug**: ~10-20% slower data loading (worth it for 2-10x more data)
- **Threshold mode**: Slightly faster/slower depending on selected points

### Recommended Settings for Different GPUs

**High-end GPU (24GB+ VRAM):**
```bash
--batch-size 4 \
--selection-mode threshold \
--max-invariant-points 2048 \
--pair-mode all_pairs \
--use-synthetic-aug \
--synthetic-ratio 0.8
```

**Mid-range GPU (12-16GB VRAM):**
```bash
--batch-size 2 \
--selection-mode hybrid \
--top-k 512 \
--pair-mode consecutive \
--use-synthetic-aug \
--synthetic-ratio 0.5
```

**Low-end GPU (8GB VRAM):**
```bash
--batch-size 1 \
--selection-mode top_k \
--top-k 256 \
--pair-mode reference_only \
--use-synthetic-aug \
--synthetic-ratio 0.3
```

---

## Programmatic API Usage

You can also use the augmentation features programmatically:

```python
from vit_colmap.dataloader import (
    HPatchesDataset,
    TrainingSampler,
    SyntheticHomographyConfig,
)

# Custom synthetic config
config = SyntheticHomographyConfig(
    rotation_range=20.0,
    scale_range=(0.85, 1.15),
    perspective_range=0.0001,
    translation_range=(0.08, 0.08),
)

# Dataset with all augmentations
dataset = HPatchesDataset(
    root_dir="data/raw/HPatches",
    pair_mode="consecutive",
    use_synthetic_aug=True,
    synthetic_ratio=0.5,
    synthetic_config=config,
)

# Sampler with threshold selection
sampler = TrainingSampler(
    top_k_invariant=512,
    selection_mode="hybrid",
    similarity_threshold=0.7,
    min_invariant_points=100,
    max_invariant_points=1024,
)
```

---

## Troubleshooting

### Issue: "Not enough points above threshold"

**Solution**: Lower `--similarity-threshold` or increase `--min-invariant-points`

```bash
--similarity-threshold 0.6 \
--min-invariant-points 50
```

### Issue: OOM errors with threshold mode

**Solution**: Reduce `--max-invariant-points` or batch size

```bash
--max-invariant-points 1024 \
--batch-size 1
```

### Issue: Synthetic samples look unrealistic

**Solution**: Use more conservative augmentation settings

```bash
--synthetic-rotation-range 15.0 \
--synthetic-scale-range 0.9 1.1 \
--synthetic-perspective-range 0.0001
```

---

## Files Modified/Created

### New Files
- `vit_colmap/dataloader/synthetic_homography.py` - Synthetic augmentation logic

### Modified Files
- `vit_colmap/dataloader/hpatches_dataset.py` - Extended pairing + synthetic integration
- `vit_colmap/dataloader/training_sampler.py` - Threshold-based selection
- `scripts/train_vit_features.py` - CLI arguments
- `vit_colmap/dataloader/__init__.py` - Exports

---

## References

For implementation details, see:
- Extended pairing: [hpatches_dataset.py:172-232](vit_colmap/dataloader/hpatches_dataset.py#L172-L232)
- Synthetic augmentation: [synthetic_homography.py](vit_colmap/dataloader/synthetic_homography.py)
- Threshold selection: [training_sampler.py:53-172](vit_colmap/dataloader/training_sampler.py#L53-L172)
