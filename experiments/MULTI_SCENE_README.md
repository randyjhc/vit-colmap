## Multi-Scene Layer Invariance Analysis for DTU Dataset

This script performs layer-wise invariance analysis across **all 124 DTU scan scenes** to avoid scene-specific bias in the results.

### Overview

The `exp_multi_scene_analysis.py` script:
1. Sweeps through all DTU scan directories
2. Randomly samples **one "_3" image** from each scan (124 images total)
3. Runs invariance tests on each image
4. **Aggregates results** to show average performance across all scenes
5. Generates plots with error bars showing scene-to-scene variance

### Why Multi-Scene Analysis?

Single-image analysis can be biased toward:
- Specific textures in that scene
- Particular lighting conditions
- Scene-specific geometric structures

By analyzing **all 124 DTU scenes**, we get:
- ✓ Robust statistics across diverse scenes
- ✓ Understanding of scene-to-scene variance
- ✓ Identification of layers that work consistently across scenes
- ✓ Unbiased assessment of invariance properties

---

## Usage

### Basic Usage

```bash
python experiments/exp_multi_scene_analysis.py \
    --dtu-path data/raw/DTU/Cleaned \
    --model dinov2_vitb14 \
    --num-points 100 \
    --output-dir outputs/multi_scene_dtu
```

### With Different Model

```bash
# Test BEiT model
python experiments/exp_multi_scene_analysis.py \
    --dtu-path data/raw/DTU/Cleaned \
    --model beit_base_patch16_224 \
    --num-points 100 \
    --output-dir outputs/multi_scene_beit

# Test larger DINOv2
python experiments/exp_multi_scene_analysis.py \
    --dtu-path data/raw/DTU/Cleaned \
    --model dinov2_vitl14 \
    --num-points 200 \
    --output-dir outputs/multi_scene_dinov2_large
```

### On CPU

```bash
python experiments/exp_multi_scene_analysis.py \
    --dtu-path data/raw/DTU/Cleaned \
    --model dinov2_vitb14 \
    --device cpu \
    --output-dir outputs/multi_scene_cpu
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dtu-path` | str | `../data/raw/DTU/Cleaned` | Path to DTU Cleaned directory |
| `--model` | str | `dinov2_vitb14` | Model name |
| `--num-points` | int | 100 | Number of pixel locations to test per image |
| `--output-dir` | str | `outputs/multi_scene_dtu_analysis` | Output directory |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--device` | str | `cuda` | Device to use (cuda or cpu) |

---

## Output Structure

```
outputs/multi_scene_dtu_analysis/
├── sampled_images.txt                    # List of 124 sampled images
├── aggregated_results/
│   ├── aggregated_summary.png            # Mean ± std across all scenes
│   ├── scene_variance_heatmap.png        # Variance heatmap
│   ├── aggregated_results.json           # Detailed JSON results
│   └── aggregated_results.csv            # Summary table
└── summary_report.txt                    # Text summary
```

### Output Files Explained

**1. `sampled_images.txt`**
- Lists all 124 images used in the analysis
- One image per scan, randomly selected from "*_3_r5000.png" files
- Useful for reproducing exact analysis

**2. `aggregated_summary.png`**
- 2×2 grid plot showing all four invariance tests
- **Y-axis**: Mean cosine similarity (averaged across all 124 scenes)
- **Error bars**: Standard deviation across scenes (shows scene-to-scene variance)
- **Red line**: Best performing layer
- Higher mean + lower std = better and more consistent performance

**3. `scene_variance_heatmap.png`**
- Heatmap showing **standard deviation across scenes** for each layer and test
- Lower values (yellow) = more consistent across scenes
- Higher values (red) = more variance across scenes
- Helps identify which layers are scene-dependent

**4. `aggregated_results.json`**
```json
{
  "metadata": {
    "num_scenes": 124,
    "model_name": "dinov2_vitb14"
  },
  "results": {
    "scale": {
      "layer_0": {
        "mean_across_scenes": 0.85,      // Average over 124 scenes
        "std_across_scenes": 0.05,       // Scene-to-scene variance
        "min_across_scenes": 0.72,       // Worst scene
        "max_across_scenes": 0.93,       // Best scene
        "median_across_scenes": 0.86,
        "mean_of_stds": 0.03             // Avg within-image variance
      },
      ...
    },
    ...
  }
}
```

**5. `aggregated_results.csv`**
- Tabular format of the JSON data
- Easy to import into spreadsheets or pandas

**6. `summary_report.txt`**
- Human-readable summary
- Best layers for each invariance property
- Statistics and interpretation

---

## Interpretation Guide

### Understanding the Metrics

**Mean Across Scenes**
- Average cosine similarity across all 124 DTU scenes
- Higher = better invariance on average
- Range: [0, 1]

**Std Across Scenes**
- Standard deviation of cosine similarities across scenes
- **Lower is better** = consistent performance across diverse scenes
- High std = performance varies a lot depending on the scene

**Min/Max Across Scenes**
- Worst and best performing scenes
- Large gap suggests scene-dependent behavior

**Mean of Stds**
- Average within-image variance (from transformations)
- Lower = more stable to transformations within each scene

### What to Look For

**Good Layer Characteristics:**
- ✓ High mean_across_scenes (> 0.85)
- ✓ Low std_across_scenes (< 0.05)
- ✓ Small min-max gap
- ✓ Low mean_of_stds

**Example Interpretation:**
```
Layer 1: mean=0.92, std=0.03
Layer 11: mean=0.88, std=0.12
```
→ Layer 1 is better: Higher average AND more consistent across scenes

---

## Performance

### Runtime Estimate

- **124 images** × **4 tests** × **~30 seconds per test** ≈ **2.5 hours**
- Progress bar shows remaining time
- Can be interrupted and resumed (currently not implemented)

### Memory Usage

- Model loaded once (shared across all images)
- Peak memory: ~4-6 GB GPU (for base models)
- Larger models (vitl14) may need 8-12 GB

### Optimization Tips

**1. Reduce num-points for faster testing:**
```bash
# Quick test with 50 points
python exp_multi_scene_analysis.py --num-points 50
```

**2. Test on subset first:**
```bash
# Modify script to test first 10 scans
# (Add --max-scans 10 argument if needed)
```

**3. Parallelize across GPUs** (future enhancement):
```bash
# Split scans across multiple GPUs
CUDA_VISIBLE_DEVICES=0 python exp_multi_scene_analysis.py --start-scan 0 --end-scan 62
CUDA_VISIBLE_DEVICES=1 python exp_multi_scene_analysis.py --start-scan 62 --end-scan 124
```

---

## Comparison with Single-Image Analysis

| Aspect | Single Image | Multi-Scene (124 images) |
|--------|-------------|--------------------------|
| **Runtime** | ~2 minutes | ~2.5 hours |
| **Bias** | Scene-specific | Unbiased across scenes |
| **Variance** | Unknown | Quantified with std |
| **Reliability** | Limited | High statistical confidence |
| **Use case** | Quick testing | Final evaluation |

---

## Example Workflow

### 1. Quick Single-Image Test
```bash
# Test on one image first
python exp_layer_invariance.py \
    --image data/raw/test_image.png \
    --num-points 100
```

### 2. Multi-Scene Analysis
```bash
# Run on all DTU scenes
python exp_multi_scene_analysis.py \
    --num-points 100 \
    --output-dir outputs/dtu_full_analysis
```

### 3. Compare Models
```bash
# Test DINOv2
python exp_multi_scene_analysis.py \
    --model dinov2_vitb14 \
    --output-dir outputs/dinov2_multi_scene

# Test BEiT
python exp_multi_scene_analysis.py \
    --model beit_base_patch16_224 \
    --output-dir outputs/beit_multi_scene

# Compare results in outputs/
```

---

## Troubleshooting

### Error: "No images found"
```bash
# Check DTU path
ls data/raw/DTU/Cleaned/scan1/*_3_r5000.png

# Verify path in command
python exp_multi_scene_analysis.py \
    --dtu-path /full/path/to/DTU/Cleaned
```

### Error: "CUDA out of memory"
```bash
# Use CPU
python exp_multi_scene_analysis.py --device cpu

# Or reduce num-points
python exp_multi_scene_analysis.py --num-points 50
```

### Error: "Failed to load image"
- Some scans may have corrupted images
- Script will skip and continue with remaining images
- Check `sampled_images.txt` for which images were used

---

## Technical Details

### Sampling Strategy

For each scan directory (`scan1`, `scan2`, ..., `scan124`):
1. Find all images matching `*_3_r5000.png` (~49 images per scan)
2. Randomly select **one** image
3. Use fixed seed for reproducibility

**Why "_3" pattern?**
- DTU images are organized by view index and lighting condition
- "_3" refers to a specific lighting condition
- Ensures consistent lighting across different scenes

### Aggregation Method

For each (layer, test_type) pair:
```python
# Collect scores from all scenes
scores = [result[layer][test] for result in all_results]

# Compute statistics
mean_across_scenes = np.mean(scores)
std_across_scenes = np.std(scores)
min_across_scenes = np.min(scores)
max_across_scenes = np.max(scores)
median_across_scenes = np.median(scores)
```

### Reused Components

The script imports and reuses functions from `exp_layer_invariance.py`:
- `LayerFeatureExtractor` - Model loading and feature extraction
- `test_scale_invariance()` - Scale invariance testing
- `test_rotation_invariance()` - Rotation invariance testing
- `test_illumination_invariance()` - Illumination invariance testing
- `test_viewpoint_invariance()` - Viewpoint invariance testing

This ensures consistency between single-image and multi-scene analyses.

---

## Citation

If using this analysis in research, please cite the DTU dataset:

```bibtex
@inproceedings{aanaes2016large,
  title={Large scale multi-view stereopsis evaluation},
  author={Aan{\ae}s, Henrik and Jensen, Rasmus Ramsbøl and Vogiatzis, George and Tola, Engin and Dahl, Anders Bjorholm},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}
```

---

## Future Enhancements

Potential improvements:
- [ ] Add `--max-scans N` to test on subset
- [ ] Add `--resume` to continue interrupted analysis
- [ ] Add `--parallel` for multi-GPU support
- [ ] Export per-scene detailed results (optional)
- [ ] Add bootstrap confidence intervals
- [ ] Compare multiple models in one run
