# Pixel Invariance Visualization

This document explains how to visualize the top N most invariant pixels across different transformation types.

## Overview

The updated `exp_layer_invariance.py` script now supports visualizing which pixels in an image have the most invariant features across transformations (scale, rotation, illumination, and viewpoint changes).

## Usage

### Basic Usage (without pixel visualization)

```bash
python experiments/exp_layer_invariance.py --image path/to/image.jpg
```

### With Pixel Visualization

```bash
python experiments/exp_layer_invariance.py \
    --image path/to/image.jpg \
    --visualize-pixels \
    --top-n 50 \
    --layer-to-visualize layer_1
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image` | str | **required** | Path to input image |
| `--model` | str | `dinov2_vitb14` | Model name (dinov2_vitb14, dinov2_vitl14, etc.) |
| `--num-points` | int | 100 | Number of random pixel locations to test |
| `--output-dir` | str | `outputs/invariance_results` | Directory to save results |
| `--device` | str | `cuda` | Device to use (cuda or cpu) |
| `--visualize-pixels` | flag | False | Generate visualization of top N invariant pixels |
| `--top-n` | int | 50 | Number of top invariant pixels to visualize |
| `--layer-to-visualize` | str | `layer_1` | Which layer to visualize (e.g., layer_0, layer_6, layer_11) |

## Examples

### Example 1: Visualize top 30 pixels from layer 6

```bash
python experiments/exp_layer_invariance.py \
    --image data/raw/test_image.png \
    --visualize-pixels \
    --top-n 30 \
    --layer-to-visualize layer_6 \
    --output-dir outputs/layer6_visualization
```

### Example 2: Test on 200 points and visualize top 100

```bash
python experiments/exp_layer_invariance.py \
    --image data/raw/test_image.png \
    --num-points 200 \
    --visualize-pixels \
    --top-n 100 \
    --layer-to-visualize layer_1
```

### Example 3: Use BEiT model and visualize last layer

```bash
python experiments/exp_layer_invariance.py \
    --image data/raw/test_image.png \
    --model beit_base_patch16_224 \
    --visualize-pixels \
    --top-n 50 \
    --layer-to-visualize layer_11
```

## Output

When `--visualize-pixels` is enabled, the script generates:

1. **Standard outputs** (always generated):
   - `invariance_summary.png` - Layer-wise invariance plots for all tests
   - `invariance_heatmap.png` - Heatmap comparing layers across tests
   - `invariance_results.json` - Numerical results in JSON format
   - `invariance_results.csv` - Summary in CSV format

2. **Pixel visualization** (new):
   - `top_N_invariant_pixels_layer_X.png` - 2×2 grid showing:
     - **Top-left**: Scale invariance top pixels
     - **Top-right**: Rotation invariance top pixels
     - **Bottom-left**: Illumination invariance top pixels
     - **Bottom-right**: Viewpoint invariance top pixels

## Interpreting the Visualization

### Markers

- **Gray circles**: All tested pixel locations
- **Colored stars**: Top N most invariant pixels
  - **Green**: Highest invariance (cosine similarity close to 1.0)
  - **Yellow**: Moderate invariance
  - **Red**: Lower invariance (among the top N)

### What to Look For

1. **Consistent locations across tests**: Pixels that appear as top invariant in multiple tests are excellent feature point candidates

2. **High average similarity**: Check the subplot title showing average cosine similarity
   - > 0.90: Excellent invariance
   - 0.80-0.90: Good invariance
   - 0.70-0.80: Moderate invariance
   - < 0.70: Poor invariance

3. **Spatial patterns**:
   - Edge regions often have high invariance
   - Texture-rich areas typically show better invariance
   - Uniform/flat regions usually have lower invariance

## Understanding Per-Pixel Similarity

Each pixel's invariance score is computed as:

1. **Per transformation**: Cosine similarity between original and transformed feature vectors
2. **Aggregation**: Average similarity across all transformations for that test type
3. **Ranking**: Pixels ranked by their average similarity score
4. **Top N**: The N pixels with highest average similarity are highlighted

## Performance Note

Enabling `--visualize-pixels` requires storing per-pixel similarity scores for all transformations, which increases memory usage slightly but has minimal impact on runtime.

## Tips

1. **Compare different layers**: Run the visualization for multiple layers to see which one produces the most stable feature points:
   ```bash
   for layer in layer_0 layer_1 layer_6 layer_11; do
       python experiments/exp_layer_invariance.py \
           --image data/raw/test_image.png \
           --visualize-pixels \
           --layer-to-visualize $layer \
           --output-dir outputs/pixel_viz_$layer
   done
   ```

2. **Test more points for statistical robustness**: Use `--num-points 500` for more comprehensive analysis

3. **Focus on specific transformations**: Look at which transformation type your pixels perform best on - this tells you what your features are most robust to

4. **Use for keypoint selection**: The top invariant pixels are excellent candidates for keypoint detection in COLMAP pipelines
