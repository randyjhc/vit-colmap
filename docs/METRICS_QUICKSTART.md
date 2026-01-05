# Metrics System Quick Start

## What's New

A comprehensive metrics extraction and export system has been added to enable evaluation and comparison of different feature extractors (SIFT vs ViT).

## Quick Usage

### 1. Run Experiments with Metrics

```bash
# SIFT baseline
./scripts/run_DTU_sift.sh scan1

# ViT
./scripts/run_DTU_vit.sh scan1

# Trainable ViT
./scripts/run_DTU_trainable_vit.sh scan1
```

### 2. Compare Results

```bash
# View comparison for default scan (scan1)
python scripts/compare_metrics.py

# Compare specific scan
python scripts/compare_metrics.py scan2

# Compare with custom dataset
python scripts/compare_metrics.py --dataset DTU scan3

# Compare 3 methods (SIFT, ViT, and a third method)
python scripts/compare_metrics.py scan1 --third-json trainable_vit.json --third-label "ViT-v1"

# Or aggregate all results
python scripts/aggregate_results.py --dataset DTU
```

### 3. View Results

```bash
# Detailed metrics (JSON)
cat data/results/DTU/scan1/sift.json
cat data/results/DTU/scan1/trainable_vit.json

# Comparison plot (PNG)
open data/results/DTU/scan1/comparison_plot.png

# Summary table (CSV)
cat data/results/DTU/summary.csv

# Comparison report (Markdown)
cat data/results/DTU/comparison_report.md
```

## What Metrics Are Captured?

### Feature Extraction
- Total keypoints, average per image
- Min/max/median keypoint counts

### Feature Matching
- Match counts (raw and after RANSAC)
- Inlier ratios
- Geometric verification stats

### 3D Reconstruction
- Registered image counts and rates
- Total 3D points
- Average track lengths
- Reprojection errors

## File Structure

```
data/results/
└── DTU/
    ├── scan1/
    │   ├── sift.json             # Detailed SIFT metrics
    │   ├── trainable_vit.json    # Detailed Trainable ViT metrics
    │   ├── vit.json              # Detailed standard ViT metrics
    │   └── comparison_plot.png   # Visual comparison plot
    ├── summary.csv               # All results
    └── comparison_report.md      # Analysis
```

## Example Output

```
METRICS COMPARISON: scan1
======================================================================

### Feature Extraction
Metric                                  SIFT            ViT        Diff
----------------------------------------------------------------------
Total keypoints                     400000.00     350000.00      -12.5%
Avg keypoints/image                   8163.27       7142.86      -12.5%

### Feature Matching
Metric                                  SIFT            ViT        Diff
----------------------------------------------------------------------
Matched pairs                         1150.00       1120.00       -2.6%
Avg inlier matches                     245.30        312.50       27.4%
Inlier ratio                             0.42          0.58       38.1%

### 3D Reconstruction
Metric                                  SIFT            ViT        Diff
----------------------------------------------------------------------
Registered images                        47.00         48.00        2.1%
Total 3D points                       35420.00      42380.00       19.6%
```

## Python API

```python
from pathlib import Path
from vit_colmap.utils.export import MetricsExporter
from vit_colmap.utils.plot_metrics import MetricsPlotter

# Load metrics
sift = MetricsExporter.load_json(Path("data/results/DTU/scan1/sift.json"))
vit = MetricsExporter.load_json(Path("data/results/DTU/scan1/trainable_vit.json"))

# Compare
print(f"SIFT: {sift.reconstruction.total_3d_points} points")
print(f"ViT:  {vit.reconstruction.total_3d_points} points")

# Generate plots with custom filenames
plotter = MetricsPlotter(
    Path("data/results/DTU"),
    sift_filename="sift.json",
    vit_filename="trainable_vit.json"
)
plotter.plot_single_scan("scan1", output_path=Path("comparison.png"))

# Compare three methods
plotter = MetricsPlotter(
    Path("data/results/DTU"),
    sift_filename="sift.json",
    vit_filename="vit.json",
    third_filename="trainable_vit.json",
    third_label="ViT-v1"
)
plotter.plot_single_scan("scan1", output_path=Path("comparison_3methods.png"))
```

## Integration with Existing Pipeline

The metrics system is fully integrated but optional:

```bash
# Run pipeline WITHOUT metrics (works as before)
python -m vit_colmap.pipeline.run_pipeline \
    --images data/images \
    --output data/outputs/scan1 \
    --db data/db/scan1.db

# Run pipeline WITH metrics (new functionality)
python -m vit_colmap.pipeline.run_pipeline \
    --images data/images \
    --output data/outputs/scan1 \
    --db data/db/scan1.db \
    --dataset DTU \
    --scene scan1 \
    --export-metrics data/results
```
