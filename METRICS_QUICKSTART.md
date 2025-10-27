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
```

### 2. Compare Results

```bash
# View comparison
python scripts/compare_metrics.py

# Or aggregate all results
python scripts/aggregate_results.py --dataset DTU
```

### 3. View Results

```bash
# Detailed metrics (JSON)
cat data/results/DTU/scan1/sift.json
cat data/results/DTU/scan1/vit.json

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
    │   ├── sift.json          # Detailed SIFT metrics
    │   └── vit.json           # Detailed ViT metrics
    ├── summary.csv            # All results
    └── comparison_report.md   # Analysis
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

# Load metrics
sift = MetricsExporter.load_json(Path("data/results/DTU/scan1/sift.json"))
vit = MetricsExporter.load_json(Path("data/results/DTU/scan1/vit.json"))

# Compare
print(f"SIFT: {sift.reconstruction.total_3d_points} points")
print(f"ViT:  {vit.reconstruction.total_3d_points} points")
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
