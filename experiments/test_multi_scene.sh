#!/bin/bash

# Test script for multi-scene DTU analysis
# This demonstrates how to run the analysis on all 124 DTU scenes

echo "=========================================="
echo "Multi-Scene DTU Invariance Analysis Test"
echo "=========================================="
echo ""

# Check if DTU data exists
if [ ! -d "../data/raw/DTU/Cleaned" ]; then
    echo "Error: DTU Cleaned directory not found at ../data/raw/DTU/Cleaned"
    echo "Please ensure DTU dataset is downloaded and extracted"
    exit 1
fi

# Count number of scans
num_scans=$(ls -d ../data/raw/DTU/Cleaned/scan*/ 2>/dev/null | wc -l)
echo "Found $num_scans DTU scan directories"
echo ""

# Estimate runtime
echo "Estimated runtime: ~2.5 hours for all 124 scans"
echo "Each scan takes about ~2 minutes (4 tests × 30 seconds)"
echo ""

read -p "Continue with full analysis? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting multi-scene analysis..."
echo "Progress will be shown with tqdm progress bar"
echo ""

# Run the analysis
python exp_multi_scene_analysis.py \
    --dtu-path ../data/raw/DTU/Cleaned \
    --model dinov2_vitb14 \
    --num-points 100 \
    --output-dir outputs/multi_scene_dtu_analysis \
    --seed 42 \
    --device cuda

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Check results in:"
echo "  - outputs/multi_scene_dtu_analysis/summary_report.txt"
echo "  - outputs/multi_scene_dtu_analysis/aggregated_results/"
echo ""
