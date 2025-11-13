#!/bin/bash

# Test script for pixel visualization feature
# This script demonstrates how to use the new pixel visualization functionality

echo "=========================================="
echo "Testing Pixel Visualization Feature"
echo "=========================================="
echo ""

# Check if test image exists
if [ ! -f "../data/raw/test_image.png" ]; then
    echo "Error: Test image not found at ../data/raw/test_image.png"
    exit 1
fi

echo "Running experiment with pixel visualization..."
echo ""

# Run with pixel visualization enabled
python exp_layer_invariance.py \
    --image ../data/raw/test_image.png \
    --num-points 100 \
    --visualize-pixels \
    --top-n 50 \
    --layer-to-visualize layer_1 \
    --output-dir outputs/test_pixel_viz \
    --device cuda

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Check the following outputs:"
echo "  - outputs/test_pixel_viz/invariance_summary.png"
echo "  - outputs/test_pixel_viz/invariance_heatmap.png"
echo "  - outputs/test_pixel_viz/top_50_invariant_pixels_layer_1.png  <-- NEW!"
echo ""
