#!/bin/bash

# Script to run ViT-COLMAP pipeline on DTU dataset
# DTU images are 1600x1200 pixels
# Using PINHOLE camera model as DTU provides calibrated intrinsics

# Usage: ./scripts/run_DTU_colmap.sh [scan_name]
# Example: ./scripts/run_DTU_colmap.sh scan1

# Check if running from project root directory
if [ ! -f "vit_colmap/__init__.py" ] && [ ! -f "pyproject.toml" ]; then
    echo "Error: This script must be run from the project root directory (vit-colmap/)"
    echo "Current directory: $(pwd)"
    echo "Expected directory structure:"
    echo "  vit-colmap/"
    echo "  ├── vit_colmap/"
    echo "  ├── scripts/"
    echo "  ├── data/"
    echo "  └── pyproject.toml"
    echo ""
    echo "Please cd to the project root and run: ./scripts/run_DTU_colmap.sh <scan_name>"
    exit 1
fi

# Parse command-line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <scan_name>"
    echo "Example: $0 scan1"
    echo ""
    echo "Available scans in data/raw/DTU/Cleaned:"
    ls -d data/raw/DTU/Cleaned/scan* 2>/dev/null | xargs -n1 basename | head -10
    exit 1
fi

SCAN_NAME="$1"
DTU_BASE="data/raw/DTU/Cleaned"
IMAGE_DIR="${DTU_BASE}/${SCAN_NAME}"
OUTPUT_DIR="data/outputs/DTU/${SCAN_NAME}"
DB_PATH="data/intermediate/DTU/${SCAN_NAME}/database.db"

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory $IMAGE_DIR does not exist"
    exit 1
fi

# Count images (only use diffuse lighting images ending with _3_r5000.png)
echo "Scanning for images in $IMAGE_DIR..."
num_images=$(ls -1 ${IMAGE_DIR}/*_3_r5000.png 2>/dev/null | wc -l)
if [ $num_images -eq 0 ]; then
    echo "Error: No images found with pattern *_3_r5000.png"
    echo "Available images:"
    ls -1 ${IMAGE_DIR}/*.png | head -5
    exit 1
fi
echo "Found $num_images images with diffuse lighting (_3_r5000.png pattern)"

# Create temporary directory with symlinks to only the diffuse lighting images
TEMP_IMAGE_DIR="data/intermediate/DTU/${SCAN_NAME}/images"
mkdir -p "$TEMP_IMAGE_DIR"
rm -f ${TEMP_IMAGE_DIR}/*.png  # Clean up old symlinks

# Clean up old database to avoid conflicts
if [ -f "$DB_PATH" ]; then
    echo "Removing existing database: $DB_PATH"
    rm -f "$DB_PATH"
fi

echo "Creating symlinks to diffuse lighting images..."
for img in ${IMAGE_DIR}/*_3_r5000.png; do
    ln -s "$(realpath $img)" "$TEMP_IMAGE_DIR/$(basename $img)"
done

# Run the pipeline
# Note: DTU provides calibrated cameras, but for now we use estimated intrinsics
# You can later modify this to load actual DTU calibration data
echo "Running COLMAP pipeline with SIFT on DTU ${SCAN_NAME}..."
python -m vit_colmap.pipeline.run_pipeline \
    --images "$TEMP_IMAGE_DIR" \
    --output "$OUTPUT_DIR" \
    --db "$DB_PATH" \
    --camera-model PINHOLE \
    --use-colmap-sift \
    --verbose

echo "Pipeline complete!"
echo "Results saved to: $OUTPUT_DIR"
