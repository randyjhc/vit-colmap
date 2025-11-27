#!/bin/bash

# Script to run ViT-COLMAP pipeline with SIFT on HPatches dataset
# HPatches sequences contain 6 images (.ppm format) with homography ground truth
# Using SIMPLE_RADIAL camera model as HPatches images are uncalibrated

# Usage: ./scripts/run_HPatches_sift.sh [sequence_name]
# Example: ./scripts/run_HPatches_sift.sh i_ajuntament
# Example: ./scripts/run_HPatches_sift.sh v_adam

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
    echo "Please cd to the project root and run: ./scripts/run_HPatches_sift.sh <sequence_name>"
    exit 1
fi

# Parse command-line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <sequence_name>"
    echo "Example: $0 i_ajuntament"
    echo "Example: $0 v_adam"
    echo ""
    echo "Available sequences in data/raw/HPatches:"
    echo "  Illumination changes (i_*):"
    ls -d data/raw/HPatches/i_* 2>/dev/null | xargs -n1 basename | head -5
    echo "  Viewpoint changes (v_*):"
    ls -d data/raw/HPatches/v_* 2>/dev/null | xargs -n1 basename | head -5
    exit 1
fi

SEQUENCE_NAME="$1"
HPATCHES_BASE="data/raw/HPatches"
SEQUENCE_DIR="${HPATCHES_BASE}/${SEQUENCE_NAME}"
OUTPUT_DIR="data/outputs/HPatches/${SEQUENCE_NAME}_sift"
DB_PATH="data/intermediate/HPatches/${SEQUENCE_NAME}/sift_database.db"
RESULTS_DIR="data/results"

# Check if sequence directory exists
if [ ! -d "$SEQUENCE_DIR" ]; then
    echo "Error: Sequence directory $SEQUENCE_DIR does not exist"
    echo ""
    echo "Available sequences:"
    ls -d ${HPATCHES_BASE}/*/ 2>/dev/null | xargs -n1 basename | head -10
    exit 1
fi

# Count images (.ppm files)
echo "Scanning for images in $SEQUENCE_DIR..."
num_images=$(ls -1 ${SEQUENCE_DIR}/*.ppm 2>/dev/null | wc -l)
if [ $num_images -eq 0 ]; then
    echo "Error: No .ppm images found in $SEQUENCE_DIR"
    echo "Available files:"
    ls -1 ${SEQUENCE_DIR}/ | head -10
    exit 1
fi
echo "Found $num_images images (.ppm files)"

# Create temporary directory with symlinks to images
TEMP_IMAGE_DIR="data/intermediate/HPatches/${SEQUENCE_NAME}/images"
mkdir -p "$TEMP_IMAGE_DIR"
rm -f ${TEMP_IMAGE_DIR}/*.ppm  # Clean up old symlinks

# Clean up old database to avoid conflicts
if [ -f "$DB_PATH" ]; then
    echo "Removing existing database: $DB_PATH"
    rm -f "$DB_PATH"
fi

echo "Creating symlinks to images..."
for img in ${SEQUENCE_DIR}/*.ppm; do
    ln -s "$(realpath $img)" "$TEMP_IMAGE_DIR/$(basename $img)"
done

# Determine sequence type for informational purposes
SEQ_TYPE="unknown"
if [[ "$SEQUENCE_NAME" == i_* ]]; then
    SEQ_TYPE="illumination"
elif [[ "$SEQUENCE_NAME" == v_* ]]; then
    SEQ_TYPE="viewpoint"
fi

echo ""
echo "Sequence Information:"
echo "  - Name: $SEQUENCE_NAME"
echo "  - Type: $SEQ_TYPE changes"
echo "  - Images: $num_images"
echo ""

# Run the pipeline with SIFT
echo "Running COLMAP pipeline with SIFT on HPatches ${SEQUENCE_NAME}..."
python -m vit_colmap.pipeline.run_pipeline \
    --images "$TEMP_IMAGE_DIR" \
    --output "$OUTPUT_DIR" \
    --db "$DB_PATH" \
    --camera-model SIMPLE_RADIAL \
    --use-colmap-sift \
    --dataset HPatches \
    --scene "$SEQUENCE_NAME" \
    --export-metrics "$RESULTS_DIR" \
    --verbose

echo ""
echo "Pipeline complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Metrics exported to: ${RESULTS_DIR}/HPatches/${SEQUENCE_NAME}/"
