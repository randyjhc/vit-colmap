#!/bin/bash

# Script to run Trainable ViT-COLMAP pipeline on DTU dataset
# DTU images are 1600x1200 pixels
# Using PINHOLE camera model as DTU provides calibrated intrinsics

# Usage: ./scripts/run_DTU_trainable_vit.sh [scan_name] [--weights path/to/weights.pt]
# Example: ./scripts/run_DTU_trainable_vit.sh scan1
# Example: ./scripts/run_DTU_trainable_vit.sh scan1 --weights models/trained_vit.pt

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
    echo "Please cd to the project root and run: ./scripts/run_DTU_trainable_vit.sh <scan_name>"
    exit 1
fi

# Parse command-line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <scan_name> [--weights path/to/weights.pt]"
    echo "Example: $0 scan1"
    echo "Example: $0 scan1 --weights models/trained_vit.pt"
    echo ""
    echo "Available scans in data/raw/DTU/Cleaned:"
    ls -d data/raw/DTU/Cleaned/scan* 2>/dev/null | xargs -n1 basename | head -10
    exit 1
fi

SCAN_NAME="$1"
shift  # Remove scan_name from arguments

# Parse optional --weights argument
WEIGHTS_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --weights)
            if [ -n "$2" ]; then
                WEIGHTS_ARG="--vit-weights $2"
                shift 2
            else
                echo "Error: --weights requires a path argument"
                exit 1
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

DTU_BASE="data/raw/DTU/Cleaned"
IMAGE_DIR="${DTU_BASE}/${SCAN_NAME}"
OUTPUT_DIR="data/outputs/DTU/${SCAN_NAME}_trainable_vit"
DB_PATH="data/intermediate/DTU/${SCAN_NAME}/trainable_vit_database.db"
RESULTS_DIR="data/results"

# Check if image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory $IMAGE_DIR does not exist"
    exit 1
fi

# Count images (detect naming pattern: standard DTU or scan0 format)
echo "Scanning for images in $IMAGE_DIR..."

# First try standard DTU pattern (diffuse lighting images ending with _3_r5000.png)
num_images=$(ls -1 ${IMAGE_DIR}/*_3_r5000.png 2>/dev/null | wc -l)
if [ $num_images -gt 0 ]; then
    IMAGE_PATTERN="*_3_r5000.png"
    echo "Found $num_images images with diffuse lighting (_3_r5000.png pattern)"
else
    # Try scan0 pattern (frame_*.png)
    num_images=$(ls -1 ${IMAGE_DIR}/frame_*.png 2>/dev/null | wc -l)
    if [ $num_images -gt 0 ]; then
        IMAGE_PATTERN="frame_*.png"
        echo "Found $num_images images with frame_*.png pattern"
    else
        echo "Error: No images found with pattern *_3_r5000.png or frame_*.png"
        echo "Available images:"
        ls -1 ${IMAGE_DIR}/*.png | head -5
        exit 1
    fi
fi

# Create temporary directory with symlinks to only the diffuse lighting images
TEMP_IMAGE_DIR="data/intermediate/DTU/${SCAN_NAME}/images"
mkdir -p "$TEMP_IMAGE_DIR"
rm -f ${TEMP_IMAGE_DIR}/*.png  # Clean up old symlinks

# Clean up old database to avoid conflicts
if [ -f "$DB_PATH" ]; then
    echo "Removing existing database: $DB_PATH"
    rm -f "$DB_PATH"
fi

echo "Creating symlinks to images..."
for img in ${IMAGE_DIR}/${IMAGE_PATTERN}; do
    ln -s "$(realpath $img)" "$TEMP_IMAGE_DIR/$(basename $img)"
done

# Run the pipeline with Trainable ViT extractor
echo "Running COLMAP pipeline with Trainable ViT on DTU ${SCAN_NAME}..."
if [ -n "$WEIGHTS_ARG" ]; then
    echo "Using custom weights: $WEIGHTS_ARG"
fi

python -m vit_colmap.pipeline.run_pipeline \
    --images "$TEMP_IMAGE_DIR" \
    --output "$OUTPUT_DIR" \
    --db "$DB_PATH" \
    --camera-model PINHOLE \
    --extractor trainable_vit \
    $WEIGHTS_ARG \
    --dataset DTU \
    --scene "$SCAN_NAME" \
    --export-metrics "$RESULTS_DIR" \
    --verbose

echo "Pipeline complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Metrics exported to: ${RESULTS_DIR}/DTU/${SCAN_NAME}/"
