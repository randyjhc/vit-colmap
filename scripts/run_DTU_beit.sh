#!/bin/bash

# ============================================
# DEPRECATED: This script is no longer functional
# ============================================
# BEiT is now only used for visualization, not COLMAP feature extraction.
#
# For COLMAP feature extraction, use:
#   - ViT extractor (default): python -m vit_colmap.pipeline.run_pipeline --images <dir> ...
#   - COLMAP SIFT: python -m vit_colmap.pipeline.run_pipeline --images <dir> --use-colmap-sift ...
#
# For BEiT layer visualization, use the simplified BEiTExtractor directly:
#   from vit_colmap.features.beit_extractor import BEiTExtractor
#   extractor = BEiTExtractor(layer_idx=9)
#   extractor.visualize_layer_rgb(image_path, output_path)
#   extractor.visualize_layer_heatmap(image_path, output_path)
# ============================================

# Script to run BEiT-COLMAP pipeline on DTU dataset
# DTU images are 1600x1200 pixels
# Using PINHOLE camera model as DTU provides calibrated intrinsics

# Usage: ./scripts/run_DTU_beit.sh [scan_name] [layer] [projection_method] [keypoint_method] [num_keypoints] [max_per_patch] [min_saliency]
# Example: ./scripts/run_DTU_beit.sh scan1 9 pca adaptive
# Example: ./scripts/run_DTU_beit.sh scan1 9 pca adaptive 2048 8 0.1
# Example: ./scripts/run_DTU_beit.sh scan1 (uses defaults: layer=9, projection=pca, keypoint=saliency)

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
    echo "Please cd to the project root and run: ./scripts/run_DTU_beit.sh <scan_name>"
    exit 1
fi

# Parse command-line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <scan_name> [layer] [projection_method] [keypoint_method] [num_keypoints] [max_per_patch] [min_saliency]"
    echo ""
    echo "Arguments:"
    echo "  scan_name          - DTU scan name (e.g., scan1, scan6)"
    echo "  layer              - BEiT layer to use (0-12, default: 9)"
    echo "  projection_method  - Descriptor projection (pca/random/learned, default: pca)"
    echo "  keypoint_method    - Keypoint selection (saliency/uniform/adaptive, default: saliency)"
    echo "  num_keypoints      - Number of keypoints per image (default: 2048)"
    echo "  max_per_patch      - Max keypoints per patch for adaptive (default: 8)"
    echo "  min_saliency       - Min saliency ratio for adaptive (default: 0.1)"
    echo ""
    echo "Examples:"
    echo "  $0 scan1                         # Use all defaults"
    echo "  $0 scan1 8                       # Layer 8, other defaults"
    echo "  $0 scan1 9 random                # Layer 9, random projection"
    echo "  $0 scan1 9 pca saliency          # Layer 9, PCA, saliency keypoints"
    echo "  $0 scan1 9 pca adaptive          # Layer 9, PCA, adaptive density"
    echo "  $0 scan1 9 pca adaptive 2048 16 0.05  # Custom adaptive params"
    echo ""
    echo "Available scans in data/raw/DTU/Cleaned:"
    ls -d data/raw/DTU/Cleaned/scan* 2>/dev/null | xargs -n1 basename | head -10
    exit 1
fi

SCAN_NAME="$1"
BEIT_LAYER="${2:-9}"                    # Default: layer 9
PROJECTION_METHOD="${3:-pca}"            # Default: pca
KEYPOINT_METHOD="${4:-saliency}"         # Default: saliency
NUM_KEYPOINTS="${5:-2048}"               # Default: 2048
MAX_PER_PATCH="${6:-8}"                  # Default: 8
MIN_SALIENCY="${7:-0.1}"                 # Default: 0.1

DTU_BASE="data/raw/DTU/Cleaned"
IMAGE_DIR="${DTU_BASE}/${SCAN_NAME}"
OUTPUT_DIR="data/outputs/DTU/${SCAN_NAME}_beit_L${BEIT_LAYER}_${PROJECTION_METHOD}_${KEYPOINT_METHOD}"
DB_PATH="data/intermediate/DTU/${SCAN_NAME}/beit_L${BEIT_LAYER}_database.db"
RESULTS_DIR="data/results"

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

# Display configuration
echo ""
echo "=========================================="
echo "BEiT-COLMAP Pipeline Configuration"
echo "=========================================="
echo "Scan:              $SCAN_NAME"
echo "BEiT Layer:        $BEIT_LAYER"
echo "Projection:        $PROJECTION_METHOD"
echo "Keypoint method:   $KEYPOINT_METHOD"
echo "Num keypoints:     $NUM_KEYPOINTS"
if [ "$KEYPOINT_METHOD" == "adaptive" ]; then
    echo "Max per patch:     $MAX_PER_PATCH"
    echo "Min saliency:      $MIN_SALIENCY"
fi
echo "Images:            $TEMP_IMAGE_DIR"
echo "Output:            $OUTPUT_DIR"
echo "Database:          $DB_PATH"
echo "=========================================="
echo ""

# Run the pipeline with BEiT extractor
# Note: DTU provides calibrated cameras, but for now we use estimated intrinsics
# You can later modify this to load actual DTU calibration data
echo "Running COLMAP pipeline with BEiT on DTU ${SCAN_NAME}..."

# Build command with optional adaptive parameters
CMD="python -m vit_colmap.pipeline.run_pipeline \
    --images \"$TEMP_IMAGE_DIR\" \
    --output \"$OUTPUT_DIR\" \
    --db \"$DB_PATH\" \
    --camera-model PINHOLE \
    --use-beit \
    --beit-layer $BEIT_LAYER \
    --projection-method $PROJECTION_METHOD \
    --keypoint-method $KEYPOINT_METHOD \
    --num-keypoints $NUM_KEYPOINTS"

# Add adaptive parameters if using adaptive method
if [ "$KEYPOINT_METHOD" == "adaptive" ]; then
    CMD="$CMD --max-keypoints-per-patch $MAX_PER_PATCH --min-patch-saliency-ratio $MIN_SALIENCY"
fi

CMD="$CMD --dataset DTU --scene \"$SCAN_NAME\" --export-metrics \"$RESULTS_DIR\" --verbose"

eval $CMD

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Database: $DB_PATH"
echo "Metrics exported to: ${RESULTS_DIR}/DTU/${SCAN_NAME}/"
echo "Configuration: BEiT Layer ${BEIT_LAYER}, ${PROJECTION_METHOD} projection, ${KEYPOINT_METHOD} keypoints"
echo "=========================================="

# Visualize keypoints for the first image
echo ""
echo "Generating keypoint visualization for first image..."
FIRST_IMAGE=$(ls -1 ${TEMP_IMAGE_DIR}/*.png 2>/dev/null | head -1)
if [ -n "$FIRST_IMAGE" ]; then
    VIS_OUTPUT_DIR="${OUTPUT_DIR}/visualizations"
    mkdir -p "$VIS_OUTPUT_DIR"

    FIRST_IMAGE_NAME=$(basename "$FIRST_IMAGE")
    VIS_OUTPUT_PATH="${VIS_OUTPUT_DIR}/${FIRST_IMAGE_NAME%.png}_keypoints.png"

    echo "  Input image: $FIRST_IMAGE_NAME"
    echo "  Output: $VIS_OUTPUT_PATH"

    # Run visualization using Python
    python -c "
from pathlib import Path
from vit_colmap.features.beit_extractor import BEiTExtractor

# Initialize extractor with same configuration
extractor = BEiTExtractor(
    layer_idx=${BEIT_LAYER},
    num_keypoints=${NUM_KEYPOINTS},
    descriptor_dim=128,
    projection_method='${PROJECTION_METHOD}',
    keypoint_method='${KEYPOINT_METHOD}',
    max_keypoints_per_patch=${MAX_PER_PATCH},
    min_patch_saliency_ratio=${MIN_SALIENCY},
)

# Visualize keypoints
extractor.visualize_keypoints(
    image_path=Path('${FIRST_IMAGE}'),
    output_path=Path('${VIS_OUTPUT_PATH}')
)
print('✓ Visualization saved')
"

    if [ $? -eq 0 ]; then
        echo "✓ Keypoint visualization complete!"
        echo "  Saved to: $VIS_OUTPUT_PATH"
    else
        echo "⚠ Visualization failed"
    fi
else
    echo "⚠ No images found for visualization"
fi

echo ""
echo "=========================================="
echo "All tasks complete!"
echo "=========================================="
