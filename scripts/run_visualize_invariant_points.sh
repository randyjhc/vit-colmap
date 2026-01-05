#!/bin/bash

# Example script to visualize invariant points on HPatches image pairs
# This shows how the training sampler selects invariant points based on cosine similarity

# python scripts/visualize_invariant_points.py \
#     --data-root data/raw/HPatches \
#     --sequence i_ajuntament \
#     --pair-idx 0 \
#     --max-points 100 \
#     --top-k-invariant 1024 \
#     --dpi 300 \
#     --font-size 6

# Other example sequences:
# --sequence v_adam
# --sequence i_contruction
# --sequence v_there

# Example with filtering and no labels:
python scripts/visualize_invariant_points.py \
    --data-root data/raw/HPatches \
    --sequence v_there \
    --pair-idx 0 \
    --top-k-invariant 20000 \
    --min-similarity 0.9 \
    --no-labels \
    --dpi 300

# Useful options:
# --max-points 100           # Show only top 100 points by similarity
# --min-similarity 0.95      # Only show points with similarity >= 0.95
# --no-labels                # Disable similarity score labels
# --show-all-labels          # Show similarity labels for all displayed points
# --output output.png        # Save to file instead of displaying
# --top-k-invariant 1024     # Select more invariant points
