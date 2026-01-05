### HPatches + SIFT
python scripts/visualize_matches.py \
    --database data/intermediate/HPatches/i_ajuntament/sift_database.db \
    --image-dir data/intermediate/HPatches/i_ajuntament/images \
    --image1 3 \
    --image2 4 \
    --max-matches 100 \
    --dpi 300 \
    --show-all-keypoints \
    --show-scores \
    --show-orientation \
    --orientation-scale 15.0

### HPatches + ViT
# Note: HPatches sequences have 6 images (1.ppm to 6.ppm)
# Best matches are typically between consecutive or nearby images
# Try different pairs: (0,1), (0,2), (1,2), etc.

# To find which pairs have matches, run:
# python scripts/visualize_matches.py --database data/intermediate/HPatches/v_yard/trainable_vit_database.db --list-matches

# python scripts/visualize_matches.py \
#     --database data/intermediate/HPatches/i_ajuntament/trainable_vit_database.db \
#     --image-dir data/intermediate/HPatches/i_ajuntament/images \
#     --image1 3 \
#     --image2 4 \
#     --max-matches 100 \
#     --dpi 300 \
#     --show-all-keypoints \
#     --show-scores \
#     --show-orientation \
#     --orientation-scale 15.0

### DTU + ViT
# python scripts/visualize_matches.py \
#     --database data/intermediate/DTU/scan1/trainable_vit_database.db \
#     --image-dir data/intermediate/DTU/scan1/images \
#     --image1 47 \
#     --image2 48 \
#     --max-matches 100 \
#     --dpi 300 \
#     --show-all-keypoints \
#     --show-scores \
#     --show-orientation \
#     --orientation-scale 15.0

### DTU + SIFT
# python scripts/visualize_matches.py \
#     --database data/intermediate/DTU/scan1/sift_database.db \
#     --image-dir data/intermediate/DTU/scan1/images \
#     --image1 47 \
#     --image2 48 \
#     --dpi 300 \
#     --show-all-keypoints \
#     --show-orientation \
#     --orientation-scale 15.0
