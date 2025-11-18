# python scripts/visualize_matches.py \
#     --database data/intermediate/DTU/scan65/trainable_vit_database.db \
#     --image-dir data/intermediate/DTU/scan65/images \
#     --image1 47 \
#     --image2 48 \
#     --max-matches 100 \
#     --dpi 300 \
#     --filter inliers \
#     --show-all-keypoints

python scripts/visualize_matches.py \
    --database data/intermediate/DTU/scan65/sift_database.db \
    --image-dir data/intermediate/DTU/scan65/images \
    --image1 47 \
    --image2 48 \
    --dpi 300 \
    --show-all-keypoints
