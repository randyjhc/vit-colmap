python scripts/train_vit_features.py \
    --data-root data/raw/HPatches \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --top-k 512 \
    --lambda-det 1.0 \
    --lambda-rot 0.5 \
    --lambda-desc 1.0 \
    --checkpoint-dir checkpoints/exp1
