python scripts/train_vit_features.py \
    --data-root data/raw/HPatches \
    --resume checkpoints/latest.pt \
    --use-amp \
    --cudnn-benchmark \
    --compile \
    --epochs 99
