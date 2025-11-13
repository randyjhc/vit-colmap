# python ./exp_layer_invariance.py --image ../data/raw/test_image.png --num-points 20000 --model microsoft/beit-base-patch16-224
# python exp_layer_invariance.py \
#     --image test_scan9.png \
#     --num-points 10000 \
#     --visualize-pixels \
#     --top-n 200 \
#     --layer-to-visualize layer_1 \
#     --model beit
python exp_layer_invariance.py \
    --image test_scan9.png \
    --num-points 10000 \
    --visualize-pixels \
    --top-n 200 \
    --layer-to-visualize layer_11
