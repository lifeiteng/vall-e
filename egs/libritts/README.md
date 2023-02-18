# Training

```
cd egs/libritts

# prepare data
# Those stages are very time-consuming
bash run.sh --stage -1 --stop-stage 3

# training
# 12G GPU --max-duration 24 --num-decoder-layers 6
bash run.sh --stage 4 --stop-stage 4 \
    --num-decoder-layers 12 \
    --max-duration 40 --use-fp16 false
```
