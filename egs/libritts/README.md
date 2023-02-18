# Training

```
cd egs/libritts

# prepare data
# Those stages are very time-consuming
bash run.sh --stage -1 --stop-stage 3

# training
# --max-duration 24 --num-decoder-layers 6 on a 12G GPU
bash run.sh --stage 4 --stop-stage 4 \
    --num-decoder-layers 12 \
    --deepspeed true --max-duration 24 --use-fp16 true
```
