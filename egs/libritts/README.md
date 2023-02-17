# Training

```
cd egs/libritts

# prepare data
# Those stages are very time-consuming
bash run.sh --stage -1 --stop-stage 3

# training
# nano
# bash run.sh --stage 4 --stop-stage 5
# raw
bash run.sh --stage 5 --stop-stage 5
# --stage 5
# you should set hparams base on the GPU memory(blow can be trained on a 12G GPU)
deepspeed bin/trainer.py --max-duration 24 --use-fp16 true \
    --decoder-dim 1024 --nhead 4 --num-decoder-layers 6 \
    --deepspeed --deepspeed_config configs/ds_zero2.config \
    --exp-dir exp/valle_ds_zero2_CPUAdam_Layer6Head4

```
