# Training

```
cd egs/libritts

# Those stages are very time-consuming
./prepare.sh

# nano: on NV GPU with 12G memory
# python3 bin/trainer.py \
#     --decoder-dim 128 --nhead 4 --num-decoder-layers 4 \
#     --max-duration 40 --model-name vallf \
#     --exp-dir exp/vallf_nano_full

python3 bin/trainer.py \
    --decoder-dim 128 --nhead 4 --num-decoder-layers 4 \
    --max-duration 40 --model-name valle \
    --exp-dir exp/valle_nano_full

# same as paper, but need more memory
python3 bin/trainer.py \
  --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
  --exp-dir exp/valle
```
