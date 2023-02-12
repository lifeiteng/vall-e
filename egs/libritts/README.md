# Training

```
cd egs/libritts

# prepare data
# Those stages are very time-consuming
bash run.sh --stage -1 --stop-stage 3

# training
# nano
bash run.sh --stage 4 --stop-stage 5
# raw
bash run.sh --stage 5 --stop-stage 5
```
