# LibriTTS

## Install deps
```
pip install librosa==0.8.1
```

## Prepare Dataset
```
cd egs/libritts

# Those stages are very time-consuming
bash run.sh --stage -1 --stop-stage 3
```


## Training

```
# 12G GPU --max-duration 24 --num-decoder-layers 6
bash run.sh --stage 4 --stop-stage 4 \
    --num-decoder-layers 12 \
    --max-duration 40 --use-fp16 true
```
![train](./demos/train.png)


## Inference
* checkpoint `exp/valle/epoch-10.pt` will be added.

```
python3 bin/infer.py \
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --model-name valle \
    --text-prompts "Go to her." \
    --audio-prompts ./prompts/61_70970_000007_000001.wav \
    --text "To get up and running quickly just follow the steps below." \
    --output-dir infer/demos \
    --checkpoint exp/valle/epoch-10.pt
```
