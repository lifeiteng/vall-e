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
#### data

```
##  train
Cut statistics:
╒═══════════════════════════╤═══════════╕
│ Cuts count:               │ 354780    │
├───────────────────────────┼───────────┤
│ Total duration (hh:mm:ss) │ 555:09:48 │
├───────────────────────────┼───────────┤
│ mean                      │ 5.6       │
├───────────────────────────┼───────────┤
│ std                       │ 4.5       │
├───────────────────────────┼───────────┤
│ min                       │ 0.1       │
├───────────────────────────┼───────────┤
│ 25%                       │ 2.3       │
├───────────────────────────┼───────────┤
│ 50%                       │ 4.3       │
├───────────────────────────┼───────────┤
│ 75%                       │ 7.6       │
├───────────────────────────┼───────────┤
│ 80%                       │ 8.7       │
├───────────────────────────┼───────────┤
│ 85%                       │ 10.0      │
├───────────────────────────┼───────────┤
│ 90%                       │ 11.8      │
├───────────────────────────┼───────────┤
│ 95%                       │ 14.8      │
├───────────────────────────┼───────────┤
│ 99%                       │ 20.9      │
├───────────────────────────┼───────────┤
│ 99.5%                     │ 23.1      │
├───────────────────────────┼───────────┤
│ 99.9%                     │ 27.4      │
├───────────────────────────┼───────────┤
│ max                       │ 43.9      │
├───────────────────────────┼───────────┤
│ Recordings available:     │ 354780    │
├───────────────────────────┼───────────┤
│ Features available:       │ 354780    │
├───────────────────────────┼───────────┤
│ Supervisions available:   │ 354780    │
╘═══════════════════════════╧═══════════╛
SUPERVISION custom fields:
Speech duration statistics:
╒══════════════════════════════╤═══════════╤══════════════════════╕
│ Total speech duration        │ 555:09:48 │ 100.00% of recording │
├──────────────────────────────┼───────────┼──────────────────────┤
│ Total speaking time duration │ 555:09:48 │ 100.00% of recording │
├──────────────────────────────┼───────────┼──────────────────────┤
│ Total silence duration       │ 00:00:01  │ 0.00% of recording   │
╘══════════════════════════════╧═══════════╧══════════════════════╛
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
