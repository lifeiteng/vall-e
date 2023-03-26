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
* Fix [segmentation fault (core dumped)](https://github.com/lifeiteng/vall-e#troubleshooting)
* Fix `h5py Unable to open object (object ...`
  * Make sure then version of h5py in `bin/tokenizer.py` and `bin/trainer.py` are same: `pip install h5py==3.8.0`

```
# 12G GPU --max-duration 24 --filter-max-duration 14 --num-decoder-layers 6
bash run.sh --stage 4 --stop-stage 4 --max-duration 40 --filter-max-duration 14 \
    --num-decoder-layers 12
```
![train](./demos/train.png)

#### Prefix Mode 0 1 2 4 for NAR Decoder
**Paper Chapter 5.1** "The average length of the waveform in LibriLight is 60 seconds. During
training, we randomly crop the waveform to a random length between 10 seconds and 20 seconds. For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds from the same utterance."
* **0**: no acoustic prompt tokens
* **1**: random prefix of current batched utterances
* **2**: random segment of current batched utterances
* **4**: same as the paper (As they randomly crop the long waveform to multiple utterances, so the same utterance means pre or post utterance in the same long waveform.)

```
# If train AR & NAR Decoders with prefix_mode 4
bash run.sh --stage 4 --stop-stage 4 --max-duration 40 --filter-max-duration 14 \
            --num-epochs 10 --start-epoch 1 --prefix_mode 4 --exp_suffix "_mode4" \
            --train-options "--train-stage 0 --dataset libritts --input-strategy PromptedPrecomputedFeatures"
```

#### Train AR Decoder and NAR Decoder individually
* try larger `--max-duration 80`
```
# Train AR Decoder
bash run.sh --stage 4 --stop-stage 4 --max-duration 80 --filter-max-duration 14 \
            --num-epochs 10 --start-epoch 1 --prefix_mode 0 \
            --train-options "--train-stage 1"

# Train NAR Decoder with  --prefix_mode 0/1/2
bash run.sh --stage 4 --stop-stage 4 --max-duration 60 --filter-max-duration 14 \
            --num-epochs 10 --start-epoch 10 --prefix_mode 0 \
            --train-options "--train-stage 2"

# If train NAR Decoder with prefix_mode 4
bash run.sh --stage 4 --stop-stage 4 --max-duration 60 --filter-max-duration 14 \
            --num-epochs 10 --start-epoch 10 --prefix_mode 4 --exp_suffix "_mode4" \
            --train-options "--train-stage 2 --dataset libritts --input-strategy PromptedPrecomputedFeatures"
```

## Inference

```
python3 bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12  \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below." \
    --checkpoint=expX4/valle/checkpoint-400000.pt
```
