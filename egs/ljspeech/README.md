# LJSpeech

## Install deps
```
pip install librosa==0.8.1

# lhotse update LJSpeech
# https://github.com/lhotse-speech/lhotse/pull/988
```

## Prepare Dataset
```
cd egs/ljspeech

bash run.sh --stage -1 --stop-stage 3 \
  --audio_extractor "Encodec" \
  --audio_feats_dir data/tokenized
```


## Training

```
python3 bin/trainer.py --max-duration 72 --filter-max-duration 14 \
      --num-buckets 6 --dtype "float32" --save-every-n 10000 \
      --model-name valle --norm-first true --add-prenet false \
      --decoder-dim 256 --nhead 8 --num-decoder-layers 6 \
      --base-lr 0.05 --warmup-steps 200 \
      --num-epochs 100 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 1 \
      --exp-dir exp/valle_Dim256H8L6_LR05
```


## Inference

```
python3 bin/infer.py --output-dir demos \
    --top-k -1 --temperature 1.0 \
    --model-name valle --norm-first true --add-prenet false \
    --decoder-dim 256 --nhead 8 --num-decoder-layers 6  \
    --text-prompts "In addition, the proposed legislation will insure." \
    --audio-prompts ./prompts/LJ049-0124_24K.wav \
    --text "To get up and running quickly just follow the steps below.|During the period the Commission was giving thought to this situation." \
    --checkpoint exp/valle_Dim256H8L6_LR05/epoch-100.pt
```
