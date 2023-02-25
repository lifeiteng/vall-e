# LJSpeech

Train TransformerTTS
* [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)

## Install deps
```
pip install librosa==0.8.1

# lhotse update LJSpeech
# https://github.com/lhotse-speech/lhotse/pull/988
```

## Prepare Dataset
```
cd egs/ljspeech

# Those stages are very time-consuming
bash run.sh --stage -1 --stop-stage 3
```


## Training

```
python3 bin/trainer.py --max-duration 100 --use-fp16 false \
      --bucketing-sampler true --save-every-n 5000 \
      --model-name Transformer --decoder-dim 384 --nhead 8 --num-decoder-layers 6 \
      --base-lr 0.05 --warmup-steps 5000 --optimizer-name ScaledAdam \
      --on-the-fly-feats false --manifest-dir data/fbank \
      --text-tokens data/fbank/unique_text_tokens.k2symbols \
      --exp-dir exp_seqtts/Transformer_Dim384H8
```


## Inference

```
python3 bin/infer.py \
    --model-name Transformer --decoder-dim 384 --nhead 8 --num-decoder-layers 6 \
    --text-prompts "" \
    --audio-prompts "" \
    --text-tokens data/fbank/unique_text_tokens.k2symbols \
    --text "To get up and running quickly just follow the steps below." \
    --output-dir infer/demos \
    --checkpoint exp_seqtts/Transformer_Dim384H8/epoch-20.pt
```
