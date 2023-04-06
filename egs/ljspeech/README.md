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

bash prepare.sh --stage -1 --stop-stage 3 \
  --audio_extractor "Encodec" \
  --audio_feats_dir data/tokenized
```


## Training
refer to [LibriTTS Training](../libritts/README.md#Training)


## Inference
refer to [LibriTTS Inference](../libritts/README.md#Inference)
