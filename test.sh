#!/usr/bin/env bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python3 valle/modules/model.py

cd egs/libritts

# VALL-F
python3 bin/infer.py \
    --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name vallf \
    --text-prompts "Go to her." \
    --audio-prompts ./prompts/61_70970_000007_000001.wav \
    --text "To get up and running quickly just follow the steps below." \
    --output-dir infer/demo_vallf_PostNorm_epoch10 \
    --checkpoint exp/vallf_nano_v41_PostNorm/epoch-10.pt

# VALL-E
python3 bin/infer.py \
    --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
    --text-prompts "Go to her." \
    --audio-prompts ./prompts/61_70970_000007_000001.wav \
    --text "To get up and running quickly just follow the steps below." \
    --output-dir infer/demo_valle_PostNorm_epoch10 \
    --checkpoint exp/valle_nano_v41_PostNorm/epoch-10.pt

# git add -f exp/vallf_nano_v41_PostNorm/epoch-10.pt
# git add -f infer/demo_vallf_PostNorm_epoch10
# git add -f exp/valle_nano_v41_PostNorm/epoch-10.pt
# git add -f infer/demo_valle_PostNorm_epoch10
