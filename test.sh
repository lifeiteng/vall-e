#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python3 valle/tests/model_test.py

cd egs/libritts

python3 bin/infer.py \
    --decoder-dim 64 --nhead 4 --num-decoder-layers 4 --model-name valle \
    --text-prompts "Go to her." \
    --audio-prompts ./prompts/61_70970_000007_000001.wav \
    --text "To test." \
    --output-dir infer/demos \
    --checkpoint ""

python3 bin/infer.py \
    --decoder-dim 64 --nhead 4 --num-decoder-layers 4 --model-name Transformer \
    --text-prompts "" \
    --audio-prompts "" \
    --text "To test." \
    --output-dir infer/demos_transformer \
    --checkpoint ""
