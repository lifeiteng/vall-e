#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python3 valle/tests/model_test.py

cd egs/libritts

python3 bin/infer.py \
    --decoder-dim 64 --nhead 4 --num-decoder-layers 4 --model-name valle \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "To test." \
    --output-dir infer/demos --top-k 10 --temperature 1.0 \
    --checkpoint ""

python3 bin/infer.py \
    --decoder-dim 64 --nhead 4 --num-decoder-layers 4 --model-name Transformer \
    --text-prompts "" \
    --audio-prompts "" \
    --text "To test." \
    --output-dir infer/demos_transformer \
    --checkpoint ""
