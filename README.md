Language : 🇺🇸 | [🇨🇳](./README.zh-CN.md)

An unofficial PyTorch implementation of VALL-E([Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)).

## Demo

* [official demo](https://valle-demo.github.io/)
* TODO: reproduced results

## Broader impacts

> Since VALL-E could synthesize speech that maintains speaker identity, it may carry potential risks in misuse of the model, such as spoofing voice identification or impersonating a specific speaker.

We will not provide well-trained models and services.

## Progress

- [x] Text and Audio Tokenizer
- [x] Dataset module and loaders
- [ ] VALL-E modules
    - [x] AR Decoder
    - [ ] NonAR Decoder
- [ ] update REAMDE.zh-CN
- [ ] Training & Debug
- [ ] Inference: In-Context Learning via Prompting


## Installation

To get up and running quickly just follow the steps below:

```
# phonemizer
apt-get install espeak-ng
## OSX: brew install espeak
pip install phonemizer

# lhotse
# https://github.com/lhotse-speech/lhotse/pull/956
pip install git+https://github.com/lhotse-speech/lhotse

# icefall
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH

# valle
git clone https://github.com/lifeiteng/valle.git
cd valle
pip install -e .
```

## Getting started

The quickest way to get started is to take a look at the detailed working code
examples found in the [examples] subdirectory.

[examples]: examples/
[paper]: https://arxiv.org/abs/2301.02111]


## Training
```
cd egs/libritts
./prepare.sh

# nano
python3 bin/trainer.py \
  --decoder-dim 128 --nhead 4 --num-decoder-layers 4 \
  --exp-dir exp/valle_nano

# same as paper
python3 bin/trainer.py \
  --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
  --exp-dir exp/valle
```


## Inference: In-Context Learning via Prompting

* TBD

## Contributing

* TBD

## Citing

To cite this repository:

```bibtex
@misc{valle,
  author={Feiteng Li},
  title={VALL-E: A neural codec language model},
  year={2023},
  url={http://github.com/lifeiteng/valle}
}
```

```bibtex
@article{VALL-E,
  title     = {Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers},
  author    = {Chengyi Wang, Sanyuan Chen, Yu Wu,
               Ziqiang Zhang, Long Zhou, Shujie Liu,
               Zhuo Chen, Yanqing Liu, Huaming Wang,
               Jinyu Li, Lei He, Sheng Zhao, Furu Wei},
  year      = {2023},
  eprint    = {2301.02111},
  archivePrefix = {arXiv},
  volume    = {abs/2301.02111},
  url       = {http://arxiv.org/abs/2301.02111},
}
```
