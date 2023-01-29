Language : 🇺🇸 | [🇨🇳](./README.zh-CN.md)

An unofficial PyTorch implementation of VALL-E([Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)).

![model](./docs/images/Overview.jpg)

## Demo

* [official demo](https://valle-demo.github.io/)
* reproduced results: comming soon

## Broader impacts

> Since VALL-E could synthesize speech that maintains speaker identity, it may carry potential risks in misuse of the model, such as spoofing voice identification or impersonating a specific speaker.

We will not provide well-trained models and services.

## Progress

<a href="https://www.buymeacoffee.com/feiteng" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Buy Me A Coffee" style="height: 40px !important;width: 145px !important;" ></a>

- [x] Text and Audio Tokenizer
- [x] Dataset module and loaders
- [x] VALL-F: `seq-to-seq + PrefixLanguageModel`
    - [x] AR Decoder
    - [x] NonAR Decoder
- [x] VALL-E: `PrefixLanguageModel`
    - [x] AR Decoder
    - [x] NonAR Decoder
- [ ] update README.zh-CN
- [x] Training
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
# https://github.com/lhotse-speech/lhotse/pull/960
pip uninstall lhotse
pip uninstall lhotse
pip install git+https://github.com/lhotse-speech/lhotse

# k2 icefall
# pip install k2
git clone https://github.com/k2-fsa/k2.git
cd k2
export K2_MAKE_ARGS="-j12"
export K2_CMAKE_ARGS="-DK2_WITH_CUDA=OFF"
python setup.py install
cd -

git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.zshrc
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.bashrc
cd -

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

# Those stages are very time-consuming
./prepare.sh

# nano: on NV GPU with 12G memory
# python3 bin/trainer.py \
#     --decoder-dim 128 --nhead 4 --num-decoder-layers 4 \
#     --max-duration 40 --model-name vallf \
#     --exp-dir exp/vallf_nano_full

python3 bin/trainer.py \
    --decoder-dim 128 --nhead 4 --num-decoder-layers 4 \
    --max-duration 40 --model-name valle \
    --exp-dir exp/valle_nano_full

# same as paper, but need more memory
python3 bin/trainer.py \
  --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 \
  --exp-dir exp/valle
```
#### Troubleshooting

* **SummaryWriter segmentation fault (core dumped)**
   * LINE `tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")`
   * FIX  [https://github.com/tensorflow/tensorboard/pull/6135/files](https://github.com/tensorflow/tensorboard/pull/6135/files)


## Inference: In-Context Learning via Prompting

* TBD

## Contributing

* Parallelize bin/tokenizer.py on multi-GPUs
* Reduce memory usage of **Training**
* Provide GPU resources (MyEmail: `lifeiteng0422@163.com`)
* <a href="https://www.buymeacoffee.com/feiteng" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Buy Me A Coffee" style="height: 40px !important;width: 145px !important;" ></a>


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
