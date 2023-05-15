Language : [🇺🇸](./README.md) | [🇨🇳](./README.zh-CN.md)

非官方 VALL-E（[Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)）开源 PyTorch 实现。

<a href="https://www.buymeacoffee.com/feiteng" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Buy Me A Coffee" style="height: 40px !important;width: 145px !important;" ></a>

我们可以在一个GPU上训练VALL-E模型。

![model](./docs/images/Overview.jpg)

## 演示

* [官方演示](https://valle-demo.github.io/)
* [重现演示](https://lifeiteng.github.io/valle/index.html)

<a href="https://www.buymeacoffee.com/feiteng" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Buy Me A Coffee" style="height: 40px !important;width: 145px !important;" ></a>

<img src="./docs/images/vallf.png" width="500" height="400">


## 更为广泛的影响

> 由于VALL-E可以合成保持说话人身份的语音，它可能会带来滥用模型的潜在风险，如欺骗语音识别或冒充特定的说话人。

为避免滥用，将不提供训练有素的模型和服务。

## 安装部署

要快速启动和运行，只需遵循以下步骤：

```
# PyTorch
pip install torch==1.13.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchmetrics==0.11.1
# fbank
pip install librosa==0.8.1

# phonemizer pypinyin
apt-get install espeak-ng
## OSX: brew install espeak
pip install phonemizer==3.2.1 pypinyin==0.48.0

# lhotse update to newest version
# https://github.com/lhotse-speech/lhotse/pull/956
# https://github.com/lhotse-speech/lhotse/pull/960
pip uninstall lhotse
pip uninstall lhotse
pip install git+https://github.com/lhotse-speech/lhotse

# k2
# find the right version in https://huggingface.co/csukuangfj/k2
pip install https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230224+cuda11.6.torch1.13.1-cp310-cp310-linux_x86_64.whl

# icefall
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.zshrc
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.bashrc
cd -
source ~/.zshrc

# valle
git clone https://github.com/lifeiteng/valle.git
cd valle
pip install -e .
```


## 培训与推理
* #### 英文例子 [examples/libritts/README.md](egs/libritts/README.md)
* #### 中文例子 [examples/aishell1/README.md](egs/aishell1/README.md)
* ### NAR解码器的前缀模式0 1 2 4
  **文件第5.1章** "LibriLight中波形的平均长度为60秒。在训练中，我们随机裁剪波形为10秒和20秒之间的随机长度。对于NAR声学提示标记，我们从同一语篇中选择一个3秒的随机片段波形。"
  * **0**: 没有声音提示符
  * **1**: 当前分批的话语中的随机前缀 **(建议使用这种方法)**
  * **2**: 随机片段的当前批次话语
  * **4**: 与论文相同（由于他们将长波形随机裁剪为多个语料，所以同一语料意味着在同一长波形中的前或后语料）。
    ```
    # 如果用prefix_mode 4训练NAR解码器
    python3 bin/trainer.py --prefix_mode 4 --dataset libritts --input-strategy PromptedPrecomputedFeatures ...
    ```

#### [演示LibriTTS](https://lifeiteng.github.io/valle/index.html) 在4G内存的GPU上进行训练

```
cd examples/libritts

# 第1步 准备数据集
bash prepare.sh --stage -1 --stop-stage 3

# 第2步 在一个拥有24GB内存的GPU上训练模型
exp_dir=exp/valle

## 训练 AR 模型
python3 bin/trainer.py --max-duration 80 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir ${exp_dir}

## 训练 NAR 模型
cp ${exp_dir}/best-valid-loss.pt ${exp_dir}/epoch-2.pt  # --start-epoch 3=2+1
python3 bin/trainer.py --max-duration 40 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
      --num-buckets 6 --dtype "float32" --save-every-n 10000 --valid-interval 20000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.05 --warmup-steps 200 --average-period 0 \
      --num-epochs 40 --start-epoch 3 --start-batch 0 --accumulate-grad-steps 4 \
      --exp-dir ${exp_dir}

# 第3步 驗証
python3 bin/infer.py --output-dir infer/demos \
    --model-name valle --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below." \
    --checkpoint=${exp_dir}/best-valid-loss.pt

# 演示驗証
https://github.com/lifeiteng/lifeiteng.github.com/blob/main/valle/run.sh#L68
```
![train](./docs/images/train.png)

#### 故障排除

* **SummaryWriter segmentation fault (core dumped)**
   * LINE `tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")`
   * FIX  [https://github.com/tensorflow/tensorboard/pull/6135/files](https://github.com/tensorflow/tensorboard/pull/6135/files)
   ```
   file=`python  -c 'import site; print(f"{site.getsitepackages()[0]}/tensorboard/summary/writer/event_file_writer.py")'`
   sed -i 's/import tf/import tensorflow_stub as tf/g' $file
   ```

#### 在自定义数据集上进行训练？
* 准备数据集給 `lhotse manifests`
  * There are plenty of references here [lhotse/recipes](https://github.com/lhotse-speech/lhotse/tree/master/lhotse/recipes)
* `python3 bin/tokenizer.py ...`
* `python3 bin/trainer.py ...`

## 做出贡献

* 在多GPU上并行处理bin/tokenizer.py
* <a href="https://www.buymeacoffee.com/feiteng" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Buy Me A Coffee" style="height: 40px !important;width: 145px !important;" ></a>

## 引用

引用该资料库的内容：

```bibtex
@misc{valle,
  author={Feiteng Li},
  title={VALL-E: A neural codec language model},
  year={2023},
  url={http://github.com/lifeiteng/vall-e}
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

