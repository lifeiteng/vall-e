FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y vim wget git libsndfile1 espeak-ng

RUN pip install torchmetrics==0.11.1 librosa==0.8.1 phonemizer==3.2.1 pypinyin==0.48.0 lhotse matplotlib h5py

RUN pip install https://huggingface.co/csukuangfj/k2/resolve/main/cuda/k2-1.23.4.dev20230224+cuda11.6.torch1.13.1-cp310-cp310-linux_x86_64.whl

RUN pip install torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /workspace
RUN git clone https://github.com/k2-fsa/icefall && \
    cd icefall && \
    pip install -r requirements.txt
ENV PYTHONPATH=/workspace/icefall:$PYTHONPATH

RUN cp /opt/conda/lib/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/

WORKDIR /workspace
RUN git clone https://github.com/lifeiteng/vall-e.git && \
    cd vall-e && \
    pip install -e .

WORKDIR /workspace/vall-e
