# See https://hub.docker.com/r/pytorch/pytorch/tags for other versions that may suit you
# Should work even if you installed cuda11.4 on your host system.
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# Install packages
RUN pip install dvc omegaconf pytorch_lightning sklearn

# 1_build.sh will download the file for you.
COPY resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

# You'll be mounting the parent directory like this:
# docker run -v `pwd`/..:/workspace/ghoti-2021 -it ghoti /bin/bash
RUN mkdir -p /workspace/ghoti-2021
VOLUME /workspace/ghoti-2021
WORKDIR /workspace/ghoti-2021
