FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    wget curl vim git build-essential \
    unzip python3-pip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf swig \
    ffmpeg cmake \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install uv && uv python install 3.10 && cd / && uv venv --python 3.10 myjax

ENV PATH "/myjax/bin:${PATH}"
RUN . /myjax/bin/activate && uv pip install --no-cache-dir jax[cuda12_pip] gymnasium[box2d]

WORKDIR /app
COPY . /app
RUN . /myjax/bin/activate && uv pip install .
