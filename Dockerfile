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
    ffmpeg cmake zlib1g-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install uv && uv python install 3.10 && cd / && uv venv --python 3.10 myjax

ENV PATH "/myjax/bin:${PATH}"

WORKDIR /app
COPY req.txt /app
RUN . /myjax/bin/activate && uv pip install -r req.txt
COPY . /app
RUN uv pip install -e .

ENV LD_LIBRARY_PATH /myjax/lib/python3.10/site-packages/courier/python:/myjax/lib/python3.10/site-packages/tensorflow:${LD_LIBRARY_PATH}
