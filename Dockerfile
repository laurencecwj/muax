FROM nvidia/cuda:12.4.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv && uv python install 3.10 && cd / && uv venv --python 3.10 myjax

ENV PATH "/myjax/bin:${PATH}"
RUN . /myjax/bin/activate && uv pip install --no-cache-dir jax[cuda12_pip] gymnasium[box2d]

WORKDIR /app
COPY . /app
RUN . /myjax/bin/activate && uv pip install .
