# Listening Brain — GPU worker (RunPod Serverless)
# Bakes model weights into the image so cold starts don't pay download cost.

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/hf
ENV TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# HF_TOKEN is required at build time to pull gated pyannote weights.
# Pass via: docker build --build-arg HF_TOKEN=hf_xxx .
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

COPY download_models.py ./
RUN python download_models.py

COPY handler.py pipeline.py ./

CMD ["python", "-u", "handler.py"]
