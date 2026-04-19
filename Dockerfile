# Listening Brain — GPU worker (RunPod Serverless)
# Bakes model weights into the image so cold starts don't pay download cost.

FROM runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models/hf
ENV TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git curl ca-certificates build-essential \
  && rm -rf /var/lib/apt/lists/*

# Rust toolchain — deepfilternet's deepfilterlib compiles from source on Python 3.12
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

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
