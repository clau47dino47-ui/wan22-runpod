# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

# PyTorch 2.1 + CUDA 12.1
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Diffusers stack
RUN pip install \
    diffusers>=0.30.0 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    sentencepiece \
    imageio[ffmpeg] \
    opencv-python-headless \
    boto3 \
    runpod

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    # HuggingFace cache → network volume (persiste tra le run)
    HF_HOME=/runpod-volume/cache/huggingface \
    HF_HUB_CACHE=/runpod-volume/cache/huggingface/hub \
    # VRAM: limita la frammentazione dell'allocatore CUDA
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" \
    # Timeout esteso per download iniziale dei pesi 14B
    RUNPOD_INIT_TIMEOUT=1800

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copia pacchetti installati nello stage builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY handler.py .

CMD ["python3.10", "-u", "handler.py"]
