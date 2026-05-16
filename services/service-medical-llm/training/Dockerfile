# ─── Trainer Docker Image ─────────────────────────────────────────────────────
# Image: arabic-medical-llm/trainer:latest
# Used by: kubeflow/pipeline.py (KubeFlow Pipeline components)
#
# Build:
#   docker build -t arabic-medical-llm/trainer:latest \
#     -f services/service-training/Dockerfile .
#
# Push to registry (required for KubeFlow):
#   docker tag arabic-medical-llm/trainer:latest \
#     your-registry/arabic-medical-llm/trainer:latest
#   docker push your-registry/arabic-medical-llm/trainer:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ───────────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python deps ───────────────────────────────────────────────────────────────
# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install in correct order (torch already in base image)
RUN pip install --no-cache-dir \
        xformers==0.0.27.post2 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --no-deps

RUN pip install --no-cache-dir \
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

RUN pip install --no-cache-dir \
        trl>=0.8.6 \
        peft>=0.10.0 \
        accelerate>=0.27.0 \
        bitsandbytes>=0.42.0 \
        datasets>=2.18.0 \
        pandas>=2.0.0 \
        python-dotenv>=1.0.0 \
        huggingface_hub>=0.22.0 \
        papermill>=2.5.0 \
        ipykernel>=6.29.0 \
        mlflow>=2.11.0

# ── Copy experiments source ───────────────────────────────────────────────────
COPY experiments/ experiments/

# ── Python path ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace

# ── Default: bash (overridden by KubeFlow component commands) ──────────────
CMD ["/bin/bash"]
