FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TOKENIZERS_PARALLELISM=false

# System dependencies — Ubuntu 22.04 ships Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /octo

# --- deps layer (cached unless requirements.txt / pins change) ---
COPY requirements.txt .

# Install JAX with CUDA 11 first so the later `jax==0.4.20` line in
# requirements.txt is already satisfied and won't pull a CPU-only jaxlib.
# Then pin loose >=  deps to versions compatible with JAX 0.4.20 / numpy 1.x / TF 2.15.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "jax[cuda11_pip]==0.4.20" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install --no-cache-dir \
        "numpy==1.24.3" \
        "scipy==1.11.4" \
        "transformers==4.34.1" \
        "gym==0.26.2" \
        "ml_collections==0.1.1" \
        "tqdm==4.66.1" \
        "absl-py==2.0.0" \
        "wandb==0.16.1" \
        "einops==0.7.0" \
        "imageio==2.33.0" \
        "moviepy==1.0.3" \
        "tensorflow_hub==0.15.0" \
        "tensorflow_text==2.15.0" \
        "plotly==5.18.0" \
        "matplotlib==3.8.2" \
        "opencv-python-headless==4.9.0.80" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mediapy

# --- project layer ---
COPY . .
RUN pip install --no-cache-dir -e .

CMD ["bash"]
