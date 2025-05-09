# 1. Image de base NVIDIA CUDA (runtime)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# 2. Répertoire de travail
WORKDIR /app

# 3. Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
    PYTORCH_USE_SAFE_MODE=0 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

# 4. Copier uniquement les dépendances pour tirer parti du cache Docker
COPY requirements.txt .

# 5. Installation système + Python en une seule couche
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip \
      build-essential ca-certificates curl wget git \
      libjpeg-dev libpng-dev libopenblas-dev libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install --upgrade pip setuptools wheel \
 && pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install --no-cache-dir dash-bootstrap-components>=1.4.0 \
 && pip3 cache purge

# 6. Création d'un user non-root
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

# 7. Copier le code de l'application
COPY --chown=appuser:appuser . .

# 8. Déclarations volumes et ports
VOLUME ["/app/data", "/app/logs", "/app/models"]
EXPOSE 5000 8000 8050 8501

# 9. Vérification GPU (optionnelle, en ligne unique pour ne pas casser le parsing)
RUN python3 -c "import torch, tensorflow as tf; print('PyTorch CUDA OK:', torch.cuda.is_available()); print('TF GPUs:', tf.config.list_physical_devices('GPU'))"

# 10. Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8050/health || exit 1

# 11. Commande par défaut (remplacée par docker-compose.yml pour le service web_app)
CMD ["python3", "-m", "ai_trading.dashboard.app"]
