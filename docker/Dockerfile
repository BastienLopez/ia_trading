# L'image de développement CUDA apporte NVVM/libdevice, requis par TensorFlow
# quand XLA compile réellement sur GPU. L'image runtime seule ne le contient pas.
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_PACKAGE=onnxruntime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_HOME=/usr/local/cuda \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    PYTHONPATH=/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      build-essential \
      ca-certificates \
      curl \
      wget \
      git \
      libjpeg-dev \
      libpng-dev \
      libopenblas-dev \
      libgomp1 \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# On copie les dépendances avant le code pour profiter du cache Docker.
COPY requirements.txt /tmp/requirements.raw.txt

# Corrections de compatibilité sans modifier le requirements.txt du projet :
# - gym-trading==0.0.1 n'existe pas sur PyPI.
# - gym-trading-env 0.3.5 force Flask 2.2.5 et entre en conflit avec Flask>=2.3.3.
# - 0.3.2 accepte la version Flask utilisée par le projet.
# - onnxruntime et onnxruntime-gpu ne doivent pas être installés ensemble.
# - la pile ONNX est verrouillée pour éviter le conflit précédemment rencontré.
RUN set -eux; \
    grep -Ev '^[[:space:]]*(gym-trading|gym-trading-env|onnx|onnxruntime|onnxruntime-gpu|tf2onnx|dash-bootstrap-components|deepspeed)([<>=!~ ].*)?([[:space:]]*#.*)?$' \
      /tmp/requirements.raw.txt > /tmp/requirements.base.txt; \
    cat /tmp/requirements.base.txt > /tmp/requirements.txt; \
    printf '%s\n' \
      'gym-trading-env==0.3.2' \
      'onnx==1.17.0' \
      'tf2onnx==1.17.0' \
      "${ONNXRUNTIME_PACKAGE}==1.23.2" \
      'dash-bootstrap-components>=1.4.0' \
      >> /tmp/requirements.txt; \
    python3 -m pip install --upgrade pip setuptools wheel; \
    python3 -m pip install --prefer-binary -r /tmp/requirements.txt; \
    python3 -m pip check; \
    rm -f /tmp/requirements.raw.txt /tmp/requirements.base.txt /tmp/requirements.txt

# Préparer les dossiers persistants avant de passer en utilisateur non-root.
RUN useradd --create-home --shell /bin/bash appuser \
 && mkdir -p \
      /app/ai_trading/info_retour/data \
      /app/ai_trading/info_retour/logs \
      /app/ai_trading/info_retour/models \
      /app/ai_trading/info_retour/cache \
 && chown -R appuser:appuser /app

COPY --chown=appuser:appuser . /app

RUN ln -sfn /app/ai_trading/info_retour/data /app/data \
 && ln -sfn /app/ai_trading/info_retour/logs /app/logs \
 && ln -sfn /app/ai_trading/info_retour/models /app/models

USER appuser

EXPOSE 5000 5555 8000 8050 8501 8786 8787

# Pas de HEALTHCHECK global : la même image exécute plusieurs services différents.
CMD ["python3", "-m", "ai_trading.dashboard.app"]
