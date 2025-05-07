# Image de base avec support CUDA
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as builder

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    CUDA_HOME="/usr/local/cuda" \
    PATH="/usr/local/cuda/bin:${PATH}"

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    wget \
    gcc \
    g++ \
    make \
    cmake \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    openmpi-bin \
    libopenmpi-dev \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Liaison symbolique de Python 3.11 en tant que Python par défaut
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Installation de TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Installation de Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:$PATH"

# Configuration du répertoire de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY pyproject.toml poetry.lock* ./

# Installation des dépendances avec Poetry
RUN if [ -f poetry.lock ]; then \
        poetry install --no-root --no-dev; \
    else \
        poetry install --no-root --no-dev; \
    fi

# Installation directe de packages liés à CUDA et DeepSpeed
RUN pip install deepspeed accelerate

# Vérifier l'installation de CUDA
RUN nvcc --version && python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Image finale optimisée pour la production
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as runtime

# Copie de TA-Lib depuis l'image builder
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Installation des dépendances minimales pour l'exécution
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev \
    openssh-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Liaison symbolique de Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Variables d'environnement pour la production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    CUDA_HOME="/usr/local/cuda" \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Copie de l'environnement virtuel et des fichiers du projet
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Création des répertoires nécessaires avec permissions
RUN mkdir -p ai_trading/info_retour/data \
    ai_trading/info_retour/logs \
    ai_trading/info_retour/models/checkpoints \
    ai_trading/info_retour/models/jit \
    test-reports

# Exposition du port
EXPOSE 8000

# Utilisateur non-root pour la sécurité en production (commentez cette ligne pour le debugging)
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Vérification de l'environnement CUDA
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# La commande par défaut lance l'application web
CMD ["python", "-m", "web_app.app"]

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Run a training session
# CMD ["python", "-m", "ai_trading.train", "--download", "--symbol", "BTC/USDT", "--timeframe", "1h", "--days", "60", "--backtest"] 