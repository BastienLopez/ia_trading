# Utilisation d'une image de base plus récente et plus légère
FROM python:3.11-slim-bookworm as builder

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gcc \
    g++ \
    make \
    python3-dev \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    openmpi-bin \
    libopenmpi-dev \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

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
COPY pyproject.toml poetry.lock ./

# Installation des dépendances avec Poetry
RUN poetry install --no-root --no-dev

# Installation directe de DeepSpeed (en dehors de Poetry pour une meilleure compatibilité)
RUN pip install deepspeed accelerate

# Image finale
FROM python:3.11-slim-bookworm as runtime

# Copie de TA-Lib depuis l'image builder
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Installation des bibliothèques système requises pour DeepSpeed
RUN apt-get update && apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Variables d'environnement pour la production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copie de l'environnement virtuel et des fichiers du projet
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Création des répertoires nécessaires
RUN mkdir -p data logs models/checkpoints

# Exposition du port
EXPOSE 8000

# Utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande par défaut
CMD ["python", "-m", "web_app.app"]

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Run a training session
# CMD ["python", "-m", "ai_trading.train", "--download", "--symbol", "BTC/USDT", "--timeframe", "1h", "--days", "60", "--backtest"] 