# Base image
FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Variables d'environnement
ENV PYTHONPATH=/app
ENV CRYPTO_PANIC_API_KEY=votre_cle_api

# Create necessary directories
RUN mkdir -p data logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "ai_trading.examples.rl_training_example"]

# Autres commandes possibles en fonction de l'utilisation :
# Pour entraîner un modèle : 
# CMD ["python", "ai_trading/train.py", "--download", "--symbol", "BTC/USDT", "--timeframe", "1h", "--days", "60", "--backtest"] 