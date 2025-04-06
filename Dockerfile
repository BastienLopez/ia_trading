FROM python:3.10-slim

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Créer les répertoires nécessaires
RUN mkdir -p logs

# Copier le code source
COPY ai_trading/ ai_trading/
COPY tradingview/ tradingview/
COPY .env .

# Exposer le port pour l'API
EXPOSE 8000

# Commande par défaut : démarrer l'API
CMD ["python", "-m", "ai_trading.api"]

# Autres commandes possibles en fonction de l'utilisation :
# Pour entraîner un modèle : 
# CMD ["python", "ai_trading/train.py", "--download", "--symbol", "BTC/USDT", "--timeframe", "1h", "--days", "60", "--backtest"] 