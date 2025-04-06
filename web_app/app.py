"""
Application web pour visualiser les résultats du trading IA
"""
import os
import json
import logging
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime, timedelta
import random
import math
from web_app.config import Config
from web_app.routes.transaction_routes import transaction_bp

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Créer le répertoire de logs s'il n'existe pas
if not os.path.exists("logs"):
    os.makedirs("logs")
    logger.info("Répertoire de logs créé")

app = Flask(__name__)
app.config.from_object(Config)

# Configuration
API_URL = os.environ.get('API_URL', 'http://localhost:8000')
logger.info(f"Configuration API_URL: {API_URL}")

# Chemin vers le fichier de données
DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'transactions.json')

SUPPORTED_CRYPTOS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", 
    "MATIC", "LINK", "UNI", "AAVE", "ATOM", "DOGE", "SHIB",
    "USDT", "USDC", "DAI"
]

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            # S'assurer que la structure des données est correcte
            if "transactions" not in data:
                data["transactions"] = []
            if "portfolio" not in data:
                data["portfolio"] = {"current_value": 0, "starting_value": 0, "assets": []}
            return data
    return {"transactions": [], "portfolio": {"current_value": 0, "starting_value": 0, "assets": []}}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_real_crypto_prices():
    """Récupère les prix réels des cryptomonnaies depuis CoinGecko"""
    try:
        logger.info("Récupération des prix depuis CoinGecko...")
        
        # Mapping des symboles vers les IDs CoinGecko
        symbol_to_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "BNB": "binancecoin",
            "SOL": "solana",
            "XRP": "ripple",
            "ADA": "cardano",
            "AVAX": "avalanche-2",
            "DOT": "polkadot",
            "MATIC": "matic-network",
            "LINK": "chainlink",
            "UNI": "uniswap",
            "AAVE": "aave",
            "ATOM": "cosmos",
            "DOGE": "dogecoin",
            "SHIB": "shiba-inu",
            "USDT": "tether",
            "USDC": "usd-coin",
            "DAI": "dai"
        }
        
        # Construire la liste des IDs pour la requête
        crypto_ids = ",".join(symbol_to_id.values())
        
        # Faire la requête à CoinGecko
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_ids}&vs_currencies=usd"
        logger.info(f"URL CoinGecko: {url}")
        
        response = requests.get(url)
        logger.info(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            coingecko_data = response.json()
            logger.info(f"Données brutes de CoinGecko: {json.dumps(coingecko_data, indent=2)}")
            
            prices = {}
            for symbol, coin_id in symbol_to_id.items():
                if coin_id in coingecko_data:
                    current_price = coingecko_data[coin_id]["usd"]
                    prices[symbol] = current_price
                    logger.info(f"Prix {symbol}: ${current_price:,.2f}")
            
            return prices
        else:
            logger.error(f"Erreur CoinGecko API: {response.status_code}")
            logger.error(f"Réponse: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des prix: {str(e)}")
        return None

def update_portfolio():
    try:
        data = load_data()
        logger.info("Mise à jour du portefeuille...")

        # Initialiser le portfolio
        portfolio = {
            "current_value": 0,
            "starting_value": data["portfolio"].get("starting_value", 0),
            "assets": []
        }

        # Si pas de transactions, retourner un portfolio vide
        if not data["transactions"]:
            logger.info("Aucune transaction trouvée, retour d'un portfolio vide")
            save_data({"transactions": [], "portfolio": portfolio})
            return portfolio

        # Récupérer les prix actuels depuis CoinGecko
        current_prices = get_real_crypto_prices()
        if not current_prices:
            logger.error("Impossible de récupérer les prix actuels")
            # Utiliser les derniers prix connus si disponibles
            if data["portfolio"]["assets"]:
                logger.info("Utilisation des derniers prix connus")
                current_prices = {asset["symbol"]: asset["current_price"] for asset in data["portfolio"]["assets"]}
            else:
                return portfolio

        logger.info("Prix actuels des cryptos:")
        for symbol, price in current_prices.items():
            logger.info(f"{symbol}: ${price:,.2f}")

        # Calculer les quantités totales par actif
        assets = {}
        for transaction in data["transactions"]:
            symbol = transaction["symbol"]
            amount = float(transaction["amount"])
            if transaction["action"] == "ACHETER":
                assets[symbol] = assets.get(symbol, 0) + amount
            else:  # VENDRE
                assets[symbol] = assets.get(symbol, 0) - amount

        # Calculer la valeur actuelle de chaque actif et la valeur totale
        total_value = 0
        assets_list = []
        
        for symbol, amount in assets.items():
            if amount > 0 and symbol in current_prices:  # Ne considérer que les actifs avec une quantité positive
                current_price = float(current_prices[symbol])
                value = amount * current_price
                total_value += value
                
                asset_info = {
                    "symbol": symbol,
                    "amount": amount,
                    "value_usd": round(value, 2),
                    "current_price": current_price
                }
                assets_list.append(asset_info)
                logger.info(f"Calcul pour {symbol}: {amount} * ${current_price:,.2f} = ${value:,.2f}")

        # Mettre à jour le portfolio
        portfolio["current_value"] = round(total_value, 2)
        portfolio["assets"] = sorted(assets_list, key=lambda x: x["value_usd"], reverse=True)

        logger.info(f"Valeur totale du portfolio: ${total_value:,.2f}")
        
        # Sauvegarder les données mises à jour
        data["portfolio"] = portfolio
        save_data(data)
        
        return portfolio

    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du portfolio: {str(e)}")
        return {
            "current_value": 0,
            "starting_value": 0,
            "assets": []
        }

# Fonction d'aide pour simuler les réponses de l'API
def simulate_api_response(endpoint, params=None):
    """Simule une réponse d'API pour le front-end quand l'API n'est pas disponible"""
    try:
        # Essayer d'abord d'appeler l'API réelle
        if endpoint == '/predict':
            url = f"{API_URL}/predict"
            response = requests.post(url, json=params, timeout=5)
            return response.json()
        elif endpoint == '/backtest':
            url = f"{API_URL}/backtest"
            response = requests.post(url, json=params, timeout=5)
            return response.json()
        elif endpoint == '/train':
            url = f"{API_URL}/train"
            response = requests.post(url, json=params, timeout=5)
            return response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Impossible de se connecter à l'API: {e}. Utilisation des données simulées.")
        
    # Si l'API n'est pas disponible, générer des données simulées
    logger.info(f"Génération de données simulées pour {endpoint}")
    
    # Données simulées pour différents endpoints
    if endpoint == '/predict':
        symbol = params.get('symbol', 'BTC/USDT')
        actions = ['ACHETER', 'VENDRE', 'ATTENDRE']
        confidence = random.uniform(0.6, 0.95)
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'recommendation': random.choice(actions),
            'confidence': round(confidence, 2),
            'latest_price': round(random.uniform(30000, 40000), 2),
            'indicators': {
                'rsi': round(random.uniform(20, 80), 2),
                'macd': round(random.uniform(-10, 10), 2),
                'ema_cross': random.choice([True, False])
            }
        }
    elif endpoint == '/backtest':
        symbol = params.get('symbol', 'BTC/USDT')
        days = params.get('days', 30)
        profit = random.uniform(2.0, 15.0)
        bh_profit = random.uniform(-5.0, 10.0)
        trades = random.randint(10, 50)
        return {
            'symbol': symbol,
            'period': f"{days} jours",
            'profit_pct': round(profit, 2),
            'buy_hold_pct': round(bh_profit, 2),
            'sharpe_ratio': round(random.uniform(0.5, 2.0), 2),
            'num_trades': trades,
            'winning_trades_pct': round(random.uniform(50, 80), 2),
            'max_drawdown_pct': round(random.uniform(5, 20), 2),
            'trades': [round(random.uniform(-3, 5), 2) for _ in range(5)]
        }
    elif endpoint == '/train':
        return {
            'status': 'success',
            'model': 'PPO',
            'training_time': f"{random.randint(10, 120)} minutes",
            'timesteps': random.randint(10000, 100000),
            'final_reward': round(random.uniform(100, 500), 2),
            'backtest_results': {
                'profit_pct': round(random.uniform(5, 20), 2),
                'buy_hold_pct': round(random.uniform(-5, 10), 2),
                'sharpe_ratio': round(random.uniform(0.5, 2.5), 2)
            }
        }
    return {'error': 'Endpoint non pris en charge pour la simulation'}

# Contexte global pour les templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

@app.route('/')
def index():
    """Page d'accueil"""
    logger.info("Accès à la page d'accueil")
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Tableau de bord principal"""
    logger.info("Accès au tableau de bord")
    return render_template('dashboard.html')

@app.route('/backtest')
def backtest():
    """Page de backtest"""
    logger.info("Accès à la page de backtest")
    return render_template('backtest.html')

@app.route('/tradingview')
def tradingview():
    """Page TradingView"""
    logger.info("Accès à la page TradingView")
    return render_template('tradingview.html')

@app.route('/get-pine-script')
def get_pine_script():
    try:
        # Chemin relatif au dossier du projet
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tradingview', 'btc_signals.pine')
        
        with open(script_path, 'r') as file:
            content = file.read()
            
        return content
    except Exception as e:
        return f"Erreur lors de la lecture du fichier: {str(e)}", 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Endpoint pour obtenir une prédiction (simulée)"""
    logger.info(f"Demande de prédiction reçue: {request.json}")
    
    try:
        params = request.json
        result = simulate_api_response('/predict', params)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Proxy pour l'API de backtest"""
    data = request.json
    logger.info(f"Requête de backtest reçue avec données: {data}")
    try:
        endpoint = f"{API_URL}/backtest"
        logger.debug(f"Envoi de la requête à: {endpoint}")
        response = requests.post(endpoint, json=data)
        logger.debug(f"Réponse API: {response.status_code}")
        if response.status_code != 200:
            logger.warning(f"API a retourné une erreur: {response.text}")
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Erreur lors du backtest: {str(e)}", exc_info=True)
        # Utiliser des données simulées en cas d'erreur pour démonstration
        logger.info("Retour de données de backtest simulées pour la démonstration")
        mock_data = {
            "error": True,
            "message": str(e),
            "mock_data": True,
            "symbol": data.get("symbol", "BTC/USDT"),
            "timeframe": data.get("timeframe", "1h"),
            "profit_pct": 8.5,
            "buy_hold_pct": 5.2,
            "sharpe_ratio": 1.2,
            "num_trades": 24,
            "max_drawdown_pct": 12.5,
            "results": []
        }
        return jsonify(mock_data), 500

@app.route('/api/market_data')
def get_market_data():
    """Récupérer les données du marché"""
    try:
        timeframe = request.args.get('timeframe', '1h')
        logger.info(f"Récupération des données de marché pour le timeframe {timeframe}")

        # Récupérer le prix actuel de Bitcoin depuis CoinGecko
        current_prices = get_real_crypto_prices()
        if not current_prices or 'BTC' not in current_prices:
            logger.error("Impossible de récupérer le prix actuel du Bitcoin")
            return jsonify({"error": "Impossible de récupérer les données de marché"}), 500

        current_price = current_prices['BTC']
        logger.info(f"Prix actuel du Bitcoin: ${current_price:,.2f}")

        # Générer l'historique des prix sur 24h
        now = datetime.now()
        prices = []
        
        # Définir le nombre de points et l'intervalle selon le timeframe
        if timeframe == '1h':
            num_points = 24
            interval = timedelta(hours=1)
        elif timeframe == '4h':
            num_points = 24
            interval = timedelta(hours=4)
        else:  # '1d'
            num_points = 24
            interval = timedelta(days=1)

        # Générer les prix historiques avec une variation réaliste
        base_price = current_price
        for i in range(num_points):
            point_time = now - (interval * (num_points - 1 - i))
            # Ajouter une variation aléatoire de ±2% maximum
            variation = random.uniform(-0.02, 0.02)
            price = base_price * (1 + variation)
            prices.append({
                "date": point_time.isoformat(),
                "price": price
            })
            # Mettre à jour le prix de base pour la prochaine itération
            base_price = price

        # S'assurer que le dernier prix correspond au prix actuel
        prices[-1]["price"] = current_price

        # Calculer les indicateurs techniques
        price_values = [p["price"] for p in prices]
        rsi = calculate_rsi(price_values)
        macd = calculate_macd(price_values)

        # Générer une recommandation basée sur les indicateurs
        recommendation = generate_recommendation(rsi, macd)

        response_data = {
            "prices": prices,
            "indicators": {
                "rsi": rsi,
                "macd": macd,
                "ema9": current_price,  # Simplifié pour l'exemple
                "ema21": current_price * 0.995  # Simplifié pour l'exemple
            },
            "last_recommendation": recommendation
        }

        logger.info("Données de marché générées avec succès")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données de marché: {str(e)}")
        return jsonify({"error": "Erreur lors de la récupération des données de marché"}), 500

def calculate_rsi(prices, periods=14):
    """Calculer le RSI"""
    if len(prices) < periods:
        return 50  # Valeur par défaut

    # Calculer les variations
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    # Séparer les gains et les pertes
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    # Calculer les moyennes
    avg_gain = sum(gains[-periods:]) / periods
    avg_loss = sum(losses[-periods:]) / periods
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26):
    """Calculer le MACD"""
    if len(prices) < slow:
        return 0  # Valeur par défaut
    
    # Calculer les EMA
    ema_fast = sum(prices[-fast:]) / fast
    ema_slow = sum(prices[-slow:]) / slow
    
    return ema_fast - ema_slow

def generate_recommendation(rsi, macd):
    """Générer une recommandation basée sur les indicateurs"""
    confidence = random.uniform(0.6, 0.9)
    
    if rsi < 30 and macd > 0:
        action = "ACHETER"
    elif rsi > 70 and macd < 0:
        action = "VENDRE"
    else:
        action = "NEUTRE"
    
    return {
        "action": action,
        "confidence": confidence
    }

@app.route('/api/portfolio', methods=['GET'])
def api_portfolio():
    """Endpoint pour récupérer les données du portfolio"""
    try:
        logger.info("Récupération des données du portfolio...")
        portfolio = update_portfolio()
        data = load_data()
        response = {
            "current_value": portfolio["current_value"],
            "starting_value": portfolio["starting_value"],
            "assets": portfolio["assets"],
            "transactions": data["transactions"]
        }
        logger.info(f"Données du portfolio renvoyées: {json.dumps(response, indent=2)}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du portfolio: {str(e)}")
        return jsonify({
            "error": "Erreur lors de la récupération du portfolio",
            "current_value": 0,
            "starting_value": 0,
            "assets": [],
            "transactions": []
        }), 500

@app.route('/api/performance', methods=['GET'])
def api_performance():
    """Récupérer les métriques de performance"""
    logger.info("Demande de métriques de performance")
    try:
        # Pour la démonstration, on utilise des données fictives mais constantes
        logger.debug("Utilisation de données fictives de performance constantes")
        
        # Générer les 5 derniers mois avec des données constantes
        current_date = datetime.now()
        months_data = []
        
        # Dictionnaire pour traduire les noms des mois en français
        month_names_fr = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 
            6: "Juin", 7: "Juillet", 8: "Août", 9: "Septembre", 
            10: "Octobre", 11: "Novembre", 12: "Décembre"
        }
        
        # Données de performance mensuelles fixes
        monthly_performance = [
            {"profit": 4.8, "buy_hold": 4.2},  # Mois actuel - 4
            {"profit": 7.2, "buy_hold": 6.3},  # Mois actuel - 3
            {"profit": 2.1, "buy_hold": 1.6},  # Mois actuel - 2
            {"profit": 4.8, "buy_hold": 3.7},  # Mois actuel - 1
            {"profit": -0.3, "buy_hold": -0.7} # Mois actuel
        ]
        
        for i in range(4, -1, -1):
            # Calculer la date pour chaque mois (en remontant dans le temps)
            month_date = current_date - timedelta(days=30 * i)
            month_name = f"{month_names_fr[month_date.month]} {month_date.year}"
            
            # Utiliser les données fixes
            months_data.append({
                "month": month_name,
                "profit_pct": monthly_performance[4-i]["profit"],
                "buy_hold_pct": monthly_performance[4-i]["buy_hold"]
            })
        
        mock_data = {
            "overall": {
                "profit_pct": 25.01,
                "buy_hold_pct": 18.50,
                "sharpe_ratio": 1.45,
                "max_drawdown_pct": 12.3,
                "win_rate": 0.68,
                "total_trades": 42
            },
            "monthly": months_data,
            "assets": [
                {"symbol": "BTC/USDT", "profit_pct": 28.5, "buy_hold_pct": 20.1, "trades": 18},
                {"symbol": "ETH/USDT", "profit_pct": 22.8, "buy_hold_pct": 17.3, "trades": 15},
                {"symbol": "SOL/USDT", "profit_pct": 15.2, "buy_hold_pct": 12.7, "trades": 9}
            ]
        }
        logger.debug(f"Données de performance générées pour {len(mock_data['assets'])} actifs")
        logger.debug(f"Données mensuelles générées pour {len(mock_data['monthly'])} mois")
        return jsonify(mock_data)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des métriques de performance: {str(e)}", exc_info=True)
        return jsonify({"error": True, "message": str(e)}), 500

@app.route('/api/transactions', methods=['GET', 'POST', 'PUT', 'DELETE'])
def handle_transactions():
    try:
        data = load_data()
        
        if request.method == 'GET':
            logger.info(f"Récupération des transactions: {json.dumps(data['transactions'], indent=2)}")
            return jsonify(data["transactions"])
            
        elif request.method == 'POST':
            try:
                transaction = request.json
                logger.info(f"Nouvelle transaction reçue: {json.dumps(transaction, indent=2)}")
                
                # Valider les données
                required_fields = ['symbol', 'action', 'amount']
                missing_fields = [field for field in required_fields if field not in transaction]
                if missing_fields:
                    error_msg = f"Champs manquants: {', '.join(missing_fields)}"
                    logger.error(f"Données manquantes dans la transaction: {error_msg}")
                    return jsonify({"error": error_msg}), 400
                
                # Valider le format des données
                try:
                    amount = float(transaction['amount'])
                    if amount <= 0:
                        raise ValueError("Le montant doit être positif")
                except ValueError as e:
                    logger.error(f"Montant invalide: {transaction['amount']}")
                    return jsonify({"error": f"Montant invalide: {str(e)}"}), 400
                
                # Récupérer le prix actuel depuis CoinGecko
                current_prices = get_real_crypto_prices()
                
                # Si CoinGecko n'est pas disponible, utiliser le dernier prix connu ou une valeur par défaut
                if not current_prices:
                    logger.warning("Impossible de récupérer les prix depuis CoinGecko, utilisation du dernier prix connu")
                    # Chercher le dernier prix connu pour ce symbole
                    last_price = None
                    for t in reversed(data['transactions']):
                        if t['symbol'] == transaction['symbol']:
                            last_price = t['price']
                            break
                    
                    if last_price is None:
                        # Si aucun prix n'est trouvé, utiliser une valeur par défaut
                        default_prices = {
                            'BTC': 50000,
                            'ETH': 3000,
                            'BNB': 400,
                            'SOL': 100,
                            'XRP': 0.5,
                            'ADA': 0.5,
                            'AVAX': 30,
                            'DOT': 7,
                            'MATIC': 1,
                            'LINK': 15,
                            'UNI': 7,
                            'AAVE': 100,
                            'ATOM': 10,
                            'DOGE': 0.1,
                            'SHIB': 0.00001,
                            'USDT': 1,
                            'USDC': 1,
                            'DAI': 1
                        }
                        last_price = default_prices.get(transaction['symbol'], 1)
                        logger.warning(f"Utilisation du prix par défaut pour {transaction['symbol']}: {last_price}")
                    
                    current_price = last_price
                else:
                    if transaction['symbol'] not in current_prices:
                        logger.error(f"Symbole non supporté: {transaction['symbol']}")
                        return jsonify({"error": f"Symbole non supporté: {transaction['symbol']}"}), 400
                    current_price = float(current_prices[transaction['symbol']])
                
                # Ajouter la transaction avec le prix actuel
                transaction["timestamp"] = datetime.now().timestamp()
                transaction["price"] = current_price
                transaction["value"] = amount * current_price
                transaction["date"] = datetime.now().isoformat()
                
                logger.info(f"Prix utilisé pour {transaction['symbol']}: ${current_price}")
                logger.info(f"Valeur de la transaction: {amount} * ${current_price} = ${transaction['value']}")
                
                data["transactions"].append(transaction)
                save_data(data)
                update_portfolio()
                
                logger.info(f"Transaction ajoutée avec succès. Nouvelles données: {json.dumps(data, indent=2)}")
                return jsonify({"success": True})
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la transaction: {str(e)}")
                return jsonify({"error": f"Erreur lors du traitement de la transaction: {str(e)}"}), 500
            
        elif request.method == 'PUT':
            updated_transaction = request.json
            logger.info(f"Transaction à mettre à jour: {json.dumps(updated_transaction, indent=2)}")
            
            # Valider les données
            required_fields = ['symbol', 'action', 'amount', 'timestamp']
            if not all(field in updated_transaction for field in required_fields):
                logger.error(f"Données manquantes dans la transaction à mettre à jour: {updated_transaction}")
                return jsonify({"error": "Données manquantes"}), 400
            
            # Récupérer le prix actuel depuis CoinGecko
            current_prices = get_real_crypto_prices()
            if not current_prices or updated_transaction['symbol'] not in current_prices:
                logger.error(f"Impossible de récupérer le prix pour {updated_transaction['symbol']}")
                return jsonify({"error": f"Impossible de récupérer le prix pour {updated_transaction['symbol']}"}), 400
            
            # Mettre à jour la transaction avec le prix actuel
            current_price = float(current_prices[updated_transaction['symbol']])
            amount = float(updated_transaction['amount'])
            updated_transaction["price"] = current_price
            updated_transaction["value"] = amount * current_price
            
            # Mettre à jour la transaction
            for i, transaction in enumerate(data["transactions"]):
                if transaction["timestamp"] == updated_transaction["timestamp"]:
                    data["transactions"][i] = updated_transaction
                    break
            
            save_data(data)
            update_portfolio()
            
            logger.info(f"Transaction mise à jour avec succès. Nouvelles données: {json.dumps(data, indent=2)}")
            return jsonify({"success": True})
            
        elif request.method == 'DELETE':
            try:
                delete_data = request.json
                if not delete_data or 'timestamp' not in delete_data:
                    logger.error("Données de suppression invalides")
                    return jsonify({"error": "Timestamp manquant"}), 400
                
                timestamp = float(delete_data['timestamp'])
                logger.info(f"Tentative de suppression de la transaction avec timestamp: {timestamp}")
                
                # Trouver et supprimer la transaction
                initial_length = len(data['transactions'])
                data['transactions'] = [t for t in data['transactions'] if t['timestamp'] != timestamp]
                
                if len(data['transactions']) == initial_length:
                    logger.error(f"Transaction non trouvée avec timestamp: {timestamp}")
                    return jsonify({"error": "Transaction non trouvée"}), 404
                
                # Sauvegarder les données mises à jour
                save_data(data)
                logger.info("Transaction supprimée avec succès")
                return jsonify({"success": True})
                
            except ValueError as e:
                logger.error(f"Erreur de format: {str(e)}")
                return jsonify({"error": "Format de timestamp invalide"}), 400
            except Exception as e:
                logger.error(f"Erreur lors de la suppression: {str(e)}")
                return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        logger.error(f"Erreur lors de la gestion des transactions: {str(e)}")
        return jsonify({"error": "Erreur lors de la gestion des transactions"}), 500

@app.route('/api/transactions', methods=['DELETE'])
def delete_transaction():
    """Supprimer une transaction"""
    try:
        data = request.get_json()
        if not data or 'timestamp' not in data:
            return jsonify({"error": "Timestamp manquant"}), 400

        timestamp = data['timestamp']
        logger.info(f"Suppression de la transaction avec timestamp: {timestamp}")

        # Charger les données actuelles
        current_data = load_data()
        
        # Trouver et supprimer la transaction
        transactions = current_data["transactions"]
        initial_length = len(transactions)
        current_data["transactions"] = [t for t in transactions if str(t["timestamp"]) != str(timestamp)]
        
        if len(current_data["transactions"]) == initial_length:
            return jsonify({"error": "Transaction non trouvée"}), 404

        # Sauvegarder les modifications dans transactions.json
        save_data(current_data)
        
        # Mettre à jour le portfolio
        updated_portfolio = update_portfolio()
        
        # Sauvegarder aussi dans config.json pour la synchronisation
        config_path = os.path.join(os.path.dirname(__file__), 'data', 'config.json')
        config_data = {
            "transactions": current_data["transactions"],
            "portfolio": updated_portfolio
        }
        
        # Créer le répertoire data s'il n'existe pas
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        logger.info("Transaction supprimée et configuration mise à jour avec succès")
        return jsonify({"success": True, "message": "Transaction supprimée avec succès"})

    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la transaction: {str(e)}")
        return jsonify({"error": "Erreur lors de la suppression de la transaction"}), 500

@app.errorhandler(404)
def page_not_found(e):
    """Gestionnaire pour les erreurs 404"""
    logger.warning(f"Page non trouvée: {request.path}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Gestionnaire pour les erreurs 500"""
    logger.error(f"Erreur serveur: {str(e)}")
    return render_template('500.html'), 500

if __name__ == "__main__":
    logger.info("Démarrage du serveur web Flask")
    app.run(host='0.0.0.0', port=5000, debug=True) 