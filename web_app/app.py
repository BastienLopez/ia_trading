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
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_very_secret')
logger.info("Application Flask initialisée")

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
    
    # Simuler les données pour le tableau de bord
    try:
        # Données du portefeuille
        portfolio_data = {
            'total_value': round(random.uniform(5000, 15000), 2),
            'daily_change_pct': round(random.uniform(-5, 5), 2),
            'assets': [
                {'symbol': 'BTC', 'amount': round(random.uniform(0.1, 0.5), 4), 'value_usd': round(random.uniform(3000, 8000), 2)},
                {'symbol': 'ETH', 'amount': round(random.uniform(1, 5), 4), 'value_usd': round(random.uniform(2000, 5000), 2)},
                {'symbol': 'USDT', 'amount': round(random.uniform(1000, 3000), 2), 'value_usd': round(random.uniform(1000, 3000), 2)}
            ]
        }
        
        # Données de performance
        performance_data = {
            'monthly_profit': round(random.uniform(-10, 20), 2),
            'yearly_profit': round(random.uniform(-5, 30), 2),
            'total_profit': round(random.uniform(0, 50), 2),
            'win_rate': round(random.uniform(50, 80), 2),
            'avg_win': round(random.uniform(2, 10), 2),
            'avg_loss': round(random.uniform(-5, -1), 2)
        }
        
        # Générer des données historiques pour le graphique
        days = 30
        price_history = []
        start_price = random.uniform(25000, 35000)
        
        for i in range(days):
            change = random.uniform(-0.05, 0.05)
            start_price = start_price * (1 + change)
            price_history.append({
                'date': (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d'),
                'price': round(start_price, 2)
            })
        
        # Dernières prédictions
        predictions = []
        for i in range(5):
            actions = ['ACHETER', 'VENDRE', 'ATTENDRE']
            weights = [0.4, 0.3, 0.3]  # Pondération pour rendre certaines actions plus probables
            action = random.choices(actions, weights=weights, k=1)[0]
            confidence = random.uniform(0.6, 0.95)
            
            # Assigner une classe CSS en fonction de l'action
            action_class = 'success' if action == 'ACHETER' else 'danger' if action == 'VENDRE' else 'warning'
            
            predictions.append({
                'symbol': random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT']),
                'timestamp': (datetime.now() - timedelta(hours=i*4)).strftime('%Y-%m-%d %H:%M'),
                'action': action,
                'action_class': action_class,
                'confidence': round(confidence, 2)
            })
        
        logger.debug("Données du tableau de bord générées avec succès")
        return render_template(
            'dashboard.html',
            portfolio=portfolio_data,
            performance=performance_data,
            price_history=price_history,
            predictions=predictions
        )
    except Exception as e:
        logger.error(f"Erreur lors de la génération des données du tableau de bord: {e}", exc_info=True)
        return render_template('dashboard.html', error=str(e))

@app.route('/backtest', methods=['GET', 'POST'])
def backtest():
    """Page de résultats de backtest"""
    logger.info("Accès à la page de backtest")
    
    if request.method == 'POST':
        logger.info(f"Soumission du formulaire de backtest: {request.form}")
        
        # Récupérer les paramètres du formulaire
        params = {
            'symbol': request.form.get('symbol', 'BTC/USDT'),
            'timeframe': request.form.get('timeframe', '1h'),
            'days': int(request.form.get('days', 30)),
            'exchange': request.form.get('exchange', 'binance')
        }
        
        # Appeler l'API simulée
        results = simulate_api_response('/backtest', params)
        
        # Générer des données pour le graphique
        chart_data = []
        initial_value = 10000
        current_value = initial_value
        
        # Créer des données de performance quotidiennes
        for i in range(params['days']):
            # Simuler la fluctuation quotidienne
            daily_change = random.uniform(-2, 3) if i % 3 != 0 else random.uniform(-4, 5)
            current_value = current_value * (1 + daily_change/100)
            
            # Ajouter à nos données de graphique
            chart_data.append({
                'date': (datetime.now() - timedelta(days=params['days']-i)).strftime('%Y-%m-%d'),
                'portfolio_value': round(current_value, 2),
                'benchmark_value': round(initial_value * (1 + results['buy_hold_pct']/100 * i/params['days']), 2)
            })
        
        # Générer des données de transactions
        trades = []
        for i in range(min(10, results['num_trades'])):
            trade_type = random.choice(['achat', 'vente'])
            profit = random.uniform(-3, 5) if trade_type == 'vente' else 0
            trades.append({
                'date': (datetime.now() - timedelta(days=random.randint(1, params['days']))).strftime('%Y-%m-%d %H:%M'),
                'type': trade_type,
                'price': round(random.uniform(20000, 40000), 2),
                'amount': round(random.uniform(0.001, 0.1), 6),
                'profit_pct': round(profit, 2),
                'profit_class': 'text-success' if profit > 0 else 'text-danger' if profit < 0 else ''
            })
        
        logger.debug("Données de backtest générées avec succès")
        return render_template('backtest.html', results=results, chart_data=chart_data, trades=trades, params=params)
    
    return render_template('backtest.html')

@app.route('/tradingview')
def tradingview():
    # Liste des symboles disponibles
    symbols = [
        {"symbol": "BINANCE:BTCUSDT", "name": "Bitcoin"},
        {"symbol": "BINANCE:ETHUSDT", "name": "Ethereum"},
        {"symbol": "BINANCE:XRPUSDT", "name": "Ripple"},
        {"symbol": "BINANCE:BNBUSDT", "name": "Binance Coin"},
        {"symbol": "BINANCE:ADAUSDT", "name": "Cardano"}
    ]
    
    default_symbol = "BINANCE:BTCUSDT"
    
    return render_template('tradingview.html', symbols=symbols, default_symbol=default_symbol)

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

@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    try:
        timeframe = request.args.get('timeframe', '1h')
        logger.info(f"Récupération des données de marché pour {timeframe}")

        # Récupérer le prix actuel depuis CoinGecko
        current_prices = get_real_crypto_prices()
        if not current_prices or 'BTC' not in current_prices:
            logger.error("Impossible de récupérer le prix du BTC depuis CoinGecko")
            return jsonify({"error": "Impossible de récupérer les prix"}), 500

        current_btc_price = float(current_prices['BTC'])
        logger.info(f"Prix BTC actuel: ${current_btc_price:,.2f}")

        # Calculer les indicateurs techniques basés sur le prix actuel
        rsi = 37.85  # Ces valeurs seront à calculer réellement plus tard
        macd = 28.9918
        ema9 = current_btc_price
        ema21 = current_btc_price * 0.995

        # Générer une recommandation basée sur les indicateurs
        if rsi < 30:
            recommendation = {"action": "ACHETER", "confidence": 0.69}
        elif rsi > 70:
            recommendation = {"action": "VENDRE", "confidence": 0.69}
        else:
            recommendation = {"action": "NEUTRE", "confidence": 0.69}

        # Créer la structure de données de marché
        market_data = {
            "prices": [
                {
                    "date": datetime.now().isoformat(),
                    "price": current_btc_price
                }
            ],
            "indicators": {
                "rsi": rsi,
                "macd": macd,
                "ema9": ema9,
                "ema21": ema21
            },
            "last_recommendation": recommendation
        }

        logger.info(f"Données de marché générées: {json.dumps(market_data, indent=2)}")
        return jsonify(market_data)

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données de marché: {str(e)}")
        return jsonify({"error": "Erreur serveur"}), 500

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
            transaction = request.json
            logger.info(f"Nouvelle transaction reçue: {json.dumps(transaction, indent=2)}")
            
            # Valider les données
            required_fields = ['symbol', 'action', 'amount']
            if not all(field in transaction for field in required_fields):
                logger.error(f"Données manquantes dans la transaction: {transaction}")
                return jsonify({"error": "Données manquantes"}), 400
            
            # Récupérer le prix actuel depuis CoinGecko
            current_prices = get_real_crypto_prices()
            if not current_prices or transaction['symbol'] not in current_prices:
                logger.error(f"Impossible de récupérer le prix pour {transaction['symbol']}")
                return jsonify({"error": f"Impossible de récupérer le prix pour {transaction['symbol']}"}), 400
            
            # Ajouter la transaction avec le prix actuel
            current_price = float(current_prices[transaction['symbol']])
            amount = float(transaction['amount'])
            
            transaction["timestamp"] = datetime.now().timestamp()
            transaction["price"] = current_price
            transaction["value"] = amount * current_price
            transaction["date"] = datetime.now().isoformat()
            
            logger.info(f"Prix actuel de {transaction['symbol']}: ${current_price}")
            logger.info(f"Valeur de la transaction: {amount} * ${current_price} = ${transaction['value']}")
            
            data["transactions"].append(transaction)
            save_data(data)
            update_portfolio()
            
            logger.info(f"Transaction ajoutée avec succès. Nouvelles données: {json.dumps(data, indent=2)}")
            return jsonify({"success": True})
            
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
            transaction_id = request.json.get("id")
            if transaction_id is None:
                logger.error("ID de transaction manquant")
                return jsonify({"error": "ID de transaction manquant"}), 400
            
            data["transactions"] = [t for t in data["transactions"] if t["timestamp"] != transaction_id]
            save_data(data)
            update_portfolio()
            
            logger.info(f"Transaction supprimée avec succès. Nouvelles données: {json.dumps(data, indent=2)}")
            return jsonify({"success": True})
            
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