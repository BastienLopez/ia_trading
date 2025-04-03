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
def api_market_data():
    """Récupérer les données de marché actuelles"""
    symbol = request.args.get('symbol', 'BTC/USDT')
    timeframe = request.args.get('timeframe', '1h')
    
    logger.info(f"Demande de données de marché: {symbol}, {timeframe}")
    try:
        # Pour la démonstration, on utilise des données fictives
        logger.debug("Génération de données fictives pour la démo")
        mock_data = {
            "prices": [
                {"date": (datetime.now() - timedelta(hours=i)).isoformat(), 
                 "price": 20000 + (i * 100) + ((i % 3) * 200)} 
                for i in range(48, 0, -1)
            ],
            "indicators": {
                "rsi": 65.2,
                "macd": 0.8,
                "signal": 0.2
            },
            "last_recommendation": {
                "action": "ACHETER",
                "confidence": 0.87,
                "timestamp": datetime.now().isoformat()
            }
        }
        logger.debug(f"Nombre d'entrées de prix générées: {len(mock_data['prices'])}")
        return jsonify(mock_data)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données de marché: {str(e)}", exc_info=True)
        return jsonify({"error": True, "message": str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def api_portfolio():
    """Récupérer les données du portefeuille"""
    logger.info("Demande de données du portefeuille")
    try:
        # Pour la démonstration, on utilise des données fictives
        logger.debug("Génération de données fictives de portefeuille")
        mock_data = {
            "current_value": 12500.75,
            "starting_value": 10000.00,
            "profit_pct": 25.01,
            "assets": [
                {"symbol": "BTC", "amount": 0.25, "value_usd": 7500.25},
                {"symbol": "ETH", "amount": 2.5, "value_usd": 3000.50},
                {"symbol": "USDT", "amount": 2000.00, "value_usd": 2000.00}
            ],
            "trade_history": [
                {"timestamp": (datetime.now() - timedelta(days=20)).isoformat(), "action": "ACHETER", "symbol": "BTC", "price": 19500, "amount": 0.15, "value": 2925.00},
                {"timestamp": (datetime.now() - timedelta(days=15)).isoformat(), "action": "ACHETER", "symbol": "ETH", "price": 1050, "amount": 2.0, "value": 2100.00},
                {"timestamp": (datetime.now() - timedelta(days=7)).isoformat(), "action": "ACHETER", "symbol": "BTC", "price": 21500, "amount": 0.1, "value": 2150.00},
                {"timestamp": (datetime.now() - timedelta(days=3)).isoformat(), "action": "ACHETER", "symbol": "ETH", "price": 1200, "amount": 0.5, "value": 600.00}
            ]
        }
        logger.debug(f"Données de portefeuille générées avec {len(mock_data['assets'])} actifs")
        return jsonify(mock_data)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données du portefeuille: {str(e)}", exc_info=True)
        return jsonify({"error": True, "message": str(e)}), 500

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