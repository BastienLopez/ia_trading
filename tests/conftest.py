"""
Fixtures pour les tests pytest
"""
import os
import sys
import logging
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from pathlib import Path

# Assurer que les modules sont dans le chemin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration du logging pour les tests
def configure_logging():
    """Configure le logging pour les tests"""
    # Créer le répertoire de logs s'il n'existe pas
    logs_dir = Path(__file__).parent.parent / 'logs'
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True)
    
    # Fichier de log pour cette session de test
    log_file = logs_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Niveau de log basé sur la variable d'environnement ou DEBUG par défaut
    log_level = os.environ.get('LOGLEVEL', 'DEBUG').upper()
    
    # Format des logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configuration de base
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Console aussi
        ]
    )
    
    # Message de début de session
    logging.info("="*80)
    logging.info("DÉBUT DE LA SESSION DE TEST")
    logging.info("="*80)

# Configurer le logging au démarrage
configure_logging()

@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Fixture pour configurer et nettoyer le logging"""
    # Déjà configuré au démarrage, mais on peut ajouter d'autres actions ici
    
    yield  # Exécution des tests
    
    # Message de fin de session
    logging.info("="*80)
    logging.info("FIN DE LA SESSION DE TEST")
    logging.info("="*80)

@pytest.fixture(autouse=True)
def log_test_info(request):
    """Logging automatique pour chaque test"""
    test_name = request.node.name
    module_name = request.module.__name__
    
    logging.info(f"DÉBUT DU TEST: {module_name}::{test_name}")
    
    yield  # Exécution du test
    
    # Après le test
    if hasattr(request.node, 'rep_call') and request.node.rep_call:
        outcome = "SUCCÈS" if request.node.rep_call.passed else "ÉCHEC"
    else:
        outcome = "INDÉTERMINÉ"
    
    logging.info(f"FIN DU TEST: {module_name}::{test_name} - Résultat: {outcome}")
    logging.info("-"*50)

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook pour capturer le résultat de chaque test pour le logging"""
    outcome = yield
    rep = outcome.get_result()
    
    # Définir un attribut sur le nœud de test pour récupérer le résultat plus tard
    setattr(item, f"rep_{rep.when}", rep)
    
    # Log détaillé en cas d'échec
    if rep.when == "call" and not rep.passed:
        logging.error(f"ÉCHEC DU TEST: {item.name}")
        if hasattr(rep, "longrepr"):
            logging.error(f"DÉTAILS: {rep.longrepr}")

@pytest.fixture
def sample_ohlcv_data():
    """Génère un petit échantillon de données OHLCV"""
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=20, freq='1h'),
        'open': np.random.rand(20) * 100 + 20000,
        'high': np.random.rand(20) * 100 + 20100,
        'low': np.random.rand(20) * 100 + 19900,
        'close': np.random.rand(20) * 100 + 20000,
        'volume': np.random.rand(20) * 1000
    })
    df.set_index('timestamp', inplace=True)
    return df

@pytest.fixture
def sample_data_with_indicators(sample_ohlcv_data):
    """Génère des données OHLCV avec des indicateurs techniques"""
    df = sample_ohlcv_data.copy()
    
    # Ajouter des indicateurs fictifs
    df['rsi'] = np.random.rand(20) * 100
    df['macd'] = np.random.rand(20) * 2 - 1
    df['signal_line'] = np.random.rand(20) * 2 - 1
    df['hist_line'] = np.random.rand(20) * 2 - 1
    df['ema9'] = np.random.rand(20) * 100 + 20000
    df['ema21'] = np.random.rand(20) * 100 + 20000
    df['tenkan'] = np.random.rand(20) * 100 + 20000
    df['kijun'] = np.random.rand(20) * 100 + 20000
    df['senkou_span_a'] = np.random.rand(20) * 100 + 20000
    df['senkou_span_b'] = np.random.rand(20) * 100 + 20000
    df['bb_upper'] = np.random.rand(20) * 100 + 20100
    df['bb_middle'] = df['close']
    df['bb_lower'] = np.random.rand(20) * 100 + 19900
    df['volume_ratio'] = np.random.rand(20)
    
    return df

@pytest.fixture
def mock_api_response():
    """Crée une réponse d'API mock standard pour les tests"""
    def _create_response(action="ATTENDRE", confidence=0.7, price=20000.0):
        return {
            "recommendation": action,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "market_data": {
                "price": price,
                "rsi": 45.0,
                "macd": 0.2,
                "volume_24h": 100000.0
            }
        }
    
    return _create_response

@pytest.fixture
def mock_backtest_results():
    """Crée des résultats de backtest mock pour les tests"""
    return {
        "symbol": "BTC/USDT",
        "period": "2023-01-01 to 2023-01-30",
        "profit_pct": 7.5,
        "buy_hold_pct": 3.2,
        "sharpe_ratio": 1.4,
        "num_trades": 12,
        "win_rate": 0.67,
        "trades_summary": [
            {"entry_time": "2023-01-01T08:00:00", "exit_time": "2023-01-02T14:00:00", "profit_pct": 2.5, "win": True},
            {"entry_time": "2023-01-03T10:00:00", "exit_time": "2023-01-04T09:00:00", "profit_pct": -1.2, "win": False},
            {"entry_time": "2023-01-05T16:00:00", "exit_time": "2023-01-10T11:00:00", "profit_pct": 5.7, "win": True}
        ]
    } 