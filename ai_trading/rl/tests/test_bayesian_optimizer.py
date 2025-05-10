import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

# Ajuster le chemin pour inclure le répertoire racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ai_trading.rl.bayesian_optimizer import BayesianOptimizer, optimize_agent_bayesian
from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.config import INFO_RETOUR_DIR

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=1000):
    """
    Génère des données synthétiques pour les tests.
    
    Args:
        n_samples: Nombre d'échantillons
        
    Returns:
        pd.DataFrame: DataFrame avec les données générées
    """
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    prices = np.linspace(5000, 50000, n_samples) + np.random.normal(0, 2000, n_samples)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 500, n_samples),
        'low': prices - np.random.uniform(0, 500, n_samples),
        'close': prices + np.random.normal(0, 200, n_samples),
        'volume': np.random.uniform(1000, 5000, n_samples)
    }, index=dates)
    
    return df


def create_env(data):
    """
    Crée un environnement de trading pour les tests.
    
    Args:
        data: Données de marché
        
    Returns:
        TradingEnvironment: Environnement de trading
    """
    return TradingEnvironment(
        df=data,
        window_size=20,
        initial_balance=10000,
        commission=0.001,
        use_risk_manager=True
    )


def test_bayesian_optimizer_initialization():
    """
    Teste l'initialisation de l'optimiseur bayésien.
    """
    data = generate_synthetic_data(n_samples=500)
    
    def env_creator():
        return create_env(data)
    
    param_space = {
        "learning_rate": (1e-4, 1e-2, "real"),
        "gamma": (0.9, 0.99, "real"),
        "batch_size": [32, 64, 128],  # Exemple de valeurs catégorielles
    }
    
    optimizer = BayesianOptimizer(
        env_creator=env_creator,
        agent_class=SACAgent,
        param_space=param_space,
        n_episodes=3,  # Valeur réduite pour les tests
        eval_episodes=2,
        n_calls=5,  # Valeur réduite pour les tests
    )
    
    assert len(optimizer.dimensions) == 3
    assert optimizer.param_names == ["learning_rate", "gamma", "batch_size"]
    

def test_bayesian_search_integration():
    """
    Teste l'intégration complète de la recherche bayésienne.
    Ce test est marqué comme lent car il peut prendre plusieurs minutes.
    """
    # Marquer ce test comme lent
    pytest.mark.slow
    
    data = generate_synthetic_data(n_samples=500)
    
    def env_creator():
        return create_env(data)
    
    # Définir un espace de paramètres simplifié pour les tests
    param_space = {
        "learning_rate": (1e-4, 1e-2, "real"),
        "gamma": (0.9, 0.99, "real"),
        "batch_size": [32, 64],  # Options réduites pour les tests
    }
    
    # Créer un répertoire temporaire pour les résultats
    save_dir = INFO_RETOUR_DIR / "test_bayesian_opt"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    optimizer = BayesianOptimizer(
        env_creator=env_creator,
        agent_class=SACAgent,
        param_space=param_space,
        n_episodes=2,  # Valeur très réduite pour les tests
        eval_episodes=1,
        n_calls=3,  # Valeur très réduite pour les tests
        save_dir=save_dir,
    )
    
    # Exécuter la recherche bayésienne
    best_params, best_score = optimizer.bayesian_search()
    
    # Vérifier que les résultats sont cohérents
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert "gamma" in best_params
    assert "batch_size" in best_params
    assert isinstance(best_score, float)
    
    # Vérifier que des fichiers de résultats ont été créés
    files = list(save_dir.glob("bayesian_*.png")) + list(save_dir.glob("bayesian_*.csv"))
    assert len(files) > 0
    
    logger.info(f"Bayesian search test completed with best score: {best_score}")
    logger.info(f"Best parameters: {best_params}")


def test_optimize_agent_bayesian():
    """
    Teste la fonction helper d'optimisation bayésienne.
    Ce test est marqué comme lent car il peut prendre plusieurs minutes.
    """
    # Marquer ce test comme lent
    pytest.mark.slow
    
    data = generate_synthetic_data(n_samples=500)
    
    # Définir un espace de paramètres simplifié pour les tests
    param_space = {
        "learning_rate": (1e-4, 1e-2, "real"),
        "gamma": (0.9, 0.99, "real"),
        "batch_size": [32, 64],  # Options réduites pour les tests
    }
    
    # Créer un répertoire temporaire pour les résultats
    save_dir = INFO_RETOUR_DIR / "test_optimize_agent"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Exécuter l'optimisation
    best_params = optimize_agent_bayesian(
        train_data=data,
        agent_class=SACAgent,
        param_space=param_space,
        n_episodes=2,  # Valeur très réduite pour les tests
        eval_episodes=1,
        n_calls=3,  # Valeur très réduite pour les tests
        save_dir=save_dir,
    )
    
    # Vérifier que les résultats sont cohérents
    assert isinstance(best_params, dict)
    assert "learning_rate" in best_params
    assert "gamma" in best_params
    assert "batch_size" in best_params
    
    logger.info(f"Optimize agent bayesian test completed with best parameters: {best_params}")


if __name__ == "__main__":
    # Exécuter les tests manuellement
    logger.info("Testing Bayesian Optimizer initialization...")
    test_bayesian_optimizer_initialization()
    
    logger.info("Testing Bayesian Search integration...")
    test_bayesian_search_integration()
    
    logger.info("Testing optimize_agent_bayesian function...")
    test_optimize_agent_bayesian()
    
    logger.info("All tests completed successfully!") 