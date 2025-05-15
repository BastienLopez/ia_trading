"""
Exemple d'optimisation des hyperparamètres pour les modèles de trading.

Ce script démontre comment optimiser les hyperparamètres des modèles
d'apprentissage automatique et d'apprentissage par renforcement.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# Import des modules d'optimisation
from ai_trading.optimization.hyperparameters.optimizer import HyperparameterOptimizer
from ai_trading.optimization.hyperparameters.search_space import SearchSpace
from ai_trading.optimization.hyperparameters.evaluation import ModelEvaluator
from ai_trading.ml.trading_signals.ml_model import TradingSignalModel
from ai_trading.rl.agents.dqn_agent import DQNAgent

def load_sample_data():
    """Charge des données d'exemple pour l'optimisation."""
    # Créer des données simulées
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simuler une tendance
    n = len(dates)
    trend = np.concatenate([
        np.linspace(10000, 12000, n//4),
        np.linspace(12000, 9000, n//4),
        np.linspace(9000, 11000, n//4),
        np.linspace(11000, 15000, n - 3*(n//4))
    ])
    noise = np.random.normal(0, 300, n)
    
    # Créer le DataFrame
    data = pd.DataFrame({
        'open': trend - np.random.uniform(0, 100, n),
        'high': trend + np.random.uniform(50, 200, n),
        'low': trend - np.random.uniform(50, 200, n),
        'close': trend + noise,
        'volume': np.random.uniform(1000, 5000, n)
    }, index=dates)
    
    # Ajouter des indicateurs techniques
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['sma_30'] = data['close'].rolling(window=30).mean()
    data['ema_10'] = data['close'].ewm(span=10, adjust=False).mean()
    data['ema_30'] = data['close'].ewm(span=30, adjust=False).mean()
    data['rsi'] = calculate_rsi(data['close'])
    data['volatility'] = data['close'].rolling(window=10).std() / data['close']
    
    # Supprimer les NaN
    data = data.dropna()
    
    return data

def calculate_rsi(prices, period=14):
    """Calcule l'indicateur RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def define_ml_search_space():
    """Définit l'espace de recherche pour les modèles ML."""
    search_space = SearchSpace()
    
    # Paramètres du modèle Random Forest
    search_space.add_parameter('n_estimators', 'int', [50, 100, 200, 300])
    search_space.add_parameter('max_depth', 'int', [3, 5, 8, 10, None])
    search_space.add_parameter('min_samples_split', 'int', [2, 5, 10])
    
    # Paramètres de prétraitement
    search_space.add_parameter('feature_selection', 'categorical', ['all', 'pca', 'rfe'])
    search_space.add_parameter('scaling', 'categorical', ['standard', 'minmax', 'robust', None])
    
    # Paramètres de la stratégie
    search_space.add_parameter('signal_threshold', 'float', [0.2, 0.3, 0.4, 0.5])
    search_space.add_parameter('window_size', 'int', [5, 10, 20])
    
    return search_space

def define_rl_search_space():
    """Définit l'espace de recherche pour les modèles RL."""
    search_space = SearchSpace()
    
    # Paramètres du réseau
    search_space.add_parameter('hidden_size', 'int', [64, 128, 256, 512])
    search_space.add_parameter('learning_rate', 'float', [1e-4, 3e-4, 1e-3], log=True)
    
    # Paramètres de l'agent
    search_space.add_parameter('gamma', 'float', [0.95, 0.97, 0.99, 0.995])
    search_space.add_parameter('batch_size', 'int', [32, 64, 128, 256])
    search_space.add_parameter('buffer_size', 'int', [10000, 50000, 100000])
    
    # Paramètres de l'environnement
    search_space.add_parameter('reward_function', 'categorical', ['sharpe', 'sortino', 'calmar', 'profit'])
    search_space.add_parameter('window_size', 'int', [10, 20, 30])
    
    return search_space

def create_ml_model(params):
    """Crée un modèle ML avec les paramètres spécifiés."""
    return TradingSignalModel(
        model_type='random_forest',
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        feature_selection=params['feature_selection'],
        scaling=params['scaling'],
        signal_threshold=params['signal_threshold'],
        window_size=params['window_size']
    )

def create_rl_model(params):
    """Crée un modèle RL avec les paramètres spécifiés."""
    return DQNAgent(
        state_dim=10,  # Fixé pour l'exemple
        action_dim=3,  # Fixé pour l'exemple
        hidden_dim=params['hidden_size'],
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        batch_size=params['batch_size'],
        buffer_size=params['buffer_size']
    )

def hyperparameter_optimization_example():
    """Exemple principal d'optimisation des hyperparamètres."""
    # Charger les données
    data = load_sample_data()
    
    # Créer le dossier de résultats
    output_dir = "examples/optimization/hyperparameters/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Optimisation des hyperparamètres pour le modèle ML
    print("=== Optimisation des hyperparamètres pour le modèle ML ===")
    
    # Définir l'espace de recherche
    ml_search_space = define_ml_search_space()
    
    # Créer l'évaluateur de modèle
    ml_evaluator = ModelEvaluator(
        metric='f1_score',  # Métrique à optimiser
        cv_folds=5,         # Validation croisée
        train_ratio=0.7     # Ratio d'entraînement/test
    )
    
    # Créer l'optimiseur
    ml_optimizer = HyperparameterOptimizer(
        search_space=ml_search_space,
        evaluator=ml_evaluator,
        model_factory=create_ml_model,
        optimization_type='grid',  # Recherche par grille
        n_trials=10,               # Nombre d'essais
        n_jobs=3                   # Parallélisation
    )
    
    # Lancer l'optimisation
    print("Lancement de l'optimisation ML...")
    ml_results = ml_optimizer.optimize(data)
    
    # Afficher les meilleurs paramètres
    print("\n=== Résultats de l'optimisation ML ===")
    print(f"Meilleur score: {ml_results['best_score']:.4f}")
    print("Meilleurs paramètres:")
    for param, value in ml_results['best_params'].items():
        print(f"  {param}: {value}")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 8))
    
    # Tracer l'évolution du score
    plt.subplot(2, 1, 1)
    plt.plot(ml_results['history']['trial'], ml_results['history']['score'])
    plt.title('Évolution du score pendant l\'optimisation ML')
    plt.xlabel('Essai')
    plt.ylabel('Score (F1)')
    plt.grid(True)
    
    # Tracer l'importance des paramètres
    plt.subplot(2, 1, 2)
    params = list(ml_results['param_importance'].keys())
    importances = list(ml_results['param_importance'].values())
    plt.barh(params, importances)
    plt.title('Importance des paramètres pour le modèle ML')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # Enregistrer le graphique
    plt.savefig(f"{output_dir}/ml_optimization_results.png")
    plt.close()
    
    # 2. Optimisation des hyperparamètres pour le modèle RL
    print("\n=== Optimisation des hyperparamètres pour le modèle RL ===")
    
    # Définir l'espace de recherche
    rl_search_space = define_rl_search_space()
    
    # Créer l'évaluateur de modèle
    rl_evaluator = ModelEvaluator(
        metric='sharpe_ratio',  # Métrique à optimiser
        cv_folds=3,             # Validation croisée
        train_ratio=0.7         # Ratio d'entraînement/test
    )
    
    # Créer l'optimiseur
    rl_optimizer = HyperparameterOptimizer(
        search_space=rl_search_space,
        evaluator=rl_evaluator,
        model_factory=create_rl_model,
        optimization_type='bayesian',  # Optimisation bayésienne
        n_trials=15,                  # Nombre d'essais
        n_jobs=3                      # Parallélisation
    )
    
    # Lancer l'optimisation
    print("Lancement de l'optimisation RL...")
    rl_results = rl_optimizer.optimize(data)
    
    # Afficher les meilleurs paramètres
    print("\n=== Résultats de l'optimisation RL ===")
    print(f"Meilleur score: {rl_results['best_score']:.4f}")
    print("Meilleurs paramètres:")
    for param, value in rl_results['best_params'].items():
        print(f"  {param}: {value}")
    
    # Visualiser les résultats
    plt.figure(figsize=(12, 8))
    
    # Tracer l'évolution du score
    plt.subplot(2, 1, 1)
    plt.plot(rl_results['history']['trial'], rl_results['history']['score'])
    plt.title('Évolution du score pendant l\'optimisation RL')
    plt.xlabel('Essai')
    plt.ylabel('Score (Sharpe Ratio)')
    plt.grid(True)
    
    # Tracer l'importance des paramètres
    plt.subplot(2, 1, 2)
    params = list(rl_results['param_importance'].keys())
    importances = list(rl_results['param_importance'].values())
    plt.barh(params, importances)
    plt.title('Importance des paramètres pour le modèle RL')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # Enregistrer le graphique
    plt.savefig(f"{output_dir}/rl_optimization_results.png")
    plt.close()
    
    print(f"\nGraphiques des résultats enregistrés dans {output_dir}")
    
    return {
        'ml_results': ml_results,
        'rl_results': rl_results
    }

if __name__ == "__main__":
    results = hyperparameter_optimization_example() 