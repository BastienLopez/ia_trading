import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime

from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.dqn_agent import DQNAgent
from ai_trading.rl.evaluation import evaluate_agent

def grid_search(df, param_grid, episodes=50, eval_episodes=10):
    """
    Effectue une recherche par grille des hyperparamètres optimaux.
    
    Args:
        df (pd.DataFrame): Données de marché
        param_grid (dict): Grille de paramètres à tester
        episodes (int): Nombre d'épisodes d'entraînement
        eval_episodes (int): Nombre d'épisodes d'évaluation
        
    Returns:
        dict: Meilleurs paramètres trouvés
        float: Meilleure performance
    """
    results = []
    best_performance = -np.inf
    best_params = None
    
    # Générer toutes les combinaisons de paramètres
    param_combinations = []
    
    def generate_combinations(params, current_idx=0, current_combo={}):
        if current_idx == len(list(param_grid.keys())):
            param_combinations.append(current_combo.copy())
            return
        
        param_name = list(param_grid.keys())[current_idx]
        for param_value in param_grid[param_name]:
            current_combo[param_name] = param_value
            generate_combinations(params, current_idx + 1, current_combo)
    
    generate_combinations(param_grid)
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for params in tqdm(param_combinations):
        # Créer l'environnement
        env_params = {k: v for k, v in params.items() if k in 
                     ['window_size', 'transaction_fee', 'action_type', 'n_discrete_actions', 'use_risk_manager']}
        env = TradingEnvironment(df=df, initial_balance=10000, **env_params)
        
        # Créer l'agent
        state_size = len(env.reset()[0])
        action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
        
        agent_params = {k: v for k, v in params.items() if k in 
                       ['learning_rate', 'gamma', 'epsilon', 'epsilon_decay', 'epsilon_min', 'batch_size']}
        agent = DQNAgent(state_size=state_size, action_size=action_size, **agent_params)
        
        # Entraîner l'agent
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
            
            if episode % 5 == 0:
                agent.update_target_model()
        
        # Évaluer l'agent
        performance = evaluate_agent(agent, env, episodes=eval_episodes)
        
        # Enregistrer les résultats
        result = {
            'params': params,
            'performance': performance
        }
        results.append(result)
        
        # Mettre à jour les meilleurs paramètres
        if performance > best_performance:
            best_performance = performance
            best_params = params
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    with open(f'results/hyperparameter_search_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return best_params, best_performance

# Exemple d'utilisation
if __name__ == "__main__":
    from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector
    
    # Utiliser une approche alternative pour charger les données
    import pandas as pd
    import numpy as np

    # Créer des données synthétiques pour les tests
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
    prices = np.linspace(5000, 50000, 1000) + np.random.normal(0, 2000, 1000)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.random.uniform(0, 500, 1000),
            "low": prices - np.random.uniform(0, 500, 1000),
            "close": prices + np.random.normal(0, 200, 1000),
            "volume": np.random.uniform(1000, 5000, 1000),
        },
        index=dates,
    )
    
    # Définir la grille de paramètres
    param_grid = {
        # Paramètres de l'environnement
        'window_size': [20, 50, 100],
        'transaction_fee': [0.001, 0.002],
        'use_risk_manager': [True, False],
        
        # Paramètres de l'agent
        'learning_rate': [0.0001, 0.001, 0.01],
        'gamma': [0.9, 0.95, 0.99],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'batch_size': [32, 64, 128]
    }
    
    # Lancer la recherche par grille
    best_params, best_performance = grid_search(df, param_grid)
    
    print(f"Best parameters: {best_params}")
    print(f"Best performance: {best_performance}") 