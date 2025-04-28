import logging
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.curriculum_learning import CurriculumLearning, CurriculumTrainer
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.utils.enhanced_data_collector import EnhancedDataCollector
from ai_trading.utils.enhanced_preprocessor import EnhancedPreprocessor

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_with_curriculum")

def load_data():
    """Charge ou collecte les données pour l'entraînement"""
    data_path = Path("data/processed/btc_processed.csv")
    
    if data_path.exists():
        logger.info(f"Chargement des données depuis {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        logger.info("Collecte des données depuis l'API...")
        # Créer le répertoire si nécessaire
        data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collecter les données
        collector = EnhancedDataCollector(
            symbols=["BTC"],
            start_date="2020-01-01",
            end_date="2022-12-31",
            timeframe="1d"
        )
        data = collector.collect_price_data()
        
        # Prétraiter les données
        preprocessor = EnhancedPreprocessor()
        df = preprocessor.preprocess(data["BTC"])
        
        # Sauvegarder les données prétraitées
        df.to_csv(data_path)
        logger.info(f"Données sauvegardées dans {data_path}")
    
    return df

def create_agent(state_size, action_size):
    """Crée un agent SAC avec les paramètres spécifiés"""
    agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        discount_factor=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=100000,
        hidden_size=256,
        train_alpha=True,
        # Paramètres pour gradient clipping et entropy regularization
        grad_clip_value=1.0,
        entropy_regularization=0.1
    )
    
    return agent

def plot_training_history(history, save_path=None):
    """Trace l'historique d'entraînement avec le curriculum learning"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Tracer les récompenses
    axs[0, 0].plot(history["episode"], history["reward"])
    axs[0, 0].set_title("Récompenses par épisode")
    axs[0, 0].set_xlabel("Épisode")
    axs[0, 0].set_ylabel("Récompense")
    axs[0, 0].grid(True)
    
    # Tracer les valeurs de portefeuille
    axs[0, 1].plot(history["episode"], history["portfolio_value"])
    axs[0, 1].set_title("Valeur du portefeuille par épisode")
    axs[0, 1].set_xlabel("Épisode")
    axs[0, 1].set_ylabel("Valeur ($)")
    axs[0, 1].grid(True)
    
    # Tracer les niveaux de difficulté
    axs[1, 0].plot(history["episode"], history["level"])
    axs[1, 0].set_title("Progression des niveaux de difficulté")
    axs[1, 0].set_xlabel("Épisode")
    axs[1, 0].set_ylabel("Niveau de difficulté")
    axs[1, 0].grid(True)
    
    # Tracer les performances (uniquement aux points d'évaluation)
    eval_episodes = []
    eval_performances = []
    
    # Filtrer les points où les performances ont été mesurées
    for i, perf in enumerate(history["performance"]):
        if not np.isnan(perf):
            # Calculer l'épisode correspondant
            eval_episodes.append(i * 10)  # Ajuster si eval_every est différent de 10
            eval_performances.append(perf)
    
    axs[1, 1].plot(eval_episodes, eval_performances, marker='o')
    axs[1, 1].set_title("Performance de l'agent au fil du temps")
    axs[1, 1].set_xlabel("Épisode")
    axs[1, 1].set_ylabel("Performance (0-1)")
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique sauvegardé dans {save_path}")
    
    plt.show()

def main():
    # Charger les données
    df = load_data()
    logger.info(f"Données chargées: {len(df)} points")
    
    # Définir les paramètres de base pour l'environnement
    env_params = {
        "initial_balance": 10000.0,
        "transaction_fee": 0.001,
        "window_size": 20,
        "include_technical_indicators": True,
        "risk_management": True,
        "normalize_observation": True,
        "reward_function": "simple",
        "action_type": "continuous",  # Pour SAC
        "n_discrete_actions": 5  # Ignoré pour action_type=continuous
    }
    
    # Paramètres pour le curriculum learning
    curriculum_params = {
        "initial_difficulty": 0.1,
        "max_difficulty": 1.0,
        "difficulty_increment": 0.1,
        "success_threshold": 0.5,
        "patience": 3,
        "curriculum_type": "mixed",  # Utiliser tous les types de curriculum
        "env_params": env_params
    }
    
    # Créer le système de curriculum learning
    curriculum = CurriculumLearning(df=df, **curriculum_params)
    
    # Créer un environnement temporaire pour déterminer la taille de l'état
    temp_env = curriculum.create_environment()
    state_size = temp_env.observation_space.shape[0]
    action_size = 1  # Pour l'environnement continu
    
    logger.info(f"État: {state_size} dimensions, Action: {action_size} dimensions")
    
    # Créer l'agent SAC
    agent = create_agent(state_size, action_size)
    
    # Créer l'entraîneur avec curriculum
    trainer = CurriculumTrainer(
        agent=agent,
        curriculum=curriculum,
        episodes_per_level=100,
        max_episodes=1000,
        eval_every=10
    )
    
    # Entraîner l'agent
    logger.info("Démarrage de l'entraînement avec curriculum learning...")
    start_time = datetime.now()
    
    training_history = trainer.train(verbose=True)
    
    training_time = datetime.now() - start_time
    logger.info(f"Entraînement terminé en {training_time}")
    
    # Créer le répertoire des modèles s'il n'existe pas
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    
    # Sauvegarder l'agent final
    model_path = save_dir / f"sac_agent_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    agent.save(str(model_path))
    logger.info(f"Agent sauvegardé dans {model_path}")
    
    # Créer le répertoire des visualisations s'il n'existe pas
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # Visualiser les résultats
    plot_path = vis_dir / f"curriculum_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_training_history(training_history, save_path=plot_path)
    
    # Évaluer l'agent final sur l'environnement le plus difficile
    curriculum.current_difficulty = curriculum.max_difficulty
    final_env = curriculum.create_environment()
    
    # Exécuter quelques épisodes d'évaluation
    n_eval = 5
    returns = []
    portfolio_values = []
    
    for i in range(n_eval):
        state, _ = final_env.reset()
        done = False
        truncated = False
        episode_return = 0
        
        while not (done or truncated):
            action = agent.get_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = final_env.step(action)
            episode_return += reward
            state = next_state
        
        returns.append(episode_return)
        portfolio_values.append(final_env.get_portfolio_value())
        
        logger.info(f"Évaluation {i+1}/{n_eval}: "
                   f"Rendement={episode_return:.2f}, "
                   f"Portefeuille final={portfolio_values[-1]:.2f}")
    
    # Afficher les statistiques finales
    avg_return = np.mean(returns)
    avg_portfolio = np.mean(portfolio_values)
    roi = (avg_portfolio / final_env.initial_balance) - 1
    
    logger.info(f"Évaluation finale (moyenne sur {n_eval} épisodes):")
    logger.info(f"Récompense moyenne: {avg_return:.2f}")
    logger.info(f"Valeur finale moyenne du portefeuille: {avg_portfolio:.2f}")
    logger.info(f"ROI moyen: {roi:.2%}")

if __name__ == "__main__":
    main() 