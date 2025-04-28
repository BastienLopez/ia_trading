import os
import sys
import numpy as np
import tensorflow as tf
import logging
import pandas as pd

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data

# Configuration du logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestGRUState")

def main():
    """
    Test pour comprendre le problème avec les états dans l'agent GRU-SAC.
    """
    # Générer des données synthétiques
    data = generate_synthetic_market_data(n_points=500, trend=0.001, volatility=0.01, start_price=100.0)
    
    # Ajouter des indicateurs techniques simples
    data['sma_10'] = data['close'].rolling(10).mean()
    data['sma_30'] = data['close'].rolling(30).mean()
    data['rsi'] = 50 + np.random.normal(0, 10, len(data))  # RSI simulé
    data = data.bfill()
    
    # Initialiser l'environnement
    env = TradingEnvironment(
        df=data,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=20,
        reward_function='sharpe',
        action_type="continuous"
    )
    
    # Obtenir la taille de l'état et de l'action
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    sequence_length = 10
    
    logger.info(f"Taille de l'état: {state_size}, Taille de l'action: {action_size}")
    
    # Créer l'agent GRU-SAC
    agent = SACAgent(
        state_size=state_size,
        action_size=action_size,
        action_bounds=[-1, 1],
        buffer_size=1000,
        batch_size=64,
        hidden_size=128,
        use_gru=True,
        sequence_length=sequence_length,
        gru_units=64,
        entropy_regularization=0.01,
        grad_clip_value=1.0
    )
    
    # Réinitialiser l'environnement et récupérer l'état initial
    state, _ = env.reset()
    
    # Afficher le type et la forme de l'état
    logger.info(f"Type d'état: {type(state)}")
    logger.info(f"Forme d'état: {state.shape if hasattr(state, 'shape') else 'N/A'}")
    
    # Inspecter le contenu et le type d'éléments dans l'état
    if isinstance(state, np.ndarray):
        logger.info(f"Type des éléments de l'état: {state.dtype}")
        logger.info(f"Premiers éléments de l'état: {state[:5]}")
        try:
            state_float = state.astype(np.float32)
            logger.info(f"Conversion en float32 réussie")
            logger.info(f"L'état contient des NaN après conversion: {np.any(np.isnan(state_float))}")
        except Exception as e:
            logger.error(f"Erreur de conversion: {e}")
    
    # Créer une séquence d'états avec conversion explicite
    try:
        state_float = np.array(state, dtype=np.float32)
        sequence = np.array([state_float] * sequence_length, dtype=np.float32)
        logger.info(f"Forme de la séquence: {sequence.shape}")
    except Exception as e:
        logger.error(f"Erreur lors de la création de la séquence: {e}")
        sequence = np.zeros((sequence_length, state_size), dtype=np.float32)
    
    try:
        # Tenter de sélectionner une action
        action = agent.act(sequence)
        logger.info(f"Action sélectionnée: {action}")
    except Exception as e:
        logger.error(f"Erreur lors de la sélection d'action: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test avec différents types de données
    logger.info("Test avec différents types de données:")
    
    # 1. Tenseur TensorFlow
    tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
    logger.info(f"Forme du tenseur TensorFlow: {tf_state.shape}")
    
    try:
        # Créer une séquence de tenseurs
        tf_sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)
        action = agent.act(tf_sequence)
        logger.info(f"Action avec tenseur TensorFlow: {action}")
    except Exception as e:
        logger.error(f"Erreur avec tenseur TensorFlow: {e}")
    
    # 2. Liste Python
    try:
        list_state = state.tolist()
        list_sequence = [list_state] * sequence_length
        action = agent.act(list_sequence)
        logger.info(f"Action avec liste Python: {action}")
    except Exception as e:
        logger.error(f"Erreur avec liste Python: {e}")
    
    # 3. DataFrame Pandas
    try:
        df_state = pd.DataFrame([state])
        action = agent.act(df_state)
        logger.info(f"Action avec DataFrame Pandas: {action}")
    except Exception as e:
        logger.error(f"Erreur avec DataFrame Pandas: {e}")
    
    logger.info("Tests terminés.")

if __name__ == "__main__":
    main() 