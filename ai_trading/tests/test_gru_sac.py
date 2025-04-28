import unittest
import numpy as np
import tensorflow as tf
import sys
import os
import logging
import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajustement du chemin pour l'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ai_trading.rl.agents.sac_agent import SACAgent, ReplayBuffer, SequenceReplayBuffer
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.data.synthetic_data_generator import generate_synthetic_market_data

class TestGRUSAC(unittest.TestCase):
    
    def setUp(self):
        """Initialise l'environnement de test avec des données synthétiques."""
        # Générer des données synthétiques
        self.data = generate_synthetic_market_data(n_points=500, trend=0.001, 
                                                   volatility=0.01, start_price=100.0)
        
        # Ajouter des indicateurs techniques (colonnes supplémentaires)
        self.data['sma_10'] = self.data['close'].rolling(10).mean()
        self.data['sma_30'] = self.data['close'].rolling(30).mean()
        self.data['rsi'] = 50 + np.random.normal(0, 10, len(self.data))  # RSI simulé
        
        # Remplir les NaN avec des valeurs appropriées
        self.data = self.data.bfill()
        
        # Initialiser l'environnement
        self.env = TradingEnvironment(
            df=self.data,
            window_size=20,
            fee=0.001,
            initial_balance=10000,
            reward_function='sharpe',
            data_start=50,
            data_end=450,
            action_type="continuous"  # Spécifier explicitement un espace d'action continu
        )
        
        # Paramètres de l'agent
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.sequence_length = 10
        
        # Créer un agent SAC avec GRU
        self.agent = SACAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            action_bounds=[-1, 1],
            buffer_size=1000,
            batch_size=64,
            hidden_size=128,
            use_gru=True,
            sequence_length=self.sequence_length,
            gru_units=64
        )
        
        # Préparer un mini-batch de séquences pour les tests
        self.batch_sequences = np.random.normal(0, 1, (32, self.sequence_length, self.state_size))
        
    def test_sequence_normalization(self):
        """Teste la normalisation des séquences d'états."""
        # Appliquer la normalisation
        normalized_sequences = self.agent._normalize_sequence_states(self.batch_sequences)
        
        # Vérifier que le résultat est un tenseur TensorFlow
        self.assertIsInstance(normalized_sequences, tf.Tensor)
        
        # Vérifier la forme
        self.assertEqual(normalized_sequences.shape, 
                         (32, self.sequence_length, self.state_size))
        
        # Vérifier que la moyenne de chaque feature est proche de 0
        means = tf.reduce_mean(normalized_sequences, axis=2)
        self.assertTrue(np.allclose(tf.reduce_mean(means).numpy(), 0, atol=1e-5))
        
        # Vérifier que l'écart-type de chaque feature est proche de 1
        stds = tf.math.reduce_std(normalized_sequences, axis=2)
        self.assertTrue(np.allclose(tf.reduce_mean(stds).numpy(), 1, atol=1e-1))
        
        # Tester avec une entrée de séquence aux valeurs constantes
        constant_sequences = np.ones((32, self.sequence_length, self.state_size))
        normalized_constant = self.agent._normalize_sequence_states(constant_sequences)
        
        # Pour des valeurs constantes, l'écart-type devrait être proche de 0
        # et la normalisation devrait produire des valeurs proches de 0
        self.assertTrue(np.allclose(normalized_constant.numpy(), 0, atol=1e-5))
        
    def test_end_to_end_gru_processing(self):
        """Teste le traitement de bout en bout avec les couches GRU."""
        # Obtenir un état initial de l'environnement
        reset_result = self.env.reset()
        # Dans gymnasium, reset() retourne (state, info)
        if isinstance(reset_result, tuple):
            state = reset_result[0]
        else:
            state = reset_result
        
        # Créer une séquence artificielle avec le bon format
        # Assurez-vous que state est un tableau numpy
        state = np.array(state, dtype=np.float32)
        # Créer une séquence de states identiques
        sequence = np.tile(state, (self.sequence_length, 1))
        # Ajouter la dimension du batch
        sequence = np.expand_dims(sequence, axis=0)
        
        # Normaliser la séquence
        normalized_seq = self.agent._normalize_sequence_states(sequence)
        
        # Vérifier que l'agent peut traiter cette séquence normalisée
        # en appelant directement l'acteur (si l'accès est possible)
        if hasattr(self.agent, 'actor'):
            action_mean, _ = self.agent.actor(normalized_seq)
            
            # Vérifier la forme de la sortie de l'acteur
            self.assertEqual(action_mean.shape, (1, self.action_size))
            
            # Vérifier que les actions sont dans les limites
            self.assertTrue(np.all(action_mean.numpy() >= -1))
            self.assertTrue(np.all(action_mean.numpy() <= 1))
            
            logger.info(f"Action moyenne produite: {action_mean.numpy()}")
        else:
            logger.warning("Test ignoré: l'acteur n'est pas directement accessible")

if __name__ == '__main__':
    unittest.main() 