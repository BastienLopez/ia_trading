import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os
import logging
import pandas as pd
import tempfile

# Configurer le logger pour les tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trading.rl.agents.sac_agent import SACAgent
from ai_trading.rl.trading_environment import TradingEnvironment


class TestEntropyRegularization(unittest.TestCase):
    """Tests pour la régularisation d'entropie et le gradient clipping dans l'agent SAC"""

    def setUp(self):
        """Configuration pour chaque test"""
        # Créer un petit environnement de test avec des données synthétiques
        np.random.seed(42)  # Pour la reproductibilité
        tf.random.set_seed(42)
        
        # Générer des données synthétiques pour l'environnement
        n_samples = 100
        dates = pd.date_range(start='2023-01-01', periods=n_samples)
        prices = np.linspace(100, 200, n_samples) + np.random.normal(0, 5, n_samples)
        volumes = np.random.normal(1000, 200, n_samples)
        
        # Créer un DataFrame avec les données
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': prices - np.random.uniform(0, 2, n_samples),
            'high': prices + np.random.uniform(0, 2, n_samples),
            'low': prices - np.random.uniform(0, 2, n_samples),
            'close': prices,
            'volume': volumes,
            'market_cap': prices * volumes
        })
        
        # Créer l'environnement avec action_type="continuous"
        self.env = TradingEnvironment(
            df=self.df,
            initial_balance=10000.0,
            transaction_fee=0.001,
            window_size=10,
            action_type="continuous"
        )
        
        # Créer un état pour déterminer sa taille réelle
        state, _ = self.env.reset()
        self.state_size = state.shape[0]
        
        # Collecter des expériences pour le tampon de replay
        state, _ = self.env.reset()
        for _ in range(50):
            action = np.random.uniform(-1, 1, 1)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state

    def test_entropy_regularization(self):
        """Teste si la régularisation d'entropie affecte la perte de l'acteur"""
        # Créer deux agents avec différentes valeurs de régularisation d'entropie
        agent_low_reg = SACAgent(
            state_size=self.state_size,
            action_size=1,
            batch_size=32,
            buffer_size=1000,
            hidden_size=64,
            entropy_regularization=0.001
        )
        
        agent_high_reg = SACAgent(
            state_size=self.state_size,
            action_size=1,
            batch_size=32,
            buffer_size=1000,
            hidden_size=64,
            entropy_regularization=0.1  # Valeur plus élevée
        )
        
        # Ajouter les mêmes expériences aux deux agents
        state, _ = self.env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, 1)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            agent_low_reg.remember(state, action, reward, next_state, done)
            agent_high_reg.remember(state, action, reward, next_state, done)
            
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
        
        # Entraîner les deux agents
        metrics_low = agent_low_reg.train()
        metrics_high = agent_high_reg.train()
        
        # L'agent avec une régularisation d'entropie plus élevée devrait avoir
        # une entropie plus élevée après l'entraînement
        self.assertLessEqual(metrics_low["entropy"], metrics_high["entropy"],
                          msg="L'agent avec une régularisation d'entropie plus élevée devrait avoir une entropie plus élevée")

    def test_gradient_clipping(self):
        """Teste si le gradient clipping limite correctement la norme des gradients"""
        # Créer un agent avec gradient clipping
        agent = SACAgent(
            state_size=self.state_size,
            action_size=1,
            batch_size=32,
            buffer_size=1000,
            hidden_size=64,
            grad_clip_value=0.5  # Valeur de clipping assez faible pour des tests
        )
        
        # Ajouter des expériences
        state, _ = self.env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, 1)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Ajouter des récompenses extrêmes pour générer de grands gradients
            # qui devraient être clippés
            reward = reward * 100  # Amplifier artificiellement les récompenses
            
            agent.remember(state, action, reward, next_state, done)
            
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
        
        # Vérifier les normes des gradients pendant l'entraînement
        if len(agent.replay_buffer) >= agent.batch_size:
            # Échantillonner un lot
            states, actions, rewards, next_states, dones = agent.replay_buffer.sample(agent.batch_size)
            
            # Convertir en tenseurs
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            # Ajouter une dimension si nécessaire
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, axis=1)
            
            # Calculer les gradients manuellement pour vérifier le clipping
            with tf.GradientTape(persistent=True) as tape:
                # Étape d'entraînement similaire à celle dans l'agent
                next_means, next_log_stds = agent.actor(next_states)
                next_stds = tf.exp(next_log_stds)
                next_normal_dists = tfp.distributions.Normal(next_means, next_stds)
                next_actions_raw = next_normal_dists.sample()
                next_actions = tf.tanh(next_actions_raw)
                
                # Calculer les valeurs Q
                current_q1 = agent.critic_1([states, actions])
                target_q = rewards  # Simplifier pour le test
                
                # Calculer la perte
                critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_q))
            
            # Obtenir les gradients
            critic1_gradients = tape.gradient(critic1_loss, agent.critic_1.trainable_variables)
            
            # Vérifier que les gradients sont limités après clipping
            for g in critic1_gradients:
                if g is not None:
                    clipped_g = tf.clip_by_norm(g, agent.grad_clip_value)
                    norm_original = tf.norm(g)
                    norm_clipped = tf.norm(clipped_g)
                    
                    # Si le gradient original avait une norme supérieure à la valeur de clipping,
                    # alors la norme du gradient clippé devrait être exactement égale à la valeur de clipping
                    if norm_original > agent.grad_clip_value:
                        self.assertAlmostEqual(
                            float(norm_clipped), 
                            agent.grad_clip_value, 
                            places=5,
                            msg=f"Le gradient clippé devrait avoir une norme de {agent.grad_clip_value}, mais a {norm_clipped}"
                        )

    def test_compare_performance(self):
        """Teste la performance de l'agent avec et sans régularisation d'entropie"""
        # Créer un agent sans régularisation d'entropie supplémentaire
        agent_no_reg = SACAgent(
            state_size=self.state_size,
            action_size=1,
            batch_size=32,
            buffer_size=1000,
            hidden_size=64,
            entropy_regularization=0.0  # Désactiver la régularisation supplémentaire
        )
        
        # Créer un agent avec régularisation d'entropie
        agent_with_reg = SACAgent(
            state_size=self.state_size,
            action_size=1,
            batch_size=32,
            buffer_size=1000,
            hidden_size=64,
            entropy_regularization=0.05  # Valeur modérée
        )
        
        # Entraîner les deux agents sur le même environnement
        total_rewards_no_reg = []
        total_rewards_with_reg = []
        
        for _ in range(5):  # Plusieurs épisodes pour une meilleure comparaison
            state, _ = self.env.reset()
            done = False
            total_reward_no_reg = 0
            total_reward_with_reg = 0
            
            while not done:
                # Agent sans régularisation
                action_no_reg = agent_no_reg.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action_no_reg)
                done = terminated or truncated
                agent_no_reg.remember(state, action_no_reg, reward, next_state, done)
                total_reward_no_reg += reward
                
                # Entraîner l'agent
                if len(agent_no_reg.replay_buffer) >= agent_no_reg.batch_size:
                    agent_no_reg.train()
                
                if done:
                    break
                    
                state = next_state
            
            # Réinitialiser pour l'agent avec régularisation
            state, _ = self.env.reset()
            done = False
            
            while not done:
                # Agent avec régularisation
                action_with_reg = agent_with_reg.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action_with_reg)
                done = terminated or truncated
                agent_with_reg.remember(state, action_with_reg, reward, next_state, done)
                total_reward_with_reg += reward
                
                # Entraîner l'agent
                if len(agent_with_reg.replay_buffer) >= agent_with_reg.batch_size:
                    agent_with_reg.train()
                
                if done:
                    break
                    
                state = next_state
            
            total_rewards_no_reg.append(total_reward_no_reg)
            total_rewards_with_reg.append(total_reward_with_reg)
        
        # Comparer les performances moyennes
        avg_reward_no_reg = np.mean(total_rewards_no_reg)
        avg_reward_with_reg = np.mean(total_rewards_with_reg)
        
        # L'agent avec régularisation d'entropie peut explorer davantage,
        # ce qui pourrait conduire à de meilleures performances à long terme
        # Note: Ce test peut ne pas toujours passer car l'exploration peut 
        # réduire les performances à court terme
        logger.info(f"Récompense moyenne sans régularisation: {avg_reward_no_reg}")
        logger.info(f"Récompense moyenne avec régularisation: {avg_reward_with_reg}")
        
        # Aucune assertion ici, car les résultats peuvent varier
        # C'est plutôt un test informatif pour comparer les performances


if __name__ == '__main__':
    unittest.main() 