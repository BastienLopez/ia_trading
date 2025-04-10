"""
Module pour le système de trading basé sur l'apprentissage par renforcement.
"""

import logging
import os
import numpy as np

from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.train import train_agent

# Configuration du logger
logger = logging.getLogger("RLTradingSystem")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class RLTradingSystem:
    """Système de trading basé sur l'apprentissage par renforcement."""

    def __init__(self, config=None):
        """Initialise le système de trading."""
        self.config = config or {}
        self.data_integrator = RLDataIntegrator()
        self.env = None
        self.agent = None
        self.history = None

        logger.info("Système de trading RL initialisé")

    def create_agent(self, agent_type="dqn", state_size=None, action_size=3, **kwargs):
        """
        Crée un agent d'apprentissage par renforcement.

        Args:
            agent_type (str): Type d'agent ('dqn' ou 'ppo')
            state_size (int): Taille de l'espace d'état
            action_size (int): Taille de l'espace d'action
            **kwargs: Arguments supplémentaires pour l'agent

        Returns:
            Agent: Agent d'apprentissage par renforcement
        """
        if state_size is None and self.env is not None:
            # Déterminer la taille de l'état à partir de l'environnement
            state_size = self.env.observation_space.shape[0]

        if agent_type.lower() == "dqn":
            from ai_trading.rl.dqn_agent import DQNAgent

            agent = DQNAgent(state_size=state_size, action_size=action_size, **kwargs)
        elif agent_type.lower() == "ppo":
            # Implémentation future pour PPO
            raise NotImplementedError("Agent PPO non implémenté")
        else:
            raise ValueError(f"Type d'agent inconnu: {agent_type}")

        return agent

    def create_environment(self, data=None, **kwargs):
        """
        Crée un environnement de trading.

        Args:
            data (DataFrame): Données de marché
            **kwargs: Arguments supplémentaires pour l'environnement

        Returns:
            TradingEnvironment: Environnement de trading
        """
        if data is None:
            # Générer des données synthétiques
            data = self.data_integrator.generate_synthetic_data(
                n_samples=100, trend="bullish", volatility=0.02, with_sentiment=True
            )

        # Créer l'environnement
        self.env = TradingEnvironment(df=data, **kwargs)

        logger.info(f"Environnement de trading créé avec {len(data)} points de données")
        return self.env

    def train(self, agent=None, env=None, **kwargs):
        """
        Entraîne l'agent sur l'environnement.

        Args:
            agent: Agent d'apprentissage par renforcement
            env: Environnement de trading
            **kwargs: Arguments supplémentaires pour l'entraînement

        Returns:
            dict: Historique d'entraînement
        """
        if agent is None:
            if self.agent is None:
                raise ValueError("Aucun agent spécifié")
            agent = self.agent
        else:
            self.agent = agent

        if env is None:
            if self.env is None:
                raise ValueError("Aucun environnement spécifié")
            env = self.env
        else:
            self.env = env

        # Entraîner l'agent
        self.history = train_agent(agent=agent, env=env, **kwargs)

        return self.history

    def evaluate(
        self,
        agent=None,
        env=None,
        num_episodes=1,
        test_data=None,
        visualize=False,
        save_dir=None,
    ):
        """
        Évalue l'agent sur l'environnement.

        Args:
            agent: Agent d'apprentissage par renforcement
            env: Environnement de trading
            num_episodes (int): Nombre d'épisodes d'évaluation
            test_data (DataFrame): Données de test (si différentes de celles de l'environnement)
            visualize (bool): Afficher les visualisations
            save_dir (str): Répertoire pour sauvegarder les visualisations

        Returns:
            dict: Résultats de l'évaluation
        """
        if agent is None:
            if self.agent is None:
                raise ValueError("Aucun agent spécifié")
            agent = self.agent

        if env is None:
            if self.env is None:
                raise ValueError("Aucun environnement spécifié")
            env = self.env

        # Si des données de test sont fournies, créer un nouvel environnement
        if test_data is not None:
            env = self.create_environment(data=test_data)

        # Évaluer l'agent
        from ai_trading.rl.evaluation import evaluate_agent

        results = evaluate_agent(agent=agent, env=env, num_episodes=num_episodes)

        # Visualiser les résultats si demandé
        if visualize and save_dir:
            from ai_trading.rl.evaluation import PerformanceVisualizer

            visualizer = PerformanceVisualizer(save_dir=save_dir)
            visualizer.create_performance_dashboard(
                results=results,
                dates=env.df.index[-len(results["portfolio_history"]) :],
                actions=results["actions"],
            )

        return results

    def save(self, path):
        """
        Sauvegarde le système de trading.

        Args:
            path (str): Chemin de sauvegarde
        """
        if self.agent is None:
            raise ValueError("Aucun agent à sauvegarder")

        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Sauvegarder l'agent
        self.agent.save(path)

        logger.info(f"Système de trading sauvegardé dans {path}")

    def load(self, path):
        """
        Charge le système de trading.

        Args:
            path (str): Chemin de chargement
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier {path} n'existe pas")

        # Créer un agent temporaire pour charger le modèle
        from ai_trading.rl.dqn_agent import DQNAgent

        self.agent = DQNAgent(state_size=1, action_size=3)  # Tailles temporaires
        self.agent.load(path)

        logger.info(f"Système de trading chargé depuis {path}")

        return self.agent

    def test_random_strategy(self, num_episodes=10):
        """
        Teste une stratégie aléatoire pour établir une référence.
        
        Args:
            num_episodes (int): Nombre d'épisodes à exécuter.
            
        Returns:
            dict: Résultats de la stratégie aléatoire.
        """
        total_rewards = []
        portfolio_values = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.env.action_space.sample()  # Action aléatoire
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Enregistrer la valeur du portefeuille
                portfolio_values.append(info['portfolio_value'])
            
            total_rewards.append(episode_reward)
        
        # Calculer les métriques
        avg_reward = np.mean(total_rewards)
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)
        
        # Calculer les métriques de trading
        final_portfolio_value = portfolio_values[-1]
        initial_portfolio_value = self.env.initial_balance
        total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
        
        return {
            'average_reward': avg_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'total_return': total_return,
            'final_portfolio_value': final_portfolio_value
        }

    def integrate_data(
        self,
        market_data=None,
        news_data=None,
        social_data=None,
        sentiment_data=None,
        split_ratio=0.8,
        **kwargs,
    ):
        """
        Intègre les données de marché, de nouvelles et de sentiment.

        Args:
            market_data (pd.DataFrame): Données de marché
            news_data (pd.DataFrame): Données de nouvelles
            social_data (pd.DataFrame): Données de médias sociaux
            sentiment_data (pd.DataFrame): Données de sentiment
            split_ratio (float): Ratio de division train/test
            **kwargs: Arguments supplémentaires

        Returns:
            tuple: (données d'entraînement, données de test)
        """
        # Si aucune donnée n'est fournie, générer des données synthétiques
        if market_data is None:
            market_data = self.data_integrator.generate_synthetic_data(
                n_samples=100, trend="random", volatility=0.02, with_sentiment=True
            )
            logger.info(f"Données synthétiques générées: {len(market_data)} points")

        # Prétraiter les données de marché
        processed_data = self.data_integrator.preprocess_market_data(market_data)

        # Intégrer les données de sentiment si disponibles
        if sentiment_data is not None:
            processed_data = self.data_integrator.integrate_sentiment_data(
                processed_data, sentiment_data
            )

        # Diviser les données en ensembles d'entraînement et de test
        train_size = int(len(processed_data) * split_ratio)
        train_data = processed_data[:train_size]
        test_data = processed_data[train_size:]

        logger.info(
            f"Données intégrées: {len(train_data)} points d'entraînement, {len(test_data)} points de test"
        )

        return train_data, test_data
