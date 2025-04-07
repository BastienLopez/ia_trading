import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from ai_trading.rl.trading_environment import TradingEnvironment
import logging
from ai_trading.rl.data_integration import RLDataIntegrator
from ai_trading.rl.train import train_agent, TrainingMonitor
from ai_trading.rl.evaluation import evaluate_agent, PerformanceVisualizer, PerformanceMetrics


class TradingCallback(BaseCallback):
    """
    Callback pour suivre les performances pendant l'entraînement
    """

    def __init__(self, verbose=0):
        super(TradingCallback, self).__init__(verbose)
        self.rewards = []
        self.portfolio_values = []

    def _on_step(self):
        self.rewards.append(self.locals["rewards"][0])
        if hasattr(self.model.env.envs[0].unwrapped, "portfolio_value"):
            self.portfolio_values.append(
                self.model.env.envs[0].unwrapped.portfolio_value
            )
        return True


class CryptoTradingEnv(gym.Env):
    """
    Environnement de trading pour l'apprentissage par renforcement
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, initial_balance=10000, transaction_fee=0.001):
        super(CryptoTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

        # Actions: 0=Attendre, 1=Acheter, 2=Vendre
        self.action_space = spaces.Discrete(3)

        # État: [RSI, MACD, Signal, Hist, EMA9, EMA21, Tenkan, Kijun, Senkou A, Senkou B, BB Top, BB Bottom, Volume Ratio, Position]
        self.observation_space = spaces.Box(
            low=np.array([0, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array(
                [
                    100,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    1,
                    1,
                ]
            ),
            dtype=np.float32,
        )

        self.reset()

    def _next_observation(self):
        """
        Récupérer l'observation suivante (état du marché)
        """
        frame = np.array(
            [
                self.df.loc[self.current_step, "rsi"] / 100.0,
                self.df.loc[self.current_step, "macd"],
                self.df.loc[self.current_step, "signal_line"],
                self.df.loc[self.current_step, "hist_line"],
                self.df.loc[self.current_step, "ema9"],
                self.df.loc[self.current_step, "ema21"],
                self.df.loc[self.current_step, "tenkan"],
                self.df.loc[self.current_step, "kijun"],
                self.df.loc[self.current_step, "senkou_span_a"],
                self.df.loc[self.current_step, "senkou_span_b"],
                self.df.loc[self.current_step, "bb_upper"],
                self.df.loc[self.current_step, "bb_lower"],
                self.df.loc[self.current_step, "volume_ratio"],
                1.0 if self.current_position > 0 else 0.0,
            ]
        )

        return frame

    def _take_action(self, action):
        """
        Exécuter l'action de trading
        """
        current_price = self.df.loc[self.current_step, "close"]

        if action == 1:  # Acheter
            if self.current_position == 0:  # Si on n'a pas de position
                # Calculer combien de crypto on peut acheter
                max_crypto = self.balance / current_price * (1 - self.transaction_fee)
                self.crypto_held = max_crypto
                self.balance = 0
                self.current_position = 1
                self.entry_price = current_price

        elif action == 2:  # Vendre
            if self.current_position == 1:  # Si on a une position
                # Calculer la valeur de la vente
                sale_amount = (
                    self.crypto_held * current_price * (1 - self.transaction_fee)
                )
                self.balance = sale_amount
                self.crypto_held = 0
                self.current_position = 0

                # Calculer le profit/perte relatif
                profit_pct = (current_price - self.entry_price) / self.entry_price
                self.trades.append(profit_pct)

        # Mettre à jour la valeur du portefeuille
        self.portfolio_value = self.balance + self.crypto_held * current_price

    def step(self, action):
        """
        Avancer d'un pas dans l'environnement
        """
        # Exécuter l'action
        self._take_action(action)

        # Avancer dans le temps
        self.current_step += 1

        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.df) - 1

        # Calculer la récompense (variation de la valeur du portefeuille)
        reward = 0
        if self.previous_portfolio_value > 0:
            reward = (
                self.portfolio_value - self.previous_portfolio_value
            ) / self.previous_portfolio_value

        self.previous_portfolio_value = self.portfolio_value

        # Obtenir la nouvelle observation
        obs = self._next_observation()

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        """
        Réinitialiser l'environnement
        """
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.current_step = 0
        self.current_position = 0
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        self.trades = []
        self.entry_price = 0

        return self._next_observation(), {}

    def render(self, mode="human"):
        """
        Afficher l'état actuel
        """
        profit = self.portfolio_value - self.initial_balance
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Profit/Loss: ${profit:.2f}")
        print(f"Position: {'LONG' if self.current_position == 1 else 'NONE'}")
        return


class RLAgent:
    """
    Agent d'apprentissage par renforcement pour le trading
    """

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None

    def train(self, episodes=100, batch_size=32, update_target_every=5, 
              save_path=None, visualize=True, checkpoint_interval=10,
              early_stopping=None, max_steps_per_episode=None,
              use_tensorboard=False, tensorboard_log_dir='./logs'):
        """
        Entraîne l'agent sur l'environnement avec des fonctionnalités avancées.
        
        Args:
            episodes (int): Nombre d'épisodes d'entraînement
            batch_size (int): Taille du batch pour l'entraînement
            update_target_every (int): Fréquence de mise à jour du modèle cible
            save_path (str, optional): Chemin pour sauvegarder le modèle
            visualize (bool): Si True, génère des visualisations
            checkpoint_interval (int): Intervalle d'épisodes entre les sauvegardes
            early_stopping (dict, optional): Paramètres pour l'arrêt anticipé
                - 'patience': Nombre d'épisodes sans amélioration
                - 'min_delta': Amélioration minimale requise
                - 'metric': Métrique à surveiller ('reward', 'portfolio_value', 'returns')
            max_steps_per_episode (int, optional): Nombre maximum d'étapes par épisode
            use_tensorboard (bool): Si True, utilise TensorBoard pour le suivi
            tensorboard_log_dir (str): Répertoire pour les logs TensorBoard
            
        Returns:
            dict: Historique d'entraînement
        """
        if not self.model or not self.environment:
            raise ValueError("Le modèle et l'environnement doivent être créés avant l'entraînement")
        
        logger.info("Démarrage de l'entraînement de l'agent RL...")
        
        # Configurer l'arrêt anticipé
        es_config = None
        if early_stopping:
            es_config = {
                'patience': early_stopping.get('patience', 10),
                'min_delta': early_stopping.get('min_delta', 0.0),
                'metric': early_stopping.get('metric', 'returns')
            }
            logger.info(f"Arrêt anticipé configuré: patience={es_config['patience']}, "
                       f"min_delta={es_config['min_delta']}, metric={es_config['metric']}")
        
        # Exécuter l'entraînement
        history = train_agent(
            agent=self.model,
            env=self.environment,
            episodes=episodes,
            batch_size=batch_size,
            update_target_every=update_target_every,
            save_path=save_path,
            visualize=visualize,
            checkpoint_interval=checkpoint_interval,
            early_stopping=es_config,
            max_steps_per_episode=max_steps_per_episode,
            use_tensorboard=use_tensorboard,
            tensorboard_log_dir=tensorboard_log_dir
        )
        
        logger.info("Entraînement terminé")
        
        # Stocker l'historique d'entraînement
        self.training_history = history
        
        return history

    def load(self, model_path):
        """
        Charger un modèle pré-entraîné
        """
        self.model = PPO.load(model_path)

    def predict(self, observation):
        """
        Prédire une action à partir d'une observation
        """
        if self.model is None:
            raise ValueError(
                "Le modèle n'est pas chargé. Utilisez load() ou train() d'abord."
            )

        action, _ = self.model.predict(observation)
        return action

    def backtest(self, df):
        """
        Tester le modèle sur des données historiques
        """
        if self.model is None:
            raise ValueError(
                "Le modèle n'est pas chargé. Utilisez load() ou train() d'abord."
            )

        # Créer l'environnement de backtest
        env = CryptoTradingEnv(df)

        # Variables de suivi
        observations = []
        actions = []
        rewards = []
        portfolio_values = []
        positions = []

        # Initialisation
        obs, _ = env.reset()
        done = False

        # Exécuter le backtest
        while not done:
            # Prédire l'action
            action, _ = self.model.predict(obs)

            # Exécuter l'action
            next_obs, reward, done, _, _ = env.step(action)

            # Enregistrer les résultats
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            portfolio_values.append(env.portfolio_value)
            positions.append(env.current_position)

            # Passer à l'état suivant
            obs = next_obs

        # Calculer les métriques de performance
        initial_value = df.iloc[0]["close"] * (
            env.initial_balance / df.iloc[0]["close"]
        )
        final_portfolio = env.portfolio_value
        bh_final_value = df.iloc[-1]["close"] * (
            env.initial_balance / df.iloc[0]["close"]
        )

        profit_pct = (final_portfolio - env.initial_balance) / env.initial_balance * 100
        bh_profit_pct = (
            (bh_final_value - env.initial_balance) / env.initial_balance * 100
        )

        sharpe_ratio = np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0

        # Créer un DataFrame pour l'analyse
        results = pd.DataFrame(
            {
                "close": df["close"][1:],
                "action": actions,
                "reward": rewards,
                "portfolio_value": portfolio_values,
                "position": positions,
            }
        )

        return {
            "results": results,
            "profit_pct": profit_pct,
            "bh_profit_pct": bh_profit_pct,
            "sharpe_ratio": sharpe_ratio,
            "trades": env.trades,
            "final_portfolio": final_portfolio,
            "buy_hold_value": bh_final_value,
        }


# Configuration du logger
logger = logging.getLogger('RLAgent')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class RLTradingSystem:
    """Interface principale pour le système de trading par RL."""
    
    def __init__(self, config=None):
        """
        Initialise le système de trading RL.
        
        Args:
            config (dict, optional): Configuration du système
        """
        self.config = config or {}
        self.agent = None
        self.environment = None
        
        logger.info("Système de trading RL initialisé")
    
    def create_environment(self, data, initial_balance=10000, transaction_fee=0.001, window_size=5):
        """
        Crée l'environnement de trading.
        
        Args:
            data (DataFrame): Données historiques de prix
            initial_balance (float): Solde initial
            transaction_fee (float): Frais de transaction
            window_size (int): Taille de la fenêtre d'observation
            
        Returns:
            TradingEnvironment: L'environnement créé
        """
        self.environment = TradingEnvironment(
            df=data,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            window_size=window_size
        )
        
        logger.info(f"Environnement de trading créé avec {len(data)} points de données")
        
        return self.environment
    
    def test_random_strategy(self, num_episodes=1):
        """
        Teste une stratégie aléatoire dans l'environnement.
        
        Args:
            num_episodes (int): Nombre d'épisodes à exécuter
            
        Returns:
            list: Historique des valeurs de portefeuille pour chaque épisode
        """
        if self.environment is None:
            raise ValueError("L'environnement doit être créé avant de tester une stratégie")
        
        results = []
        
        for episode in range(num_episodes):
            obs = self.environment.reset()
            done = False
            episode_rewards = []
            
            while not done:
                action = np.random.randint(0, 3)  # Action aléatoire
                obs, reward, done, info = self.environment.step(action)
                episode_rewards.append(reward)
            
            final_value = self.environment.get_portfolio_value()
            returns = (final_value / self.environment.initial_balance) - 1
            
            results.append({
                'episode': episode,
                'final_value': final_value,
                'returns': returns,
                'avg_reward': np.mean(episode_rewards),
                'portfolio_history': self.environment.get_portfolio_history()
            })
            
            logger.info(f"Épisode {episode}: Valeur finale=${final_value:.2f}, "
                       f"Rendement={returns*100:.2f}%")
        
        return results

    def integrate_data(self, market_data, sentiment_data=None, window_size=5, test_split=0.2):
        """
        Intègre les données de marché et de sentiment pour l'apprentissage par renforcement.
        
        Args:
            market_data (DataFrame): Données de marché prétraitées
            sentiment_data (DataFrame, optional): Données de sentiment
            window_size (int): Taille de la fenêtre d'observation
            test_split (float): Proportion des données à utiliser pour le test
            
        Returns:
            tuple: (train_data, test_data) DataFrames prêts pour l'RL
        """
        integrator = RLDataIntegrator()
        return integrator.integrate_data(
            market_data=market_data,
            sentiment_data=sentiment_data,
            window_size=window_size,
            test_split=test_split
        )

    def evaluate(self, test_data=None, num_episodes=1, visualize=True, save_dir='results/evaluation'):
        """
        Évalue l'agent sur des données de test.
        
        Args:
            test_data (DataFrame, optional): Données de test. Si None, utilise l'environnement actuel.
            num_episodes (int): Nombre d'épisodes d'évaluation
            visualize (bool): Si True, génère des visualisations
            save_dir (str): Répertoire pour sauvegarder les visualisations
            
        Returns:
            dict: Résultats de l'évaluation
        """
        if not self.agent:
            raise ValueError("L'agent doit être créé avant l'évaluation")
        
        # Créer un nouvel environnement avec les données de test si fournies
        if test_data is not None:
            test_env = self.create_environment(
                data=test_data,
                initial_balance=self.environment.initial_balance,
                transaction_fee=self.environment.transaction_fee,
                window_size=self.environment.window_size
            )
        else:
            test_env = self.environment
        
        # Évaluer l'agent
        results = evaluate_agent(self.agent, test_env, num_episodes=num_episodes)
        
        # Visualiser les résultats si demandé
        if visualize:
            visualizer = PerformanceVisualizer(save_dir=save_dir)
            visualizer.create_performance_dashboard(
                results=results,
                dates=test_data.index if test_data is not None else None,
                actions=results['actions'],
                trades=results['trades']
            )
        
        return results
