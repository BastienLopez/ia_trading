import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


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

    def train(self, df, total_timesteps=100000, save_path=None):
        """
        Entraîner l'agent sur les données historiques
        """
        # Créer l'environnement de trading
        env = DummyVecEnv([lambda: CryptoTradingEnv(df)])

        # Initialiser le modèle PPO
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

        # Callback pour suivre les performances
        callback = TradingCallback()

        # Entraîner le modèle
        model.learn(total_timesteps=total_timesteps, callback=callback)

        # Sauvegarder le modèle
        if save_path:
            model.save(save_path)

        self.model = model

        # Retourner les métriques d'entraînement
        return {
            "rewards": callback.rewards,
            "portfolio_values": callback.portfolio_values,
        }

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
