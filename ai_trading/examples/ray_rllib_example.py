#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de Ray RLlib pour l'entraînement distribué d'un agent RL sur un environnement de trading.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from ai_trading.utils.ray_rllib_optimizer import (
    RayRLlibOptimizer,
    optimize_rl_with_ray
)

# Définition d'un environnement simplifié de trading
class SimpleTradingEnv(gym.Env):
    """
    Environnement simplifié de trading pour démontrer l'utilisation de Ray RLlib.
    L'agent peut acheter, vendre ou tenir une position sur un actif.
    """
    
    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        
        # Charger les données ou générer des données factices
        if "data_path" in config and os.path.exists(config["data_path"]):
            self.data = pd.read_csv(config["data_path"])
        else:
            # Générer des données factices
            self.data = self._generate_dummy_data(
                n_steps=config.get("n_steps", 1000),
                volatility=config.get("volatility", 0.01)
            )
        
        # S'assurer que nous avons suffisamment de données pour la fenêtre
        if len(self.data) < 20:
            raise ValueError("Pas assez de données pour l'environnement de trading")
        
        # Paramètres de l'environnement
        self.window_size = config.get("window_size", 10)
        self.initial_balance = config.get("initial_balance", 10000.0)
        self.transaction_fee_percent = config.get("transaction_fee_percent", 0.001)
        
        # Définir l'espace d'observation (fenêtre des prix + indicateurs + état du portefeuille)
        self.obs_features = 5  # OHLCV
        self.obs_dim = self.window_size * self.obs_features + 2  # +2 pour le solde et la position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        
        # Définir l'espace d'action (0: tenir, 1: acheter, 2: vendre)
        self.action_space = spaces.Discrete(3)
        
        # État interne
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.returns = []
        self.nav_history = [self.initial_balance]
        self.prev_action = 0
        
        # Normalisation
        self.price_scaler = None
        self.volume_scaler = None
        
        # Traiter les données dès l'initialisation pour préparer les scalers
        self._process_data()
        
        # Préparer une observation vide (zéros) pour les cas où nous atteignons la fin des données
        self.dummy_observation = np.zeros(self.obs_dim, dtype=np.float32)
    
    def _generate_dummy_data(self, n_steps=1000, volatility=0.01):
        """Génère des données de prix factices."""
        # Générer des prix suivant un mouvement brownien
        returns = np.random.normal(0, volatility, n_steps)
        price = 100.0  # Prix initial
        prices = [price]
        
        for r in returns:
            price *= (1 + r)
            prices.append(price)
        
        prices = np.array(prices)
        
        # Créer OHLCV
        df = pd.DataFrame()
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(prices[0])
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.003, n_steps+1))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.003, n_steps+1))
        df['volume'] = np.random.lognormal(10, 1, n_steps+1)
        
        return df
    
    def _process_data(self):
        """Normalise les données."""
        if self.price_scaler is None:
            # Calculer les statistiques pour la normalisation
            price_mean = self.data.iloc[:, 0:4].mean().mean()
            price_std = self.data.iloc[:, 0:4].std().mean()
            volume_mean = self.data['volume'].mean()
            volume_std = self.data['volume'].std()
            
            self.price_scaler = {"mean": price_mean, "std": price_std}
            self.volume_scaler = {"mean": volume_mean, "std": volume_std}
    
    def _calculate_reward(self, action):
        """Calcule la récompense pour l'action prise."""
        # Rendement de l'actif
        current_price = self.data.iloc[self.current_step]['close']
        prev_price = self.data.iloc[self.current_step - 1]['close'] if self.current_step > 0 else current_price
        asset_return = (current_price - prev_price) / prev_price
        
        # Récompense basée sur le changement de valeur nette du portefeuille
        old_nav = self.nav_history[-1]
        new_nav = self.balance + self.position * current_price
        
        # Appliquer des frais de transaction si l'agent a acheté ou vendu
        if action == 1 and self.position <= 0:  # Achat
            new_nav -= abs(current_price * self.transaction_fee_percent)
        elif action == 2 and self.position >= 0:  # Vendre
            new_nav -= abs(current_price * self.transaction_fee_percent)
        
        portfolio_return = (new_nav - old_nav) / old_nav if old_nav > 0 else 0
        
        # Mettre à jour l'historique de la VNI
        self.nav_history.append(new_nav)
        
        # Récompense = rendement du portefeuille
        reward = portfolio_return
        
        # Ajouter une pénalité pour les transactions trop fréquentes
        if (action == 1 and self.prev_action == 2) or (action == 2 and self.prev_action == 1):
            reward -= 0.001  # Pénalité pour "overtrading"
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement."""
        super().reset(seed=seed)
        
        # Traiter les données si ce n'est pas déjà fait
        if self.price_scaler is None:
            self._process_data()
        
        # Réinitialiser l'état
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = self.window_size
        self.nav_history = [self.initial_balance]
        self.prev_action = 0  # Pas de position initiale
        
        # Obtenir l'observation initiale
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def _get_observation(self):
        """Construit l'observation à partir de l'état actuel."""
        # Vérifier si nous sommes à l'intérieur des limites des données
        if self.current_step >= len(self.data) or self.current_step < self.window_size:
            return self.dummy_observation
            
        # Extraire la fenêtre courante des données
        frame = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Vérifier que le frame a la bonne taille
        if len(frame) != self.window_size:
            # Si nous n'avons pas assez de données, retourner une observation vide
            return self.dummy_observation
        
        # Normaliser les prix et volumes
        normalized_prices = (frame.iloc[:, 0:4].values - self.price_scaler["mean"]) / self.price_scaler["std"]
        normalized_volumes = (frame['volume'].values - self.volume_scaler["mean"]) / self.volume_scaler["std"]
        
        # Convertir le DataFrame en matrice Numpy et aplatir
        prices_feature = normalized_prices.flatten()
        volume_feature = normalized_volumes.flatten()
        
        # Ajouter les informations de portefeuille (balance normalisée et position)
        balance_feature = [self.balance / self.initial_balance]
        position_feature = [self.position]
        
        # Concaténer toutes les features
        observation = np.concatenate([prices_feature, volume_feature, balance_feature, position_feature])
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """
        Exécute une étape dans l'environnement.
        
        Args:
            action: 0 (rien), 1 (acheter), 2 (vendre)
            
        Returns:
            Tuple (observation, reward, done, truncated, info)
        """
        # Vérifier si l'épisode est déjà terminé
        if self.current_step >= len(self.data) - 1:
            return self.dummy_observation, 0.0, True, False, {"terminal": True}
        
        # Obtenir le prix actuel
        current_price = self.data.iloc[self.current_step]['close']
        
        # Exécuter l'action
        if action == 1:  # Acheter
            if self.position <= 0:
                # Calculer la quantité à acheter (utiliser tout le solde)
                max_qty = self.balance / current_price
                qty = max_qty * 0.95  # Garder une petite marge pour les frais
                
                # Calculer les frais
                fee = qty * current_price * self.transaction_fee_percent
                
                # Mettre à jour le solde et la position
                self.balance -= (qty * current_price + fee)
                self.position += qty
        
        elif action == 2:  # Vendre
            if self.position >= 0:
                # Vendre toute la position
                qty = self.position
                
                # Calculer les frais
                fee = qty * current_price * self.transaction_fee_percent
                
                # Mettre à jour le solde et la position
                self.balance += (qty * current_price - fee)
                self.position = 0
        
        # Calculer la récompense
        reward = self._calculate_reward(action)
        
        # Enregistrer l'action pour la prochaine étape
        self.prev_action = action
        
        # Passer à l'étape suivante
        self.current_step += 1
        
        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Préparer l'observation suivante
        observation = self._get_observation()
        
        # Informations supplémentaires
        info = {
            "balance": self.balance,
            "position": self.position,
            "nav": self.nav_history[-1],
            "return": reward
        }
        
        return observation, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Affiche l'état actuel de l'environnement."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Price: {self.data.iloc[self.current_step]['close']:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"NAV: {self.nav_history[-1]:.2f}")
            print(f"Return: {(self.nav_history[-1] / self.initial_balance - 1) * 100:.2f}%")
            print("-" * 50)


def train_agent(
    algorithm="PPO",
    num_workers=2,
    num_gpus=0,
    train_iterations=50,
    save_dir="ray_checkpoints"
):
    """Entraîne un agent RL avec Ray RLlib."""
    # Créer le répertoire de sauvegarde avec un chemin absolu
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration pour les frais de transaction
    env_config = {
        "n_steps": 1000,
        "window_size": 10,
        "initial_balance": 10000.0,
        "transaction_fee_percent": 0.001
    }
    
    # Fonction de création d'environnement
    def env_creator(config):
        return SimpleTradingEnv(config)
    
    # Créer l'optimiseur Ray RLlib
    optimizer = RayRLlibOptimizer(
        env_creator=env_creator,
        algorithm=algorithm,
        num_workers=num_workers,
        num_gpus=num_gpus,
        config_overrides={
            "env_config": env_config,
            # Désactiver complètement la nouvelle API stack pour éviter les problèmes de compatibilité
            "api_stack": {
                "enable_rl_module_and_learner": False,
                "enable_env_runner_and_connector_v2": False
            },
            # Configuration du modèle pour l'ancienne API
            "model": {
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu"
            },
            "lr": 1e-4,
            # Désactiver temporairement la validation pour Ray 2.45.0
            "experimental": {
                "_validate_config": False
            },
            # Désactiver complètement les RLModules
            "_enable_rl_module_api": False
        },
        checkpoint_dir=save_dir
    )
    
    # Entraîner l'agent
    start_time = time.time()
    train_results = optimizer.train(num_iterations=train_iterations)
    train_time = time.time() - start_time
    
    print(f"Entraînement terminé en {train_time:.2f} secondes")
    
    # Sauvegarde le modèle final (on ne fait pas d'évaluation pour éviter les erreurs)
    final_path = os.path.join(save_dir, "final_model")
    optimizer.save_model(final_path)
    print(f"Modèle final sauvegardé dans: {final_path}")
    
    # Nettoyer les ressources
    optimizer.cleanup()
    
    # Retourner les résultats sans évaluation
    return optimizer, train_results, {"message": "Évaluation désactivée pour éviter les erreurs avec Ray 2.45.0"}


def main(args):
    print("=" * 50)
    print(f"Entraînement d'un agent RL avec Ray RLlib - {args.algorithm}")
    print(f"Nombre de workers: {args.num_workers}")
    print(f"Nombre de GPUs: {args.num_gpus}")
    print(f"Nombre d'itérations: {args.iterations}")
    print("=" * 50)
    
    # Entraîner l'agent
    optimizer, train_results, eval_results = train_agent(
        algorithm=args.algorithm,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        train_iterations=args.iterations,
        save_dir=args.save_dir
    )
    
    print("\nExemple Ray RLlib terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exemple d'utilisation de Ray RLlib pour l'entraînement distribué")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC", "DQN"], help="Algorithme à utiliser")
    parser.add_argument("--num_workers", type=int, default=2, help="Nombre de workers parallèles")
    parser.add_argument("--num_gpus", type=float, default=0, help="Nombre de GPUs à utiliser")
    parser.add_argument("--iterations", type=int, default=50, help="Nombre d'itérations d'entraînement")
    parser.add_argument("--save_dir", type=str, default="ray_checkpoints", help="Répertoire pour les checkpoints")
    
    args = parser.parse_args()
    main(args) 