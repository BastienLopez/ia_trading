#!/usr/bin/env python
"""
Module pour l'entraînement d'agents de trading RL sur plusieurs périodes temporelles
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Union

import pandas as pd
import tensorflow as tf

from ai_trading.data.market_data import MarketDataFetcher as EnhancedMarketDataCollector


# Nous utiliserons une classe mock pour EnhancedSentimentCollector puisqu'elle n'existe pas
class EnhancedSentimentCollector:
    def collect_data(self, coins=None, days=None):
        return pd.DataFrame()  # Retourne un DataFrame vide


from ai_trading.rl.trading_environment import TradingEnvironment


# Mock pour MultiEnvTrading
class MultiEnvTrading(TradingEnvironment):
    def __init__(
        self,
        market_data=None,
        sentiment_data=None,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=20,
        is_training=True,
    ):
        pass


from ai_trading.rl.agents.sac_agent import SACAgent


# Mock for missing classes
class GRUSACAgent(SACAgent):
    pass


# Mock for reward functions
def calculate_sharpe_ratio(returns):
    return 0.0


def calculate_sortino_ratio(returns):
    return 0.0


# Mock for TensorboardCallback
class TensorboardCallback:
    pass


# Configuration du logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INFO_RETOUR_DIR = Path(__file__).parent.parent / "info_retour"
INFO_RETOUR_DIR.mkdir(exist_ok=True)


class MultiPeriodTrainer:
    """
    Classe pour l'entraînement d'agents de trading RL sur plusieurs périodes temporelles
    avec un curriculum d'apprentissage (entraînement progressif des périodes courtes aux longues).
    """

    def __init__(
        self,
        symbol: str,
        days: int,
        periods: List[int],
        agent_type: str = "sac",
        use_gru: bool = False,
        initial_balance: float = 10000.0,
        use_curriculum: bool = True,
        epochs_per_period: int = 5,
        episodes_per_epoch: int = 10,
        validation_ratio: float = 0.2,
        include_sentiment: bool = True,
        sequence_length: int = 10,
        gru_units: int = 64,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        tau: float = 0.005,
        batch_size: int = 64,
        buffer_size: int = 100000,
        reward_scaling: float = 1.0,
        action_type: str = "continuous",
        save_dir: Union[str, Path] = None,
    ):
        """
        Initialise le MultiPeriodTrainer.

        Args:
            symbol: Symbole de la crypto-monnaie (ex: "BTC")
            days: Nombre de jours de données à collecter
            periods: Liste des périodes en minutes (ex: [5, 15, 60, 240, 1440])
            agent_type: Type d'agent à utiliser ("sac" ou "n_step_sac")
            use_gru: Utiliser une architecture GRU pour l'agent
            initial_balance: Solde initial pour l'environnement de trading
            use_curriculum: Utiliser un curriculum d'apprentissage
            epochs_per_period: Nombre d'époques d'entraînement par période
            episodes_per_epoch: Nombre d'épisodes par époque
            validation_ratio: Ratio de données pour la validation
            include_sentiment: Inclure les données de sentiment
            sequence_length: Longueur de séquence pour les modèles GRU
            gru_units: Nombre d'unités GRU pour les modèles GRU
            actor_lr: Taux d'apprentissage pour le réseau actor
            critic_lr: Taux d'apprentissage pour le réseau critic
            tau: Facteur de mise à jour pour les réseaux cibles
            batch_size: Taille du batch pour l'entraînement
            buffer_size: Taille du buffer de replay
            reward_scaling: Facteur d'échelle pour les récompenses
            action_type: Type d'espace d'action ("continuous" ou "discrete")
            save_dir: Répertoire de sauvegarde pour les modèles et les métriques
        """
        self.symbol = symbol
        self.days = days
        self.periods = sorted(periods, reverse=True)  # Du plus grand au plus petit
        self.agent_type = agent_type
        self.use_gru = use_gru
        self.initial_balance = initial_balance
        self.use_curriculum = use_curriculum
        self.epochs_per_period = epochs_per_period
        self.episodes_per_epoch = episodes_per_epoch
        self.validation_ratio = validation_ratio
        self.include_sentiment = include_sentiment
        self.sequence_length = sequence_length
        self.gru_units = gru_units
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.reward_scaling = reward_scaling
        self.action_type = action_type

        # Créer le répertoire de sauvegarde s'il n'existe pas
        self.save_dir = (
            Path(save_dir) if save_dir else INFO_RETOUR_DIR / "models" / "multi_period"
        )
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # Créer des sous-répertoires pour chaque période
        for period in self.periods:
            period_dir = self.save_dir / f"{self.symbol}_{period}min"
            period_dir.mkdir(exist_ok=True)

        # Initialiser les collecteurs de données
        self.market_collector = EnhancedMarketDataCollector()
        self.sentiment_collector = (
            EnhancedSentimentCollector() if include_sentiment else None
        )

        # Créer un writer TensorBoard pour le suivi
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.save_dir / f"{self.symbol}_multi_period_{current_time}"
        self.summary_writer = tf.summary.create_file_writer(str(log_dir))

        # Dictionnaire pour stocker les données par période
        self.data = {}

        # Dictionnaire pour stocker les datasets d'entraînement et de validation
        self.datasets = {}

        # Dictionnaire pour stocker les agents par période
        self.agents = {}

        # Dictionnaire pour stocker les environnements par période
        self.environments = {}

        # Dictionnaire pour stocker les métriques d'entraînement
        self.metrics = {}

    def collect_data(self):
        """Collecte les données de marché et de sentiment."""
        market_data = self.market_collector.collect_data(
            symbol=self.symbol, days=self.days, periods=self.periods
        )

        sentiment_data = None
        if self.include_sentiment and self.sentiment_collector:
            sentiment_data = self.sentiment_collector.collect_data(
                coins=[self.symbol], days=self.days
            )

        return market_data, sentiment_data

    def create_env(self, market_data, sentiment_data=None):
        """Crée un environnement de trading."""
        return MultiEnvTrading(
            market_data=market_data,
            sentiment_data=sentiment_data,
            initial_balance=self.initial_balance,
            transaction_fee=0.001,  # Valeur par défaut
            window_size=20,  # Valeur par défaut
            is_training=True,
        )

    def create_agent(self, env):
        """Crée un agent RL adapté à l'environnement et aux paramètres choisis.

        Args:
            env: L'environnement de trading

        Returns:
            Agent de trading (SACAgent ou GRUSACAgent)
        """
        # Récupérer les informations de l'environnement
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        action_bounds = (
            float(env.action_space.low[0]),
            float(env.action_space.high[0]),
        )

        # Stocker la référence à l'agent pour pouvoir le sauvegarder/charger
        if self.use_gru:
            # Utiliser GRUSACAgent si use_gru est True
            self.current_agent = GRUSACAgent(
                state_size=state_size,
                action_size=action_size,
                action_bounds=action_bounds,
                actor_lr=self.actor_lr,
                critic_lr=self.critic_lr,
                tau=self.tau,
                sequence_length=self.sequence_length,
                gru_units=self.gru_units,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
            )
        else:
            # Utiliser SACAgent standard
            self.current_agent = SACAgent(
                state_size=state_size,
                action_size=action_size,
                action_bounds=action_bounds,
                actor_lr=self.actor_lr,
                critic_lr=self.critic_lr,
                tau=self.tau,
                batch_size=self.batch_size,
                buffer_size=self.buffer_size,
            )

        return self.current_agent

    def prepare_datasets(self, market_data, sentiment_data, validation_ratio):
        """Prépare les datasets d'entraînement et de validation.

        Args:
            market_data: Données de marché
            sentiment_data: Données de sentiment
            validation_ratio: Ratio pour la validation

        Returns:
            tuple: (train_market, train_sentiment, val_market, val_sentiment)
        """
        # Calculer l'index de séparation
        split_idx = int(len(market_data) * (1 - validation_ratio))

        # Séparer les données de marché
        train_market = market_data.iloc[:split_idx]
        val_market = market_data.iloc[split_idx:]

        # Séparer les données de sentiment si disponibles
        train_sentiment = None
        val_sentiment = None
        if sentiment_data is not None:
            train_sentiment = sentiment_data.iloc[:split_idx]
            val_sentiment = sentiment_data.iloc[split_idx:]

        return train_market, train_sentiment, val_market, val_sentiment

    def save_current_agent(self, custom_name=None):
        """Sauvegarde l'agent courant.

        Args:
            custom_name: Nom personnalisé pour la sauvegarde

        Returns:
            str: Chemin du fichier de sauvegarde
        """
        if not hasattr(self, "current_agent") or self.current_agent is None:
            raise ValueError("Aucun agent courant à sauvegarder")

        # Créer le nom du fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = custom_name if custom_name else f"agent_{timestamp}"
        save_path = self.save_dir / f"{name}.h5"

        # Sauvegarder l'agent
        self.current_agent.save_weights(str(save_path))
        return str(save_path)

    def load_agent(self, path):
        """Charge un agent sauvegardé.

        Args:
            path: Chemin vers le fichier de sauvegarde
        """
        if not hasattr(self, "current_agent") or self.current_agent is None:
            raise ValueError("Aucun agent courant pour charger les poids")

        self.current_agent.load_weights(path)
