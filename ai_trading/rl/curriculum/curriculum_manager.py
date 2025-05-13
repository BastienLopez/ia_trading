"""
Gestionnaire de curriculum learning pour l'apprentissage progressif.

Ce module implémente un système de curriculum learning qui :
- Augmente progressivement la difficulté des tâches
- Adapte les récompenses selon le niveau
- Gère la progression de l'agent
"""

import logging
from typing import Dict

import numpy as np

from ..trading_environment import TradingEnvironment

logger = logging.getLogger(__name__)


class CurriculumStage:
    """Représente un niveau dans le curriculum."""

    def __init__(
        self,
        name: str,
        difficulty: float,
        reward_scale: float,
        market_volatility: float,
        trading_fee: float,
        slippage: float,
        min_performance: float,
    ):
        """
        Initialise un niveau du curriculum.

        Args:
            name: Nom du niveau
            difficulty: Difficulté (0-1)
            reward_scale: Facteur d'échelle des récompenses
            market_volatility: Volatilité du marché pour ce niveau
            trading_fee: Frais de trading
            slippage: Slippage simulé
            min_performance: Performance minimale pour passer au niveau suivant
        """
        self.name = name
        self.difficulty = difficulty
        self.reward_scale = reward_scale
        self.market_volatility = market_volatility
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.min_performance = min_performance


class CurriculumManager:
    """Gestionnaire de curriculum learning."""

    def __init__(self, env: TradingEnvironment):
        """
        Initialise le gestionnaire de curriculum.

        Args:
            env: Environnement de trading
        """
        self.env = env
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        self.performance_history = []

        # Définir les niveaux du curriculum
        self.stages = [
            CurriculumStage(
                name="Débutant",
                difficulty=0.2,
                reward_scale=2.0,
                market_volatility=0.5,
                trading_fee=0.0001,
                slippage=0.0001,
                min_performance=0.1,
            ),
            CurriculumStage(
                name="Intermédiaire",
                difficulty=0.5,
                reward_scale=1.5,
                market_volatility=0.8,
                trading_fee=0.001,
                slippage=0.001,
                min_performance=0.2,
            ),
            CurriculumStage(
                name="Avancé",
                difficulty=0.8,
                reward_scale=1.2,
                market_volatility=1.0,
                trading_fee=0.002,
                slippage=0.002,
                min_performance=0.3,
            ),
            CurriculumStage(
                name="Expert",
                difficulty=1.0,
                reward_scale=1.0,
                market_volatility=1.2,
                trading_fee=0.003,
                slippage=0.003,
                min_performance=0.4,
            ),
        ]

    @property
    def current_stage(self) -> CurriculumStage:
        """Retourne le niveau actuel."""
        return self.stages[self.current_stage_idx]

    def adapt_environment(self) -> None:
        """Adapte l'environnement au niveau actuel."""
        stage = self.current_stage

        # Adapter les paramètres de l'environnement
        self.env.trading_fee = stage.trading_fee
        self.env.slippage = stage.slippage

        # Adapter la volatilité du marché (si applicable)
        if hasattr(self.env, "market_volatility"):
            self.env.market_volatility = stage.market_volatility

        logger.info(f"Environnement adapté au niveau {stage.name}")

    def scale_reward(self, reward: float) -> float:
        """
        Adapte la récompense selon le niveau actuel.

        Args:
            reward: Récompense originale

        Returns:
            float: Récompense adaptée
        """
        return reward * self.current_stage.reward_scale

    def update_progress(self, episode_return: float) -> bool:
        """
        Met à jour la progression et vérifie si on peut passer au niveau suivant.

        Args:
            episode_return: Retour de l'épisode

        Returns:
            bool: True si passage au niveau suivant
        """
        self.episodes_in_stage += 1
        self.performance_history.append(episode_return)

        # Garder uniquement les 10 derniers épisodes
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]

        # Vérifier si on peut passer au niveau suivant
        if self.episodes_in_stage >= 10:
            avg_performance = np.mean(self.performance_history)
            if avg_performance >= self.current_stage.min_performance:
                if self.current_stage_idx < len(self.stages) - 1:
                    self.current_stage_idx += 1
                    self.episodes_in_stage = 0
                    self.performance_history = []
                    self.adapt_environment()
                    logger.info(f"Passage au niveau {self.current_stage.name}")
                    return True

        return False

    def get_progress_info(self) -> Dict:
        """
        Retourne les informations sur la progression.

        Returns:
            Dict: Informations sur la progression
        """
        return {
            "stage_name": self.current_stage.name,
            "stage_difficulty": self.current_stage.difficulty,
            "episodes_in_stage": self.episodes_in_stage,
            "avg_performance": (
                np.mean(self.performance_history) if self.performance_history else 0
            ),
            "min_performance_required": self.current_stage.min_performance,
        }

    def reset(self) -> None:
        """Réinitialise le curriculum au niveau débutant."""
        self.current_stage_idx = 0
        self.episodes_in_stage = 0
        self.performance_history = []
        self.adapt_environment()
        logger.info("Curriculum réinitialisé au niveau débutant")
