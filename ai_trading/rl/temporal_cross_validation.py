import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class TemporalCrossValidator:
    """
    Classe pour effectuer une validation croisée temporelle sur les données de trading.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
        train_size: float = 0.7,
        test_size: float = 0.2,
        gap: int = 0,
        shuffle: bool = False,
    ):
        """
        Initialise le validateur croisé temporel.

        Args:
            data: DataFrame contenant les données de marché
            n_splits: Nombre de splits pour la validation croisée
            train_size: Proportion des données pour l'entraînement
            test_size: Proportion des données pour le test
            gap: Nombre de périodes entre l'entraînement et le test
            shuffle: Si True, mélange les données avant de les diviser
        """
        self.df = data
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.shuffle = shuffle

        # Vérifier que les tailles sont valides
        assert (
            train_size + test_size <= 1.0
        ), "La somme de train_size et test_size doit être <= 1"
        assert n_splits > 0, "n_splits doit être > 0"

    def split(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Génère les splits pour la validation croisée temporelle.

        Returns:
            Liste de tuples (train_data, test_data)
        """
        splits = []
        total_size = len(self.df)
        train_samples = int(total_size * self.train_size)
        test_samples = int(total_size * self.test_size)

        for i in range(self.n_splits):
            # Calculer les indices de début et de fin pour chaque split
            start_idx = i * (total_size // self.n_splits)
            train_end_idx = start_idx + train_samples
            test_start_idx = train_end_idx + self.gap
            test_end_idx = test_start_idx + test_samples

            # Vérifier que nous ne dépassons pas la taille du DataFrame
            if test_end_idx > total_size:
                continue

            # Extraire les données d'entraînement et de test
            train_data = self.df.iloc[start_idx:train_end_idx]
            test_data = self.df.iloc[test_start_idx:test_end_idx]

            if self.shuffle:
                train_data = train_data.sample(frac=1)
                test_data = test_data.sample(frac=1)

            splits.append((train_data, test_data))

        return splits

    def evaluate(
        self,
        agent,
        env_class,
        metrics: List[str] = ["sharpe_ratio", "max_drawdown", "profit_pct"],
        **env_kwargs,
    ) -> Dict[str, List[float]]:
        """
        Évalue l'agent sur tous les splits de validation croisée.

        Args:
            agent: Agent RL à évaluer
            env_class: Classe de l'environnement de trading
            metrics: Liste des métriques à calculer
            **env_kwargs: Arguments supplémentaires pour l'environnement

        Returns:
            Dictionnaire contenant les métriques pour chaque split
        """
        results = {metric: [] for metric in metrics}
        splits = self.split()

        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Évaluation sur le split {i+1}/{len(splits)}")

            # Créer et entraîner l'agent sur les données d'entraînement
            train_env = env_class(df=train_data, **env_kwargs)
            agent.train(train_env)

            # Évaluer sur les données de test
            test_env = env_class(df=test_data, **env_kwargs)
            test_results = agent.evaluate(test_env)

            # Enregistrer les métriques
            for metric in metrics:
                if metric in test_results:
                    results[metric].append(test_results[metric])

        return results
