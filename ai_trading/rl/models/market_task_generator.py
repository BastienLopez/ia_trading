"""
Module de génération de tâches pour le meta-learning en trading.

Ce module fournit des outils pour:
- Générer des tâches à partir de données de différents marchés
- Créer des ensembles de support et de requête pour le meta-learning
- Échantillonner des tâches similaires pour l'apprentissage par transfert
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class MarketDataset(Dataset):
    """
    Dataset pour les données de marché avec prise en charge des séquences temporelles.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        targets: Optional[Union[pd.Series, np.ndarray, torch.Tensor]] = None,
        sequence_length: int = 50,
        step_size: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialise le dataset.

        Args:
            data: Données d'entrée (DataFrame, ndarray ou tensor)
            targets: Valeurs cibles (optionnel si inclus dans data)
            sequence_length: Longueur des séquences temporelles
            step_size: Pas entre les séquences consécutives
            transform: Transformation à appliquer aux données
            target_transform: Transformation à appliquer aux cibles
        """
        # Convertir les données en tenseurs torch
        if isinstance(data, pd.DataFrame):
            self.data = torch.tensor(data.values, dtype=torch.float32)
        elif isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = data.float()  # Assurer que c'est un tensor float32

        # Traiter les cibles
        if targets is None:
            # Utiliser la dernière colonne comme cible par défaut
            self.features = self.data[:, :-1]
            self.targets = self.data[:, -1].unsqueeze(1)
        else:
            self.features = self.data
            if isinstance(targets, pd.Series):
                self.targets = torch.tensor(
                    targets.values, dtype=torch.float32
                ).unsqueeze(1)
            elif isinstance(targets, np.ndarray):
                self.targets = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
            else:
                self.targets = targets.float().reshape(-1, 1)

        self.sequence_length = sequence_length
        self.step_size = step_size
        self.transform = transform
        self.target_transform = target_transform

        # Calculer le nombre de séquences disponibles
        self.num_sequences = max(0, len(self.features) - sequence_length + 1)

        # Indices des séquences (avec step_size)
        self.indices = range(0, self.num_sequences, step_size)

    def __len__(self) -> int:
        """Retourne le nombre de séquences."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Récupère une séquence par son index.

        Args:
            idx: Index de la séquence

        Returns:
            Tuple (séquence, cible)
        """
        # Convertir l'index en position de début de séquence
        start_idx = self.indices[idx]
        seq_features = self.features[start_idx : start_idx + self.sequence_length]

        # La cible est la valeur juste après la séquence
        seq_target = self.targets[start_idx + self.sequence_length - 1]

        # Appliquer les transformations si nécessaire
        if self.transform:
            seq_features = self.transform(seq_features)

        if self.target_transform:
            seq_target = self.target_transform(seq_target)

        return seq_features, seq_target


class MarketTaskGenerator:
    """
    Générateur de tâches pour le meta-learning à partir de données de marché.

    Cette classe permet de créer des tâches pour l'apprentissage meta-learning
    où chaque tâche correspond à un marché ou à une période différente.
    """

    def __init__(
        self,
        market_data: Dict[str, pd.DataFrame],
        target_column: str = "close",
        feature_columns: Optional[List[str]] = None,
        sequence_length: int = 50,
        prediction_horizon: int = 1,
        normalization: str = "global",  # "none", "global", "local", "task"
        random_seed: int = 42,
    ):
        """
        Initialise le générateur de tâches.

        Args:
            market_data: Dictionnaire de DataFrames avec les données par marché
            target_column: Nom de la colonne cible
            feature_columns: Liste des colonnes à utiliser comme features
            sequence_length: Longueur des séquences temporelles
            prediction_horizon: Horizon de prédiction (en pas de temps)
            normalization: Stratégie de normalisation des données
            random_seed: Graine pour la génération aléatoire
        """
        self.market_data = market_data
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalization = normalization
        self.random_seed = random_seed

        # Initialiser le générateur aléatoire
        np.random.seed(random_seed)

        # Préparer les données pour chaque marché
        self.prepared_data = {}
        self._prepare_data()

        logger.info(f"Générateur de tâches initialisé avec {len(market_data)} marchés")

    def _prepare_data(self) -> None:
        """
        Prépare les données pour tous les marchés.
        """
        # Calculer les statistiques globales si nécessaire
        if self.normalization == "global":
            all_features = []
            for market_name, df in self.market_data.items():
                features = (
                    df[self.feature_columns]
                    if self.feature_columns
                    else df.drop(columns=[self.target_column])
                )
                all_features.append(features)

            # Concaténer tous les features
            all_features_df = pd.concat(all_features)

            # Calculer les statistiques globales
            self.global_mean = all_features_df.mean()
            self.global_std = all_features_df.std()

            logger.info("Statistiques globales calculées pour la normalisation")

        # Préparer les données pour chaque marché
        for market_name, df in self.market_data.items():
            # Sélectionner les colonnes de features
            if self.feature_columns:
                features = df[self.feature_columns].values
            else:
                features = df.drop(columns=[self.target_column]).values

            # Créer la cible décalée (pour la prédiction)
            target = (
                df[self.target_column]
                .shift(-self.prediction_horizon)
                .values[: -self.prediction_horizon]
            )

            # Ajuster les features en conséquence
            features = features[: -self.prediction_horizon]

            # Normaliser les features
            if self.normalization == "global":
                # Normalisation globale (mêmes stats pour tous les marchés)
                features = (features - self.global_mean.values) / self.global_std.values
            elif self.normalization == "local":
                # Normalisation locale (stats spécifiques à chaque marché)
                features = (features - np.mean(features, axis=0)) / np.std(
                    features, axis=0
                )

            # Enregistrer dans le dictionnaire
            self.prepared_data[market_name] = {
                "features": features,
                "target": target,
                "length": len(features),
            }

            logger.debug(
                f"Données préparées pour {market_name}: {len(features)} points"
            )

    def generate_task(
        self,
        market_name: Optional[str] = None,
        support_size: int = 20,
        query_size: int = 20,
        task_type: str = "random",  # "random", "consecutive", "regime"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Génère une tâche à partir d'un marché spécifique.

        Args:
            market_name: Nom du marché à utiliser (None = choix aléatoire)
            support_size: Taille de l'ensemble de support
            query_size: Taille de l'ensemble de query
            task_type: Stratégie d'échantillonnage des points

        Returns:
            Tuple (support_x, support_y, query_x, query_y)
        """
        # Choisir un marché aléatoirement si non spécifié
        if market_name is None:
            market_name = np.random.choice(list(self.prepared_data.keys()))
        elif market_name not in self.prepared_data:
            raise ValueError(f"Marché '{market_name}' non disponible")

        market_data = self.prepared_data[market_name]

        # Nombre total de séquences possibles
        n_samples = market_data["length"] - self.sequence_length + 1

        if n_samples < support_size + query_size:
            logger.warning(
                f"Pas assez de données dans {market_name}, ajustement des tailles"
            )
            total_size = min(n_samples, support_size + query_size)
            support_size = total_size // 2
            query_size = total_size - support_size

        # Échantillonner les indices de début selon la stratégie
        if task_type == "consecutive":
            # Échantillonnage consécutif (pour le transfer par périodes)
            start_idx = np.random.randint(0, n_samples - support_size - query_size + 1)
            support_indices = list(range(start_idx, start_idx + support_size))
            query_indices = list(
                range(start_idx + support_size, start_idx + support_size + query_size)
            )

        elif task_type == "regime":
            # Tenter d'identifier des régimes similaires (ex. par volatilité)
            try:
                # Utiliser la volatilité comme indicateur de régime
                volatility = np.std(
                    np.diff(market_data["target"]) / market_data["target"][:-1], axis=0
                )

                # Identifier les périodes par fenêtre
                window_volatility = []
                for i in range(n_samples):
                    if i + self.sequence_length <= len(volatility):
                        window_volatility.append(
                            np.mean(volatility[i : i + self.sequence_length])
                        )
                    else:
                        window_volatility.append(np.mean(volatility[i:]))

                # Classifier par quartiles
                quartiles = np.percentile(window_volatility, [25, 50, 75])
                regimes = np.digitize(window_volatility, quartiles)

                # Choisir un régime
                chosen_regime = np.random.randint(0, 4)
                regime_indices = np.where(regimes == chosen_regime)[0]

                if len(regime_indices) < support_size + query_size:
                    # Pas assez d'indices dans ce régime, revenir à l'échantillonnage aléatoire
                    task_type = "random"
                else:
                    # Mélanger les indices de ce régime
                    np.random.shuffle(regime_indices)
                    support_indices = regime_indices[:support_size]
                    query_indices = regime_indices[
                        support_size : support_size + query_size
                    ]
            except:
                # En cas d'erreur, revenir à l'échantillonnage aléatoire
                logger.warning(
                    "Échec de l'identification des régimes, utilisation de l'échantillonnage aléatoire"
                )
                task_type = "random"

        if task_type == "random":
            # Échantillonnage aléatoire
            all_indices = np.random.choice(
                n_samples, support_size + query_size, replace=False
            )
            support_indices = all_indices[:support_size]
            query_indices = all_indices[support_size:]

        # Créer les séquences d'entrée/sortie
        support_x = []
        support_y = []
        query_x = []
        query_y = []

        # Remplir l'ensemble de support
        for idx in support_indices:
            x_seq = market_data["features"][idx : idx + self.sequence_length]
            y_val = market_data["target"][idx + self.sequence_length - 1]

            support_x.append(x_seq)
            support_y.append(y_val)

        # Remplir l'ensemble de query
        for idx in query_indices:
            x_seq = market_data["features"][idx : idx + self.sequence_length]
            y_val = market_data["target"][idx + self.sequence_length - 1]

            query_x.append(x_seq)
            query_y.append(y_val)

        # Convertir en tenseurs
        support_x = torch.tensor(np.array(support_x), dtype=torch.float32)
        support_y = torch.tensor(np.array(support_y), dtype=torch.float32).reshape(
            -1, 1
        )
        query_x = torch.tensor(np.array(query_x), dtype=torch.float32)
        query_y = torch.tensor(np.array(query_y), dtype=torch.float32).reshape(-1, 1)

        # Normaliser les cibles si task_normalization
        if self.normalization == "task":
            # Normalisation par tâche (aide à généraliser entre différentes échelles de prix)
            support_mean = torch.mean(support_y)
            support_std = torch.std(support_y)

            if support_std == 0:
                support_std = 1.0

            support_y = (support_y - support_mean) / support_std
            query_y = (query_y - support_mean) / support_std

        return support_x, support_y, query_x, query_y

    def generate_multi_market_batch(
        self,
        batch_size: int,
        markets: Optional[List[str]] = None,
        support_size: int = 20,
        query_size: int = 20,
        task_type: str = "random",
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Génère un batch de tâches à partir de plusieurs marchés.

        Args:
            batch_size: Nombre de tâches à générer
            markets: Liste des marchés à utiliser (None = tous)
            support_size: Taille de l'ensemble de support
            query_size: Taille de l'ensemble de query
            task_type: Stratégie d'échantillonnage des points

        Returns:
            Liste de tuples (support_x, support_y, query_x, query_y)
        """
        batch = []

        # Utiliser tous les marchés disponibles si non spécifié
        if markets is None:
            markets = list(self.prepared_data.keys())

        # Vérifier les marchés spécifiés
        markets = [m for m in markets if m in self.prepared_data]
        if not markets:
            raise ValueError("Aucun marché valide spécifié")

        # Générer les tâches
        for _ in range(batch_size):
            # Choisir un marché aléatoirement
            market_name = np.random.choice(markets)

            # Générer une tâche
            task = self.generate_task(market_name, support_size, query_size, task_type)
            batch.append(task)

        return batch

    def create_market_dataloader(
        self,
        market_name: str,
        batch_size: int = 32,
        shuffle: bool = True,
        test_size: float = 0.2,
        use_prepared_data: bool = True,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Crée un DataLoader pour un marché spécifique.

        Args:
            market_name: Nom du marché
            batch_size: Taille des batchs
            shuffle: Si True, mélange les données
            test_size: Proportion des données pour le test (0 = pas de split)
            use_prepared_data: Si True, utilise les données préparées (normalisées)

        Returns:
            Tuple (train_loader, test_loader ou None)
        """
        if market_name not in self.prepared_data:
            raise ValueError(f"Marché '{market_name}' non disponible")

        market_data = self.prepared_data[market_name]

        if use_prepared_data:
            features = market_data["features"]
            target = market_data["target"]
        else:
            # Reprendre les données brutes
            df = self.market_data[market_name]

            if self.feature_columns:
                features = df[self.feature_columns].values
            else:
                features = df.drop(columns=[self.target_column]).values

            target = (
                df[self.target_column]
                .shift(-self.prediction_horizon)
                .values[: -self.prediction_horizon]
            )
            features = features[: -self.prediction_horizon]

        # Créer le dataset avec séquences
        dataset = MarketDataset(
            data=features,
            targets=target,
            sequence_length=self.sequence_length,
            step_size=1,
        )

        # Split train/test si nécessaire
        if test_size > 0:
            test_len = int(len(dataset) * test_size)
            train_len = len(dataset) - test_len

            train_dataset, test_dataset = torch.utils.data.random_split(
                dataset,
                [train_len, test_len],
                generator=torch.Generator().manual_seed(self.random_seed),
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
            )

            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
            )

            return train_loader, test_loader
        else:
            # Pas de split
            train_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
            )

            return train_loader, None
