from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class MarketRegimeClassifier:
    """
    Classe pour classifier les différents régimes de marché basés sur la volatilité,
    le momentum et le volume.
    """

    def __init__(self, n_regimes: int = 4):
        """
        Initialise le classifieur de régimes de marché.

        Args:
            n_regimes (int): Nombre de régimes de marché à identifier
        """
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=42)
        self.is_fitted = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrait les caractéristiques pertinentes pour l'identification des régimes.

        Args:
            df (pd.DataFrame): DataFrame avec les données de marché

        Returns:
            pd.DataFrame: DataFrame avec les caractéristiques extraites
        """
        features = pd.DataFrame(index=df.index)

        # Volatilité (ATR sur 14 périodes normalisé)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift(1))
        low_close = np.abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features["volatility"] = atr / df["close"]

        # Momentum (RSI)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features["momentum"] = 100 - (100 / (1 + rs))

        # Volume relatif
        features["rel_volume"] = df["volume"] / df["volume"].rolling(20).mean()

        # Direction du prix (tendance)
        features["trend"] = df["close"].pct_change(20)

        return features.dropna()

    def fit(self, df: pd.DataFrame) -> "MarketRegimeClassifier":
        """
        Entraîne le classifieur sur les données historiques.

        Args:
            df (pd.DataFrame): DataFrame avec les données de marché

        Returns:
            MarketRegimeClassifier: self
        """
        features = self.extract_features(df)
        self.model.fit(features)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prédit les régimes de marché pour les données fournies.

        Args:
            df (pd.DataFrame): DataFrame avec les données de marché

        Returns:
            np.ndarray: Tableau des régimes de marché prédits
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Le modèle doit être entraîné avant de faire des prédictions."
            )

        features = self.extract_features(df)
        return self.model.predict(features)

    def get_regime_characteristics(
        self, df: pd.DataFrame
    ) -> Dict[int, Dict[str, float]]:
        """
        Calcule les caractéristiques moyennes de chaque régime de marché.

        Args:
            df (pd.DataFrame): DataFrame avec les données de marché

        Returns:
            Dict[int, Dict[str, float]]: Dictionnaire avec les caractéristiques moyennes par régime
        """
        features = self.extract_features(df)
        regimes = self.predict(df)

        regime_stats = {}
        for regime in range(self.n_regimes):
            regime_data = features[regimes == regime]
            regime_stats[regime] = {
                "volatility": regime_data["volatility"].mean(),
                "momentum": regime_data["momentum"].mean(),
                "volume": regime_data["rel_volume"].mean(),
                "trend": regime_data["trend"].mean(),
                "size": len(regime_data),
            }

        return regime_stats


class RobustnessEvaluator:
    """
    Évalue la robustesse d'une stratégie de trading sous différentes conditions de marché.
    """

    def __init__(self, classifier: MarketRegimeClassifier = None):
        """
        Initialise l'évaluateur de robustesse.

        Args:
            classifier (MarketRegimeClassifier, optional): Classifieur de régimes de marché
        """
        self.classifier = classifier or MarketRegimeClassifier()

    def evaluate(
        self,
        df: pd.DataFrame,
        returns: pd.Series,
        metrics: List[Callable[[pd.Series], float]],
    ) -> Dict[int, Dict[str, float]]:
        """
        Évalue la performance de la stratégie dans différents régimes de marché.

        Args:
            df (pd.DataFrame): DataFrame avec les données de marché
            returns (pd.Series): Série temporelle des rendements de la stratégie
            metrics (List[Callable]): Liste des fonctions de métrique à appliquer aux rendements

        Returns:
            Dict[int, Dict[str, float]]: Dictionnaire des métriques par régime de marché
        """
        if not self.classifier.is_fitted:
            self.classifier.fit(df)

        regimes = self.classifier.predict(df)

        # S'assurer que les indices correspondent
        regimes_df = pd.DataFrame(
            {"regime": regimes}, index=self.classifier.extract_features(df).index
        )
        aligned_returns = returns.reindex(regimes_df.index)

        results = {}
        for regime in range(self.classifier.n_regimes):
            regime_returns = aligned_returns[regimes_df["regime"] == regime]

            if len(regime_returns) == 0:
                continue

            regime_metrics = {}
            for metric_func in metrics:
                metric_name = metric_func.__name__
                regime_metrics[metric_name] = metric_func(regime_returns)

            results[regime] = regime_metrics

        return results

    def stress_test(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        scenarios: List[Dict[str, float]],
        metrics: List[Callable[[pd.Series], float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Teste la stratégie sous différents scénarios de stress.

        Args:
            df (pd.DataFrame): DataFrame avec les données de marché
            strategy_func (Callable): Fonction qui prend df en entrée et retourne les rendements
            scenarios (List[Dict]): Liste de scénarios avec des facteurs de modification
            metrics (List[Callable]): Liste des fonctions de métrique à appliquer aux rendements

        Returns:
            Dict[str, Dict[str, float]]: Résultats des tests de stress par scénario
        """
        results = {}

        # Scénario de base
        base_returns = strategy_func(df.copy())
        base_metrics = {}
        for metric_func in metrics:
            metric_name = metric_func.__name__
            base_metrics[metric_name] = metric_func(base_returns)
        results["base"] = base_metrics

        # Scénarios de stress
        for i, scenario in enumerate(scenarios):
            scenario_name = f"scenario_{i}"
            modified_df = df.copy()

            # Appliquer les modifications du scénario
            for column, factor in scenario.items():
                if column in modified_df.columns:
                    modified_df[column] = modified_df[column] * factor

            # Exécuter la stratégie sur les données modifiées
            scenario_returns = strategy_func(modified_df)

            # Calculer les métriques
            scenario_metrics = {}
            for metric_func in metrics:
                metric_name = metric_func.__name__
                scenario_metrics[metric_name] = metric_func(scenario_returns)

            results[scenario_name] = scenario_metrics

        return results
