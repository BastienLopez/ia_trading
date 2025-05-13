"""
Stratégie d'arbitrage statistique (pairs trading).

Ce module implémente une stratégie d'arbitrage statistique qui exploite
les relations statistiques entre paires d'actifs corrélés.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from ai_trading.strategies.arbitrage.base import ArbitrageStrategy

# Configuration du logging
logger = logging.getLogger(__name__)


class StatisticalArbitrageStrategy(ArbitrageStrategy):
    """
    Stratégie d'arbitrage statistique basée sur le pairs trading.

    Cette stratégie identifie des paires d'actifs co-intégrés et exploite
    les déviations temporaires de leur relation d'équilibre.
    """

    def __init__(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
        lookback_period: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        max_position_duration: int = 20,
        min_profit_threshold: float = 0.003,
        transaction_fee: float = 0.001,
        risk_tolerance: float = 0.5,
        max_position_size: float = 1000.0,
        max_positions: int = 5,
        rebalance_frequency: int = 24,  # heures
        z_score_window: int = 20,
    ):
        """
        Initialise une stratégie d'arbitrage statistique.

        Args:
            pairs: Liste de paires d'actifs à surveiller [(asset1, asset2), ...]
            lookback_period: Période d'observation pour le calcul des relations statistiques (jours)
            entry_threshold: Seuil de z-score pour l'entrée en position (2.0 = 2 écarts-types)
            exit_threshold: Seuil de z-score pour la sortie de position (0.5 = 0.5 écart-type)
            max_position_duration: Durée maximale d'une position en jours
            min_profit_threshold: Seuil minimum de profit pour exécuter un arbitrage (0.3% par défaut)
            transaction_fee: Frais de transaction par opération (0.1% par défaut)
            risk_tolerance: Tolérance au risque entre 0 et 1 (0.5 par défaut)
            max_position_size: Taille maximale d'une position en USD
            max_positions: Nombre maximum de positions simultanées
            rebalance_frequency: Fréquence de rééquilibrage des modèles en heures
            z_score_window: Fenêtre de calcul du z-score (jours)
        """
        super().__init__(
            name="Arbitrage Statistique",
            min_profit_threshold=min_profit_threshold,
            transaction_fee=transaction_fee,
            risk_tolerance=risk_tolerance,
            max_position_size=max_position_size,
            max_positions=max_positions,
        )

        self.pairs = pairs or []
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position_duration = max_position_duration
        self.rebalance_frequency = rebalance_frequency
        self.z_score_window = z_score_window

        # Historique des prix pour chaque actif
        self.price_history = {}

        # Modèles de régression pour chaque paire
        self.pair_models = {}

        # Z-scores actuels
        self.current_z_scores = {}

        # Dernière mise à jour des modèles
        self.last_model_update = datetime.now()

        logger.info(
            f"Stratégie d'arbitrage statistique initialisée avec {len(self.pairs)} paires"
        )

    def update_price_history(self, market_data: Dict):
        """
        Met à jour l'historique des prix pour les actifs concernés.

        Args:
            market_data: Dictionnaire avec les prix actuels des actifs
        """
        # Obtenir l'horodatage actuel
        timestamp = datetime.now()

        # Extraire tous les actifs uniques des paires
        all_assets = set()
        for asset1, asset2 in self.pairs:
            all_assets.add(asset1)
            all_assets.add(asset2)

        # Mettre à jour l'historique des prix pour chaque actif
        for asset in all_assets:
            if asset in market_data and "close" in market_data[asset]:
                price = market_data[asset]["close"]

                # Initialiser l'historique si nécessaire
                if asset not in self.price_history:
                    self.price_history[asset] = pd.DataFrame(
                        columns=["timestamp", "price"]
                    )

                # Ajouter le nouveau prix
                new_data = pd.DataFrame(
                    [[timestamp, price]], columns=["timestamp", "price"]
                )
                self.price_history[asset] = pd.concat(
                    [self.price_history[asset], new_data], ignore_index=True
                )

                # Supprimer les données anciennes (garder seulement lookback_period jours)
                max_age = timestamp - pd.Timedelta(days=self.lookback_period)
                self.price_history[asset] = self.price_history[asset][
                    self.price_history[asset]["timestamp"] >= max_age
                ]

    def update_models(self, force: bool = False):
        """
        Met à jour les modèles statistiques pour chaque paire.

        Args:
            force: Si True, force la mise à jour même si la période de rebalance n'est pas écoulée
        """
        current_time = datetime.now()
        time_since_update = (
            current_time - self.last_model_update
        ).total_seconds() / 3600  # en heures

        # Vérifier si une mise à jour est nécessaire
        if not force and time_since_update < self.rebalance_frequency:
            return

        logger.info("Mise à jour des modèles d'arbitrage statistique")

        # Mettre à jour les modèles pour chaque paire
        for asset1, asset2 in self.pairs:
            # Vérifier si les deux actifs ont suffisamment de données
            if asset1 not in self.price_history or asset2 not in self.price_history:
                continue

            df1 = self.price_history[asset1]
            df2 = self.price_history[asset2]

            if len(df1) < 30 or len(df2) < 30:  # Au moins 30 points de données
                continue

            # Fusionner les données sur les horodatages
            merged = pd.merge(df1, df2, on="timestamp", suffixes=("_1", "_2"))

            if len(merged) < 30:
                continue

            # Tester la cointégration
            _, pvalue, _ = coint(merged["price_1"], merged["price_2"])

            # Si la paire est cointégrée (p < 0.05)
            if pvalue < 0.05:
                # Ajuster le modèle de régression linéaire
                X = sm.add_constant(merged["price_1"])
                model = sm.OLS(merged["price_2"], X).fit()

                # Calculer les résidus et le z-score
                merged["residuals"] = model.resid
                merged["z_score"] = self._calculate_z_score(merged["residuals"])

                # Enregistrer le modèle et le z-score actuel
                pair_key = f"{asset1}_{asset2}"
                self.pair_models[pair_key] = {
                    "asset1": asset1,
                    "asset2": asset2,
                    "beta": model.params[1],
                    "alpha": model.params[0],
                    "half_life": self._calculate_half_life(merged["residuals"]),
                    "std_dev": merged["residuals"].std(),
                    "last_update": current_time,
                    "pvalue": pvalue,
                }

                # Stocker le z-score actuel
                if not merged["z_score"].empty:
                    self.current_z_scores[pair_key] = merged["z_score"].iloc[-1]

                logger.debug(
                    f"Modèle mis à jour pour {asset1}/{asset2}: "
                    f"beta={model.params[1]:.4f}, "
                    f"alpha={model.params[0]:.4f}, "
                    f"z_score={self.current_z_scores[pair_key]:.4f}, "
                    f"p-value={pvalue:.4f}"
                )

        # Mettre à jour l'horodatage de la dernière mise à jour
        self.last_model_update = current_time

    def _calculate_z_score(self, residuals: pd.Series) -> pd.Series:
        """
        Calcule le z-score d'une série de résidus.

        Args:
            residuals: Série des résidus

        Returns:
            Z-scores normalisés
        """
        # Utiliser une fenêtre mobile pour le calcul des z-scores
        window = min(self.z_score_window, len(residuals))
        rolling_mean = residuals.rolling(window=window).mean()
        rolling_std = residuals.rolling(window=window).std()
        z_score = (residuals - rolling_mean) / rolling_std

        return z_score

    def _calculate_half_life(self, residuals: pd.Series) -> float:
        """
        Calcule la demi-vie de la série de résidus (vitesse de mean-reversion).

        Args:
            residuals: Série des résidus

        Returns:
            Demi-vie estimée (jours)
        """
        residuals = residuals.dropna()
        if len(residuals) <= 1:
            return 0

        # Calculer la différence laggée
        lagged_residuals = residuals.shift(1)
        delta = residuals - lagged_residuals

        # Régresser delta ~ résidus
        X = sm.add_constant(lagged_residuals.iloc[1:])
        y = delta.iloc[1:]

        model = sm.OLS(y, X).fit()

        # Coefficient de régression
        beta = model.params[1]

        if beta >= 0:
            # Pas de mean-reversion
            return 999

        # Calculer la demi-vie: ln(2) / taux de décroissance
        half_life = -np.log(2) / beta

        return max(1, min(half_life, 365))  # Limiter entre 1 et 365 jours

    def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """
        Trouve les opportunités d'arbitrage statistique dans les données de marché.

        Args:
            market_data: Dictionnaire avec les prix actuels des actifs
                Format attendu: {
                    'BTC': {'close': 40000, 'open': 39800, 'high': 40100, 'low': 39700, 'volume': 100},
                    'ETH': {'close': 3000, 'open': 2950, 'high': 3050, 'low': 2900, 'volume': 500},
                    ...
                }

        Returns:
            Liste d'opportunités d'arbitrage détectées
        """
        opportunities = []

        # Vérifier le format des données
        if not isinstance(market_data, dict):
            logger.error(f"Format market_data invalide: {type(market_data)}")
            return opportunities

        # Mettre à jour l'historique des prix
        self.update_price_history(market_data)

        # Mettre à jour les modèles si nécessaire
        self.update_models()

        # Parcourir chaque paire modélisée
        for pair_key, model in self.pair_models.items():
            asset1 = model["asset1"]
            asset2 = model["asset2"]

            # Vérifier si les deux actifs ont des données de prix actuelles
            if asset1 not in market_data or asset2 not in market_data:
                continue

            if "close" not in market_data[asset1] or "close" not in market_data[asset2]:
                continue

            # Récupérer les prix actuels
            price1 = market_data[asset1]["close"]
            price2 = market_data[asset2]["close"]

            # Calculer le prix théorique de asset2 basé sur le modèle
            theo_price2 = model["alpha"] + model["beta"] * price1

            # Calculer l'écart par rapport au prix théorique
            spread = price2 - theo_price2

            # Calculer le z-score actuel
            z_score = spread / model["std_dev"] if model["std_dev"] > 0 else 0

            # Mettre à jour le z-score actuel
            self.current_z_scores[pair_key] = z_score

            # Vérifier si le z-score dépasse le seuil d'entrée
            if abs(z_score) > self.entry_threshold:
                # Déterminer la direction de la position
                direction = "short_long" if z_score > 0 else "long_short"

                # Direction: short_long = short asset2, long asset1
                # Direction: long_short = long asset2, short asset1

                # Estimer le profit potentiel (basé sur la demi-vie et le z-score)
                expected_return = min(abs(z_score) * 0.01, 0.05)  # Plafonner à 5%

                # Créer une opportunité
                opportunity = {
                    "id": str(uuid.uuid4()),
                    "type": "statistical",
                    "pair_key": pair_key,
                    "asset1": asset1,
                    "asset2": asset2,
                    "price1": price1,
                    "price2": price2,
                    "theo_price2": theo_price2,
                    "z_score": z_score,
                    "direction": direction,
                    "profit_pct": expected_return * 100,  # En pourcentage
                    "timestamp": datetime.now().timestamp(),
                    "beta": model["beta"],
                    "half_life": model["half_life"],
                    "num_transactions": 4,  # 2 pour l'entrée, 2 pour la sortie
                    "confidence": self._calculate_confidence(z_score, model),
                }

                # Ajouter si la différence est suffisante après frais
                if self.is_profitable(opportunity):
                    opportunities.append(opportunity)
                    logger.debug(
                        f"Opportunité statistique détectée: {asset1}/{asset2}, "
                        f"Z-score: {z_score:.2f}, Direction: {direction}, "
                        f"Profit attendu: {expected_return*100:.2f}%"
                    )

        # Trier les opportunités par profit potentiel
        opportunities.sort(key=lambda x: x["profit_pct"], reverse=True)

        # Ajouter ces opportunités à l'historique
        self.historical_opportunities.extend(opportunities)

        return opportunities

    def calculate_profit(self, opportunity: Dict) -> float:
        """
        Calcule le profit attendu pour une opportunité d'arbitrage.

        Args:
            opportunity: Dictionnaire représentant une opportunité d'arbitrage

        Returns:
            Profit attendu en pourcentage
        """
        # Récupérer le profit brut estimé
        profit_pct = opportunity.get("profit_pct", 0)

        # Pour l'arbitrage statistique, on doit tenir compte de l'incertitude
        # sur le timing de la convergence
        confidence = opportunity.get("confidence", 0.5)

        # Ajuster le profit en fonction de la confiance
        adjusted_profit = profit_pct * confidence

        return adjusted_profit

    def _calculate_confidence(self, z_score: float, model: Dict) -> float:
        """
        Calcule un score de confiance pour l'opportunité d'arbitrage.

        Args:
            z_score: Z-score actuel
            model: Modèle statistique de la paire

        Returns:
            Score de confiance entre 0 et 1
        """
        # Plus le z-score est extrême, plus on est confiant que la paire convergera
        z_score_confidence = min(0.9, abs(z_score) / 10.0)

        # Plus la p-value est basse, plus on est confiant dans la cointégration
        pvalue = model.get("pvalue", 1.0)
        pvalue_confidence = max(
            0, 1.0 - pvalue * 10
        )  # 0 pour p>=0.1, 0.5 pour p=0.05, 0.9 pour p=0.01

        # Plus la demi-vie est courte, plus on est confiant dans une convergence rapide
        half_life = model.get("half_life", 100)
        half_life_confidence = max(0.1, min(0.9, 10.0 / half_life))

        # Combiner les différentes confiances
        combined_confidence = (
            z_score_confidence * 0.4
            + pvalue_confidence * 0.3
            + half_life_confidence * 0.3
        )

        return max(0.1, min(0.95, combined_confidence))

    def get_summary(self) -> Dict:
        """
        Fournit un résumé des performances de la stratégie.

        Returns:
            Dictionnaire avec le résumé des performances
        """
        # Récupérer les données de performance
        performance_df = self.get_historical_performance()

        # Calculer les métriques
        total_trades = len(performance_df)

        if total_trades > 0:
            total_profit_pct = performance_df["expected_profit"].sum()
            avg_profit_pct = performance_df["expected_profit"].mean()
            total_profit_abs = performance_df["profit_abs"].sum()

            return {
                "strategy_name": self.name,
                "total_trades": total_trades,
                "active_positions": len(self.positions),
                "total_profit_pct": total_profit_pct,
                "avg_profit_pct": avg_profit_pct,
                "total_profit_abs": total_profit_abs,
                "pairs_monitored": len(self.pair_models),
            }
        else:
            return {
                "strategy_name": self.name,
                "total_trades": 0,
                "active_positions": 0,
                "total_profit_pct": 0,
                "avg_profit_pct": 0,
                "total_profit_abs": 0,
                "pairs_monitored": len(self.pair_models),
            }
