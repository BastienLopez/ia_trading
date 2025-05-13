"""
Stratégie d'arbitrage de taux de financement (funding rate arbitrage).

Ce module implémente une stratégie d'arbitrage qui exploite les taux de
financement élevés des contrats à terme perpétuels sur les marchés de crypto.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ai_trading.strategies.arbitrage.base import ArbitrageStrategy

# Configuration du logging
logger = logging.getLogger(__name__)


class FundingRateArbitrageStrategy(ArbitrageStrategy):
    """
    Stratégie d'arbitrage basée sur les taux de financement des contrats à terme.

    Cette stratégie exploite les taux de financement élevés des contrats
    à terme perpétuels en prenant des positions opposées sur le spot et
    le marché à terme pour capturer les paiements de financement tout
    en étant neutre par rapport au marché.
    """

    def __init__(
        self,
        min_funding_rate: float = 0.01,  # 1% minimum de taux de financement
        min_profit_threshold: float = 0.002,
        transaction_fee: float = 0.001,
        risk_tolerance: float = 0.5,
        max_position_size: float = 1000.0,
        max_positions: int = 5,
        max_holding_periods: int = 3,  # Nombre maximum de périodes de financement à maintenir
        exchanges: Optional[List[str]] = None,
    ):
        """
        Initialise une stratégie d'arbitrage de taux de financement.

        Args:
            min_funding_rate: Taux de financement minimum pour considérer une opportunité
            min_profit_threshold: Seuil minimum de profit pour exécuter un arbitrage (0.2% par défaut)
            transaction_fee: Frais de transaction par opération (0.1% par défaut)
            risk_tolerance: Tolérance au risque entre 0 et 1 (0.5 par défaut)
            max_position_size: Taille maximale d'une position en USD
            max_positions: Nombre maximum de positions simultanées
            max_holding_periods: Nombre maximum de périodes de financement à maintenir
            exchanges: Liste des plateformes d'échange à surveiller
        """
        super().__init__(
            name="Arbitrage de Taux de Financement",
            min_profit_threshold=min_profit_threshold,
            transaction_fee=transaction_fee,
            risk_tolerance=risk_tolerance,
            max_position_size=max_position_size,
            max_positions=max_positions,
        )

        self.min_funding_rate = min_funding_rate
        self.max_holding_periods = max_holding_periods
        self.exchanges = exchanges or ["binance", "ftx", "bybit", "okex", "deribit"]

        # Historique des taux de financement
        self.funding_rate_history = {}

        logger.info(f"Stratégie d'arbitrage de taux de financement initialisée")
        logger.debug(f"Taux de financement minimum: {min_funding_rate*100:.2f}%")

    def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """
        Trouve les opportunités d'arbitrage de taux de financement dans les données de marché.

        Args:
            market_data: Dictionnaire avec les données de marché
                Format attendu: {
                    'BTC/USDT': {'spot': {'price': 40000}, 'perp': {'price': 40100, 'funding_rate': 0.02}},
                    'ETH/USDT': {'spot': {'price': 3000}, 'perp': {'price': 3010, 'funding_rate': 0.015}},
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

        # Mettre à jour l'historique des taux de financement
        self._update_funding_rate_history(market_data)

        # Parcourir chaque paire d'actifs
        for symbol, data in market_data.items():
            # Vérifier si la paire a des données spot et perpetual
            if not isinstance(data, dict) or "spot" not in data or "perp" not in data:
                continue

            spot_data = data["spot"]
            perp_data = data["perp"]

            # Vérifier si les données nécessaires sont présentes
            if (
                "price" not in spot_data
                or "price" not in perp_data
                or "funding_rate" not in perp_data
                or "next_funding_time" not in perp_data
            ):
                continue

            # Extraire les données
            spot_price = spot_data["price"]
            perp_price = perp_data["price"]
            funding_rate = perp_data["funding_rate"]
            next_funding_time = perp_data["next_funding_time"]
            funding_interval = perp_data.get(
                "funding_interval", 8
            )  # en heures, typiquement 8h
            exchange = perp_data.get("exchange", "unknown")

            # Ignorer si le taux de financement est trop faible
            if abs(funding_rate) < self.min_funding_rate:
                continue

            # Calculer la direction de l'arbitrage
            if funding_rate > 0:
                # Taux positif: short perp, long spot
                direction = "short_perp_long_spot"
                expected_funding = funding_rate
            else:
                # Taux négatif: long perp, short spot
                direction = "long_perp_short_spot"
                expected_funding = -funding_rate

            # Calculer le profit attendu en tenant compte de la durée
            # Nous pouvons tenir la position pendant plusieurs périodes de financement
            expected_periods = min(
                self.max_holding_periods,
                self._predict_funding_persistence(symbol, funding_rate),
            )

            # Profit basé sur le financement attendu sur plusieurs périodes
            expected_profit = expected_funding * expected_periods

            # Calculer la base (différence entre perp et spot)
            basis = perp_price / spot_price - 1

            # Ajuster le profit en fonction de la base (si on doit fermer avec un prix différent)
            adjusted_profit = expected_profit

            if direction == "short_perp_long_spot" and basis > 0:
                # Si on short le perp qui est plus cher, on gagne aussi la base
                adjusted_profit += basis
            elif direction == "long_perp_short_spot" and basis < 0:
                # Si on long le perp qui est moins cher, on gagne aussi la base
                adjusted_profit -= basis

            # Créer une opportunité
            opportunity = {
                "id": str(uuid.uuid4()),
                "type": "funding_rate",
                "symbol": symbol,
                "exchange": exchange,
                "spot_price": spot_price,
                "perp_price": perp_price,
                "funding_rate": funding_rate,
                "next_funding_time": next_funding_time,
                "funding_interval": funding_interval,
                "basis": basis * 100,  # en pourcentage
                "direction": direction,
                "expected_periods": expected_periods,
                "profit_pct": adjusted_profit * 100,  # en pourcentage
                "timestamp": datetime.now().timestamp(),
                "num_transactions": 4,  # 2 à l'ouverture, 2 à la fermeture
                "confidence": self._calculate_confidence(symbol, funding_rate, basis),
            }

            # Ajouter si la différence est suffisante après frais
            if self.is_profitable(opportunity):
                opportunities.append(opportunity)
                logger.debug(
                    f"Opportunité de taux de financement détectée sur {symbol}: "
                    f"Taux: {funding_rate*100:+.4f}%, Direction: {direction}, "
                    f"Profit attendu: {adjusted_profit*100:.4f}%"
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

        # Pour l'arbitrage de taux de financement, nous devons tenir compte du risque de
        # changement du taux pendant la période d'arbitrage
        confidence = opportunity.get("confidence", 0.7)

        # Ajuster le profit en fonction de la confiance
        adjusted_profit = profit_pct * confidence

        return adjusted_profit

    def _update_funding_rate_history(self, market_data: Dict):
        """
        Met à jour l'historique des taux de financement.

        Args:
            market_data: Données de marché
        """
        current_time = datetime.now()

        for symbol, data in market_data.items():
            if (
                isinstance(data, dict)
                and "perp" in data
                and "funding_rate" in data["perp"]
            ):
                funding_rate = data["perp"]["funding_rate"]
                exchange = data["perp"].get("exchange", "unknown")

                # Clé pour l'historique: "symbol_exchange"
                history_key = f"{symbol}_{exchange}"

                # Initialiser l'historique si nécessaire
                if history_key not in self.funding_rate_history:
                    self.funding_rate_history[history_key] = []

                # Ajouter le nouveau taux avec l'horodatage
                self.funding_rate_history[history_key].append(
                    {"timestamp": current_time, "funding_rate": funding_rate}
                )

                # Limiter l'historique aux 30 dernières entrées
                if len(self.funding_rate_history[history_key]) > 30:
                    self.funding_rate_history[history_key] = self.funding_rate_history[
                        history_key
                    ][-30:]

    def _predict_funding_persistence(self, symbol: str, current_rate: float) -> int:
        """
        Prédit combien de périodes le taux de financement actuel pourrait persister.

        Args:
            symbol: Symbole de l'actif
            current_rate: Taux de financement actuel

        Returns:
            Estimation du nombre de périodes pendant lesquelles le taux persistera
        """
        # Par défaut, prédire une seule période
        default_periods = 1

        # Chercher l'historique pour ce symbole (toutes plateformes confondues)
        relevant_history = []
        for key, history in self.funding_rate_history.items():
            if key.startswith(symbol + "_"):
                relevant_history.extend(history)

        # Si pas assez d'historique, retourner la valeur par défaut
        if len(relevant_history) < 3:
            return default_periods

        # Trier par horodatage
        relevant_history.sort(key=lambda x: x["timestamp"])

        # Calculer la tendance récente
        recent_rates = [entry["funding_rate"] for entry in relevant_history[-3:]]
        trend = np.mean(np.diff(recent_rates))

        # Si le taux a tendance à augmenter (en valeur absolue)
        if (current_rate > 0 and trend > 0) or (current_rate < 0 and trend < 0):
            # Le taux semble se renforcer, on peut prédire plus de périodes
            return min(3, default_periods + 1)

        # Si le taux a tendance à diminuer (en valeur absolue)
        if (current_rate > 0 and trend < 0) or (current_rate < 0 and trend > 0):
            # Le taux semble s'affaiblir, prédire moins de périodes
            return default_periods

        # Par défaut, retourner la valeur conservatrice
        return default_periods

    def _calculate_confidence(
        self, symbol: str, funding_rate: float, basis: float
    ) -> float:
        """
        Calcule un score de confiance pour l'opportunité d'arbitrage.

        Args:
            symbol: Symbole de l'actif
            funding_rate: Taux de financement actuel
            basis: Différence de prix entre perpétuel et spot (en décimal)

        Returns:
            Score de confiance entre 0 et 1
        """
        # Base de confiance
        confidence = 0.8

        # Si le taux de financement est très élevé, réduire la confiance
        # car il est moins probable qu'il persiste
        if abs(funding_rate) > 0.05:  # 5%
            confidence -= min(0.3, (abs(funding_rate) - 0.05) * 3)

        # Si la base est élevée et dans la même direction que le financement,
        # c'est un signe plus fort de déséquilibre de marché
        if (funding_rate > 0 and basis > 0) or (funding_rate < 0 and basis < 0):
            confidence += min(0.1, abs(basis) * 5)

        # Limiter la confiance entre 0.1 et 1
        return max(0.1, min(1.0, confidence))

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
                "min_funding_rate": f"{self.min_funding_rate*100:.2f}%",
                "max_holding_periods": self.max_holding_periods,
            }
        else:
            return {
                "strategy_name": self.name,
                "total_trades": 0,
                "active_positions": 0,
                "total_profit_pct": 0,
                "avg_profit_pct": 0,
                "total_profit_abs": 0,
                "min_funding_rate": f"{self.min_funding_rate*100:.2f}%",
                "max_holding_periods": self.max_holding_periods,
            }
