"""
Stratégie d'arbitrage triangulaire.

Ce module implémente une stratégie d'arbitrage triangulaire qui exploite
les incohérences de prix entre trois paires de trading sur une même plateforme
d'échange.
"""

import itertools
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from ai_trading.strategies.arbitrage.base import ArbitrageStrategy

# Configuration du logging
logger = logging.getLogger(__name__)


class TriangularArbitrageStrategy(ArbitrageStrategy):
    """
    Stratégie d'arbitrage triangulaire sur une même plateforme.

    Cette stratégie détecte les opportunités d'arbitrage en utilisant
    trois paires de trading pour former un "triangle" qui commence et
    se termine par la même devise (ex: BTC → ETH → USDT → BTC).
    """

    def __init__(
        self,
        exchange: str,
        base_currencies: Optional[List[str]] = None,
        min_profit_threshold: float = 0.002,
        transaction_fee: float = 0.001,
        risk_tolerance: float = 0.5,
        max_position_size: float = 1000.0,
        max_positions: int = 5,
        slippage_estimate: float = 0.0005,
        max_triangles: int = 100,
    ):
        """
        Initialise une stratégie d'arbitrage triangulaire.

        Args:
            exchange: Nom de la plateforme d'échange
            base_currencies: Liste des devises de base pour former des triangles (ex: ["BTC", "USDT", "ETH"])
            min_profit_threshold: Seuil minimum de profit pour exécuter un arbitrage (0.2% par défaut)
            transaction_fee: Frais de transaction par opération (0.1% par défaut)
            risk_tolerance: Tolérance au risque entre 0 et 1 (0.5 par défaut)
            max_position_size: Taille maximale d'une position en USD
            max_positions: Nombre maximum de positions simultanées
            slippage_estimate: Estimation du glissement (slippage) lors de l'exécution
            max_triangles: Nombre maximum de triangles à analyser
        """
        super().__init__(
            name="Arbitrage Triangulaire",
            min_profit_threshold=min_profit_threshold,
            transaction_fee=transaction_fee,
            risk_tolerance=risk_tolerance,
            max_position_size=max_position_size,
            max_positions=max_positions,
        )

        self.exchange = exchange
        self.base_currencies = base_currencies or ["BTC", "USDT", "ETH", "USDC", "BNB"]
        self.slippage_estimate = slippage_estimate
        self.max_triangles = max_triangles

        # Cache des triangles identifiés
        self.triangles = []
        self.available_symbols = set()

        logger.info(f"Stratégie d'arbitrage triangulaire initialisée sur {exchange}")
        logger.debug(f"Devises de base: {', '.join(self.base_currencies)}")

    def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """
        Trouve les opportunités d'arbitrage triangulaire dans les données de marché.

        Args:
            market_data: Dictionnaire avec les prix des actifs
                Format attendu: {
                    'BTC/USDT': {'ask': 40000, 'bid': 39900, 'last': 39950, 'volume': 100},
                    'ETH/BTC': {'ask': 0.075, 'bid': 0.074, 'last': 0.0745, 'volume': 50},
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

        # Mettre à jour la liste des symboles disponibles
        self.available_symbols = self._extract_symbols(market_data)

        # Identifier ou mettre à jour les triangles possibles
        if not self.triangles:
            self.triangles = self._find_triangles()
            logger.info(f"Identifié {len(self.triangles)} triangles possibles")

        # Examiner chaque triangle pour des opportunités
        for triangle in self.triangles[
            : self.max_triangles
        ]:  # Limiter le nombre de triangles à examiner
            # Vérifier si toutes les paires du triangle sont disponibles
            if not all(pair in market_data for pair in triangle["pairs"]):
                continue

            # Calculer le profit potentiel
            profit, details = self._calculate_triangle_profit(triangle, market_data)

            if profit > 0:
                # Créer une opportunité
                opportunity = {
                    "id": str(uuid.uuid4()),
                    "type": "triangular",
                    "exchange": self.exchange,
                    "triangle": triangle["name"],
                    "pairs": triangle["pairs"],
                    "profit_pct": profit * 100,  # Convertir en pourcentage
                    "timestamp": datetime.now().timestamp(),
                    "details": details,
                    "path": triangle["path"],
                    "num_transactions": 3,  # trois transactions pour un triangle
                    "confidence": self._calculate_confidence(
                        triangle, market_data, profit
                    ),
                }

                # Ajouter si la différence est suffisante après frais
                if self.is_profitable(opportunity):
                    opportunities.append(opportunity)
                    logger.debug(
                        f"Opportunité triangulaire détectée sur {triangle['name']}: "
                        f"Profit: {profit*100:.4f}%, "
                        f"Paires: {', '.join(triangle['pairs'])}"
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
        profit_pct = opportunity.get("profit_pct", 0)

        # Ajustement pour le glissement
        # Pour un triangle, nous avons 3 transactions, donc 3x le slippage
        adjusted_profit = profit_pct - (self.slippage_estimate * 100 * 3)

        return max(0, adjusted_profit)

    def _extract_symbols(self, market_data: Dict) -> Set[str]:
        """
        Extrait la liste des symboles disponibles.

        Args:
            market_data: Données de marché

        Returns:
            Ensemble des symboles disponibles
        """
        symbols = set()
        for pair in market_data.keys():
            if "/" in pair:
                base, quote = pair.split("/")
                symbols.add(base)
                symbols.add(quote)
        return symbols

    def _find_triangles(self) -> List[Dict]:
        """
        Identifie tous les triangles possibles à partir des symboles disponibles.

        Returns:
            Liste des triangles identifiés
        """
        triangles = []

        # Pour chaque devise de base, trouver les triangles possibles
        for base in self.base_currencies:
            if base not in self.available_symbols:
                continue

            # Identifier toutes les paires qui incluent cette devise de base
            pairs_with_base = []
            for pair in [p for p in self.available_symbols if "/" in p]:
                try:
                    base_currency, quote_currency = pair.split("/")
                    if base_currency == base or quote_currency == base:
                        pairs_with_base.append(pair)
                except ValueError:
                    continue

            # Pour chaque paire avec base, trouver des triangles possibles
            for pair1, pair2 in itertools.combinations(pairs_with_base, 2):
                # Extraire les devises des paires
                currencies = set()
                currencies.update(pair1.split("/"))
                currencies.update(pair2.split("/"))

                # Nous avons besoin de trois devises distinctes
                if len(currencies) != 3:
                    continue

                # Trouver la troisième devise (qui n'est pas base)
                other_currencies = currencies - {base}
                if len(other_currencies) != 2:
                    continue

                curr1, curr2 = other_currencies

                # Vérifier si la troisième paire existe
                third_pair = f"{curr1}/{curr2}"
                third_pair_reverse = f"{curr2}/{curr1}"

                if third_pair in self.available_symbols:
                    pairs = [pair1, pair2, third_pair]
                    triangle_name = f"{base}-{curr1}-{curr2}"
                    path = self._determine_path(base, curr1, curr2, pairs)
                    triangles.append(
                        {
                            "name": triangle_name,
                            "base": base,
                            "currencies": [base, curr1, curr2],
                            "pairs": pairs,
                            "path": path,
                        }
                    )
                elif third_pair_reverse in self.available_symbols:
                    pairs = [pair1, pair2, third_pair_reverse]
                    triangle_name = f"{base}-{curr1}-{curr2}"
                    path = self._determine_path(base, curr1, curr2, pairs)
                    triangles.append(
                        {
                            "name": triangle_name,
                            "base": base,
                            "currencies": [base, curr1, curr2],
                            "pairs": pairs,
                            "path": path,
                        }
                    )

        return triangles

    def _determine_path(
        self, base: str, curr1: str, curr2: str, pairs: List[str]
    ) -> List[Dict]:
        """
        Détermine le chemin d'exécution optimal pour un triangle.

        Args:
            base: Devise de base
            curr1: Première devise intermédiaire
            curr2: Deuxième devise intermédiaire
            pairs: Liste des paires pour le triangle

        Returns:
            Chemin d'exécution sous forme de liste d'étapes
        """
        path = []
        current_currency = base

        # Nous voulons construire un chemin qui commence et finit par la devise de base
        for _ in range(3):  # Un triangle a 3 étapes
            found = False
            for pair in pairs:
                base_currency, quote_currency = pair.split("/")

                # Si nous pouvons utiliser cette paire pour continuer le chemin
                if base_currency == current_currency:
                    direction = "sell"
                    next_currency = quote_currency
                    price_key = "bid"  # Pour vendre, on utilise le bid (prix d'achat de la plateforme)
                    found = True
                elif quote_currency == current_currency:
                    direction = "buy"
                    next_currency = base_currency
                    price_key = "ask"  # Pour acheter, on utilise le ask (prix de vente de la plateforme)
                    found = True

                if found:
                    path.append(
                        {
                            "pair": pair,
                            "direction": direction,
                            "from": current_currency,
                            "to": next_currency,
                            "price_key": price_key,
                        }
                    )
                    current_currency = next_currency
                    pairs = [p for p in pairs if p != pair]  # Retirer la paire utilisée
                    break

        return path

    def _calculate_triangle_profit(
        self, triangle: Dict, market_data: Dict
    ) -> Tuple[float, Dict]:
        """
        Calcule le profit potentiel pour un triangle donné.

        Args:
            triangle: Dictionnaire représentant un triangle
            market_data: Données de marché

        Returns:
            Profit potentiel (en décimal) et détails du calcul
        """
        path = triangle["path"]

        # Valeur initiale (normalisée à 1.0)
        value = 1.0
        details = []

        # Suivre le chemin et calculer la valeur finale
        for step in path:
            pair = step["pair"]
            direction = step["direction"]
            price_key = step["price_key"]

            # Vérifier si les données de prix sont disponibles
            if pair not in market_data or price_key not in market_data[pair]:
                return 0, {}

            # Récupérer le prix
            price = market_data[pair][price_key]

            # Mettre à jour la valeur en fonction de la direction
            if direction == "buy":
                # Acheter: diviser par le prix (ask)
                value = value / price
            else:
                # Vendre: multiplier par le prix (bid)
                value = value * price

            # Appliquer les frais de transaction
            value = value * (1 - self.transaction_fee)

            # Enregistrer les détails de cette étape
            details.append(
                {
                    "pair": pair,
                    "direction": direction,
                    "price": price,
                    "value_after": value,
                    "fee_applied": self.transaction_fee,
                }
            )

        # Le profit est la différence par rapport à la valeur initiale de 1.0
        profit = value - 1.0

        return profit, {"steps": details, "final_value": value}

    def _calculate_confidence(
        self, triangle: Dict, market_data: Dict, profit: float
    ) -> float:
        """
        Calcule un score de confiance pour l'opportunité d'arbitrage.

        Args:
            triangle: Dictionnaire représentant un triangle
            market_data: Données de marché
            profit: Profit calculé

        Returns:
            Score de confiance entre 0 et 1
        """
        # Base de confiance
        confidence = 0.9

        # Réduire la confiance en fonction de la volatilité des prix
        volatility_penalty = 0
        for pair in triangle["pairs"]:
            if pair in market_data:
                # Calculer une estimation simplifiée de la volatilité
                price_data = market_data[pair]
                if "ask" in price_data and "bid" in price_data:
                    spread = (price_data["ask"] - price_data["bid"]) / price_data["ask"]
                    volatility_penalty += min(
                        0.1, spread * 10
                    )  # Pénalité proportionnelle au spread

        confidence -= volatility_penalty

        # Ajuster la confiance en fonction du profit (plus le profit est élevé, moins il est probable)
        if profit > 0.01:  # Si le profit est supérieur à 1%
            profit_penalty = min(
                0.2, (profit - 0.01) * 10
            )  # Pénalité proportionnelle au profit
            confidence -= profit_penalty

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
                "exchange": self.exchange,
                "total_trades": total_trades,
                "active_positions": len(self.positions),
                "total_profit_pct": total_profit_pct,
                "avg_profit_pct": avg_profit_pct,
                "total_profit_abs": total_profit_abs,
                "triangles_monitored": len(self.triangles),
            }
        else:
            return {
                "strategy_name": self.name,
                "exchange": self.exchange,
                "total_trades": 0,
                "active_positions": 0,
                "total_profit_pct": 0,
                "avg_profit_pct": 0,
                "total_profit_abs": 0,
                "triangles_monitored": len(self.triangles),
            }
