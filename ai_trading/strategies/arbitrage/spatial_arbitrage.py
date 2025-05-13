"""
Stratégie d'arbitrage spatial (cross-exchange).

Ce module implémente une stratégie d'arbitrage qui exploite les
différences de prix d'un même actif entre différentes plateformes d'échange.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ai_trading.strategies.arbitrage.base import ArbitrageStrategy

# Configuration du logging
logger = logging.getLogger(__name__)


class SpatialArbitrageStrategy(ArbitrageStrategy):
    """
    Stratégie d'arbitrage spatial entre différentes plateformes d'échange.

    Cette stratégie détecte les différences de prix pour un même actif
    entre différentes plateformes d'échange et exécute des arbitrages
    lorsque ces différences dépassent un certain seuil.
    """

    def __init__(
        self,
        exchanges: List[str],
        min_profit_threshold: float = 0.005,
        transaction_fee: float = 0.001,
        risk_tolerance: float = 0.5,
        max_position_size: float = 1000.0,
        max_positions: int = 5,
        consider_withdrawal_fees: bool = True,
        withdrawal_fees: Optional[Dict[str, float]] = None,
        withdrawal_time_minutes: Optional[Dict[str, float]] = None,
        slippage_estimate: float = 0.001,
        max_execution_time_seconds: float = 10.0,
    ):
        """
        Initialise une stratégie d'arbitrage spatial.

        Args:
            exchanges: Liste des plateformes d'échange à surveiller
            min_profit_threshold: Seuil minimum de profit pour exécuter un arbitrage (0.5% par défaut)
            transaction_fee: Frais de transaction par opération (0.1% par défaut)
            risk_tolerance: Tolérance au risque entre 0 et 1 (0.5 par défaut)
            max_position_size: Taille maximale d'une position en USD
            max_positions: Nombre maximum de positions simultanées
            consider_withdrawal_fees: Si True, considère les frais de retrait entre plateformes
            withdrawal_fees: Dictionnaire des frais de retrait par plateforme
            withdrawal_time_minutes: Dictionnaire des temps de retrait par plateforme
            slippage_estimate: Estimation du glissement (slippage) lors de l'exécution
            max_execution_time_seconds: Temps maximum d'exécution accepté
        """
        super().__init__(
            name="Arbitrage Spatial",
            min_profit_threshold=min_profit_threshold,
            transaction_fee=transaction_fee,
            risk_tolerance=risk_tolerance,
            max_position_size=max_position_size,
            max_positions=max_positions,
        )

        self.exchanges = exchanges
        self.consider_withdrawal_fees = consider_withdrawal_fees
        self.withdrawal_fees = withdrawal_fees or {}
        self.withdrawal_time_minutes = withdrawal_time_minutes or {}
        self.slippage_estimate = slippage_estimate
        self.max_execution_time_seconds = max_execution_time_seconds

        # Paramètres par défaut pour les plateformes
        self._set_default_exchange_params()

        logger.info(
            f"Stratégie d'arbitrage spatial initialisée avec {len(exchanges)} plateformes"
        )
        logger.debug(f"Plateformes surveillées: {', '.join(exchanges)}")

    def _set_default_exchange_params(self):
        """
        Définit des paramètres par défaut pour les plateformes d'échange.
        """
        default_withdrawal_fee = 0.0005  # 0.05% ou équivalent
        default_withdrawal_time = 30.0  # 30 minutes

        # Paramètres spécifiques connus pour certaines plateformes
        known_withdrawal_fees = {
            "binance": 0.0001,  # BTC exemple
            "coinbase": 0.0003,
            "kraken": 0.00015,
            "kucoin": 0.0002,
            "ftx": 0.0001,
            "huobi": 0.0004,
            "okex": 0.0002,
            "bybit": 0.0003,
        }

        known_withdrawal_times = {
            "binance": 15.0,  # minutes
            "coinbase": 60.0,
            "kraken": 20.0,
            "kucoin": 30.0,
            "ftx": 10.0,
            "huobi": 40.0,
            "okex": 30.0,
            "bybit": 25.0,
        }

        # Définir les paramètres par défaut si non spécifiés
        for exchange in self.exchanges:
            if exchange not in self.withdrawal_fees:
                self.withdrawal_fees[exchange] = known_withdrawal_fees.get(
                    exchange.lower(), default_withdrawal_fee
                )

            if exchange not in self.withdrawal_time_minutes:
                self.withdrawal_time_minutes[exchange] = known_withdrawal_times.get(
                    exchange.lower(), default_withdrawal_time
                )

    def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """
        Trouve les opportunités d'arbitrage spatial dans les données de marché.

        Cette méthode analyse les prix d'un même actif sur différentes plateformes
        et identifie les opportunités d'arbitrage lorsque la différence de prix
        dépasse le seuil minimum de profit.

        Args:
            market_data: Dictionnaire avec les prix des actifs sur différentes plateformes
                Format attendu: {
                    'BTC/USDT': {
                        'binance': {'ask': 40000, 'bid': 39900, 'last': 39950, 'volume': 100, 'timestamp': 1234567890},
                        'coinbase': {'ask': 40100, 'bid': 40000, 'last': 40050, 'volume': 80, 'timestamp': 1234567890},
                        ...
                    },
                    'ETH/USDT': {
                        ...
                    },
                    ...
                }

        Returns:
            Liste d'opportunités d'arbitrage détectées
        """
        opportunities = []

        # Vérifier si market_data est correctement formaté
        if not isinstance(market_data, dict):
            logger.error(f"Format market_data invalide: {type(market_data)}")
            return opportunities

        # Parcourir chaque paire d'actifs
        for symbol, exchange_data in market_data.items():
            # Ignorer si les données sont incomplètes
            if not isinstance(exchange_data, dict) or len(exchange_data) < 2:
                continue

            # Extraire les prix pour chaque plateforme
            exchange_prices = {}
            for exchange, price_data in exchange_data.items():
                if exchange in self.exchanges and isinstance(price_data, dict):
                    # Utiliser le prix ask pour acheter, bid pour vendre
                    if "ask" in price_data and "bid" in price_data:
                        exchange_prices[exchange] = {
                            "ask": price_data["ask"],
                            "bid": price_data["bid"],
                            "timestamp": price_data.get(
                                "timestamp", datetime.now().timestamp()
                            ),
                        }

            # Si moins de 2 plateformes ont des données, passer à la suite
            if len(exchange_prices) < 2:
                continue

            # Trouver les arbitrages potentiels
            for buy_exchange, buy_data in exchange_prices.items():
                for sell_exchange, sell_data in exchange_prices.items():
                    if buy_exchange == sell_exchange:
                        continue

                    # Calculer le potentiel d'arbitrage
                    buy_price = buy_data["ask"]
                    sell_price = sell_data["bid"]
                    price_diff = (sell_price / buy_price) - 1

                    # Si le prix d'achat est inférieur au prix de vente (profitable)
                    if price_diff > 0:
                        # Créer une opportunité
                        opportunity = {
                            "id": str(uuid.uuid4()),
                            "type": "spatial",
                            "symbol": symbol,
                            "buy_exchange": buy_exchange,
                            "sell_exchange": sell_exchange,
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "price_diff_pct": price_diff * 100,  # en %
                            "timestamp": min(
                                buy_data.get("timestamp", 0),
                                sell_data.get("timestamp", 0),
                            ),
                            "num_transactions": 2,  # achat puis vente
                            "estimated_execution_time": self._estimate_execution_time(
                                buy_exchange, sell_exchange
                            ),
                            "withdrawal_fee": self._get_withdrawal_fee(
                                buy_exchange, sell_exchange, symbol
                            ),
                            "confidence": self._calculate_confidence(
                                buy_exchange, sell_exchange, buy_data, sell_data
                            ),
                        }

                        # Ajouter si la différence est suffisante après frais
                        if self.is_profitable(opportunity):
                            opportunities.append(opportunity)
                            logger.debug(
                                f"Opportunité détectée sur {symbol}: "
                                f"Achat à {buy_price:.2f} sur {buy_exchange}, "
                                f"Vente à {sell_price:.2f} sur {sell_exchange}, "
                                f"Différence: {price_diff*100:.2f}%"
                            )

        # Trier les opportunités par profit potentiel
        opportunities.sort(key=lambda x: x["price_diff_pct"], reverse=True)

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
        price_diff_pct = opportunity.get("price_diff_pct", 0)

        # Ajustement pour le glissement
        adjusted_diff = price_diff_pct - (
            self.slippage_estimate * 100 * 2
        )  # 2 pour achat et vente

        # Ajustement pour les frais de retrait si nécessaire
        if self.consider_withdrawal_fees:
            withdrawal_fee_pct = opportunity.get("withdrawal_fee", 0) * 100
            adjusted_diff -= withdrawal_fee_pct

        return max(0, adjusted_diff)

    def _get_withdrawal_fee(
        self, buy_exchange: str, sell_exchange: str, symbol: str
    ) -> float:
        """
        Obtient les frais de retrait entre deux plateformes.

        Args:
            buy_exchange: Plateforme d'achat
            sell_exchange: Plateforme de vente
            symbol: Symbole de l'actif

        Returns:
            Frais de retrait estimés en pourcentage décimal
        """
        if not self.consider_withdrawal_fees:
            return 0.0

        # Récupérer les frais de la plateforme d'achat (pour le retrait)
        withdrawal_fee = self.withdrawal_fees.get(buy_exchange, 0.0005)

        # Ajuster en fonction de l'actif (simplification)
        # En réalité, chaque actif a des frais de retrait différents
        if "BTC" in symbol:
            fee_multiplier = 1.0
        elif "ETH" in symbol:
            fee_multiplier = 0.8
        else:
            fee_multiplier = 0.5

        return withdrawal_fee * fee_multiplier

    def _estimate_execution_time(self, buy_exchange: str, sell_exchange: str) -> float:
        """
        Estime le temps d'exécution total pour l'arbitrage.

        Args:
            buy_exchange: Plateforme d'achat
            sell_exchange: Plateforme de vente

        Returns:
            Temps estimé en secondes
        """
        # Temps de base pour les transactions
        base_time = 5.0  # secondes

        # Si l'arbitrage nécessite un transfert entre plateformes
        if self.consider_withdrawal_fees:
            # Convertir les minutes en secondes
            withdrawal_time = self.withdrawal_time_minutes.get(buy_exchange, 30.0) * 60
            return base_time + withdrawal_time

        return base_time

    def _calculate_confidence(
        self, buy_exchange: str, sell_exchange: str, buy_data: Dict, sell_data: Dict
    ) -> float:
        """
        Calcule un score de confiance pour l'opportunité d'arbitrage.

        Args:
            buy_exchange: Plateforme d'achat
            sell_exchange: Plateforme de vente
            buy_data: Données de prix pour l'achat
            sell_data: Données de prix pour la vente

        Returns:
            Score de confiance entre 0 et 1
        """
        # Initialiser avec un score de base
        confidence = 0.8

        # Réduire la confiance si les données sont anciennes
        current_time = datetime.now().timestamp()
        buy_time_diff = current_time - buy_data.get("timestamp", current_time)
        sell_time_diff = current_time - sell_data.get("timestamp", current_time)

        max_time_diff = 60.0  # 60 secondes
        if buy_time_diff > max_time_diff or sell_time_diff > max_time_diff:
            time_penalty = min(
                0.5, max(buy_time_diff, sell_time_diff) / (max_time_diff * 10)
            )
            confidence -= time_penalty

        # Réduire la confiance si le temps d'exécution est long
        execution_time = self._estimate_execution_time(buy_exchange, sell_exchange)
        if execution_time > self.max_execution_time_seconds:
            time_penalty = min(
                0.3, (execution_time - self.max_execution_time_seconds) / 300.0
            )
            confidence -= time_penalty

        # Garantir que la confiance reste entre 0 et 1
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
                "exchanges_monitored": self.exchanges,
            }
        else:
            return {
                "strategy_name": self.name,
                "total_trades": 0,
                "active_positions": 0,
                "total_profit_pct": 0,
                "avg_profit_pct": 0,
                "total_profit_abs": 0,
                "exchanges_monitored": self.exchanges,
            }
