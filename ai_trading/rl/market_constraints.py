import logging

import numpy as np

logger = logging.getLogger(__name__)


class MarketConstraints:
    """Gère les contraintes de marché réalistes pour l'environnement de trading."""

    def __init__(self):
        self.orderbook_depth = {}
        self.base_slippage = 0.001  # Slippage de base
        self.max_recovery_time = 100  # Temps de récupération maximal
        self.min_impact = 0.001  # Impact minimal
        self.max_impact = 0.05  # Impact maximal

    def calculate_slippage(self, symbol, action_value, volume, volatility, avg_volume):
        """Calcule le slippage pour une transaction."""
        if volume <= 0:
            return 0.0

        # Calcul du slippage de base
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        volatility_factor = max(0, volatility)  # Éviter les valeurs négatives
        base_slip = self.base_slippage * (1 + volume_ratio) * (1 + volatility_factor)

        if symbol in self.orderbook_depth:
            # Utilisation des données du carnet d'ordres
            depth = self.orderbook_depth[symbol]
            spread = depth.get("spread_pct", 0.001)
            total_volume = depth.get("total_volume", avg_volume)
            volume_imbalance = abs(depth.get("volume_imbalance", 0))

            # Facteur de liquidité basé sur le volume total disponible
            liquidity_factor = avg_volume / total_volume if total_volume > 0 else 1.0

            # Ajustement du slippage en fonction des conditions de marché
            market_slip = (
                base_slip * (1 + spread) * (1 + volume_imbalance) * liquidity_factor
            )
            return min(market_slip, 0.01)  # Limite à 1%

        return min(base_slip, 0.01)  # Limite à 1%

    def calculate_market_impact(self, symbol, action_value, volume, price, avg_volume):
        """Calcule l'impact sur le marché d'une transaction."""
        if volume <= 0:
            return 0.0, 1

        # Calcul du ratio de volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Impact proportionnel au ratio de volume
        impact = min(self.max_impact, self.min_impact * (1 + volume_ratio))

        # Temps de récupération proportionnel au ratio de volume
        recovery_time = int(self.max_recovery_time * volume_ratio)
        recovery_time = max(min(recovery_time, self.max_recovery_time), 1)

        return impact, recovery_time

    def calculate_execution_delay(
        self, symbol: str, action_value: float, volume: float, avg_volume: float
    ) -> int:
        """
        Calcule le délai d'exécution pour une transaction.

        Args:
            symbol (str): Symbole de l'actif
            action_value (float): Valeur de l'action (-1 à 1)
            volume (float): Volume de la transaction
            avg_volume (float): Volume moyen sur la période

        Returns:
            int: Délai d'exécution en pas de temps
        """
        # Volume nul = pas de délai
        if volume <= 0 or abs(action_value) < 1e-6:
            return 0

        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        base_delay = 1

        # Ajustement en fonction du volume
        volume_factor = np.log1p(volume_ratio)

        # Ajustement en fonction de la profondeur du carnet
        if symbol in self.orderbook_depth:
            depth = self.orderbook_depth[symbol]
            market_load = depth.get("volume_imbalance", 0)
            load_factor = np.exp(abs(market_load)) - 1
        else:
            load_factor = 0

        total_delay = int(base_delay * (1 + volume_factor) * (1 + load_factor))
        return min(total_delay, 10)  # Limite à 10 pas

    def update_orderbook_depth(self, symbol, depth_data):
        """Met à jour les données de profondeur du carnet d'ordres."""
        self.orderbook_depth[symbol] = depth_data
