from typing import Dict, List

import numpy as np

from ..market_constraints import MarketConstraints


class MultiAssetTradingEnvironment:
    def __init__(self, assets: List[str], initial_balance: float = 100000.0):
        self.assets = assets
        self.initial_balance = initial_balance
        self.market_constraints = MarketConstraints()
        self.asset_correlations = {}
        self.allocation_history = []  # Ajout de l'historique des allocations
        self.min_diversification_factor = 0.2
        self.max_diversification_factor = 2.0
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.holdings = {asset: 0.0 for asset in self.assets}
        self.portfolio_value = self.initial_balance
        self.slippage_value = 0.0
        self.asset_correlations = self._calculate_correlations()
        self.allocation_history = []  # Réinitialisation de l'historique
        return self._get_state()

    def step(self, actions: Dict[str, float]):
        """Exécute une étape de trading.

        Args:
            actions (Dict[str, float]): Actions pour chaque actif (-1 à 1)

        Returns:
            tuple: (état, récompense, terminé, info)
        """
        old_portfolio_value = self.portfolio_value
        total_slippage = 0.0

        # Exécution des actions pour chaque actif
        for asset, action in actions.items():
            if asset not in self.assets:
                continue

            # Calcul du slippage et de l'impact marché
            volume = abs(action) * self.portfolio_value
            price = self._get_current_price(asset)
            avg_volume = self._get_average_volume(asset)
            volatility = self._get_volatility(asset)

            slippage = self.market_constraints.calculate_slippage(
                asset, action, volume, volatility, avg_volume
            )
            impact, _ = self.market_constraints.calculate_market_impact(
                asset, action, volume, price, avg_volume
            )

            # Application du slippage et de l'impact
            effective_price = price * (1 + slippage + impact)
            quantity = (action * self.portfolio_value) / effective_price

            # Mise à jour des positions
            self.holdings[asset] += quantity
            self.balance -= quantity * effective_price
            total_slippage += slippage * volume

        # Mise à jour des valeurs du portfolio
        self.portfolio_value = self.balance + sum(
            self.holdings[asset] * self._get_current_price(asset)
            for asset in self.assets
        )
        self.slippage_value = total_slippage

        # Calcul de la récompense
        reward = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        reward = self._diversification_reward(reward)

        # Vérification de fin d'épisode
        done = self.portfolio_value <= 0  # L'épisode se termine si le portfolio est vide

        # Mise à jour de l'historique des allocations
        current_weights = {
            asset: (self.holdings[asset] * self._get_current_price(asset)) / max(self.portfolio_value, 1e-6)
            for asset in self.assets
        }
        self.allocation_history.append(current_weights)

        return self._get_state(), reward, done, self._get_info()

    def _get_state(self):
        """Retourne l'état actuel de l'environnement."""
        state = {
            "balance": self.balance,
            "portfolio_value": self.portfolio_value,
            "holdings": self.holdings.copy(),
            "prices": {asset: self._get_current_price(asset) for asset in self.assets},
            "slippage": self.slippage_value,
        }
        return state

    def _diversification_reward(self, base_reward: float) -> float:
        """Calcule la récompense de diversification.

        Args:
            base_reward (float): La récompense de base à ajuster.

        Returns:
            float: La récompense ajustée selon la diversification.
        """
        total_value = sum(
            self.holdings[asset] * self._get_current_price(asset)
            for asset in self.assets
        )

        # Si le portefeuille est vide, retourner la récompense de base
        if total_value <= 0:
            return base_reward

        # Calcul des poids du portfolio
        weights = {
            asset: (self.holdings[asset] * self._get_current_price(asset)) / total_value
            for asset in self.assets
        }

        # Calcul de l'indice de diversification (HHI inversé)
        hhi = sum(w * w for w in weights.values())
        n = len(weights)
        diversification_index = (1 - hhi) / (1 - 1/n) if n > 1 else 0

        # Calcul de la pénalité de corrélation
        correlation_penalty = 0.0
        if self.asset_correlations and len(weights) > 1:
            weighted_correlations = []
            for i, (asset1, w1) in enumerate(weights.items()):
                for j, (asset2, w2) in enumerate(weights.items()):
                    if i < j and w1 > 0 and w2 > 0:
                        corr = abs(self.asset_correlations.get((asset1, asset2), 0))
                        weighted_corr = corr * w1 * w2 * 5.0  # Augmentation de l'impact
                        weighted_correlations.append(weighted_corr)
            
            if weighted_correlations:
                correlation_penalty = sum(weighted_correlations)

        # Calcul du facteur de diversification
        if diversification_index < 0.5:
            diversification_factor = max(
                self.min_diversification_factor,
                0.5 * (1.0 + diversification_index - correlation_penalty)
            )
        else:
            diversification_factor = min(
                self.max_diversification_factor,
                1.0 + diversification_index - correlation_penalty
            )

        # Stockage des métriques pour les tests
        self.last_diversification_metrics = {
            "diversification_index": diversification_index,
            "correlation_penalty": correlation_penalty,
            "weights": weights,
            "diversification_factor": diversification_factor
        }

        return base_reward * diversification_factor

    def _calculate_correlations(self):
        """Calcule les corrélations entre les actifs."""
        correlations = {}
        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets[i + 1:], i + 1):
                returns1 = self._get_returns(asset1)
                returns2 = self._get_returns(asset2)
                if len(returns1) > 1 and len(returns2) > 1:
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    correlations[(asset1, asset2)] = correlation
                    correlations[(asset2, asset1)] = correlation  # Ajout de la symétrie
            correlations[(asset1, asset1)] = 1.0  # Corrélation avec soi-même
        return correlations

    def _get_current_price(self, asset: str) -> float:
        """Retourne le prix actuel d'un actif."""
        # À implémenter avec les données réelles
        return 100.0  # Prix fictif pour l'exemple

    def _get_average_volume(self, asset: str) -> float:
        """Retourne le volume moyen d'un actif."""
        # À implémenter avec les données réelles
        return 1000.0  # Volume fictif pour l'exemple

    def _get_volatility(self, asset: str) -> float:
        """Retourne la volatilité d'un actif."""
        # À implémenter avec les données réelles
        return 0.02  # Volatilité fictive pour l'exemple

    def _get_returns(self, asset: str) -> np.ndarray:
        """Retourne les rendements historiques d'un actif."""
        # À implémenter avec les données réelles
        return np.array([0.01, -0.01, 0.02])  # Rendements fictifs pour l'exemple

    def _get_info(self):
        """Retourne des informations supplémentaires sur l'épisode."""
        return {
            "portfolio_value": self.portfolio_value,
            "holdings": self.holdings.copy(),
            "slippage": self.slippage_value,
            "allocation_history": self.allocation_history
        }
