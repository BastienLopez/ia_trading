"""
Système d'allocation complet pour la gestion de portefeuille crypto.

Ce module intègre:
1. Optimisation de portefeuille multi-facteurs
2. Gestion dynamique du risque
3. Rééquilibrage adaptatif
4. Allocation basée sur plusieurs critères (rendement, risque, corrélations)
5. Intégration avec les signaux de trading
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ai_trading.risk.risk_metrics import (
    calculate_es,
    calculate_var,
    calculate_volatility,
)
from ai_trading.rl.models.portfolio_optimization import FactorModel, PortfolioOptimizer
from ai_trading.rl.portfolio_allocator import PortfolioAllocator
from ai_trading.rl.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Résultat d'une allocation de portefeuille."""

    weights: Dict[str, float]  # Poids optimaux par actif
    expected_return: float  # Rendement attendu
    volatility: float  # Volatilité attendue
    sharpe_ratio: float  # Ratio de Sharpe
    var: float  # Value-at-Risk (VaR)
    es: float  # Expected Shortfall (ES)
    factor_exposures: Optional[Dict[str, float]] = None  # Expositions aux facteurs
    risk_contributions: Optional[Dict[str, float]] = None  # Contributions au risque


class CompleteAllocationSystem:
    """
    Système d'allocation complet intégrant modèles multi-facteurs, gestion du risque,
    et optimisation multi-objectifs pour un portefeuille crypto.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
        factor_model: Optional[FactorModel] = None,
        lookback_window: int = 90,
        risk_free_rate: float = 0.01,
        rebalance_threshold: float = 0.05,
        max_single_asset_weight: float = 0.25,
        min_single_asset_weight: float = 0.0,
        risk_budget_method: str = "equal",
        optimization_method: str = "sharpe",
        risk_aversion: float = 2.0,
    ):
        """
        Initialise le système d'allocation complet.

        Args:
            returns: DataFrame des rendements historiques
            prices: DataFrame des prix historiques (optionnel)
            factor_model: Modèle de facteurs (optionnel)
            lookback_window: Fenêtre d'historique pour les calculs
            risk_free_rate: Taux sans risque annualisé
            rebalance_threshold: Seuil de déviation pour le rééquilibrage (5%)
            max_single_asset_weight: Poids maximum pour un actif individuel
            min_single_asset_weight: Poids minimum pour un actif individuel
            risk_budget_method: Méthode de budgétisation du risque ('equal', 'momentum', 'custom')
            optimization_method: Méthode d'optimisation ('sharpe', 'min_risk', 'risk_parity', 'max_return')
            risk_aversion: Coefficient d'aversion au risque
        """
        self.returns = returns
        self.prices = prices
        self.assets = list(returns.columns)
        self.num_assets = len(self.assets)
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.rebalance_threshold = rebalance_threshold
        self.max_single_asset_weight = max_single_asset_weight
        self.min_single_asset_weight = min_single_asset_weight
        self.risk_budget_method = risk_budget_method
        self.optimization_method = optimization_method
        self.risk_aversion = risk_aversion

        # Initialiser l'optimiseur de portefeuille
        self.optimizer = PortfolioOptimizer(
            returns=returns,
            factor_model=factor_model,
            lookback_window=lookback_window,
            risk_free_rate=risk_free_rate,
        )

        # Initialiser l'allocateur de portefeuille pour les ajustements
        self.allocator = PortfolioAllocator(method="smart")

        # Initialiser le gestionnaire de risque
        self.risk_manager = RiskManager()

        # État actuel du portefeuille
        self.current_weights = None
        self.market_regime = "normal"  # normal, high_vol, bearish, bullish
        self.last_allocation_time = None

        logger.info(
            f"Système d'allocation complet initialisé avec {self.num_assets} actifs"
        )

    def detect_market_regime(self) -> str:
        """
        Détecte le régime de marché actuel basé sur les données récentes.

        Returns:
            Régime de marché ('normal', 'high_vol', 'bearish', 'bullish')
        """
        # Extraire les données récentes selon la fenêtre de lookback
        recent_returns = self.returns.iloc[-self.lookback_window :]

        # Calculer les métriques clés
        volatility = recent_returns.std() * np.sqrt(252)  # Annualisée
        mean_returns = recent_returns.mean() * 252  # Annualisés
        drawdowns = recent_returns.cumsum().cummax() - recent_returns.cumsum()

        # Volatilité moyenne du marché
        market_vol = volatility.mean()

        # Rendement moyen du marché
        market_return = mean_returns.mean()

        # Drawdown moyen
        market_drawdown = drawdowns.max().mean()

        # Déterminer le régime
        if market_vol > 0.40:  # Volatilité très élevée (40%+)
            regime = "high_vol"
        elif market_return < -0.20 and market_drawdown > 0.15:  # Baisse significative
            regime = "bearish"
        elif market_return > 0.20:  # Hausse significative
            regime = "bullish"
        else:
            regime = "normal"

        logger.info(
            f"Régime de marché détecté: {regime} (Vol: {market_vol:.2f}, Ret: {market_return:.2f})"
        )
        return regime

    def create_constraints(self) -> List[Dict]:
        """
        Crée les contraintes pour l'optimisation selon le régime de marché.

        Returns:
            Liste des contraintes pour l'optimiseur
        """
        constraints = []

        # Contraintes de base: limites sur les poids individuels
        bounds = []
        for asset in self.assets:
            lower_bound = self.min_single_asset_weight
            upper_bound = self.max_single_asset_weight

            # Ajuster les limites selon le régime de marché
            if self.market_regime == "high_vol":
                # Réduire l'exposition maximale en période de haute volatilité
                upper_bound = min(upper_bound, 0.15)
            elif self.market_regime == "bearish":
                # Permettre des poids plus faibles et limiter les maximums en marché baissier
                upper_bound = min(upper_bound, 0.20)

            bounds.append((lower_bound, upper_bound))

        return bounds

    def optimize_allocation(
        self, current_weights: Optional[Dict[str, float]] = None
    ) -> AllocationResult:
        """
        Optimise l'allocation du portefeuille selon la méthode choisie.

        Args:
            current_weights: Poids actuels du portefeuille (pour minimiser le turnover)

        Returns:
            Résultat d'allocation optimisé
        """
        # Détecter le régime de marché
        self.market_regime = self.detect_market_regime()

        # Préparer les contraintes
        bounds = self.create_constraints()

        # Préparer les poids actuels
        current_weights_series = None
        if current_weights:
            current_weights_series = pd.Series(current_weights)

        # Définir les contraintes spécifiques selon le régime de marché
        constraints = []

        # En période de haute volatilité, limiter la volatilité globale
        if self.market_regime == "high_vol":
            target_risk = 0.15  # 15% volatilité annualisée max
            if self.optimization_method == "max_return":
                result = self.optimizer.optimize_portfolio(
                    objective="max_return",
                    target_risk=target_risk,
                    constraints=constraints,
                    current_weights=current_weights_series,
                )
            else:
                result = self.optimizer.optimize_portfolio(
                    objective="min_risk",
                    constraints=constraints,
                    current_weights=current_weights_series,
                )

        # En marché baissier, favoriser la minimisation du risque
        elif self.market_regime == "bearish":
            result = self.optimizer.optimize_portfolio(
                objective="min_risk",
                constraints=constraints,
                current_weights=current_weights_series,
                use_var=True,  # Utiliser la VaR comme mesure de risque
            )

        # En marché haussier, maximiser le rendement sous contrainte de risque
        elif self.market_regime == "bullish":
            target_risk = 0.25  # 25% volatilité annualisée max
            result = self.optimizer.optimize_portfolio(
                objective="max_return",
                target_risk=target_risk,
                constraints=constraints,
                current_weights=current_weights_series,
            )

        # Régime normal, utiliser la méthode d'optimisation par défaut
        else:
            if self.optimization_method == "sharpe":
                result = self.optimizer.optimize_portfolio(
                    objective="sharpe",
                    constraints=constraints,
                    current_weights=current_weights_series,
                )
            elif self.optimization_method == "min_risk":
                result = self.optimizer.optimize_portfolio(
                    objective="min_risk",
                    constraints=constraints,
                    current_weights=current_weights_series,
                )
            elif self.optimization_method == "risk_parity":
                # Implémentation de la parité des risques
                target_risk_contribs = {
                    asset: 1.0 / self.num_assets for asset in self.assets
                }
                result = self.optimize_risk_parity(
                    target_risk_contribs, current_weights_series
                )
            else:  # max_return
                result = self.optimizer.optimize_portfolio(
                    objective="max_return",
                    target_risk=0.30,  # 30% volatilité annualisée max par défaut
                    constraints=constraints,
                    current_weights=current_weights_series,
                )

        # Mise à jour des poids actuels
        self.current_weights = result["weights"]

        # Création du résultat formaté
        allocation_result = AllocationResult(
            weights=result["weights"],
            expected_return=result["expected_return"],
            volatility=result["volatility"],
            sharpe_ratio=result["sharpe_ratio"],
            var=result["var"],
            es=result["es"],
            factor_exposures=result.get("factor_exposures"),
            risk_contributions=result.get("risk_contributions"),
        )

        return allocation_result

    def optimize_risk_parity(
        self,
        target_risk_contribs: Dict[str, float],
        current_weights: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Optimise le portefeuille selon le principe de parité des risques.

        Args:
            target_risk_contribs: Contributions cibles au risque pour chaque actif
            current_weights: Poids actuels du portefeuille

        Returns:
            Résultat d'optimisation
        """
        from scipy.optimize import minimize

        def risk_parity_objective(weights, cov_matrix, target_contribs):
            weights = pd.Series(weights, index=cov_matrix.index)
            port_variance = weights.dot(cov_matrix).dot(weights)
            port_vol = np.sqrt(port_variance)

            # Calculer les contributions actuelles au risque
            marginal_contribs = cov_matrix.dot(weights) / port_vol
            risk_contribs = weights * marginal_contribs

            # Calculer l'écart par rapport aux contributions cibles
            target_contribs_series = pd.Series(target_contribs)
            error = ((risk_contribs - target_contribs_series) ** 2).sum()

            return error

        # Contraintes: somme des poids = 1
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        # Bornes: poids entre min et max
        bounds = [
            (self.min_single_asset_weight, self.max_single_asset_weight)
            for _ in range(self.num_assets)
        ]

        # Poids initiaux
        if current_weights is not None:
            initial_weights = current_weights.values
        else:
            initial_weights = np.ones(self.num_assets) / self.num_assets

        # Optimisation
        result = minimize(
            risk_parity_objective,
            initial_weights,
            args=(self.optimizer.cov_matrix, target_risk_contribs),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Récupérer les poids optimaux
        optimal_weights = pd.Series(result["x"], index=self.assets)

        # Calculer les métriques de performance
        expected_return = (optimal_weights * self.optimizer.expected_returns).sum()
        variance = optimal_weights.dot(self.optimizer.cov_matrix).dot(optimal_weights)
        volatility = np.sqrt(variance)
        sharpe_ratio = (expected_return - self.risk_free_rate / 252) / volatility

        # Calculer VaR et ES
        var = calculate_var(self.returns, confidence=0.95) * np.sqrt(
            optimal_weights.dot(self.optimizer.cov_matrix).dot(optimal_weights)
        )
        es = calculate_es(self.returns, confidence=0.95) * np.sqrt(
            optimal_weights.dot(self.optimizer.cov_matrix).dot(optimal_weights)
        )

        # Expositions aux facteurs
        factor_exposures = None
        if self.optimizer.factor_model:
            exposures = self.optimizer.factor_model.exposures
            factor_exposures = optimal_weights @ exposures

        # Calculer les contributions au risque
        marginal_contribs = self.optimizer.cov_matrix.dot(optimal_weights) / volatility
        risk_contribs = optimal_weights * marginal_contribs
        risk_contribs = risk_contribs / risk_contribs.sum()  # Normaliser

        return {
            "weights": optimal_weights,
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "var": var,
            "es": es,
            "factor_exposures": factor_exposures,
            "risk_contributions": risk_contribs,
        }

    def should_rebalance(self, current_weights: Dict[str, float]) -> bool:
        """
        Détermine si un rééquilibrage est nécessaire.

        Args:
            current_weights: Poids actuels du portefeuille

        Returns:
            True si un rééquilibrage est recommandé
        """
        if not self.current_weights:
            return True

        # Calculer la déviation maximale par rapport aux poids optimaux
        max_deviation = 0
        for asset in self.assets:
            target_weight = self.current_weights.get(asset, 0)
            current_weight = current_weights.get(asset, 0)
            deviation = abs(target_weight - current_weight)
            max_deviation = max(max_deviation, deviation)

        # Vérifier si la déviation dépasse le seuil
        if max_deviation > self.rebalance_threshold:
            logger.info(
                f"Rééquilibrage recommandé: déviation max = {max_deviation:.2%}"
            )
            return True

        return False

    def get_rebalance_plan(
        self, current_weights: Dict[str, float], capital: float
    ) -> Dict[str, float]:
        """
        Génère un plan de rééquilibrage pour atteindre l'allocation optimale.

        Args:
            current_weights: Poids actuels du portefeuille
            capital: Capital total du portefeuille

        Returns:
            Dictionnaire des ajustements à effectuer par actif (montants)
        """
        if not self.current_weights:
            self.optimize_allocation(current_weights)

        # Calculer les montants actuels
        current_amounts = {
            asset: weight * capital for asset, weight in current_weights.items()
        }

        # Calculer les montants cibles
        target_amounts = {
            asset: weight * capital for asset, weight in self.current_weights.items()
        }

        # Calculer les ajustements
        adjustments = {}
        for asset in self.assets:
            current = current_amounts.get(asset, 0)
            target = target_amounts.get(asset, 0)
            adjustments[asset] = target - current

        return adjustments

    def integrate_signals(self, trading_signals: Dict[str, float]) -> AllocationResult:
        """
        Intègre les signaux de trading dans l'allocation du portefeuille.

        Args:
            trading_signals: Forces des signaux par actif (entre -1 et 1)

        Returns:
            Allocation optimisée intégrant les signaux
        """
        # Créer des poids modifiés par les signaux
        signal_weights = {}
        for asset in self.assets:
            signal = trading_signals.get(asset, 0)
            weight = 0.5 + 0.5 * signal  # Transformer [-1, 1] en [0, 1]
            signal_weights[asset] = weight

        # Utiliser l'allocateur pour ajuster les poids
        if self.current_weights:
            volatilities = {
                asset: calculate_volatility(self.returns[asset])
                for asset in self.assets
            }
            recent_returns = {
                asset: self.returns[asset].mean() * 252 for asset in self.assets
            }

            adjusted_weights = self.allocator.allocate(
                signal_weights,
                self.assets,
                volatilities=volatilities,
                returns=recent_returns,
            )
        else:
            adjusted_weights = signal_weights

        # Optimiser avec les poids ajustés comme point de départ
        return self.optimize_allocation(adjusted_weights)

    def stress_test(
        self,
        scenario: str = "market_crash",
        custom_returns: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Teste l'allocation actuelle sous différents scénarios de stress.

        Args:
            scenario: Type de scénario ('market_crash', 'high_volatility', 'custom')
            custom_returns: Rendements personnalisés pour le scénario 'custom'

        Returns:
            Résultats du stress test
        """
        if not self.current_weights:
            logger.warning("Aucune allocation actuelle pour le stress test")
            return {}

        # Définir les scénarios
        if scenario == "market_crash":
            # Simuler un crash de marché (-30% sur tous les actifs avec corrélations élevées)
            shock_returns = pd.DataFrame(index=range(20), columns=self.assets)
            base_crash = -0.30  # -30%

            for asset in self.assets:
                # Ajouter un bruit aléatoire autour du crash de base
                asset_crash = base_crash + np.random.normal(0, 0.05)
                daily_returns = asset_crash / 20  # Répartir sur 20 jours
                shock_returns[asset] = np.random.normal(
                    daily_returns, abs(daily_returns * 0.2), 20
                )

        elif scenario == "high_volatility":
            # Simuler une période de haute volatilité
            shock_returns = pd.DataFrame(index=range(20), columns=self.assets)

            for asset in self.assets:
                # Volatilité 3x plus élevée que la normale
                normal_vol = self.returns[asset].std()
                high_vol = normal_vol * 3
                shock_returns[asset] = np.random.normal(0, high_vol, 20)

        elif scenario == "custom" and custom_returns is not None:
            shock_returns = custom_returns
        else:
            logger.error(f"Scénario de stress test non reconnu: {scenario}")
            return {}

        # Calculer les performances du portefeuille actuel sous ce scénario
        weights_series = pd.Series(self.current_weights)
        portfolio_returns = shock_returns.dot(weights_series)

        # Calculer les métriques de risque
        max_drawdown = (
            portfolio_returns.cumsum().cummax() - portfolio_returns.cumsum()
        ).max()
        total_return = portfolio_returns.sum()
        volatility = portfolio_returns.std() * np.sqrt(252)
        var = calculate_var(portfolio_returns, confidence=0.95)
        es = calculate_es(portfolio_returns, confidence=0.95)

        return {
            "scenario": scenario,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "var_95": var,
            "es_95": es,
            "portfolio_returns": portfolio_returns,
        }
