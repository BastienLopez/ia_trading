"""
Module de système d'allocation complet de portefeuille.

Ce module implémente un système d'allocation de portefeuille qui:
1. Détecte le régime de marché actuel
2. Optimise l'allocation selon le régime détecté
3. Gère le rééquilibrage du portefeuille
4. Intègre les signaux de trading
5. Effectue des stress tests et simulations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import scipy.optimize as sco
from scipy import stats

# Configuration du logging
logger = logging.getLogger(__name__)

# Import des modules requis
try:
    from ai_trading.rl.models.portfolio_optimization import FactorModel
except ImportError:
    # Modèle simplifié pour éviter les dépendances
    class FactorModel:
        """Modèle de facteurs simplifié."""
        
        def __init__(self, name, factors, exposures, specific_returns, specific_risks, factor_cov):
            self.name = name
            self.factors = factors
            self.exposures = exposures
            self.specific_returns = specific_returns
            self.specific_risks = specific_risks
            self.factor_cov = factor_cov
        
        def get_factor_exposures(self, asset):
            """Retourne les expositions aux facteurs pour un actif."""
            return self.exposures.loc[asset].values

class MarketRegime(str, Enum):
    """Types de régimes de marché."""
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    BEARISH = "bearish"
    BULLISH = "bullish"

@dataclass
class AllocationResult:
    """Résultat d'une optimisation d'allocation."""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var: float  # Value at Risk
    es: float   # Expected Shortfall
    regime: MarketRegime
    method: str
    
    def __post_init__(self):
        """Validation après initialisation."""
        # Vérifier que les poids somment à 1
        if not np.isclose(sum(self.weights), 1.0, atol=1e-4):
            logger.warning(f"Les poids ne somment pas à 1: {sum(self.weights)}")

class CompleteAllocationSystem:
    """
    Système complet d'allocation de portefeuille.
    """
    
    def __init__(self, returns: pd.DataFrame, prices: pd.DataFrame = None, 
                 factor_model: FactorModel = None, lookback_window: int = 252,
                 risk_free_rate: float = 0.0, rebalance_threshold: float = 0.05,
                 max_single_asset_weight: float = 0.3, min_single_asset_weight: float = 0.0,
                 risk_budget_method: str = "equal", optimization_method: str = "sharpe"):
        """
        Initialise le système d'allocation.
        
        Args:
            returns: DataFrame des rendements journaliers
            prices: DataFrame des prix (optionnel)
            factor_model: Modèle de facteurs (optionnel)
            lookback_window: Fenêtre d'historique pour les calculs
            risk_free_rate: Taux sans risque annualisé
            rebalance_threshold: Seuil pour déclencher un rééquilibrage
            max_single_asset_weight: Poids maximum par actif
            min_single_asset_weight: Poids minimum par actif
            risk_budget_method: Méthode de budgétisation du risque ('equal', 'relative_vol', 'custom')
            optimization_method: Méthode d'optimisation ('sharpe', 'min_var', 'custom')
        """
        self.returns = returns
        self.prices = prices if prices is not None else (1 + returns).cumprod()
        self.factor_model = factor_model
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.rebalance_threshold = rebalance_threshold
        self.max_single_asset_weight = max_single_asset_weight
        self.min_single_asset_weight = min_single_asset_weight
        self.risk_budget_method = risk_budget_method
        self.optimization_method = optimization_method
        
        # Attributs calculés
        self.num_assets = len(returns.columns)
        self.assets = list(returns.columns)
        self.market_regime = self.detect_market_regime()
        self.current_weights = None
        self.historical_allocations = pd.DataFrame(columns=self.assets + ['date', 'regime', 'method'])
        
        logger.info(f"Système d'allocation initialisé avec {self.num_assets} actifs")
    
    def detect_market_regime(self) -> MarketRegime:
        """
        Détecte le régime de marché actuel.
        
        Returns:
            Type de régime de marché
        """
        # Utiliser les données récentes selon la fenêtre lookback
        recent_returns = self.returns.iloc[-self.lookback_window:]
        
        # Indicateurs clés pour la détection
        avg_return = recent_returns.mean().mean() * 252  # Annualisé
        vol = recent_returns.std().mean() * np.sqrt(252)  # Annualisé
        recent_trend = recent_returns.iloc[-20:].mean().mean() * 252  # Tendance sur 20 jours
        
        # Analyser les indicateurs pour déterminer le régime
        if vol > 0.3:  # Seuil arbitraire pour haute volatilité
            regime = MarketRegime.HIGH_VOL
        elif avg_return < -0.05 or recent_trend < -0.1:  # Seuil baissier
            regime = MarketRegime.BEARISH
        elif avg_return > 0.15 and recent_trend > 0.2:  # Seuil haussier
            regime = MarketRegime.BULLISH
        else:
            regime = MarketRegime.NORMAL
        
        logger.info(f"Régime de marché détecté: {regime}")
        self.market_regime = regime
        return regime
    
    def optimize_allocation(self, custom_constraints: List = None) -> AllocationResult:
        """
        Optimise l'allocation selon le régime de marché.
        
        Args:
            custom_constraints: Contraintes supplémentaires pour l'optimisation
            
        Returns:
            Résultat de l'allocation
        """
        # Adapter la méthode d'optimisation selon le régime
        if self.market_regime == MarketRegime.HIGH_VOL:
            method = "min_var"  # Minimiser la variance en période de haute volatilité
        elif self.market_regime == MarketRegime.BEARISH:
            method = "min_var_with_hedge"  # Minimiser la variance avec protection
        elif self.market_regime == MarketRegime.BULLISH:
            method = "max_return_with_risk"  # Maximiser le rendement avec contrainte de risque
        else:
            method = self.optimization_method
        
        # Récupérer les données récentes
        recent_returns = self.returns.iloc[-self.lookback_window:]
        
        # Calculer les statistiques nécessaires
        mean_returns = recent_returns.mean() * 252  # Annualisé
        cov_matrix = recent_returns.cov() * 252  # Annualisée
        
        # Définir les contraintes de base
        bounds = [(self.min_single_asset_weight, self.max_single_asset_weight) for _ in range(self.num_assets)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Somme des poids = 1
        
        if custom_constraints:
            constraints.extend(custom_constraints)
        
        # Fonction objectif selon la méthode
        if method == "min_var":
            obj_function = lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        elif method == "sharpe":
            obj_function = lambda weights: -self._calculate_sharpe_ratio(weights, mean_returns, cov_matrix)
        elif method == "min_var_with_hedge":
            obj_function = lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) - \
                                          0.5 * np.dot(weights, mean_returns)
        elif method == "max_return_with_risk":
            obj_function = lambda weights: -np.dot(weights, mean_returns) + \
                                          0.2 * np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        else:
            raise ValueError(f"Méthode d'optimisation non prise en charge: {method}")
        
        # Initialiser avec des poids égaux
        initial_weights = np.array([1 / self.num_assets] * self.num_assets)
        
        # Effectuer l'optimisation
        result = sco.minimize(obj_function, initial_weights, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
        
        if not result['success']:
            logger.warning(f"L'optimisation n'a pas convergé: {result['message']}")
            # Fallback sur l'allocation équipondérée
            optimal_weights = pd.Series(initial_weights, index=self.assets)
        else:
            optimal_weights = pd.Series(result['x'], index=self.assets)
        
        # Calculer les métriques de l'allocation
        portfolio_return = np.dot(optimal_weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calcul de VaR et ES
        var_95 = self._calculate_var(optimal_weights, mean_returns, cov_matrix, alpha=0.05)
        es_95 = self._calculate_es(optimal_weights, mean_returns, cov_matrix, alpha=0.05)
        
        # Créer et retourner le résultat
        allocation_result = AllocationResult(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var=var_95,
            es=es_95,
            regime=self.market_regime,
            method=method
        )
        
        # Mettre à jour les poids courants
        self.current_weights = optimal_weights
        
        # Enregistrer dans l'historique
        self._record_allocation(allocation_result)
        
        return allocation_result
    
    def risk_parity(self) -> AllocationResult:
        """
        Implémente une allocation en parité de risque.
        
        Returns:
            Résultat de l'allocation
        """
        # Récupérer les données récentes
        recent_returns = self.returns.iloc[-self.lookback_window:]
        
        # Calculer les statistiques nécessaires
        mean_returns = recent_returns.mean() * 252  # Annualisé
        cov_matrix = recent_returns.cov() * 252  # Annualisée
        
        # Pour la parité de risque, on inverse la volatilité de chaque actif
        volatilities = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1 / volatilities
        
        # Normaliser pour obtenir des poids qui somment à 1
        weights = inv_vol / np.sum(inv_vol)
        
        # Créer une série avec les noms des actifs
        risk_parity_weights = pd.Series(weights, index=self.assets)
        
        # Calculer les métriques de l'allocation
        portfolio_return = np.dot(risk_parity_weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(risk_parity_weights.T, np.dot(cov_matrix, risk_parity_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calcul de VaR et ES
        var_95 = self._calculate_var(risk_parity_weights, mean_returns, cov_matrix, alpha=0.05)
        es_95 = self._calculate_es(risk_parity_weights, mean_returns, cov_matrix, alpha=0.05)
        
        # Créer et retourner le résultat
        allocation_result = AllocationResult(
            weights=risk_parity_weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var=var_95,
            es=es_95,
            regime=self.market_regime,
            method="risk_parity"
        )
        
        # Mettre à jour les poids courants
        self.current_weights = risk_parity_weights
        
        # Enregistrer dans l'historique
        self._record_allocation(allocation_result)
        
        return allocation_result
    
    def optimize_different_regimes(self) -> Dict[MarketRegime, AllocationResult]:
        """
        Optimise l'allocation pour différents régimes de marché.
        
        Returns:
            Dictionnaire des résultats d'allocation par régime
        """
        results = {}
        original_regime = self.market_regime
        
        for regime in MarketRegime:
            self.market_regime = regime
            results[regime] = self.optimize_allocation()
        
        # Restaurer le régime original
        self.market_regime = original_regime
        
        return results
    
    def rebalance_need(self, current_prices: pd.Series = None) -> Tuple[bool, pd.Series]:
        """
        Détermine si un rééquilibrage est nécessaire.
        
        Args:
            current_prices: Prix actuels des actifs (optionnel)
            
        Returns:
            Tuple (besoin de rééquilibrage, déviations des poids)
        """
        if self.current_weights is None:
            logger.warning("Pas d'allocation courante définie")
            return False, None
        
        # Utiliser les derniers prix disponibles si non fournis
        if current_prices is None:
            current_prices = self.prices.iloc[-1]
        
        # Calculer la valeur actuelle de chaque position
        position_values = self.current_weights * 1  # Valeur normalisée du portefeuille
        
        # Calculer les nouveaux poids basés sur l'évolution des prix
        price_changes = current_prices / self.prices.iloc[-1]
        new_position_values = position_values * price_changes
        new_portfolio_value = new_position_values.sum()
        drift_weights = new_position_values / new_portfolio_value
        
        # Calculer les déviations
        weight_deviations = drift_weights - self.current_weights
        max_deviation = weight_deviations.abs().max()
        
        # Déterminer si un rééquilibrage est nécessaire
        need_rebalance = max_deviation > self.rebalance_threshold
        
        if need_rebalance:
            logger.info(f"Rééquilibrage nécessaire, déviation maximale: {max_deviation:.4f}")
        
        return need_rebalance, weight_deviations
    
    def get_rebalance_plan(self, target_weights: pd.Series = None) -> pd.DataFrame:
        """
        Génère un plan de rééquilibrage.
        
        Args:
            target_weights: Poids cibles pour le rééquilibrage (optionnel)
            
        Returns:
            DataFrame avec les actions de rééquilibrage
        """
        if self.current_weights is None:
            logger.warning("Pas d'allocation courante définie")
            return pd.DataFrame()
        
        if target_weights is None:
            # Utiliser l'optimisation pour le régime actuel
            target_allocation = self.optimize_allocation()
            target_weights = target_allocation.weights
        
        # Calculer les déviations par rapport aux poids cibles
        deviations = target_weights - self.current_weights
        
        # Créer le plan de rééquilibrage
        rebalance_plan = pd.DataFrame({
            'asset': self.assets,
            'current_weight': self.current_weights.values,
            'target_weight': target_weights.values,
            'deviation': deviations.values,
            'action': ['buy' if d > 0 else 'sell' if d < 0 else 'hold' for d in deviations.values]
        })
        
        # Trier par amplitude de déviation (décroissante)
        rebalance_plan = rebalance_plan.sort_values(by='deviation', key=abs, ascending=False)
        
        return rebalance_plan
    
    def integrate_signals(self, trading_signals: Dict[str, float]) -> AllocationResult:
        """
        Intègre des signaux de trading dans l'allocation.
        
        Args:
            trading_signals: Dictionnaire {actif: score de signal}
            
        Returns:
            Résultat de l'allocation ajustée
        """
        if not trading_signals:
            logger.warning("Aucun signal de trading fourni")
            return self.optimize_allocation()
        
        # Récupérer les données récentes
        recent_returns = self.returns.iloc[-self.lookback_window:]
        
        # Calculer les statistiques nécessaires
        mean_returns = recent_returns.mean() * 252  # Annualisé
        cov_matrix = recent_returns.cov() * 252  # Annualisée
        
        # Ajuster les rendements attendus en fonction des signaux
        adjusted_returns = mean_returns.copy()
        for asset, signal in trading_signals.items():
            if asset in adjusted_returns.index:
                # Ajuster le rendement attendu selon le signal
                adjustment = signal * 0.1  # Facteur d'impact du signal
                adjusted_returns[asset] += adjustment
                logger.info(f"Rendement ajusté pour {asset}: {mean_returns[asset]:.4f} -> {adjusted_returns[asset]:.4f}")
        
        # Définir les contraintes
        bounds = [(self.min_single_asset_weight, self.max_single_asset_weight) for _ in range(self.num_assets)]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Fonction objectif pour maximiser le rendement ajusté avec pénalité de variance
        obj_function = lambda weights: -(np.dot(weights, adjusted_returns) - \
                                       0.5 * np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
        
        # Initialiser avec les poids actuels ou égaux
        initial_weights = self.current_weights.values if self.current_weights is not None else \
                         np.array([1 / self.num_assets] * self.num_assets)
        
        # Effectuer l'optimisation
        result = sco.minimize(obj_function, initial_weights, method='SLSQP', 
                             bounds=bounds, constraints=constraints)
        
        if not result['success']:
            logger.warning(f"L'optimisation avec signaux n'a pas convergé: {result['message']}")
            # Fallback sur l'allocation courante ou équipondérée
            signal_weights = pd.Series(initial_weights, index=self.assets)
        else:
            signal_weights = pd.Series(result['x'], index=self.assets)
        
        # Calculer les métriques de l'allocation
        portfolio_return = np.dot(signal_weights, adjusted_returns)
        portfolio_vol = np.sqrt(np.dot(signal_weights.T, np.dot(cov_matrix, signal_weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calcul de VaR et ES
        var_95 = self._calculate_var(signal_weights, adjusted_returns, cov_matrix, alpha=0.05)
        es_95 = self._calculate_es(signal_weights, adjusted_returns, cov_matrix, alpha=0.05)
        
        # Créer et retourner le résultat
        allocation_result = AllocationResult(
            weights=signal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            var=var_95,
            es=es_95,
            regime=self.market_regime,
            method="signal_integrated"
        )
        
        # Mettre à jour les poids courants
        self.current_weights = signal_weights
        
        # Enregistrer dans l'historique
        self._record_allocation(allocation_result)
        
        return allocation_result
    
    def stress_test(self, scenario_name: str, shock_factors: Dict[str, float]) -> pd.DataFrame:
        """
        Effectue un stress test sur l'allocation actuelle.
        
        Args:
            scenario_name: Nom du scénario de stress
            shock_factors: Dictionnaire {actif: facteur de choc}
            
        Returns:
            DataFrame avec les résultats du stress test
        """
        if self.current_weights is None:
            logger.warning("Pas d'allocation courante définie pour le stress test")
            return pd.DataFrame()
        
        # Récupérer les données récentes
        recent_returns = self.returns.iloc[-self.lookback_window:]
        
        # Calculer les statistiques de base
        mean_returns = recent_returns.mean() * 252
        cov_matrix = recent_returns.cov() * 252
        
        # Appliquer les chocs aux rendements moyens
        shocked_returns = mean_returns.copy()
        for asset, shock in shock_factors.items():
            if asset in shocked_returns.index:
                shocked_returns[asset] += shock
        
        # Calculer les métriques sous stress
        stressed_portfolio_return = np.dot(self.current_weights, shocked_returns)
        stressed_portfolio_vol = np.sqrt(np.dot(self.current_weights.T, np.dot(cov_matrix, self.current_weights)))
        stressed_sharpe = (stressed_portfolio_return - self.risk_free_rate) / stressed_portfolio_vol \
                         if stressed_portfolio_vol > 0 else 0
        
        # Calcul de VaR et ES sous stress
        stressed_var_95 = self._calculate_var(self.current_weights, shocked_returns, cov_matrix, alpha=0.05)
        stressed_es_95 = self._calculate_es(self.current_weights, shocked_returns, cov_matrix, alpha=0.05)
        
        # Calculer les métriques normales pour comparaison
        normal_portfolio_return = np.dot(self.current_weights, mean_returns)
        normal_portfolio_vol = np.sqrt(np.dot(self.current_weights.T, np.dot(cov_matrix, self.current_weights)))
        normal_sharpe = (normal_portfolio_return - self.risk_free_rate) / normal_portfolio_vol \
                       if normal_portfolio_vol > 0 else 0
        normal_var_95 = self._calculate_var(self.current_weights, mean_returns, cov_matrix, alpha=0.05)
        normal_es_95 = self._calculate_es(self.current_weights, mean_returns, cov_matrix, alpha=0.05)
        
        # Créer le rapport de stress test
        stress_results = pd.DataFrame({
            'metric': ['return', 'volatility', 'sharpe_ratio', 'var_95', 'es_95'],
            'normal': [normal_portfolio_return, normal_portfolio_vol, normal_sharpe, normal_var_95, normal_es_95],
            'stressed': [stressed_portfolio_return, stressed_portfolio_vol, stressed_sharpe, stressed_var_95, stressed_es_95],
            'delta': [stressed_portfolio_return - normal_portfolio_return,
                     stressed_portfolio_vol - normal_portfolio_vol,
                     stressed_sharpe - normal_sharpe,
                     stressed_var_95 - normal_var_95,
                     stressed_es_95 - normal_es_95],
            'delta_pct': [(stressed_portfolio_return - normal_portfolio_return) / abs(normal_portfolio_return) * 100 if normal_portfolio_return != 0 else float('inf'),
                         (stressed_portfolio_vol - normal_portfolio_vol) / normal_portfolio_vol * 100 if normal_portfolio_vol != 0 else float('inf'),
                         (stressed_sharpe - normal_sharpe) / normal_sharpe * 100 if normal_sharpe != 0 else float('inf'),
                         (stressed_var_95 - normal_var_95) / normal_var_95 * 100 if normal_var_95 != 0 else float('inf'),
                         (stressed_es_95 - normal_es_95) / normal_es_95 * 100 if normal_es_95 != 0 else float('inf')]
        })
        
        stress_results['scenario'] = scenario_name
        
        return stress_results
    
    def _calculate_sharpe_ratio(self, weights, mean_returns, cov_matrix):
        """Calcule le ratio de Sharpe pour une allocation donnée."""
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        if portfolio_vol == 0:
            return 0
        
        return (portfolio_return - self.risk_free_rate) / portfolio_vol
    
    def _calculate_var(self, weights, mean_returns, cov_matrix, alpha=0.05):
        """Calcule la Value-at-Risk pour une allocation donnée."""
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # VaR paramétrique (hypothèse de normalité)
        z_score = stats.norm.ppf(alpha)
        var = -portfolio_return + z_score * portfolio_vol
        return var
    
    def _calculate_es(self, weights, mean_returns, cov_matrix, alpha=0.05):
        """Calcule l'Expected Shortfall pour une allocation donnée."""
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # ES paramétrique (hypothèse de normalité)
        z_score = stats.norm.ppf(alpha)
        es = -portfolio_return + portfolio_vol * stats.norm.pdf(z_score) / alpha
        return es
    
    def _record_allocation(self, allocation_result):
        """Enregistre une allocation dans l'historique."""
        record = allocation_result.weights.copy()
        record['date'] = pd.Timestamp.now()
        record['regime'] = allocation_result.regime
        record['method'] = allocation_result.method
        
        self.historical_allocations = pd.concat([self.historical_allocations, 
                                                pd.DataFrame([record])], 
                                               ignore_index=True)
