"""
Module d'optimisation de portefeuille multi-facteurs.

Ce module permet d'optimiser l'allocation d'actifs en prenant en compte:
- Multiples facteurs de risque et rendement
- Modèles de risque avancés
- Contraintes personnalisables
- Techniques d'optimisation robustes
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.optimize as sco

from ai_trading.risk.risk_metrics import calculate_es, calculate_var

logger = logging.getLogger(__name__)


@dataclass
class FactorModel:
    """
    Modèle de facteurs pour l'analyse et l'optimisation de portefeuille.

    Attributes:
        name: Nom du modèle
        factors: DataFrame des rendements des facteurs
        exposures: DataFrame des expositions aux facteurs par actif
        specific_returns: DataFrame des rendements spécifiques (non expliqués par les facteurs)
        specific_risks: Série des risques spécifiques par actif
        factor_cov: Matrice de covariance des facteurs
    """

    name: str
    factors: pd.DataFrame
    exposures: pd.DataFrame
    specific_returns: pd.DataFrame
    specific_risks: pd.Series
    factor_cov: pd.DataFrame

    @property
    def assets(self) -> List[str]:
        """Liste des actifs dans le modèle."""
        return list(self.exposures.index)

    @property
    def factor_names(self) -> List[str]:
        """Liste des noms de facteurs."""
        return list(self.factors.columns)


class PortfolioOptimizer:
    """
    Optimiseur de portefeuille multi-facteurs pour l'allocation stratégique d'actifs.

    Cette classe permet d'optimiser un portefeuille en tenant compte de multiples facteurs
    de risque et de rendement, avec différentes contraintes et objectifs personnalisables.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        factor_model: Optional[FactorModel] = None,
        lookback_window: int = 252,
        risk_free_rate: float = 0.0,
        transaction_costs: Optional[pd.Series] = None,
    ):
        """
        Initialise l'optimiseur de portefeuille.

        Args:
            returns: DataFrame des rendements historiques des actifs
            factor_model: Modèle de facteurs (optionnel)
            lookback_window: Fenêtre d'historique pour le calcul des statistiques
            risk_free_rate: Taux sans risque annualisé
            transaction_costs: Coûts de transaction par actif (en % du montant)
        """
        self.returns = returns
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.factor_model = factor_model
        self.transaction_costs = transaction_costs

        # Paramètres calculés
        self.assets = list(returns.columns)
        self.num_assets = len(self.assets)
        self.expected_returns = self._estimate_expected_returns()
        self.cov_matrix = self._estimate_covariance_matrix()

        logger.info(
            f"Optimiseur de portefeuille initialisé avec {self.num_assets} actifs"
        )

    def _estimate_expected_returns(self) -> pd.Series:
        """
        Estime les rendements attendus des actifs.

        Returns:
            Série des rendements attendus pour chaque actif
        """
        if self.factor_model:
            # Utiliser le modèle de facteurs pour estimer les rendements
            factor_means = self.factor_model.factors.iloc[
                -self.lookback_window :
            ].mean()
            expected_returns = self.factor_model.exposures @ factor_means
            expected_returns += self.factor_model.specific_returns.iloc[
                -self.lookback_window :
            ].mean()
        else:
            # Méthode simple: moyenne historique
            expected_returns = self.returns.iloc[-self.lookback_window :].mean()

        return expected_returns

    def _estimate_covariance_matrix(self) -> pd.DataFrame:
        """
        Estime la matrice de covariance des rendements des actifs.

        Returns:
            Matrice de covariance
        """
        if self.factor_model:
            # Utiliser le modèle de facteurs pour calculer la covariance
            factor_exposures = self.factor_model.exposures
            factor_cov = self.factor_model.factor_cov

            # Composante factorielle de la covariance
            factor_component = factor_exposures @ factor_cov @ factor_exposures.T

            # Ajouter les risques spécifiques (matrice diagonale)
            specific_component = pd.DataFrame(
                np.diag(self.factor_model.specific_risks**2),
                index=self.factor_model.assets,
                columns=self.factor_model.assets,
            )

            cov_matrix = factor_component + specific_component

        else:
            # Utiliser la covariance empirique avec shrinkage
            returns_data = self.returns.iloc[-self.lookback_window :]
            cov_matrix = self._shrink_covariance(returns_data.cov())

        return cov_matrix

    def _shrink_covariance(
        self, sample_cov: pd.DataFrame, shrinkage_factor: float = 0.2
    ) -> pd.DataFrame:
        """
        Applique le shrinkage à la matrice de covariance pour améliorer la stabilité.

        Args:
            sample_cov: Matrice de covariance empirique
            shrinkage_factor: Facteur de shrinkage (entre 0 et 1)

        Returns:
            Matrice de covariance avec shrinkage
        """
        # Créer la cible de shrinkage (matrice diagonale des variances)
        target = np.diag(np.diag(sample_cov.values))
        target = pd.DataFrame(
            target, index=sample_cov.index, columns=sample_cov.columns
        )

        # Appliquer le shrinkage
        shrunk_cov = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target

        return shrunk_cov

    def optimize_portfolio(
        self,
        objective: str = "sharpe",
        risk_aversion: float = 1.0,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        constraints: Optional[List[Dict]] = None,
        current_weights: Optional[pd.Series] = None,
        use_var: bool = False,
        var_confidence: float = 0.95,
    ) -> Dict:
        """
        Optimise le portefeuille selon l'objectif et les contraintes spécifiés.

        Args:
            objective: Objectif d'optimisation ('sharpe', 'min_risk', 'max_return', 'utility')
            risk_aversion: Coefficient d'aversion au risque (pour l'utilité)
            target_return: Rendement cible (pour min_risk avec contrainte de rendement)
            target_risk: Risque cible (pour max_return avec contrainte de risque)
            constraints: Liste de contraintes personnalisées
            current_weights: Poids actuels du portefeuille (pour minimiser les transactions)
            use_var: Si True, utilise la VaR comme mesure de risque, sinon la variance
            var_confidence: Niveau de confiance pour le calcul de la VaR

        Returns:
            Résultats de l'optimisation (poids, performance attendue, métriques de risque)
        """
        # Initialiser la fonction objectif et les contraintes
        if objective == "sharpe":
            objective_function = self._negative_sharpe_ratio
        elif objective == "min_risk":
            objective_function = (
                self._portfolio_variance if not use_var else self._portfolio_var
            )
        elif objective == "max_return":
            objective_function = self._negative_portfolio_return
        elif objective == "utility":
            objective_function = lambda w: self._utility_function(
                w, risk_aversion, use_var
            )
        else:
            raise ValueError(f"Objectif d'optimisation non reconnu: {objective}")

        # Contraintes de base: somme des poids = 1
        base_constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

        # Ajouter des contraintes supplémentaires
        all_constraints = base_constraints
        if constraints:
            all_constraints.extend(constraints)

        # Ajouter contrainte de rendement cible si spécifiée
        if target_return is not None and objective == "min_risk":
            all_constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: self._portfolio_return(x) - target_return,
                }
            )

        # Ajouter contrainte de risque cible si spécifiée
        if target_risk is not None and objective == "max_return":
            risk_fun = self._portfolio_variance if not use_var else self._portfolio_var
            all_constraints.append(
                {"type": "eq", "fun": lambda x: np.sqrt(risk_fun(x)) - target_risk}
            )

        # Bornes des poids (par défaut entre 0 et 1)
        bounds = tuple((0, 1) for _ in range(self.num_assets))

        # Poids initiaux pour l'optimisation
        if current_weights is not None:
            initial_weights = current_weights.values
        else:
            initial_weights = np.ones(self.num_assets) / self.num_assets

        # Exécuter l'optimisation
        optimization_result = sco.minimize(
            objective_function,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=all_constraints,
            options={"maxiter": 1000, "ftol": 1e-8},
        )

        if not optimization_result["success"]:
            logger.warning(f"L'optimisation a échoué: {optimization_result['message']}")

        # Récupérer les poids optimaux
        optimal_weights = pd.Series(optimization_result["x"], index=self.assets)

        # Arrondir les poids très faibles à zéro
        optimal_weights[optimal_weights < 1e-5] = 0
        if optimal_weights.sum() > 0:
            optimal_weights = optimal_weights / optimal_weights.sum()  # Renormaliser

        # Calculer les métriques du portefeuille optimisé
        expected_return = self._portfolio_return(optimal_weights)
        volatility = np.sqrt(self._portfolio_variance(optimal_weights))
        sharpe_ratio = (
            (expected_return - self.risk_free_rate) / volatility
            if volatility > 0
            else 0
        )

        # Calculer les métriques de risque avancées
        var_value = self._portfolio_var(optimal_weights, var_confidence)
        es_value = self._portfolio_es(optimal_weights, var_confidence)

        # Calculer le turnover si les poids actuels sont fournis
        turnover = 0
        if current_weights is not None:
            turnover = np.sum(np.abs(optimal_weights - current_weights))

        # Calculer le coût de transaction estimé
        transaction_cost = 0
        if self.transaction_costs is not None and current_weights is not None:
            cost_series = self.transaction_costs.reindex(self.assets).fillna(0)
            transaction_cost = np.sum(
                np.abs(optimal_weights - current_weights) * cost_series
            )

        # Calculer les expositions aux facteurs si un modèle de facteurs est disponible
        factor_exposures = None
        if self.factor_model:
            factor_exposures = optimal_weights @ self.factor_model.exposures

        return {
            "weights": optimal_weights,
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "var": var_value,
            "es": es_value,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "factor_exposures": factor_exposures,
            "objective_value": optimization_result["fun"],
            "convergence": optimization_result["success"],
        }

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calcule le rendement attendu du portefeuille.

        Args:
            weights: Poids des actifs

        Returns:
            Rendement attendu
        """
        return np.sum(self.expected_returns * weights)

    def _portfolio_variance(self, weights: np.ndarray) -> float:
        """
        Calcule la variance du portefeuille.

        Args:
            weights: Poids des actifs

        Returns:
            Variance du portefeuille
        """
        return weights.T @ self.cov_matrix.values @ weights

    def _portfolio_var(self, weights: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calcule la Value-at-Risk (VaR) du portefeuille.

        Args:
            weights: Poids des actifs
            confidence: Niveau de confiance

        Returns:
            VaR au niveau de confiance spécifié
        """
        portfolio_returns = self.returns @ weights
        return calculate_var(portfolio_returns, confidence=confidence)

    def _portfolio_es(self, weights: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calcule l'Expected Shortfall (ES) du portefeuille.

        Args:
            weights: Poids des actifs
            confidence: Niveau de confiance

        Returns:
            ES au niveau de confiance spécifié
        """
        portfolio_returns = self.returns @ weights
        return calculate_es(portfolio_returns, confidence=confidence)

    def _negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calcule l'opposé du ratio de Sharpe (pour minimisation).

        Args:
            weights: Poids des actifs

        Returns:
            Négatif du ratio de Sharpe
        """
        port_return = self._portfolio_return(weights)
        port_risk = np.sqrt(self._portfolio_variance(weights))

        # Éviter la division par zéro
        if port_risk == 0:
            return -1000.0  # Valeur arbitrairement grande négative

        sharpe = (port_return - self.risk_free_rate) / port_risk
        return -sharpe  # Négatif car on minimise

    def _negative_portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calcule l'opposé du rendement attendu du portefeuille (pour minimisation).

        Args:
            weights: Poids des actifs

        Returns:
            Négatif du rendement attendu
        """
        return -self._portfolio_return(weights)

    def _utility_function(
        self, weights: np.ndarray, risk_aversion: float, use_var: bool = False
    ) -> float:
        """
        Calcule la fonction d'utilité du portefeuille.

        Args:
            weights: Poids des actifs
            risk_aversion: Coefficient d'aversion au risque
            use_var: Si True, utilise la VaR comme mesure de risque, sinon la variance

        Returns:
            Valeur de la fonction d'utilité
        """
        port_return = self._portfolio_return(weights)

        if use_var:
            risk_measure = self._portfolio_var(weights)
        else:
            risk_measure = self._portfolio_variance(weights)

        return port_return - risk_aversion * risk_measure

    def efficient_frontier(
        self,
        num_portfolios: int = 50,
        min_return: Optional[float] = None,
        max_return: Optional[float] = None,
        constraints: Optional[List[Dict]] = None,
        use_var: bool = False,
    ) -> pd.DataFrame:
        """
        Génère la frontière efficiente.

        Args:
            num_portfolios: Nombre de portefeuilles à générer
            min_return: Rendement minimal
            max_return: Rendement maximal
            constraints: Contraintes supplémentaires
            use_var: Si True, utilise la VaR comme mesure de risque

        Returns:
            DataFrame avec les portefeuilles sur la frontière efficiente
        """
        # Déterminer les bornes de rendement
        if min_return is None:
            min_vol_port = self.optimize_portfolio(
                objective="min_risk", constraints=constraints, use_var=use_var
            )
            min_return = min_vol_port["expected_return"]

        if max_return is None:
            max_ret_port = self.optimize_portfolio(
                objective="max_return", constraints=constraints
            )
            max_return = max_ret_port["expected_return"]

        # Générer les niveaux de rendement cible
        target_returns = np.linspace(min_return, max_return, num_portfolios)

        # Calculer les portefeuilles optimaux pour chaque niveau de rendement
        efficient_portfolios = []

        for target_return in target_returns:
            portfolio = self.optimize_portfolio(
                objective="min_risk",
                target_return=target_return,
                constraints=constraints,
                use_var=use_var,
            )

            # Ajouter le niveau de rendement cible
            portfolio["target_return"] = target_return
            efficient_portfolios.append(portfolio)

        # Extraire les données clés pour le DataFrame de retour
        frontier_data = []

        for port in efficient_portfolios:
            row = {
                "expected_return": port["expected_return"],
                "volatility": port["volatility"],
                "sharpe_ratio": port["sharpe_ratio"],
                "var": port["var"],
                "es": port["es"],
                "target_return": port["target_return"],
            }

            # Ajouter les poids
            for asset, weight in port["weights"].items():
                row[f"weight_{asset}"] = weight

            # Ajouter les expositions aux facteurs si disponibles
            if port["factor_exposures"] is not None:
                for factor, exposure in port["factor_exposures"].items():
                    row[f"exposure_{factor}"] = exposure

            frontier_data.append(row)

        return pd.DataFrame(frontier_data)

    def factor_risk_contribution(self, weights: pd.Series) -> pd.DataFrame:
        """
        Calcule la contribution de chaque facteur au risque total du portefeuille.

        Args:
            weights: Poids des actifs

        Returns:
            DataFrame avec les contributions au risque par facteur
        """
        if self.factor_model is None:
            raise ValueError("Un modèle de facteurs est requis pour cette analyse")

        # Exposition du portefeuille aux facteurs
        portfolio_exposures = weights @ self.factor_model.exposures

        # Contribution de chaque facteur au risque
        factor_cov = self.factor_model.factor_cov
        factor_risks = np.sqrt(np.diag(factor_cov))

        # Calculer la volatilité totale du portefeuille
        portfolio_risk = np.sqrt(self._portfolio_variance(weights))

        # Calculer la contribution marginale au risque pour chaque facteur
        factor_mcr = factor_cov @ portfolio_exposures

        # Transformer en contribution totale au risque
        factor_contributions = portfolio_exposures * factor_mcr / portfolio_risk

        # Calculer le risque spécifique (non expliqué par les facteurs)
        specific_risk = np.sqrt(sum((weights * self.factor_model.specific_risks) ** 2))
        specific_contribution = specific_risk**2 / portfolio_risk

        # Créer le DataFrame résultat
        result = pd.DataFrame(
            {
                "exposure": portfolio_exposures,
                "risk": factor_risks,
                "mcr": factor_mcr / portfolio_risk,  # Contribution marginale
                "contribution": factor_contributions,  # Contribution absolue
                "contribution_pct": factor_contributions
                / portfolio_risk,  # Contribution relative
            }
        )

        # Ajouter le risque spécifique
        specific_row = pd.Series(
            {
                "exposure": 1.0,
                "risk": specific_risk,
                "mcr": specific_risk / portfolio_risk,
                "contribution": specific_contribution,
                "contribution_pct": specific_contribution / portfolio_risk,
            },
            name="specific",
        )

        return pd.concat([result, specific_row.to_frame().T])


class BlackLittermanOptimizer(PortfolioOptimizer):
    """
    Optimiseur de portefeuille utilisant le modèle Black-Litterman.

    Cette classe étend l'optimiseur de base en incorporant le modèle Black-Litterman
    qui permet d'intégrer des vues subjectives avec des informations de marché.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_weights: pd.Series,
        risk_aversion: float = 2.5,
        tau: float = 0.025,
        factor_model: Optional[FactorModel] = None,
        lookback_window: int = 252,
        risk_free_rate: float = 0.0,
        transaction_costs: Optional[pd.Series] = None,
    ):
        """
        Initialise l'optimiseur Black-Litterman.

        Args:
            returns: DataFrame des rendements historiques des actifs
            market_weights: Poids des actifs dans le portefeuille de marché
            risk_aversion: Coefficient d'aversion au risque du marché
            tau: Paramètre d'incertitude sur les rendements d'équilibre
            factor_model: Modèle de facteurs (optionnel)
            lookback_window: Fenêtre d'historique pour le calcul des statistiques
            risk_free_rate: Taux sans risque annualisé
            transaction_costs: Coûts de transaction par actif (en % du montant)
        """
        super().__init__(
            returns, factor_model, lookback_window, risk_free_rate, transaction_costs
        )

        self.market_weights = market_weights.reindex(self.assets).fillna(0)
        self.risk_aversion = risk_aversion
        self.tau = tau

        # Calculer les rendements d'équilibre implicites
        self.equilibrium_returns = self._calculate_implied_returns()

        logger.info("Optimiseur Black-Litterman initialisé")

    def _calculate_implied_returns(self) -> pd.Series:
        """
        Calcule les rendements d'équilibre implicites selon le CAPM.

        Returns:
            Série des rendements d'équilibre implicites
        """
        return self.risk_aversion * self.cov_matrix @ self.market_weights

    def incorporate_views(
        self,
        view_matrix: pd.DataFrame,
        view_returns: pd.Series,
        view_confidences: pd.Series,
    ) -> pd.Series:
        """
        Incorpore des vues subjectives dans le modèle Black-Litterman.

        Args:
            view_matrix: Matrice de vues (lignes = vues, colonnes = actifs)
            view_returns: Rendements attendus pour chaque vue
            view_confidences: Confiance dans chaque vue (1/variance)

        Returns:
            Rendements combinés (postérieurs) après intégration des vues
        """
        # Préparer les données
        pi = self.equilibrium_returns.values  # Rendements d'équilibre
        Sigma = self.cov_matrix.values  # Matrice de covariance
        P = view_matrix.values  # Matrice de vues
        Q = view_returns.values  # Rendements des vues

        # Construire la matrice de covariance des vues
        Omega = np.diag(1 / view_confidences)

        # Calculer les rendements postérieurs
        A = self.tau * Sigma
        B = P @ A @ P.T + Omega
        C = np.linalg.inv(B)
        D = P @ pi
        E = Q - D

        # Formule de Black-Litterman
        posterior_returns = pi + A @ P.T @ C @ E

        # Retourner comme série
        return pd.Series(posterior_returns, index=self.assets)

    def optimize_bl_portfolio(
        self,
        view_matrix: Optional[pd.DataFrame] = None,
        view_returns: Optional[pd.Series] = None,
        view_confidences: Optional[pd.Series] = None,
        objective: str = "sharpe",
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        constraints: Optional[List[Dict]] = None,
        current_weights: Optional[pd.Series] = None,
        use_var: bool = False,
    ) -> Dict:
        """
        Optimise le portefeuille en utilisant le modèle Black-Litterman.

        Args:
            view_matrix: Matrice de vues (optionnelle)
            view_returns: Rendements attendus pour chaque vue (optionnel)
            view_confidences: Confiance dans chaque vue (optionnel)
            objective: Objectif d'optimisation
            target_return: Rendement cible
            target_risk: Risque cible
            constraints: Contraintes supplémentaires
            current_weights: Poids actuels
            use_var: Si True, utilise la VaR comme mesure de risque

        Returns:
            Résultats de l'optimisation
        """
        # Si des vues sont fournies, les incorporer
        if (
            view_matrix is not None
            and view_returns is not None
            and view_confidences is not None
        ):
            # Temporairement remplacer les rendements attendus
            original_returns = self.expected_returns
            self.expected_returns = self.incorporate_views(
                view_matrix, view_returns, view_confidences
            )

            # Optimiser avec les rendements mis à jour
            result = self.optimize_portfolio(
                objective=objective,
                risk_aversion=self.risk_aversion,
                target_return=target_return,
                target_risk=target_risk,
                constraints=constraints,
                current_weights=current_weights,
                use_var=use_var,
            )

            # Restaurer les rendements originaux
            self.expected_returns = original_returns

            # Ajouter des informations spécifiques à Black-Litterman
            result["equilibrium_returns"] = self.equilibrium_returns
            result["posterior_returns"] = self.expected_returns

            return result
        else:
            # Sans vues, utiliser directement les rendements d'équilibre
            original_returns = self.expected_returns
            self.expected_returns = self.equilibrium_returns

            result = self.optimize_portfolio(
                objective=objective,
                risk_aversion=self.risk_aversion,
                target_return=target_return,
                target_risk=target_risk,
                constraints=constraints,
                current_weights=current_weights,
                use_var=use_var,
            )

            # Restaurer les rendements originaux
            self.expected_returns = original_returns

            # Ajouter des informations spécifiques à Black-Litterman
            result["equilibrium_returns"] = self.equilibrium_returns

            return result
