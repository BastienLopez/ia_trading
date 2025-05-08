"""
Gestionnaire de risques avancé avec VaR et allocation de capital adaptative pour l'AI Trading.
"""

import logging
from datetime import datetime

import numpy as np
from scipy import stats

from ai_trading.config import VISUALIZATION_DIR
from ai_trading.rl.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Dossier pour les visualisations de risque
RISK_VIZ_DIR = VISUALIZATION_DIR / "risk"
RISK_VIZ_DIR.mkdir(parents=True, exist_ok=True)


class AdvancedRiskManager(RiskManager):
    """
    Gestionnaire de risques avancé avec VaR et allocation de capital adaptative.

    Étend le RiskManager de base avec des fonctionnalités avancées comme:
    - Value-at-Risk (VaR) paramétrique et historique
    - Conditional Value-at-Risk (CVaR) / Expected Shortfall
    - Allocation de capital adaptative basée sur le risque
    """

    def __init__(self, config=None):
        """
        Initialise le gestionnaire de risques avancé.

        Args:
            config (dict, optional): Configuration du gestionnaire de risques
                - var_confidence_level (float): Niveau de confiance pour le VaR (0.95, 0.99)
                - var_horizon (int): Horizon temporel pour le VaR en jours
                - var_method (str): Méthode de calcul du VaR ('parametric', 'historical', 'monte_carlo')
                - max_var_limit (float): Limite maximale de VaR en % du capital
                - cvar_confidence_level (float): Niveau de confiance pour le CVaR
                - adaptive_capital_allocation (bool): Activer l'allocation adaptative du capital
                - kelly_fraction (float): Fraction de Kelly pour l'allocation (0.5 = demi-Kelly)
                - max_drawdown_limit (float): Limite maximale de drawdown en % du capital
                - stress_test_scenarios (list): Liste des scénarios de stress test
                - risk_parity_weights (bool): Utiliser la parité des risques pour la pondération
                - use_multilevel_risk (bool): Activer la gestion multi-niveaux des risques
                - strategy_risk_weight (float): Poids du risque au niveau de la stratégie
                - portfolio_risk_weight (float): Poids du risque au niveau du portefeuille
                - market_risk_weight (float): Poids du risque au niveau du marché
                - max_correlation_exposure (float): Exposition maximale aux actifs corrélés
        """
        # Initialiser avec la classe parent
        super().__init__(config)

        # Paramètres VaR
        self.var_confidence_level = self.config.get("var_confidence_level", 0.95)
        self.var_horizon = self.config.get("var_horizon", 1)  # en jours
        self.var_method = self.config.get("var_method", "parametric")
        self.max_var_limit = self.config.get("max_var_limit", 0.05)  # 5% du capital

        # Paramètres CVaR
        self.cvar_confidence_level = self.config.get("cvar_confidence_level", 0.95)

        # Paramètres d'allocation de capital
        self.use_adaptive_allocation = self.config.get(
            "adaptive_capital_allocation", True
        )
        self.kelly_fraction = self.config.get(
            "kelly_fraction", 0.5
        )  # Demi-Kelly par défaut
        self.max_drawdown_limit = self.config.get(
            "max_drawdown_limit", 0.20
        )  # 20% max drawdown

        # Paramètres de diversification
        self.risk_parity_weights = self.config.get("risk_parity_weights", False)

        # Paramètres pour la gestion multi-niveaux des risques
        self.use_multilevel_risk = self.config.get("use_multilevel_risk", False)
        self.strategy_risk_weight = self.config.get(
            "strategy_risk_weight", 0.4
        )  # 40% poids niveau stratégie
        self.portfolio_risk_weight = self.config.get(
            "portfolio_risk_weight", 0.3
        )  # 30% poids niveau portefeuille
        self.market_risk_weight = self.config.get(
            "market_risk_weight", 0.3
        )  # 30% poids niveau marché
        self.max_correlation_exposure = self.config.get(
            "max_correlation_exposure", 0.7
        )  # 70% exposition max aux actifs corrélés

        # Historique des métriques de risque
        self.var_history = []
        self.cvar_history = []
        self.allocation_history = []
        self.drawdown_history = []
        self.multilevel_risk_history = []

        logger.info(
            f"Gestionnaire de risques avancé initialisé avec méthode VaR: {self.var_method}"
        )

    def calculate_parametric_var(self, returns, confidence_level=None, horizon=None):
        """
        Calcule la Value-at-Risk paramétrique basée sur la distribution normale.

        Args:
            returns (np.array): Rendements historiques
            confidence_level (float, optional): Niveau de confiance (0.95, 0.99)
            horizon (int, optional): Horizon temporel pour le VaR en jours

        Returns:
            float: Valeur de la VaR (en % du capital)
        """
        if confidence_level is None:
            confidence_level = self.var_confidence_level

        if horizon is None:
            horizon = self.var_horizon

        if len(returns) < 30:
            logger.warning(
                f"Données insuffisantes pour calculer la VaR paramétrique (besoin d'au moins 30 points, reçu {len(returns)})"
            )
            return 0.05  # Valeur par défaut prudente

        # Calculer la moyenne et l'écart-type des rendements
        mu = np.mean(returns)
        sigma = np.std(returns)

        # Facteur z pour le niveau de confiance (distribution normale)
        z = stats.norm.ppf(1 - confidence_level)

        # Calculer la VaR
        var = -(mu * horizon + z * sigma * np.sqrt(horizon))

        logger.debug(
            f"VaR paramétrique ({confidence_level*100}%, {horizon}j): {var:.2%}"
        )
        return var

    def calculate_historical_var(self, returns, confidence_level=None, horizon=None):
        """
        Calcule la Value-at-Risk historique basée sur les quantiles empiriques.

        Args:
            returns (np.array): Rendements historiques
            confidence_level (float, optional): Niveau de confiance (0.95, 0.99)
            horizon (int, optional): Horizon temporel pour le VaR en jours

        Returns:
            float: Valeur de la VaR (en % du capital)
        """
        if confidence_level is None:
            confidence_level = self.var_confidence_level

        if horizon is None:
            horizon = self.var_horizon

        if len(returns) < 30:
            logger.warning(
                f"Données insuffisantes pour calculer la VaR historique (besoin d'au moins 30 points, reçu {len(returns)})"
            )
            return 0.05  # Valeur par défaut prudente

        # Pour un horizon > 1, simuler des rendements sur l'horizon
        if horizon > 1:
            horizon_returns = []
            for i in range(len(returns) - horizon + 1):
                # Rendement composé sur l'horizon
                horizon_return = np.prod(1 + returns[i : i + horizon]) - 1
                horizon_returns.append(horizon_return)

            if len(horizon_returns) < 10:
                logger.warning(
                    f"Données insuffisantes pour calculer la VaR historique sur {horizon} jours"
                )
                # Approximer avec la règle de la racine carrée du temps
                var_one_day = np.percentile(returns, 100 * (1 - confidence_level))
                return -var_one_day * np.sqrt(horizon)

            returns_to_use = horizon_returns
        else:
            returns_to_use = returns

        # Calculer le quantile correspondant au niveau de confiance
        var = -np.percentile(returns_to_use, 100 * (1 - confidence_level))

        logger.debug(f"VaR historique ({confidence_level*100}%, {horizon}j): {var:.2%}")
        return var

    def calculate_monte_carlo_var(
        self, returns, confidence_level=None, horizon=None, n_simulations=10000
    ):
        """
        Calcule la Value-at-Risk par simulation Monte Carlo.

        Args:
            returns (np.array): Rendements historiques
            confidence_level (float, optional): Niveau de confiance (0.95, 0.99)
            horizon (int, optional): Horizon temporel pour le VaR en jours
            n_simulations (int): Nombre de simulations Monte Carlo

        Returns:
            float: Valeur de la VaR (en % du capital)
        """
        if confidence_level is None:
            confidence_level = self.var_confidence_level

        if horizon is None:
            horizon = self.var_horizon

        if len(returns) < 30:
            logger.warning(
                f"Données insuffisantes pour calculer la VaR Monte Carlo (besoin d'au moins 30 points, reçu {len(returns)})"
            )
            return 0.05  # Valeur par défaut prudente

        # Calculer la moyenne et l'écart-type des rendements
        mu = np.mean(returns)
        sigma = np.std(returns)

        # Simuler des scénarios de rendements
        np.random.seed(42)  # Pour reproductibilité
        simulated_returns = np.random.normal(
            mu * horizon, sigma * np.sqrt(horizon), n_simulations
        )

        # Calculer le quantile correspondant au niveau de confiance
        var = -np.percentile(simulated_returns, 100 * (1 - confidence_level))

        logger.debug(
            f"VaR Monte Carlo ({confidence_level*100}%, {horizon}j): {var:.2%}"
        )
        return var

    def calculate_var(self, df, method=None, confidence_level=None, horizon=None):
        """
        Calcule la Value-at-Risk selon la méthode spécifiée.

        Args:
            df (pd.DataFrame): DataFrame avec les prix historiques (colonne 'close' requise)
            method (str, optional): Méthode de calcul ('parametric', 'historical', 'monte_carlo')
            confidence_level (float, optional): Niveau de confiance (0.95, 0.99)
            horizon (int, optional): Horizon temporel pour le VaR en jours

        Returns:
            float: Valeur de la VaR (en % du capital)
        """
        if method is None:
            method = self.var_method

        if confidence_level is None:
            confidence_level = self.var_confidence_level

        if horizon is None:
            horizon = self.var_horizon

        # Vérifier que les données nécessaires sont disponibles
        if df is None or len(df) < 2 or "close" not in df.columns:
            logger.warning("Données insuffisantes pour calculer la VaR")
            return 0.05  # Valeur par défaut prudente

        # Calculer les rendements journaliers
        returns = df["close"].pct_change().dropna().values

        # Calculer la VaR selon la méthode spécifiée
        if method == "parametric":
            var = self.calculate_parametric_var(returns, confidence_level, horizon)
        elif method == "historical":
            var = self.calculate_historical_var(returns, confidence_level, horizon)
        elif method == "monte_carlo":
            var = self.calculate_monte_carlo_var(returns, confidence_level, horizon)
        else:
            logger.warning(
                f"Méthode VaR inconnue: {method}, utilisation de la méthode paramétrique"
            )
            var = self.calculate_parametric_var(returns, confidence_level, horizon)

        # Enregistrer dans l'historique
        self.var_history.append(
            {
                "timestamp": df.index[-1] if hasattr(df.index, "__getitem__") else None,
                "var": var,
                "method": method,
                "confidence_level": confidence_level,
                "horizon": horizon,
            }
        )

        return var

    def calculate_cvar(self, df, confidence_level=None, horizon=None):
        """
        Calcule la Conditional Value-at-Risk (CVaR) ou Expected Shortfall.

        Args:
            df (pd.DataFrame): DataFrame avec les prix historiques (colonne 'close' requise)
            confidence_level (float, optional): Niveau de confiance (0.95, 0.99)
            horizon (int, optional): Horizon temporel pour le CVaR en jours

        Returns:
            float: Valeur de la CVaR (en % du capital)
        """
        if confidence_level is None:
            confidence_level = self.cvar_confidence_level

        if horizon is None:
            horizon = self.var_horizon

        # Vérifier que les données nécessaires sont disponibles
        if df is None or len(df) < 2 or "close" not in df.columns:
            logger.warning("Données insuffisantes pour calculer la CVaR")
            return 0.08  # Valeur par défaut prudente

        # Calculer les rendements journaliers
        returns = df["close"].pct_change().dropna().values

        # Calculer le seuil VaR
        var_threshold = np.percentile(returns, 100 * (1 - confidence_level))

        # Isoler les rendements inférieurs au seuil VaR
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            logger.warning(
                "Pas assez de données dans la queue de distribution pour calculer la CVaR"
            )
            return -1.5 * var_threshold  # Approximation

        # Calculer la moyenne des rendements de la queue
        cvar = -np.mean(tail_returns)

        # Ajuster pour l'horizon temporel (approximation par racine carrée du temps)
        cvar_adjusted = cvar * np.sqrt(horizon)

        # Enregistrer dans l'historique
        self.cvar_history.append(
            {
                "timestamp": df.index[-1] if hasattr(df.index, "__getitem__") else None,
                "cvar": cvar_adjusted,
                "confidence_level": confidence_level,
                "horizon": horizon,
            }
        )

        logger.debug(f"CVaR ({confidence_level*100}%, {horizon}j): {cvar_adjusted:.2%}")
        return cvar_adjusted

    def kelly_criterion(self, win_probability, win_loss_ratio):
        """
        Calcule la fraction optimale du capital à allouer selon le critère de Kelly.

        Args:
            win_probability (float): Probabilité de gain
            win_loss_ratio (float): Ratio du gain moyen sur la perte moyenne

        Returns:
            float: Fraction optimale du capital à allouer
        """
        kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio

        # Appliquer la fraction de Kelly configurée (généralement 0.5 pour demi-Kelly)
        kelly_fraction *= self.kelly_fraction

        # Limiter à [0, 1]
        kelly_fraction = max(0, min(1, kelly_fraction))

        return kelly_fraction

    def calculate_win_probability_and_ratio(self, df, lookback=100):
        """
        Calcule la probabilité de gain et le ratio gain/perte sur les données historiques.

        Args:
            df (pd.DataFrame): DataFrame avec les prix historiques (colonne 'close' requise)
            lookback (int): Nombre de jours à considérer

        Returns:
            tuple: (probabilité de gain, ratio gain/perte)
        """
        if df is None or len(df) < 2 or "close" not in df.columns:
            logger.warning("Données insuffisantes pour calculer les statistiques")
            return 0.5, 1.0  # Valeurs par défaut neutres

        # Calculer les rendements journaliers
        returns = df["close"].pct_change().dropna()

        # Limiter au lookback
        if len(returns) > lookback:
            returns = returns.iloc[-lookback:]

        # Calculer les statistiques
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_probability = (
            len(positive_returns) / len(returns) if len(returns) > 0 else 0.5
        )

        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.01
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.01

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        return win_probability, win_loss_ratio

    def adaptive_capital_allocation(
        self, df, position_type="long", current_capital=None
    ):
        """
        Calcule l'allocation adaptative du capital basée sur le risque.

        Args:
            df (pd.DataFrame): DataFrame avec les prix historiques
            position_type (str): Type de position ('long' ou 'short')
            current_capital (float, optional): Capital disponible

        Returns:
            float: Fraction du capital à allouer (0-1)
        """
        if not self.use_adaptive_allocation:
            return self.max_position_size  # Utiliser la taille de position fixe

        # Calculer la VaR
        var = self.calculate_var(df)

        # Calculer le CVaR
        cvar = self.calculate_cvar(df)

        # Calculer probabilité de gain et ratio gain/perte
        win_probability, win_loss_ratio = self.calculate_win_probability_and_ratio(df)

        # Calcul de l'allocation Kelly
        kelly_allocation = self.kelly_criterion(win_probability, win_loss_ratio)

        # Ajuster l'allocation en fonction de la VaR
        var_scaling = (
            1.0 - (var / self.max_var_limit) if self.max_var_limit > 0 else 1.0
        )
        var_scaling = max(0.1, min(1.0, var_scaling))  # Limiter entre 0.1 et 1.0

        # Allocation finale
        allocation = kelly_allocation * var_scaling

        # Limiter à la taille maximale de position
        allocation = min(allocation, self.max_position_size)

        # Inverser pour les positions short
        if position_type == "short":
            allocation = -allocation

        # Enregistrer dans l'historique
        self.allocation_history.append(
            {
                "timestamp": df.index[-1] if hasattr(df.index, "__getitem__") else None,
                "allocation": allocation,
                "kelly": kelly_allocation,
                "var_scaling": var_scaling,
                "var": var,
                "cvar": cvar,
                "win_probability": win_probability,
                "win_loss_ratio": win_loss_ratio,
            }
        )

        logger.info(
            f"Allocation adaptative: {allocation:.2%} (Kelly: {kelly_allocation:.2%}, VaR scaling: {var_scaling:.2f})"
        )
        return allocation

    def risk_parity_allocation(self, asset_variances):
        """
        Calcule l'allocation selon le principe de parité des risques.

        Args:
            asset_variances (list): Liste des variances des actifs

        Returns:
            np.array: Poids des actifs selon la parité des risques
        """
        if not all(v > 0 for v in asset_variances):
            logger.warning(
                "Variances négatives ou nulles détectées, utilisation d'une allocation égale"
            )
            return np.ones(len(asset_variances)) / len(asset_variances)

        # Calculer l'inverse des variances
        inv_variances = 1.0 / np.array(asset_variances)

        # Normaliser pour obtenir des poids
        weights = inv_variances / np.sum(inv_variances)

        return weights

    def calculate_maximum_drawdown(self, portfolio_values):
        """
        Calcule le drawdown maximum sur l'historique du portefeuille.

        Args:
            portfolio_values (list): Historique des valeurs du portefeuille

        Returns:
            float: Drawdown maximum (en % du capital)
        """
        if len(portfolio_values) < 2:
            return 0.0

        # Convertir en numpy array pour faciliter les calculs
        values = np.array(portfolio_values)

        # Calculer les pics cumulatifs
        peaks = np.maximum.accumulate(values)

        # Calculer les drawdowns
        drawdowns = (values - peaks) / peaks

        # Calculer le drawdown maximum
        max_drawdown = abs(np.min(drawdowns))

        # Enregistrer dans l'historique
        self.drawdown_history.append(
            {
                "timestamp": datetime.now(),
                "max_drawdown": max_drawdown,
                "portfolio_value": values[-1],
            }
        )

        return max_drawdown

    def should_stop_trading(self, portfolio_values):
        """
        Détermine si le trading doit être arrêté en raison de risques excessifs.

        Args:
            portfolio_values (list): Historique des valeurs du portefeuille

        Returns:
            bool: True si le trading doit être arrêté
        """
        # Calculer le drawdown maximum
        max_drawdown = self.calculate_maximum_drawdown(portfolio_values)

        # Vérifier si le drawdown dépasse la limite
        if max_drawdown > self.max_drawdown_limit:
            logger.warning(
                f"Drawdown maximum ({max_drawdown:.2%}) dépasse la limite ({self.max_drawdown_limit:.2%})"
            )
            return True

        # Si le dernier VaR calculé dépasse la limite
        if (
            len(self.var_history) > 0
            and self.var_history[-1]["var"] > self.max_var_limit
        ):
            logger.warning(
                f"VaR ({self.var_history[-1]['var']:.2%}) dépasse la limite ({self.max_var_limit:.2%})"
            )
            return True

        return False

    def allocation_with_risk_limits(
        self, df, position_type="long", portfolio_values=None
    ):
        """
        Calcule l'allocation avec toutes les contraintes de risque.

        Args:
            df (pd.DataFrame): DataFrame avec les prix historiques
            position_type (str): Type de position ('long' ou 'short')
            portfolio_values (list, optional): Historique des valeurs du portefeuille

        Returns:
            float: Fraction du capital à allouer (-1 à 1)
        """
        # Vérifier si le trading doit être arrêté
        if portfolio_values is not None and self.should_stop_trading(portfolio_values):
            logger.warning("Trading arrêté en raison de risques excessifs")
            return 0.0

        # Calculer l'allocation adaptative
        allocation = self.adaptive_capital_allocation(df, position_type)

        # Calculer la VaR pour cette allocation
        var = self.calculate_var(df)
        position_var = allocation * var

        # Vérifier si la VaR de la position dépasse la limite
        if position_var > self.max_var_limit:
            # Ajuster l'allocation pour respecter la limite de VaR
            adjusted_allocation = (self.max_var_limit / var) if var > 0 else 0.0
            logger.info(
                f"Allocation ajustée pour respecter la limite VaR: {allocation:.2%} -> {adjusted_allocation:.2%}"
            )
            allocation = adjusted_allocation

        return allocation

    def multilevel_risk_management(
        self, df, market_data=None, portfolio_data=None, correlation_matrix=None
    ):
        """
        Applique une gestion multi-niveaux des risques (stratégie, portefeuille, marché).

        Cette approche considère différentes couches de risque:
        1. Niveau stratégie: Évalue le risque spécifique à l'instrument/stratégie actuel
        2. Niveau portefeuille: Évalue le risque global du portefeuille et la concentration
        3. Niveau marché: Évalue les conditions de marché générales et la volatilité systémique

        Args:
            df (pd.DataFrame): Données de l'instrument spécifique
            market_data (dict): Données de marché générales avec indices et indicateurs macro
            portfolio_data (dict): Données sur la composition et performance du portefeuille
            correlation_matrix (pd.DataFrame): Matrice de corrélation entre actifs

        Returns:
            dict: Allocation et métriques de risque multi-niveaux
        """
        if not self.use_multilevel_risk:
            logger.debug("Gestion multi-niveaux des risques non activée")
            return {"allocation": self.max_position_size, "risk_score": 0.5}

        logger.info("Application de la gestion multi-niveaux des risques")

        # 1. Évaluation du risque au niveau de la stratégie
        strategy_risk = self._evaluate_strategy_risk(df)

        # 2. Évaluation du risque au niveau du portefeuille
        portfolio_risk = self._evaluate_portfolio_risk(
            portfolio_data, correlation_matrix
        )

        # 3. Évaluation du risque au niveau du marché
        market_risk = self._evaluate_market_risk(market_data)

        # Calcul du score de risque global pondéré
        weighted_risk = (
            self.strategy_risk_weight * strategy_risk
            + self.portfolio_risk_weight * portfolio_risk
            + self.market_risk_weight * market_risk
        )

        # Normaliser le score entre 0 et 1 (1 = risque maximal)
        risk_score = min(max(weighted_risk, 0), 1)

        # Calculer l'allocation en fonction du score de risque inverse (1 - risk_score)
        risk_adjusted_allocation = self.max_position_size * (1 - risk_score)

        # Enregistrer dans l'historique
        risk_data = {
            "timestamp": datetime.now(),
            "strategy_risk": strategy_risk,
            "portfolio_risk": portfolio_risk,
            "market_risk": market_risk,
            "weighted_risk": weighted_risk,
            "risk_score": risk_score,
            "allocation": risk_adjusted_allocation,
        }
        self.multilevel_risk_history.append(risk_data)

        logger.info(
            f"Score de risque multi-niveaux: {risk_score:.2f}, Allocation: {risk_adjusted_allocation:.2%}"
        )

        return {
            "allocation": risk_adjusted_allocation,
            "risk_score": risk_score,
            "strategy_risk": strategy_risk,
            "portfolio_risk": portfolio_risk,
            "market_risk": market_risk,
            "risk_data": risk_data,
        }

    def _evaluate_strategy_risk(self, df):
        """
        Évalue le risque au niveau de la stratégie/instrument spécifique.

        Args:
            df (pd.DataFrame): Données de l'instrument

        Returns:
            float: Score de risque au niveau de la stratégie (0-1)
        """
        if df is None or len(df) < 30:
            return 0.5  # Risque moyen par défaut si données insuffisantes

        try:
            # Calculer la VaR et CVaR spécifiques à l'instrument
            var = self.calculate_var(df)
            cvar = self.calculate_cvar(df)

            # Calculer la volatilité récente (30 derniers jours)
            recent_returns = df["close"].pct_change().dropna().iloc[-30:]
            recent_volatility = recent_returns.std() * np.sqrt(252)  # Annualisée

            # Calculer le score de risque basé sur le ratio CVaR/VaR et la volatilité
            cvar_var_ratio = cvar / var if var > 0 else 2.0
            volatility_score = min(
                recent_volatility / 0.5, 1.0
            )  # Normaliser, avec 50% de volatilité annuelle comme maximum

            # Combiner les métriques (plus le score est élevé, plus le risque est élevé)
            strategy_risk = 0.5 * volatility_score + 0.5 * min(
                cvar_var_ratio - 1.0, 1.0
            )

            return min(max(strategy_risk, 0), 1)  # Limiter entre 0 et 1

        except Exception as e:
            logger.warning(
                f"Erreur lors de l'évaluation du risque de stratégie: {str(e)}"
            )
            return 0.5  # Risque moyen par défaut en cas d'erreur

    def _evaluate_portfolio_risk(self, portfolio_data, correlation_matrix):
        """
        Évalue le risque au niveau du portefeuille.

        Args:
            portfolio_data (dict): Données sur la composition et performance du portefeuille
            correlation_matrix (pd.DataFrame): Matrice de corrélation entre actifs

        Returns:
            float: Score de risque au niveau du portefeuille (0-1)
        """
        if portfolio_data is None:
            return 0.5  # Risque moyen par défaut si données insuffisantes

        try:
            # 1. Évaluer la concentration du portefeuille
            if "weights" in portfolio_data:
                weights = np.array(portfolio_data["weights"])
                # Calcul de l'indice de concentration HHI (Herfindahl-Hirschman Index)
                hhi = np.sum(weights**2)
                # Normaliser le HHI entre 0 et 1
                concentration_risk = (
                    min((hhi - 1 / len(weights)) / (1 - 1 / len(weights)), 1.0)
                    if len(weights) > 1
                    else 1.0
                )
            else:
                concentration_risk = 0.5

            # 2. Évaluer l'exposition aux actifs corrélés
            if correlation_matrix is not None and "assets" in portfolio_data:
                assets = portfolio_data["assets"]
                # Calculer la moyenne des corrélations entre actifs détenus
                correlations = []
                for i in range(len(assets)):
                    for j in range(i + 1, len(assets)):
                        if (
                            assets[i] in correlation_matrix.index
                            and assets[j] in correlation_matrix.columns
                        ):
                            correlations.append(
                                abs(correlation_matrix.loc[assets[i], assets[j]])
                            )

                if correlations:
                    avg_correlation = sum(correlations) / len(correlations)
                    correlation_risk = min(
                        avg_correlation / self.max_correlation_exposure, 1.0
                    )
                else:
                    correlation_risk = (
                        0.3  # Valeur par défaut si corrélations non disponibles
                    )
            else:
                correlation_risk = 0.3

            # 3. Évaluer la performance historique du portefeuille
            if "returns" in portfolio_data:
                returns = np.array(portfolio_data["returns"])
                if len(returns) > 0:
                    # Calculer le ratio de Sharpe (simplifié)
                    sharpe = (
                        np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                    )
                    performance_risk = max(
                        1 - (sharpe + 1) / 2, 0
                    )  # Normaliser entre 0 et 1
                else:
                    performance_risk = 0.5
            else:
                performance_risk = 0.5

            # Combiner les métriques (plus le score est élevé, plus le risque est élevé)
            portfolio_risk = (
                0.4 * concentration_risk
                + 0.3 * correlation_risk
                + 0.3 * performance_risk
            )

            return min(max(portfolio_risk, 0), 1)  # Limiter entre 0 et 1

        except Exception as e:
            logger.warning(
                f"Erreur lors de l'évaluation du risque de portefeuille: {str(e)}"
            )
            return 0.5  # Risque moyen par défaut en cas d'erreur

    def _evaluate_market_risk(self, market_data):
        """
        Évalue le risque au niveau du marché global.

        Args:
            market_data (dict): Données de marché générales

        Returns:
            float: Score de risque au niveau du marché (0-1)
        """
        if market_data is None:
            return 0.5  # Risque moyen par défaut si données insuffisantes

        try:
            market_risk_factors = []

            # 1. Indicateurs de volatilité du marché
            if "vix" in market_data:
                # VIX comme indicateur de volatilité, normalisé (20+ est considéré comme élevé)
                vix_risk = min(market_data["vix"] / 40, 1.0)
                market_risk_factors.append(vix_risk)

            # 2. Indicateurs de sentiment
            if "fear_greed_index" in market_data:
                # Indice Fear & Greed (0-100, 0=peur extrême, 100=avidité extrême)
                # Convertir en risque: 0 ou 100 = haut risque, 50 = risque bas
                fgi = market_data["fear_greed_index"]
                fgi_risk = min(abs(fgi - 50) / 50, 1.0)
                market_risk_factors.append(fgi_risk)

            # 3. Écarts de rendement (spreads) et indicateurs macro
            if "credit_spread" in market_data:
                # Écart de crédit normalisé (3%+ est considéré comme élevé)
                spread_risk = min(market_data["credit_spread"] / 0.03, 1.0)
                market_risk_factors.append(spread_risk)

            # 4. Liquidité du marché
            if "market_liquidity" in market_data:
                # Indicateur de liquidité normalisé (0=très liquide, 1=illiquide)
                liquidity_risk = market_data["market_liquidity"]
                market_risk_factors.append(liquidity_risk)

            # 5. Tendance du marché
            if "market_trend" in market_data:
                # -1 à 1, où -1 = forte baisse, 1 = forte hausse
                # Convertir en risque: -1 = haut risque, 1 = risque moyen
                trend = market_data["market_trend"]
                trend_risk = 0.5 * (1 - trend)  # Tendance négative = risque plus élevé
                market_risk_factors.append(trend_risk)

            # Calculer la moyenne des facteurs de risque, ou utiliser une valeur par défaut
            if market_risk_factors:
                market_risk = sum(market_risk_factors) / len(market_risk_factors)
            else:
                market_risk = 0.5

            return min(max(market_risk, 0), 1)  # Limiter entre 0 et 1

        except Exception as e:
            logger.warning(f"Erreur lors de l'évaluation du risque de marché: {str(e)}")
            return 0.5  # Risque moyen par défaut en cas d'erreur
