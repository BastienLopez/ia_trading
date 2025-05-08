"""
Métriques de risque pour l'évaluation et la gestion des portefeuilles.

Ce module fournit diverses mesures de risque financier, notamment:
- Value-at-Risk (VaR)
- Expected Shortfall (ES) / Conditional Value-at-Risk (CVaR)
- Drawdown
- Volatilité
- Ratios de performance ajustés au risque
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def calculate_var(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95,
    method: str = "historical",
    lookback: int = None,
) -> float:
    """
    Calcule la Value-at-Risk (VaR) d'une série de rendements.

    Args:
        returns: Série temporelle des rendements
        confidence: Niveau de confiance (ex: 0.95 pour 95%)
        method: Méthode de calcul ('historical', 'parametric', 'cornish_fisher')
        lookback: Période de lookback pour le calcul (utilise toutes les données si None)

    Returns:
        Valeur de la VaR (en valeur positive)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    if lookback is not None:
        returns = returns[-lookback:]

    if method == "historical":
        # Méthode historique (non-paramétrique)
        return -np.percentile(returns, 100 * (1 - confidence))

    elif method == "parametric":
        # Méthode paramétrique (supposant une distribution normale)
        mu = np.mean(returns)
        sigma = np.std(returns)
        return -(mu + stats.norm.ppf(1 - confidence) * sigma)

    elif method == "cornish_fisher":
        # Méthode de Cornish-Fisher (ajustement pour skewness et kurtosis)
        mu = np.mean(returns)
        sigma = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        z_c = stats.norm.ppf(1 - confidence)
        z_cf = (
            z_c
            + (z_c**2 - 1) * skew / 6
            + (z_c**3 - 3 * z_c) * kurt / 24
            - (2 * z_c**3 - 5 * z_c) * skew**2 / 36
        )

        return -(mu + sigma * z_cf)

    else:
        raise ValueError(
            f"Méthode non reconnue: {method}. Utilisez 'historical', 'parametric', ou 'cornish_fisher'"
        )


def calculate_es(
    returns: Union[pd.Series, np.ndarray],
    confidence: float = 0.95,
    method: str = "historical",
    lookback: int = None,
) -> float:
    """
    Calcule l'Expected Shortfall (ES), aussi appelé Conditional Value-at-Risk (CVaR).

    Args:
        returns: Série temporelle des rendements
        confidence: Niveau de confiance (ex: 0.95 pour 95%)
        method: Méthode de calcul ('historical', 'parametric')
        lookback: Période de lookback pour le calcul (utilise toutes les données si None)

    Returns:
        Valeur de l'ES (en valeur positive)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    if lookback is not None:
        returns = returns[-lookback:]

    if method == "historical":
        # Méthode historique (non-paramétrique)
        var = calculate_var(returns, confidence, "historical")
        return -np.mean(returns[returns <= -var])

    elif method == "parametric":
        # Méthode paramétrique (supposant une distribution normale)
        mu = np.mean(returns)
        sigma = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence)
        return -(mu + sigma * stats.norm.pdf(z_score) / (1 - confidence))

    else:
        raise ValueError(
            f"Méthode non reconnue: {method}. Utilisez 'historical' ou 'parametric'"
        )


def calculate_drawdown(
    returns: Union[pd.Series, np.ndarray],
) -> Tuple[float, float, int]:
    """
    Calcule le drawdown maximal d'une série de rendements.

    Args:
        returns: Série temporelle des rendements

    Returns:
        Tuple (drawdown maximal, durée maximale du drawdown, temps de récupération)
    """
    # Convertir en Series si nécessaire
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # Calculer l'évolution du cumul des rendements (wealth index)
    wealth_index = (1 + returns).cumprod()

    # Calculer le maximum roulant
    running_max = wealth_index.cummax()

    # Calculer le drawdown en pourcentage
    drawdown = (wealth_index / running_max - 1) * 100

    # Trouver le drawdown maximal
    max_drawdown = drawdown.min()

    # Calculer la durée du drawdown
    is_drawdown = drawdown < 0
    drawdown_periods = is_drawdown.astype(int)

    # Trouver le début et la fin des périodes de drawdown
    drawdown_starts = (drawdown_periods.diff() == 1).astype(int)
    drawdown_ends = (drawdown_periods.diff() == -1).astype(int)

    # Calculer les durées des drawdowns
    durations = []
    start_idx = None

    for i, (start, end) in enumerate(zip(drawdown_starts, drawdown_ends)):
        if start == 1:
            start_idx = i
        if end == 1 and start_idx is not None:
            durations.append(i - start_idx)
            start_idx = None

    # Si on est encore en drawdown à la fin de la série
    if start_idx is not None:
        durations.append(len(returns) - start_idx)

    max_duration = max(durations) if durations else 0

    # Calculer le temps de récupération après le drawdown maximal
    max_dd_idx = drawdown.idxmin()
    recovery_time = 0

    if max_dd_idx < len(returns) - 1:
        # Trouver la première fois où on revient au niveau précédent après le max drawdown
        for i in range(max_dd_idx + 1, len(returns)):
            if wealth_index.iloc[i] >= running_max.iloc[max_dd_idx]:
                recovery_time = i - max_dd_idx
                break

    return max_drawdown, max_duration, recovery_time


def calculate_volatility(
    returns: Union[pd.Series, np.ndarray],
    annualization_factor: int = 252,
    lookback: Optional[int] = None,
) -> float:
    """
    Calcule la volatilité annualisée des rendements.

    Args:
        returns: Série temporelle des rendements
        annualization_factor: Facteur d'annualisation (252 pour données journalières)
        lookback: Période de lookback pour le calcul (utilise toutes les données si None)

    Returns:
        Volatilité annualisée
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    if lookback is not None:
        returns = returns[-lookback:]

    return np.std(returns) * np.sqrt(annualization_factor)


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    Calcule le ratio de Sharpe annualisé.

    Args:
        returns: Série temporelle des rendements
        risk_free_rate: Taux sans risque (annualisé)
        annualization_factor: Facteur d'annualisation (252 pour données journalières)

    Returns:
        Ratio de Sharpe annualisé
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Convertir le taux sans risque à la même fréquence que les rendements
    rf_period = risk_free_rate / annualization_factor

    excess_returns = returns - rf_period
    mean_excess_return = np.mean(excess_returns)
    volatility = np.std(returns)

    if volatility == 0:
        return 0

    sharpe = mean_excess_return / volatility

    # Annualiser le ratio de Sharpe
    return sharpe * np.sqrt(annualization_factor)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Calcule le ratio de Sortino annualisé.

    Args:
        returns: Série temporelle des rendements
        risk_free_rate: Taux sans risque (annualisé)
        annualization_factor: Facteur d'annualisation (252 pour données journalières)
        target_return: Rendement minimal acceptable (MAR)

    Returns:
        Ratio de Sortino annualisé
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Convertir le taux sans risque à la même fréquence que les rendements
    rf_period = risk_free_rate / annualization_factor

    excess_returns = returns - rf_period
    mean_excess_return = np.mean(excess_returns)

    # Calculer les rendements négatifs par rapport à la cible
    negative_returns = returns[returns < target_return] - target_return

    # Calculer la semi-déviation
    if len(negative_returns) == 0:
        downside_deviation = 0
    else:
        downside_deviation = np.sqrt(np.mean(negative_returns**2))

    if downside_deviation == 0:
        return 0

    sortino = mean_excess_return / downside_deviation

    # Annualiser le ratio
    return sortino * np.sqrt(annualization_factor)


def calculate_risk_contribution(
    weights: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Calcule la contribution au risque de chaque actif dans un portefeuille.

    Args:
        weights: Poids des actifs
        cov_matrix: Matrice de covariance

    Returns:
        Contributions au risque (en pourcentage)
    """
    # Calculer le risque total du portefeuille
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Calculer la contribution marginale au risque
    marginal_contribution = cov_matrix @ weights

    # Calculer la contribution au risque
    risk_contribution = weights * marginal_contribution / portfolio_volatility

    # Normaliser pour obtenir des pourcentages
    return risk_contribution / portfolio_volatility * 100
