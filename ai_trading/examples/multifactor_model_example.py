"""
Exemple d'implémentation et d'utilisation des modèles multi-facteurs.

Ce script démontre:
1. Différents types de modèles multi-facteurs (statistiques, fondamentaux, mixtes)
2. Analyse d'attribution de performance par facteur
3. Décomposition du risque par facteur
4. Optimisation basée sur les vues factorielles
"""

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_trading.rl.models.portfolio_optimization import (
    BlackLittermanOptimizer,
    FactorModel,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_test_data(n_assets=8, n_days=500):
    """
    Génère des données de test pour les modèles multi-facteurs.

    Args:
        n_assets: Nombre d'actifs
        n_days: Nombre de jours

    Returns:
        Dictionnaire avec les prix et rendements
    """
    np.random.seed(42)

    # Définir les facteurs sous-jacents
    # 1. Facteur marché (tendance générale)
    market_factor = np.random.normal(0.0005, 0.01, n_days)
    market_factor = np.cumsum(market_factor)

    # 2. Facteur taille (petite vs grande capitalisation)
    size_factor = np.random.normal(0.0002, 0.008, n_days)

    # 3. Facteur momentum (tendance récente)
    momentum_factor = np.zeros(n_days)
    for i in range(20, n_days):
        momentum_factor[i] = 0.7 * momentum_factor[i - 1] + 0.3 * np.random.normal(
            0.0001, 0.006
        )

    # 4. Facteur volatilité
    volatility_factor = abs(np.random.normal(0, 0.007, n_days))
    for i in range(1, n_days):
        volatility_factor[i] = (
            0.9 * volatility_factor[i - 1] + 0.1 * volatility_factor[i]
        )

    # Créer une matrice de facteurs
    factors = np.column_stack(
        [market_factor, size_factor, momentum_factor, volatility_factor]
    )

    # Générer des expositions aux facteurs pour chaque actif
    # Ces expositions détermineront comment chaque actif réagit à chaque facteur
    exposures = np.zeros((n_assets, 4))
    for i in range(n_assets):
        # Chaque actif a une exposition aléatoire à chaque facteur
        exposures[i, 0] = np.random.uniform(0.5, 1.5)  # Exposition au marché
        exposures[i, 1] = np.random.uniform(-1.0, 1.0)  # Exposition à la taille
        exposures[i, 2] = np.random.uniform(-0.8, 0.8)  # Exposition au momentum
        exposures[i, 3] = np.random.uniform(
            -1.2, 0.2
        )  # Exposition à la volatilité (négative généralement)

    # Générer les rendements spécifiques (non expliqués par les facteurs)
    specific_returns = np.random.normal(0, 0.01, (n_days, n_assets))

    # Générer les rendements totaux
    returns = np.zeros((n_days, n_assets))
    for d in range(n_days):
        for a in range(n_assets):
            # Rendement = expositions aux facteurs * rendements des facteurs + rendement spécifique
            returns[d, a] = np.dot(exposures[a], factors[d]) + specific_returns[d, a]

    # Convertir en DataFrame
    dates = pd.date_range(end="2023-12-31", periods=n_days)
    asset_names = [f"Asset{i+1}" for i in range(n_assets)]

    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)

    # Convertir en prix
    prices_df = 100 * np.cumprod(1 + returns, axis=0)
    prices_df = pd.DataFrame(prices_df, index=dates, columns=asset_names)

    # Créer les facteurs de rendement comme DataFrame
    factor_names = ["Market", "Size", "Momentum", "Volatility"]
    factors_df = pd.DataFrame(factors, index=dates, columns=factor_names)

    # Créer les expositions comme DataFrame
    exposures_df = pd.DataFrame(exposures, index=asset_names, columns=factor_names)

    # Créer les rendements spécifiques comme DataFrame
    specific_returns_df = pd.DataFrame(
        specific_returns, index=dates, columns=asset_names
    )

    return {
        "prices": prices_df,
        "returns": returns_df,
        "factors": factors_df,
        "exposures": exposures_df,
        "specific_returns": specific_returns_df,
    }


def create_statistical_factor_model(
    returns: pd.DataFrame, n_factors: int = 4
) -> FactorModel:
    """
    Crée un modèle de facteurs statistiques (PCA).

    Args:
        returns: DataFrame des rendements
        n_factors: Nombre de facteurs à extraire

    Returns:
        Modèle de facteurs
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    logger.info("Création du modèle de facteurs statistiques...")

    # Standardiser les rendements
    returns_std = (returns - returns.mean()) / returns.std()

    # Extraire les facteurs avec PCA
    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(returns_std)

    # Convertir en DataFrame
    factor_names = [f"StatFactor{i+1}" for i in range(n_factors)]
    factors_df = pd.DataFrame(factor_returns, index=returns.index, columns=factor_names)

    # Calculer les expositions (bêtas) pour chaque actif
    exposures = pd.DataFrame(index=returns.columns, columns=factor_names)
    specific_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
    specific_risks = pd.Series(index=returns.columns)

    for asset in returns.columns:
        # Régresser les rendements sur les facteurs
        reg = LinearRegression()
        reg.fit(factor_returns, returns_std[asset])

        # Stocker les expositions
        exposures.loc[asset, :] = reg.coef_

        # Calculer les rendements spécifiques
        asset_returns = returns[asset].values
        factor_component = reg.predict(factor_returns)
        specific_return = (
            asset_returns
            - factor_component * returns[asset].std()
            + returns[asset].mean()
        )
        specific_returns[asset] = specific_return

        # Stocker le risque spécifique
        specific_risks[asset] = np.std(specific_return)

    # Calculer la matrice de covariance des facteurs
    factor_cov = pd.DataFrame(
        np.cov(factor_returns, rowvar=False), index=factor_names, columns=factor_names
    )

    logger.info(f"Modèle de facteurs statistiques créé avec {n_factors} facteurs")

    return FactorModel(
        name="Statistical Factor Model",
        factors=factors_df,
        exposures=exposures,
        specific_returns=specific_returns,
        specific_risks=specific_risks,
        factor_cov=factor_cov,
    )


def create_fundamental_factor_model(
    returns: pd.DataFrame, economic_data: Dict[str, pd.Series]
) -> FactorModel:
    """
    Crée un modèle de facteurs fondamentaux basé sur des variables économiques.

    Args:
        returns: DataFrame des rendements
        economic_data: Dictionnaire de séries temporelles pour différentes variables économiques

    Returns:
        Modèle de facteurs
    """
    from sklearn.linear_model import LinearRegression

    logger.info("Création du modèle de facteurs fondamentaux...")

    # Créer un DataFrame pour les facteurs économiques
    factor_names = list(economic_data.keys())
    factors_df = pd.DataFrame({name: data for name, data in economic_data.items()})

    # Calculer les expositions pour chaque actif
    exposures = pd.DataFrame(index=returns.columns, columns=factor_names)
    specific_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
    specific_risks = pd.Series(index=returns.columns)

    # Aligner les données économiques avec les rendements
    aligned_factors = factors_df.reindex(returns.index)

    for asset in returns.columns:
        # Régresser les rendements sur les facteurs
        reg = LinearRegression()
        # Supprimer les NaN
        valid_idx = ~np.isnan(aligned_factors).any(axis=1)
        X = aligned_factors.loc[valid_idx].values
        y = returns.loc[valid_idx, asset].values

        reg.fit(X, y)

        # Stocker les expositions
        exposures.loc[asset, :] = reg.coef_

        # Calculer les rendements spécifiques
        predicted = pd.Series(index=returns.index, dtype=float)
        predicted.loc[valid_idx] = reg.predict(X)

        specific_return = returns[asset] - predicted
        specific_returns[asset] = specific_return

        # Stocker le risque spécifique
        specific_risks[asset] = specific_return.std()

    # Calculer la matrice de covariance des facteurs
    valid_idx = ~np.isnan(aligned_factors).any(axis=1)
    factor_data = aligned_factors.loc[valid_idx].values
    factor_cov = pd.DataFrame(
        np.cov(factor_data, rowvar=False), index=factor_names, columns=factor_names
    )

    logger.info(
        f"Modèle de facteurs fondamentaux créé avec {len(factor_names)} facteurs"
    )

    return FactorModel(
        name="Fundamental Factor Model",
        factors=factors_df,
        exposures=exposures,
        specific_returns=specific_returns,
        specific_risks=specific_risks,
        factor_cov=factor_cov,
    )


def create_hybrid_factor_model(
    returns: pd.DataFrame,
    statistical_model: FactorModel,
    fundamental_model: FactorModel,
) -> FactorModel:
    """
    Combine les modèles de facteurs statistiques et fondamentaux en un modèle hybride.

    Args:
        returns: DataFrame des rendements
        statistical_model: Modèle de facteurs statistiques
        fundamental_model: Modèle de facteurs fondamentaux

    Returns:
        Modèle de facteurs hybride
    """
    logger.info("Création du modèle de facteurs hybride...")

    # Combiner les facteurs en alignant les dates
    all_dates = returns.index
    stat_factors = statistical_model.factors.reindex(all_dates)
    fund_factors = fundamental_model.factors.reindex(all_dates)

    # Créer un DataFrame combiné pour tous les facteurs
    combined_factors = pd.concat([stat_factors, fund_factors], axis=1)

    # Créer un DataFrame pour les expositions combinées
    all_factor_names = list(combined_factors.columns)
    combined_exposures = pd.DataFrame(index=returns.columns, columns=all_factor_names)

    # Remplir les expositions
    for asset in returns.columns:
        for factor in statistical_model.factor_names:
            if factor in combined_exposures.columns:
                combined_exposures.loc[asset, factor] = statistical_model.exposures.loc[
                    asset, factor
                ]

        for factor in fundamental_model.factor_names:
            if factor in combined_exposures.columns:
                combined_exposures.loc[asset, factor] = fundamental_model.exposures.loc[
                    asset, factor
                ]

    # Calculer les rendements spécifiques moyennés des deux modèles
    specific_returns = (
        statistical_model.specific_returns + fundamental_model.specific_returns
    ) / 2

    # Calculer les risques spécifiques moyennés
    specific_risks = pd.Series(index=returns.columns)
    for asset in returns.columns:
        specific_risks[asset] = specific_returns[asset].std()

    # Calculer la matrice de covariance des facteurs combinés
    valid_idx = ~np.isnan(combined_factors).any(axis=1)
    factor_data = combined_factors.loc[valid_idx].values
    factor_cov = pd.DataFrame(
        np.cov(factor_data, rowvar=False),
        index=all_factor_names,
        columns=all_factor_names,
    )

    logger.info(
        f"Modèle de facteurs hybride créé avec {len(all_factor_names)} facteurs"
    )

    return FactorModel(
        name="Hybrid Factor Model",
        factors=combined_factors,
        exposures=combined_exposures,
        specific_returns=specific_returns,
        specific_risks=specific_risks,
        factor_cov=factor_cov,
    )


def analyze_factor_contribution(
    returns: pd.DataFrame, factor_model: FactorModel, period: slice = None
) -> pd.DataFrame:
    """
    Analyse la contribution de chaque facteur aux rendements historiques.

    Args:
        returns: DataFrame des rendements
        factor_model: Modèle de facteurs
        period: Période d'analyse (slice)

    Returns:
        DataFrame des contributions par facteur
    """
    logger.info("Analyse des contributions des facteurs aux rendements...")

    if period is None:
        period = slice(None)

    # Sélectionner la période
    period_returns = returns.loc[period]
    period_factors = factor_model.factors.loc[period]

    # Calculer les rendements moyens des actifs
    asset_returns = period_returns.mean()

    # Calculer les rendements moyens des facteurs
    factor_returns = period_factors.mean()

    # Créer un DataFrame pour les résultats
    result = pd.DataFrame(index=returns.columns)
    result["total_return"] = asset_returns

    # Calculer la contribution de chaque facteur
    for factor in factor_model.factor_names:
        exposures = factor_model.exposures[factor]
        factor_return = factor_returns[factor]
        result[f"{factor}_contrib"] = exposures * factor_return

    # Calculer la contribution spécifique
    specific_contrib = pd.Series(index=returns.columns)
    for asset in returns.columns:
        specific_contrib[asset] = factor_model.specific_returns.loc[
            period, asset
        ].mean()

    result["specific_contrib"] = specific_contrib

    # Calculer la contribution totale des facteurs
    factor_contrib_cols = [f"{factor}_contrib" for factor in factor_model.factor_names]
    result["factor_contrib_total"] = result[factor_contrib_cols].sum(axis=1)

    # Vérifier la différence
    result["unexplained"] = (
        result["total_return"]
        - result["factor_contrib_total"]
        - result["specific_contrib"]
    )

    return result


def visualize_factor_contributions(contribution_analysis: pd.DataFrame):
    """
    Visualise les contributions des facteurs aux rendements.

    Args:
        contribution_analysis: Résultat de l'analyse des contributions
    """
    plt.figure(figsize=(14, 8))

    # Identifier les colonnes de contribution des facteurs
    factor_cols = [
        col
        for col in contribution_analysis.columns
        if "_contrib" in col and col != "specific_contrib"
    ]

    # Créer un DataFrame pour la visualisation
    viz_data = contribution_analysis[factor_cols + ["specific_contrib"]].copy()

    # Tri des actifs par rendement total
    sorted_assets = contribution_analysis.sort_values(
        "total_return", ascending=False
    ).index
    viz_data = viz_data.reindex(sorted_assets)

    # Créer le graphique à barres empilées
    ax = viz_data.plot(kind="bar", stacked=True, figsize=(14, 8))

    # Ajouter les rendements totaux comme ligne
    ax2 = ax.twinx()
    total_returns = contribution_analysis.loc[sorted_assets, "total_return"]
    ax2.plot(range(len(total_returns)), total_returns, "ko-", linewidth=2, markersize=8)

    # Légendes et labels
    ax.set_xlabel("Actifs")
    ax.set_ylabel("Contribution au rendement")
    ax2.set_ylabel("Rendement total")

    plt.title("Décomposition des rendements par facteur")
    plt.tight_layout()
    plt.savefig("factor_contributions.png")

    logger.info(
        "Graphique des contributions des facteurs enregistré dans 'factor_contributions.png'"
    )


def optimize_with_factor_tilts(
    returns: pd.DataFrame,
    factor_model: FactorModel,
    factor_views: Dict[str, float],
    view_confidences: Dict[str, float],
) -> Dict:
    """
    Optimise un portefeuille en incorporant des vues sur les facteurs.

    Args:
        returns: DataFrame des rendements
        factor_model: Modèle de facteurs
        factor_views: Dictionnaire des vues sur les rendements des facteurs
        view_confidences: Dictionnaire des niveaux de confiance dans les vues

    Returns:
        Résultat de l'optimisation
    """
    logger.info("Optimisation avec tilts factoriels...")

    # Calculer les poids de marché (égaux pour l'exemple)
    n_assets = len(returns.columns)
    market_weights = pd.Series(1 / n_assets, index=returns.columns)

    # Initialiser l'optimiseur Black-Litterman
    bl_optimizer = BlackLittermanOptimizer(
        returns=returns,
        market_weights=market_weights,
        risk_aversion=2.5,
        factor_model=factor_model,
    )

    # Convertir les vues sur les facteurs en vues sur les actifs
    view_matrix = pd.DataFrame(
        0, index=list(factor_views.keys()), columns=returns.columns
    )
    view_returns_series = pd.Series(index=list(factor_views.keys()))
    view_confidences_series = pd.Series(index=list(factor_views.keys()))

    for factor, view in factor_views.items():
        # La vue est appliquée proportionnellement aux expositions
        view_matrix.loc[factor] = factor_model.exposures[factor]
        view_returns_series[factor] = view
        view_confidences_series[factor] = view_confidences.get(factor, 0.5)

    # Optimiser avec les vues
    result = bl_optimizer.optimize_bl_portfolio(
        view_matrix=view_matrix,
        view_returns=view_returns_series,
        view_confidences=view_confidences_series,
        objective="sharpe",
    )

    # Calculer les expositions du portefeuille optimisé aux facteurs
    weights = result["weights"]
    factor_exposures = weights @ factor_model.exposures

    logger.info("Expositions du portefeuille optimisé aux facteurs:")
    for factor, exposure in factor_exposures.items():
        logger.info(f"  {factor}: {exposure:.4f}")

    return result


def run_multifactor_model_example():
    """
    Exécute l'exemple complet de modèles multi-facteurs.
    """
    logger.info("Démarrage de l'exemple de modèles multi-facteurs...")

    # 1. Générer les données de test
    data = generate_test_data()
    returns = data["returns"]

    # 2. Créer un modèle de facteurs statistiques
    stat_model = create_statistical_factor_model(returns)

    # 3. Créer des facteurs économiques simulés
    economic_data = {}

    # Simuler les rendements du marché global
    market_index = pd.Series(
        np.random.normal(0.0006, 0.01, len(returns)), index=returns.index
    )
    market_index = market_index.cumsum()
    economic_data["MarketIndex"] = market_index

    # Simuler les taux d'intérêt
    interest_rate = pd.Series(index=returns.index)
    base_rate = 0.02  # 2%
    for i, date in enumerate(returns.index):
        if i == 0:
            interest_rate[date] = base_rate
        else:
            # Petite variation aléatoire autour de la valeur précédente
            interest_rate[date] = interest_rate.iloc[i - 1] + np.random.normal(
                0, 0.0005
            )
    economic_data["InterestRate"] = interest_rate

    # Simuler l'inflation
    inflation = pd.Series(index=returns.index)
    base_inflation = 0.03  # 3%
    for i, date in enumerate(returns.index):
        if i == 0:
            inflation[date] = base_inflation
        else:
            # Variation avec persistance
            inflation[date] = 0.95 * inflation.iloc[i - 1] + 0.05 * np.random.normal(
                base_inflation, 0.002
            )
    economic_data["Inflation"] = inflation

    # 4. Créer un modèle de facteurs fondamentaux
    fundamental_model = create_fundamental_factor_model(returns, economic_data)

    # 5. Créer un modèle hybride
    hybrid_model = create_hybrid_factor_model(returns, stat_model, fundamental_model)

    # 6. Analyser les contributions des facteurs
    # Diviser l'historique en deux périodes
    mid_point = len(returns) // 2
    period1 = slice(0, mid_point)
    period2 = slice(mid_point, None)

    # Analyser les contributions sur les deux périodes
    contrib1 = analyze_factor_contribution(returns, hybrid_model, period1)
    contrib2 = analyze_factor_contribution(returns, hybrid_model, period2)

    # Visualiser les contributions de la première période
    visualize_factor_contributions(contrib1)

    # 7. Optimiser avec des vues sur les facteurs
    # Définir des vues sur les facteurs
    factor_views = {
        "MarketIndex": 0.10,  # +10% de rendement annualisé
        "InterestRate": -0.02,  # -2% (impact négatif)
        "StatFactor1": 0.05,  # +5%
        "Size": 0.03,  # +3%
    }

    # Définir les niveaux de confiance
    view_confidences = {
        "MarketIndex": 0.7,  # 70% de confiance
        "InterestRate": 0.8,  # 80% de confiance
        "StatFactor1": 0.5,  # 50% de confiance
        "Size": 0.6,  # 60% de confiance
    }

    # Optimiser avec les vues
    optimized = optimize_with_factor_tilts(
        returns, hybrid_model, factor_views, view_confidences
    )

    # Analyser le portefeuille optimisé
    weights = optimized["weights"]
    expected_return = optimized["expected_return"] * 252  # Annualisé
    volatility = optimized["volatility"] * np.sqrt(252)  # Annualisée
    sharpe = optimized["sharpe_ratio"]

    logger.info("\nPortefeuille optimisé avec vues factorielles:")
    logger.info(f"  Rendement attendu (annualisé): {expected_return:.2%}")
    logger.info(f"  Volatilité (annualisée): {volatility:.2%}")
    logger.info(f"  Ratio de Sharpe: {sharpe:.2f}")

    # Décomposition du risque par facteur
    risk_decomposition = optimized.get("risk_decomposition", {})

    if risk_decomposition:
        logger.info("\nDécomposition du risque par facteur:")
        for factor, contribution in risk_decomposition.items():
            logger.info(f"  {factor}: {contribution:.2%}")

    logger.info("\nExemple des modèles multi-facteurs terminé avec succès!")


if __name__ == "__main__":
    run_multifactor_model_example()
