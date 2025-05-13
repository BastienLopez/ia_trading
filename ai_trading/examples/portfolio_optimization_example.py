"""
Exemple d'utilisation de l'optimisation de portefeuille multi-facteurs.

Cet exemple démontre:
1. L'optimisation de portefeuille standard (minimum variance, maximum Sharpe)
2. L'optimisation avec contraintes personnalisées
3. L'utilisation du modèle de facteurs
4. L'optimisation Black-Litterman avec intégration de vues
5. L'analyse du risque multi-facteurs
"""

import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ai_trading.rl.models.portfolio_optimization import (
    BlackLittermanOptimizer,
    FactorModel,
    PortfolioOptimizer,
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_crypto_data() -> pd.DataFrame:
    """
    Charge les données historiques de crypto-monnaies.

    Returns:
        DataFrame des prix journaliers
    """
    try:
        # Essayer de charger des données réelles si disponibles
        from ai_trading.examples.meta_learning_example import load_market_data

        markets = [
            "BTC-USD",
            "ETH-USD",
            "XRP-USD",
            "ADA-USD",
            "SOL-USD",
            "BNB-USD",
            "DOGE-USD",
            "DOT-USD",
            "MATIC-USD",
            "LINK-USD",
        ]

        market_data = load_market_data(markets)

        # Extraire les prix de clôture
        prices = pd.DataFrame()
        for market, df in market_data.items():
            if "close" in df.columns:
                prices[market] = df["close"]

        if not prices.empty:
            logger.info(
                f"Données réelles chargées pour {len(prices.columns)} crypto-monnaies"
            )
            return prices

    except Exception as e:
        logger.warning(f"Erreur lors du chargement des données réelles: {e}")

    # Si les données réelles ne sont pas disponibles, générer des données synthétiques
    logger.warning("Utilisation de données synthétiques pour l'exemple")
    return generate_synthetic_prices()


def generate_synthetic_prices(n_assets: int = 10, n_days: int = 500) -> pd.DataFrame:
    """
    Génère des prix synthétiques pour l'exemple.

    Args:
        n_assets: Nombre d'actifs
        n_days: Nombre de jours

    Returns:
        DataFrame des prix journaliers
    """
    np.random.seed(42)

    # Paramètres pour la génération
    mean_returns = np.random.normal(
        0.0005, 0.0002, n_assets
    )  # Rendements journaliers moyens
    volatilities = np.random.uniform(0.01, 0.04, n_assets)  # Volatilités journalières

    # Matrice de corrélation (avec des corrélations plus élevées pour simuler un marché crypto)
    correlations = np.random.uniform(0.3, 0.8, (n_assets, n_assets))
    np.fill_diagonal(correlations, 1.0)
    correlations = (correlations + correlations.T) / 2  # Assurer la symétrie

    # Convertir en matrice de covariance
    cov_matrix = np.outer(volatilities, volatilities) * correlations

    # Générer les rendements journaliers
    daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)

    # Convertir en prix
    prices = 100 * np.cumprod(1 + daily_returns, axis=0)

    # Créer le DataFrame
    dates = pd.date_range(end="2023-12-31", periods=n_days)
    asset_names = [f"Crypto{i+1}" for i in range(n_assets)]

    return pd.DataFrame(prices, index=dates, columns=asset_names)


def calculate_returns(prices: pd.DataFrame, period: str = "D") -> pd.DataFrame:
    """
    Calcule les rendements à partir des prix.

    Args:
        prices: DataFrame des prix
        period: Période des rendements ('D' pour journalier, 'W' pour hebdomadaire, etc.)

    Returns:
        DataFrame des rendements
    """
    if period == "D":
        return prices.pct_change().dropna()
    else:
        return prices.resample(period).last().pct_change().dropna()


def create_factor_model(returns: pd.DataFrame, n_factors: int = 3) -> FactorModel:
    """
    Crée un modèle de facteurs statistiques à partir des rendements.

    Args:
        returns: DataFrame des rendements
        n_factors: Nombre de facteurs à extraire

    Returns:
        Modèle de facteurs
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    # Standardiser les rendements
    returns_std = (returns - returns.mean()) / returns.std()

    # Extraire les facteurs avec PCA
    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(returns_std)

    # Convertir en DataFrame
    factor_names = [f"Factor{i+1}" for i in range(n_factors)]
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

    return FactorModel(
        name="Statistical Factor Model",
        factors=factors_df,
        exposures=exposures,
        specific_returns=specific_returns,
        specific_risks=specific_risks,
        factor_cov=factor_cov,
    )


def run_basic_optimization(prices: pd.DataFrame) -> Dict:
    """
    Exécute une optimisation de portefeuille basique.

    Args:
        prices: DataFrame des prix

    Returns:
        Résultats de l'optimisation
    """
    logger.info("Exécution de l'optimisation de portefeuille basique")

    # Calculer les rendements
    returns = calculate_returns(prices)

    # Initialiser l'optimiseur
    optimizer = PortfolioOptimizer(
        returns=returns,
        risk_free_rate=0.01,  # 1% annualisé
        lookback_window=252,  # 1 an de données
    )

    # 1. Portefeuille de variance minimale
    min_var_portfolio = optimizer.optimize_portfolio(objective="min_risk")

    # 2. Portefeuille avec ratio de Sharpe maximal
    max_sharpe_portfolio = optimizer.optimize_portfolio(objective="sharpe")

    # 3. Portefeuille avec rendement cible
    target_return = 0.15  # 15% annualisé
    daily_target = target_return / 252  # Approximation en rendement journalier
    target_return_portfolio = optimizer.optimize_portfolio(
        objective="min_risk", target_return=daily_target
    )

    # Afficher les résultats
    logger.info("\nPordfeuille à variance minimale:")
    logger.info(f"Rendement attendu: {min_var_portfolio['expected_return']*252:.2%}")
    logger.info(f"Volatilité: {min_var_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {min_var_portfolio['sharpe_ratio']:.2f}")
    logger.info(f"VaR (95%): {min_var_portfolio['var']:.2%}")

    logger.info("\nPortefeuille à Sharpe maximal:")
    logger.info(f"Rendement attendu: {max_sharpe_portfolio['expected_return']*252:.2%}")
    logger.info(f"Volatilité: {max_sharpe_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {max_sharpe_portfolio['sharpe_ratio']:.2f}")
    logger.info(f"VaR (95%): {max_sharpe_portfolio['var']:.2%}")

    logger.info("\nPortefeuille avec rendement cible:")
    logger.info(
        f"Rendement attendu: {target_return_portfolio['expected_return']*252:.2%}"
    )
    logger.info(f"Volatilité: {target_return_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {target_return_portfolio['sharpe_ratio']:.2f}")
    logger.info(f"VaR (95%): {target_return_portfolio['var']:.2%}")

    # Visualisation des poids
    visualize_portfolio_weights(
        [
            ("Min Variance", min_var_portfolio["weights"]),
            ("Max Sharpe", max_sharpe_portfolio["weights"]),
            ("Target Return", target_return_portfolio["weights"]),
        ]
    )

    return {
        "min_var": min_var_portfolio,
        "max_sharpe": max_sharpe_portfolio,
        "target_return": target_return_portfolio,
    }


def run_constrained_optimization(prices: pd.DataFrame) -> Dict:
    """
    Exécute une optimisation de portefeuille avec contraintes.

    Args:
        prices: DataFrame des prix

    Returns:
        Résultats de l'optimisation
    """
    logger.info("Exécution de l'optimisation de portefeuille avec contraintes")

    # Calculer les rendements
    returns = calculate_returns(prices)

    # Initialiser l'optimiseur
    optimizer = PortfolioOptimizer(
        returns=returns,
        risk_free_rate=0.01,  # 1% annualisé
        lookback_window=252,  # 1 an de données
    )

    # Définir des contraintes personnalisées

    # 1. Contrainte de diversification: max 20% par actif
    max_allocation_constraint = {
        "type": "ineq",
        "fun": lambda x: 0.20 - np.max(x),  # Aucun actif ne dépasse 20%
    }

    # 2. Contrainte de groupe: max 40% dans le top 3 des cryptos
    top_3_constraint = {
        "type": "ineq",
        "fun": lambda x: 0.40 - np.sum(x[:3]),  # Top 3 limités à 40%
    }

    # 3. Contrainte de groupe: min 30% dans les cryptos plus petites
    small_cap_constraint = {
        "type": "ineq",
        "fun": lambda x: np.sum(x[5:]) - 0.30,  # Au moins 30% dans les plus petites
    }

    # Liste de toutes les contraintes
    all_constraints = [
        max_allocation_constraint,
        top_3_constraint,
        small_cap_constraint,
    ]

    # Optimiser avec contraintes
    constrained_portfolio = optimizer.optimize_portfolio(
        objective="sharpe", constraints=all_constraints
    )

    # Optimiser sans contraintes pour comparaison
    unconstrained_portfolio = optimizer.optimize_portfolio(objective="sharpe")

    # Afficher les résultats
    logger.info("\nPortefeuille contraint:")
    logger.info(
        f"Rendement attendu: {constrained_portfolio['expected_return']*252:.2%}"
    )
    logger.info(f"Volatilité: {constrained_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {constrained_portfolio['sharpe_ratio']:.2f}")

    logger.info("\nPortefeuille non contraint:")
    logger.info(
        f"Rendement attendu: {unconstrained_portfolio['expected_return']*252:.2%}"
    )
    logger.info(f"Volatilité: {unconstrained_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {unconstrained_portfolio['sharpe_ratio']:.2f}")

    # Visualisation des poids
    visualize_portfolio_weights(
        [
            ("Contraint", constrained_portfolio["weights"]),
            ("Non contraint", unconstrained_portfolio["weights"]),
        ]
    )

    return {
        "constrained": constrained_portfolio,
        "unconstrained": unconstrained_portfolio,
    }


def run_factor_optimization(prices: pd.DataFrame) -> Dict:
    """
    Exécute une optimisation de portefeuille avec modèle de facteurs.

    Args:
        prices: DataFrame des prix

    Returns:
        Résultats de l'optimisation
    """
    logger.info("Exécution de l'optimisation de portefeuille basée sur les facteurs")

    # Calculer les rendements
    returns = calculate_returns(prices)

    # Créer un modèle de facteurs
    factor_model = create_factor_model(returns, n_factors=3)

    # Initialiser l'optimiseur avec le modèle de facteurs
    optimizer = PortfolioOptimizer(
        returns=returns,
        factor_model=factor_model,
        risk_free_rate=0.01,
        lookback_window=252,
    )

    # Optimiser le portefeuille
    factor_portfolio = optimizer.optimize_portfolio(objective="sharpe")

    # Optimiser en utilisant directement la matrice de covariance empirique
    standard_optimizer = PortfolioOptimizer(
        returns=returns, risk_free_rate=0.01, lookback_window=252
    )
    standard_portfolio = standard_optimizer.optimize_portfolio(objective="sharpe")

    # Afficher les résultats
    logger.info("\nPortefeuille basé sur les facteurs:")
    logger.info(f"Rendement attendu: {factor_portfolio['expected_return']*252:.2%}")
    logger.info(f"Volatilité: {factor_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {factor_portfolio['sharpe_ratio']:.2f}")

    logger.info("\nPortefeuille standard:")
    logger.info(f"Rendement attendu: {standard_portfolio['expected_return']*252:.2%}")
    logger.info(f"Volatilité: {standard_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {standard_portfolio['sharpe_ratio']:.2f}")

    # Visualisation des poids
    visualize_portfolio_weights(
        [
            ("Modèle de facteurs", factor_portfolio["weights"]),
            ("Standard", standard_portfolio["weights"]),
        ]
    )

    # Analyser la contribution des facteurs au risque
    factor_contribution = optimizer.factor_risk_contribution(
        factor_portfolio["weights"]
    )

    # Visualiser la contribution des facteurs
    visualize_factor_contribution(factor_contribution)

    return {
        "factor_portfolio": factor_portfolio,
        "standard_portfolio": standard_portfolio,
        "factor_contribution": factor_contribution,
    }


def run_black_litterman_optimization(prices: pd.DataFrame) -> Dict:
    """
    Exécute une optimisation de portefeuille avec le modèle Black-Litterman.

    Args:
        prices: DataFrame des prix

    Returns:
        Résultats de l'optimisation
    """
    logger.info("Exécution de l'optimisation Black-Litterman")

    # Calculer les rendements
    returns = calculate_returns(prices)

    # Définir les poids du marché (capitalisation approximative)
    market_weights = pd.Series(
        {
            col: weight
            for col, weight in zip(
                returns.columns,
                np.array(
                    [0.40, 0.25, 0.10, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02][
                        : len(returns.columns)
                    ]
                ),
            )
        }
    )

    # Initialiser l'optimiseur Black-Litterman
    bl_optimizer = BlackLittermanOptimizer(
        returns=returns,
        market_weights=market_weights,
        risk_aversion=2.5,
        tau=0.025,
        risk_free_rate=0.01,
        lookback_window=252,
    )

    # Optimiser avec les rendements d'équilibre (sans vues)
    market_portfolio = bl_optimizer.optimize_bl_portfolio(objective="sharpe")

    # Définir des vues sur certains actifs
    # Vue 1: La crypto 1 va surperformer de 5% annualisé
    # Vue 2: La crypto 3 va sous-performer de 3% annualisé
    # Vue 3: La crypto 2 va surperformer la crypto 4 de 2% annualisé

    # Matrice de vues (P)
    view_matrix = pd.DataFrame(
        0, index=["view1", "view2", "view3"], columns=returns.columns
    )
    view_matrix.loc["view1", returns.columns[0]] = 1  # Crypto 1
    view_matrix.loc["view2", returns.columns[2]] = 1  # Crypto 3
    view_matrix.loc["view3", returns.columns[1]] = 1  # Crypto 2
    view_matrix.loc["view3", returns.columns[3]] = -1  # moins Crypto 4

    # Rendements attendus pour les vues (annualisés puis convertis en journalier)
    view_returns = pd.Series(
        {
            "view1": 0.05 / 252,  # 5% annualisé
            "view2": -0.03 / 252,  # -3% annualisé
            "view3": 0.02 / 252,  # 2% annualisé
        }
    )

    # Confiance dans les vues (inverse de la variance)
    view_confidences = pd.Series(
        {
            "view1": 0.5,  # Confiance moyenne
            "view2": 0.3,  # Confiance faible
            "view3": 0.7,  # Confiance élevée
        }
    )

    # Optimiser avec les vues
    views_portfolio = bl_optimizer.optimize_bl_portfolio(
        view_matrix=view_matrix,
        view_returns=view_returns,
        view_confidences=view_confidences,
        objective="sharpe",
    )

    # Afficher les résultats
    logger.info("\nPortefeuille d'équilibre (sans vues):")
    logger.info(f"Rendement attendu: {market_portfolio['expected_return']*252:.2%}")
    logger.info(f"Volatilité: {market_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {market_portfolio['sharpe_ratio']:.2f}")

    logger.info("\nPortefeuille avec vues:")
    logger.info(f"Rendement attendu: {views_portfolio['expected_return']*252:.2%}")
    logger.info(f"Volatilité: {views_portfolio['volatility']*np.sqrt(252):.2%}")
    logger.info(f"Ratio de Sharpe: {views_portfolio['sharpe_ratio']:.2f}")

    # Visualisation des poids
    visualize_portfolio_weights(
        [
            ("Équilibre", market_portfolio["weights"]),
            ("Avec vues", views_portfolio["weights"]),
            ("Marché", market_weights),
        ]
    )

    return {"market_portfolio": market_portfolio, "views_portfolio": views_portfolio}


def run_efficient_frontier(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Génère et visualise la frontière efficiente.

    Args:
        prices: DataFrame des prix

    Returns:
        DataFrame contenant les portefeuilles de la frontière efficiente
    """
    logger.info("Génération de la frontière efficiente")

    # Calculer les rendements
    returns = calculate_returns(prices)

    # Initialiser l'optimiseur
    optimizer = PortfolioOptimizer(
        returns=returns, risk_free_rate=0.01, lookback_window=252
    )

    # Générer la frontière efficiente
    efficient_frontier = optimizer.efficient_frontier(num_portfolios=50)

    # Trouver le portefeuille à Sharpe maximal
    max_sharpe_idx = efficient_frontier["sharpe_ratio"].idxmax()
    max_sharpe_portfolio = efficient_frontier.iloc[max_sharpe_idx]

    # Trouver le portefeuille à variance minimale
    min_var_idx = efficient_frontier["volatility"].idxmin()
    min_var_portfolio = efficient_frontier.iloc[min_var_idx]

    # Annualiser les rendements et volatilités pour l'affichage
    efficient_frontier["expected_return_annual"] = (
        efficient_frontier["expected_return"] * 252
    )
    efficient_frontier["volatility_annual"] = efficient_frontier[
        "volatility"
    ] * np.sqrt(252)

    # Visualiser la frontière efficiente
    plt.figure(figsize=(10, 6))
    plt.scatter(
        efficient_frontier["volatility_annual"],
        efficient_frontier["expected_return_annual"],
        c=efficient_frontier["sharpe_ratio"],
        cmap="viridis",
        marker="o",
        alpha=0.7,
    )

    # Marquer les portefeuilles spéciaux
    plt.scatter(
        min_var_portfolio["volatility_annual"],
        min_var_portfolio["expected_return_annual"],
        marker="*",
        color="red",
        s=300,
        label="Minimum Variance",
    )

    plt.scatter(
        max_sharpe_portfolio["volatility_annual"],
        max_sharpe_portfolio["expected_return_annual"],
        marker="*",
        color="green",
        s=300,
        label="Maximum Sharpe",
    )

    # Tracer la CML (Capital Market Line)
    x_min, x_max = plt.xlim()
    y_min = 0.01  # Risk-free rate
    y_max = max_sharpe_portfolio["expected_return_annual"] + max_sharpe_portfolio[
        "sharpe_ratio"
    ] * (x_max - max_sharpe_portfolio["volatility_annual"])
    plt.plot([0, x_max], [y_min, y_max], "k--", label="Capital Market Line")

    plt.colorbar(label="Ratio de Sharpe")
    plt.xlabel("Volatilité annualisée")
    plt.ylabel("Rendement attendu annualisé")
    plt.title("Frontière Efficiente")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("efficient_frontier.png")
    plt.close()

    logger.info("Frontière efficiente sauvegardée: efficient_frontier.png")

    return efficient_frontier


def visualize_portfolio_weights(portfolios: List[Tuple[str, pd.Series]]) -> None:
    """
    Visualise les poids des portefeuilles.

    Args:
        portfolios: Liste de tuples (nom_portefeuille, poids)
    """
    plt.figure(figsize=(12, 6))

    n_portfolios = len(portfolios)
    bar_width = 0.8 / n_portfolios

    for i, (name, weights) in enumerate(portfolios):
        positions = np.arange(len(weights)) + i * bar_width
        plt.bar(positions, weights.values, width=bar_width, label=name, alpha=0.7)

    plt.xlabel("Actifs")
    plt.ylabel("Poids")
    plt.title("Allocation des Portefeuilles")
    plt.xticks(
        np.arange(len(portfolios[0][1])) + bar_width * (n_portfolios - 1) / 2,
        portfolios[0][1].index,
        rotation=45,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("portfolio_weights.png")
    plt.close()

    logger.info("Visualisation des poids sauvegardée: portfolio_weights.png")


def visualize_factor_contribution(factor_contribution: pd.DataFrame) -> None:
    """
    Visualise la contribution des facteurs au risque du portefeuille.

    Args:
        factor_contribution: DataFrame des contributions des facteurs
    """
    plt.figure(figsize=(10, 6))

    # Trier par contribution
    sorted_contrib = factor_contribution.sort_values("contribution", ascending=False)

    # Créer la visualisation
    plt.bar(
        sorted_contrib.index,
        sorted_contrib["contribution_pct"],
        alpha=0.7,
        color=sns.color_palette("viridis", len(sorted_contrib)),
    )

    plt.xlabel("Facteurs")
    plt.ylabel("Contribution au risque (%)")
    plt.title("Contribution des facteurs au risque du portefeuille")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("factor_contribution.png")
    plt.close()

    logger.info(
        "Visualisation de la contribution des facteurs sauvegardée: factor_contribution.png"
    )


def run_portfolio_optimization_example() -> None:
    """Exemple principal d'optimisation de portefeuille."""
    logger.info("Démarrage de l'exemple d'optimisation de portefeuille multi-facteurs")

    # Charger les données
    prices = load_crypto_data()
    logger.info(
        f"Données chargées pour {len(prices.columns)} actifs sur {len(prices)} jours"
    )

    # Exécuter les différentes optimisations
    basic_results = run_basic_optimization(prices)
    constrained_results = run_constrained_optimization(prices)
    factor_results = run_factor_optimization(prices)
    bl_results = run_black_litterman_optimization(prices)

    # Générer la frontière efficiente
    efficient_frontier = run_efficient_frontier(prices)

    logger.info("Exemple d'optimisation de portefeuille terminé")


if __name__ == "__main__":
    run_portfolio_optimization_example()
