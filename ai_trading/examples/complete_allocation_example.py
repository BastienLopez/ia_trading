"""
Exemple d'utilisation du système d'allocation complet avec modèles multi-facteurs.

Ce script démontre:
1. Chargement et préparation des données
2. Création d'un modèle multi-facteurs
3. Initialisation et configuration du système d'allocation
4. Optimisation selon différents régimes de marché
5. Intégration des signaux de trading
6. Rééquilibrage et stress tests
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ai_trading.portfolio.complete_allocation_system import CompleteAllocationSystem
from ai_trading.rl.models.portfolio_optimization import FactorModel

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_crypto_data() -> dict:
    """
    Charge les données historiques de crypto-monnaies.

    Returns:
        Dictionnaire avec deux DataFrames: 'prices' et 'returns'
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
            returns = prices.pct_change().dropna()
            return {"prices": prices, "returns": returns}

    except Exception as e:
        logger.warning(f"Erreur lors du chargement des données réelles: {e}")

    # Si les données réelles ne sont pas disponibles, générer des données synthétiques
    logger.warning("Utilisation de données synthétiques pour l'exemple")
    return generate_synthetic_data()


def generate_synthetic_data(n_assets=10, n_days=500):
    """
    Génère des données synthétiques pour l'exemple.

    Args:
        n_assets: Nombre d'actifs
        n_days: Nombre de jours

    Returns:
        Dictionnaire avec deux DataFrames: 'prices' et 'returns'
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

    # Simuler différents régimes de marché
    # 1. Régime normal (déjà généré)
    # 2. Haute volatilité
    high_vol_period = slice(100, 150)
    daily_returns[high_vol_period] *= 2.5

    # 3. Marché baissier
    bear_period = slice(200, 250)
    daily_returns[bear_period] -= 0.015  # Rendements négatifs

    # 4. Marché haussier
    bull_period = slice(300, 350)
    daily_returns[bull_period] += 0.012  # Rendements positifs élevés

    # Convertir en DataFrame
    dates = pd.date_range(end="2023-12-31", periods=n_days)
    asset_names = [f"Crypto{i+1}" for i in range(n_assets)]
    returns_df = pd.DataFrame(daily_returns, index=dates, columns=asset_names)

    # Convertir en prix
    prices_df = 100 * np.cumprod(1 + daily_returns, axis=0)
    prices_df = pd.DataFrame(prices_df, index=dates, columns=asset_names)

    return {"prices": prices_df, "returns": returns_df}


def create_multi_factor_model(returns_data: pd.DataFrame) -> FactorModel:
    """
    Crée un modèle multi-facteurs à partir des rendements.

    Ce modèle inclut:
    1. Facteurs statistiques (via PCA)
    2. Facteurs de risque communs

    Args:
        returns_data: DataFrame des rendements

    Returns:
        Modèle de facteurs
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    logger.info("Création du modèle multi-facteurs...")

    # 1. Extraire les facteurs statistiques via PCA
    n_stat_factors = 3
    returns_std = (returns_data - returns_data.mean()) / returns_data.std()

    pca = PCA(n_components=n_stat_factors)
    stat_factors = pca.fit_transform(returns_std)

    # 2. Créer des facteurs macro (simulés pour l'exemple)
    n_days = len(returns_data)

    # Simuler un facteur "tendance du marché"
    market_trend = np.cumsum(np.random.normal(0.0003, 0.01, n_days))

    # Simuler un facteur "volatilité du marché"
    market_vol = abs(np.random.normal(0, 0.01, n_days))
    for i in range(1, n_days):
        market_vol[i] = 0.9 * market_vol[i - 1] + 0.1 * market_vol[i]

    # Simuler un facteur "sentiment du marché"
    market_sentiment = np.sin(np.linspace(0, 8 * np.pi, n_days)) * 0.01

    # Combiner tous les facteurs
    all_factors = np.column_stack(
        [
            stat_factors,  # Facteurs statistiques
            market_trend,  # Tendance
            market_vol,  # Volatilité
            market_sentiment,  # Sentiment
        ]
    )

    # Nommer les facteurs
    factor_names = [f"StatFactor{i+1}" for i in range(n_stat_factors)]
    factor_names.extend(["MarketTrend", "MarketVolatility", "MarketSentiment"])

    # Créer le DataFrame des facteurs
    factors_df = pd.DataFrame(
        all_factors, index=returns_data.index, columns=factor_names
    )

    # Calculer les expositions (bêtas) pour chaque actif
    exposures = pd.DataFrame(index=returns_data.columns, columns=factor_names)
    specific_returns = pd.DataFrame(
        index=returns_data.index, columns=returns_data.columns
    )
    specific_risks = pd.Series(index=returns_data.columns)

    for asset in returns_data.columns:
        # Régresser les rendements sur les facteurs
        reg = LinearRegression()
        asset_returns = returns_std[asset].values
        reg.fit(all_factors, asset_returns)

        # Stocker les expositions
        exposures.loc[asset, :] = reg.coef_

        # Calculer les rendements spécifiques
        factor_component = reg.predict(all_factors)
        specific_return = asset_returns - factor_component
        specific_returns[asset] = specific_return

        # Stocker le risque spécifique
        specific_risks[asset] = np.std(specific_return)

    # Calculer la matrice de covariance des facteurs
    factor_cov = pd.DataFrame(
        np.cov(all_factors, rowvar=False), index=factor_names, columns=factor_names
    )

    logger.info(f"Modèle multi-facteurs créé avec {len(factor_names)} facteurs")

    return FactorModel(
        name="Multi-Factor Model",
        factors=factors_df,
        exposures=exposures,
        specific_returns=specific_returns,
        specific_risks=specific_risks,
        factor_cov=factor_cov,
    )


def visualize_factor_exposures(factor_model: FactorModel):
    """
    Visualise les expositions aux facteurs par actif.

    Args:
        factor_model: Modèle de facteurs
    """
    plt.figure(figsize=(12, 8))

    # Créer une heatmap des expositions
    sns.heatmap(
        factor_model.exposures,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
    )

    plt.title("Expositions aux facteurs par actif")
    plt.tight_layout()
    plt.savefig("factor_exposures.png")
    logger.info(
        "Graphique des expositions aux facteurs enregistré dans 'factor_exposures.png'"
    )


def visualize_portfolio_weights(portfolios):
    """
    Visualise les poids des portefeuilles pour différentes configurations.

    Args:
        portfolios: Liste de tuples (nom, poids)
    """
    plt.figure(figsize=(12, 8))

    n_portfolios = len(portfolios)
    bar_width = 0.8 / n_portfolios

    for i, (name, weights) in enumerate(portfolios):
        x = np.arange(len(weights))
        plt.bar(
            x + i * bar_width - 0.4 + bar_width / 2,
            weights.values,
            width=bar_width,
            label=name,
        )

    plt.xlabel("Actifs")
    plt.ylabel("Poids")
    plt.title("Poids des portefeuilles pour différentes configurations")
    plt.xticks(np.arange(len(weights)), weights.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("portfolio_weights.png")
    logger.info(
        "Graphique des poids de portefeuille enregistré dans 'portfolio_weights.png'"
    )


def visualize_stress_tests(stress_results):
    """
    Visualise les résultats des stress tests.

    Args:
        stress_results: Dictionnaire des résultats par scénario
    """
    plt.figure(figsize=(14, 10))

    # Sous-figure 1: Comparaison des métriques de risque
    plt.subplot(2, 1, 1)

    scenarios = list(stress_results.keys())
    metrics = ["total_return", "max_drawdown", "var_95", "es_95"]
    metric_labels = ["Rendement total", "Drawdown max", "VaR 95%", "ES 95%"]

    x = np.arange(len(metrics))
    bar_width = 0.8 / len(scenarios)

    for i, scenario in enumerate(scenarios):
        values = [
            stress_results[scenario]["total_return"],
            stress_results[scenario]["max_drawdown"],
            stress_results[scenario]["var_95"],
            stress_results[scenario]["es_95"],
        ]
        plt.bar(
            x + i * bar_width - 0.4 + bar_width / 2,
            values,
            width=bar_width,
            label=scenario,
        )

    plt.xlabel("Métriques")
    plt.ylabel("Valeur")
    plt.title("Comparaison des métriques de risque par scénario")
    plt.xticks(x, metric_labels)
    plt.legend()

    # Sous-figure 2: Rendements cumulés
    plt.subplot(2, 1, 2)

    for scenario in scenarios:
        returns = stress_results[scenario]["portfolio_returns"]
        plt.plot((1 + returns).cumprod() - 1, label=scenario)

    plt.xlabel("Jours")
    plt.ylabel("Rendement cumulé")
    plt.title("Rendements cumulés par scénario")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("stress_test_results.png")
    logger.info(
        "Graphique des résultats des stress tests enregistré dans 'stress_test_results.png'"
    )


def run_complete_allocation_example():
    """
    Exécute l'exemple complet du système d'allocation de portefeuille.
    """
    logger.info("Démarrage de l'exemple du système d'allocation complet...")

    # 1. Charger les données
    data = load_crypto_data()
    returns = data["returns"]
    prices = data["prices"]

    # 2. Créer le modèle multi-facteurs
    factor_model = create_multi_factor_model(returns)

    # Visualiser les expositions aux facteurs
    visualize_factor_exposures(factor_model)

    # 3. Initialiser le système d'allocation
    allocation_system = CompleteAllocationSystem(
        returns=returns,
        prices=prices,
        factor_model=factor_model,
        lookback_window=90,
        risk_free_rate=0.01,
        rebalance_threshold=0.05,
        max_single_asset_weight=0.25,
        min_single_asset_weight=0.02,  # Au moins 2% par actif
        risk_budget_method="equal",
        optimization_method="sharpe",
        risk_aversion=2.0,
    )

    # 4. Optimiser l'allocation pour différents régimes de marché
    logger.info("Optimisation pour différents régimes de marché...")

    # Régime normal
    allocation_system.market_regime = "normal"
    normal_allocation = allocation_system.optimize_allocation()

    # Régime haute volatilité
    allocation_system.market_regime = "high_vol"
    high_vol_allocation = allocation_system.optimize_allocation()

    # Régime baissier
    allocation_system.market_regime = "bearish"
    bearish_allocation = allocation_system.optimize_allocation()

    # Régime haussier
    allocation_system.market_regime = "bullish"
    bullish_allocation = allocation_system.optimize_allocation()

    # Afficher les résultats
    logger.info("\nRésultats des optimisations par régime de marché:")

    for regime, allocation in [
        ("Normal", normal_allocation),
        ("Haute volatilité", high_vol_allocation),
        ("Baissier", bearish_allocation),
        ("Haussier", bullish_allocation),
    ]:
        logger.info(f"\nRégime: {regime}")
        logger.info(
            f"  Rendement attendu (annualisé): {allocation.expected_return * 252:.2%}"
        )
        logger.info(
            f"  Volatilité (annualisée): {allocation.volatility * np.sqrt(252):.2%}"
        )
        logger.info(f"  Ratio de Sharpe: {allocation.sharpe_ratio:.2f}")
        logger.info(f"  VaR 95%: {allocation.var:.2%}")
        logger.info(f"  ES 95%: {allocation.es:.2%}")

    # Visualiser les poids
    visualize_portfolio_weights(
        [
            ("Normal", pd.Series(normal_allocation.weights)),
            ("Haute Vol.", pd.Series(high_vol_allocation.weights)),
            ("Baissier", pd.Series(bearish_allocation.weights)),
            ("Haussier", pd.Series(bullish_allocation.weights)),
        ]
    )

    # 5. Intégrer des signaux de trading
    logger.info("\nIntégration des signaux de trading...")

    # Créer des signaux hypothétiques
    trading_signals = {}
    for asset in allocation_system.assets:
        # Générer un signal aléatoire entre -1 et 1
        trading_signals[asset] = np.random.uniform(-1, 1)

    # Intégrer les signaux
    signal_allocation = allocation_system.integrate_signals(trading_signals)

    logger.info("\nAllocation avec signaux de trading:")
    logger.info(
        f"  Rendement attendu (annualisé): {signal_allocation.expected_return * 252:.2%}"
    )
    logger.info(
        f"  Volatilité (annualisée): {signal_allocation.volatility * np.sqrt(252):.2%}"
    )
    logger.info(f"  Ratio de Sharpe: {signal_allocation.sharpe_ratio:.2f}")

    # 6. Tester le rééquilibrage
    logger.info("\nTest de rééquilibrage...")

    # Simuler une dérive des poids
    current_weights = normal_allocation.weights.copy()
    assets = list(current_weights.keys())

    # Augmenter certains poids, réduire d'autres
    for i, asset in enumerate(assets[:3]):
        current_weights[asset] += 0.05

    for i, asset in enumerate(assets[3:6]):
        current_weights[asset] -= 0.03

    # Normaliser pour que la somme soit 1
    total = sum(current_weights.values())
    current_weights = {k: v / total for k, v in current_weights.items()}

    # Vérifier si un rééquilibrage est nécessaire
    needs_rebalance = allocation_system.should_rebalance(current_weights)
    logger.info(f"Rééquilibrage nécessaire: {needs_rebalance}")

    if needs_rebalance:
        # Obtenir le plan de rééquilibrage
        capital = 100000  # $100,000
        adjustments = allocation_system.get_rebalance_plan(current_weights, capital)

        logger.info("\nPlan de rééquilibrage:")
        for asset, amount in adjustments.items():
            if abs(amount) > 100:  # Afficher uniquement les ajustements significatifs
                logger.info(f"  {asset}: {amount:.2f} USD")

    # 7. Effectuer des stress tests
    logger.info("\nExécution des stress tests...")

    # Définir le portefeuille actuel
    allocation_system.current_weights = normal_allocation.weights

    # Tester différents scénarios
    stress_results = {}

    # Scénario 1: Crash de marché
    stress_results["Crash"] = allocation_system.stress_test(scenario="market_crash")

    # Scénario 2: Haute volatilité
    stress_results["Haute Vol."] = allocation_system.stress_test(
        scenario="high_volatility"
    )

    # Scénario 3: Personnalisé (simulation d'une forte hausse suivie d'une correction)
    custom_scenario = pd.DataFrame(index=range(20), columns=allocation_system.assets)

    for asset in allocation_system.assets:
        # 10 premiers jours: hausse
        custom_scenario.loc[:9, asset] = np.random.normal(0.02, 0.03, 10)
        # 10 jours suivants: correction
        custom_scenario.loc[10:, asset] = np.random.normal(-0.015, 0.025, 10)

    stress_results["Hausse-Correction"] = allocation_system.stress_test(
        scenario="custom", custom_returns=custom_scenario
    )

    # Afficher les résultats
    logger.info("\nRésultats des stress tests:")
    for scenario, results in stress_results.items():
        logger.info(f"\nScénario: {scenario}")
        logger.info(f"  Rendement total: {results['total_return']:.2%}")
        logger.info(f"  Drawdown maximum: {results['max_drawdown']:.2%}")
        logger.info(f"  Volatilité: {results['volatility']:.2%}")
        logger.info(f"  VaR 95%: {results['var_95']:.2%}")
        logger.info(f"  ES 95%: {results['es_95']:.2%}")

    # Visualiser les résultats des stress tests
    visualize_stress_tests(stress_results)

    logger.info("\nExemple du système d'allocation complet terminé avec succès!")


if __name__ == "__main__":
    run_complete_allocation_example()
