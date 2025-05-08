"""
Exemple d'utilisation du gestionnaire de risques avancé avec VaR et allocation adaptative.
"""

import logging
import os
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.config import VISUALIZATION_DIR
from ai_trading.risk.advanced_risk_manager import AdvancedRiskManager
from ai_trading.utils.data_loader import load_crypto_data

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dossier pour les visualisations
EXAMPLES_VIZ_DIR = VISUALIZATION_DIR / "examples"
EXAMPLES_VIZ_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(
    days=365, volatility=0.02, drift=0.0001, crash_period=None, crash_severity=0.2
):
    """
    Génère des données de marché synthétiques avec une tendance, volatilité et crash optionnel.

    Args:
        days (int): Nombre de jours à générer
        volatility (float): Volatilité quotidienne
        drift (float): Tendance quotidienne
        crash_period (tuple, optional): Période du crash (start, end)
        crash_severity (float): Sévérité du crash (de 0 à 1)

    Returns:
        pd.DataFrame: DataFrame avec des données de marché synthétiques
    """
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), periods=days, freq="D"
    )

    # Simulation des rendements avec mouvement brownien géométrique
    np.random.seed(42)  # Pour reproductibilité
    returns = np.random.normal(drift, volatility, days)

    # Simuler un crash si spécifié
    if crash_period:
        start, end = crash_period
        crash_length = end - start
        # Crash progressif
        crash_returns = np.linspace(0, -crash_severity / crash_length, crash_length)
        returns[start:end] = crash_returns

    # Convertir les rendements en prix
    price = 100  # Prix initial
    prices = [price]
    for r in returns:
        price *= 1 + r
        prices.append(price)
    prices = prices[:-1]  # Ajuster la longueur

    # Créer le DataFrame
    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["open"] = df["close"].shift(1).fillna(prices[0] * 0.99)
    df["high"] = df["close"] * (1 + np.random.uniform(0, 0.01, days))
    df["low"] = df["close"] * (1 - np.random.uniform(0, 0.01, days))
    df["volume"] = np.random.uniform(1000, 5000, days)

    return df


def plot_var_and_returns(returns, var_history, title="Value-at-Risk vs. Returns"):
    """
    Trace l'historique de la VaR et des rendements.

    Args:
        returns (pd.Series): Série des rendements
        var_history (list): Historique des VaR calculées
        title (str): Titre du graphique
    """
    plt.figure(figsize=(12, 6))

    # Tracer les rendements
    plt.plot(
        returns.index,
        returns.values,
        label="Rendements quotidiens",
        color="blue",
        alpha=0.5,
    )

    # Extraire les données de l'historique VaR
    var_values = [item["var"] for item in var_history]
    cvar_values = [
        item.get("cvar", item["var"] * 1.2) for item in var_history
    ]  # Utiliser VaR * 1.2 si CVaR non disponible

    # Tracer la VaR négative (pour comparaison avec les rendements négatifs)
    plt.plot(
        returns.index[-len(var_values) :],
        [-v for v in var_values],
        label="VaR (95%)",
        color="red",
        linewidth=2,
    )

    # Tracer la CVaR négative
    plt.plot(
        returns.index[-len(cvar_values) :],
        [-v for v in cvar_values],
        label="CVaR (95%)",
        color="orange",
        linewidth=2,
        linestyle="--",
    )

    # Ajouter des lignes horizontales pour référence
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Rendement / VaR")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Enregistrer le graphique
    plt.savefig(EXAMPLES_VIZ_DIR / "var_vs_returns.png")
    plt.close()


def plot_allocation_history(
    allocation_history, title="Allocation du capital au fil du temps"
):
    """
    Trace l'historique des allocations de capital.

    Args:
        allocation_history (list): Historique des allocations
        title (str): Titre du graphique
    """
    plt.figure(figsize=(12, 6))

    # Extraire les données
    dates = [
        item["timestamp"]
        for item in allocation_history
        if item["timestamp"] is not None
    ]
    allocations = [
        item["allocation"]
        for item in allocation_history
        if item["timestamp"] is not None
    ]
    kelly_values = [
        item["kelly"] for item in allocation_history if item["timestamp"] is not None
    ]
    var_scaling = [
        item["var_scaling"]
        for item in allocation_history
        if item["timestamp"] is not None
    ]

    if not dates:  # Si pas de timestamps valides, utiliser des indices
        dates = range(len(allocation_history))
        allocations = [item["allocation"] for item in allocation_history]
        kelly_values = [item["kelly"] for item in allocation_history]
        var_scaling = [item["var_scaling"] for item in allocation_history]

    # Tracer les allocations
    plt.plot(dates, allocations, label="Allocation finale", color="green", linewidth=2)
    plt.plot(
        dates,
        kelly_values,
        label="Allocation Kelly",
        color="blue",
        linewidth=1,
        linestyle="--",
    )
    plt.plot(
        dates,
        var_scaling,
        label="Ajustement VaR",
        color="red",
        linewidth=1,
        linestyle=":",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Allocation (% du capital)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Enregistrer le graphique
    plt.savefig(EXAMPLES_VIZ_DIR / "allocation_history.png")
    plt.close()


def plot_portfolio_drawdown(
    portfolio_values, drawdown_history, title="Portfolio et Maximum Drawdown"
):
    """
    Trace l'historique du portefeuille et du drawdown maximum.

    Args:
        portfolio_values (list): Historique des valeurs du portefeuille
        drawdown_history (list): Historique des drawdowns maximums
        title (str): Titre du graphique
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Tracer la valeur du portefeuille
    ax1.plot(
        range(len(portfolio_values)),
        portfolio_values,
        label="Valeur du portefeuille",
        color="blue",
    )
    ax1.set_title(title)
    ax1.set_ylabel("Valeur")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calculer le drawdown
    peaks = np.maximum.accumulate(portfolio_values)
    drawdowns = (np.array(portfolio_values) - peaks) / peaks

    # Tracer le drawdown
    ax2.fill_between(
        range(len(drawdowns)), 0, drawdowns, color="red", alpha=0.3, label="Drawdown"
    )
    ax2.plot(range(len(drawdowns)), drawdowns, color="red", alpha=0.5)

    # Tracer le drawdown maximum
    max_dd = min(drawdowns)
    ax2.axhline(
        y=max_dd, color="darkred", linestyle="--", label=f"Max DD: {max_dd:.2%}"
    )

    # Tracer la limite de drawdown
    if len(drawdown_history) > 0:
        ax2.axhline(
            y=-drawdown_history[0]["max_drawdown"],
            color="orange",
            linestyle=":",
            label=f'Limite: {drawdown_history[0]["max_drawdown"]:.2%}',
        )

    ax2.set_xlabel("Jours")
    ax2.set_ylabel("Drawdown")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Enregistrer le graphique
    plt.savefig(EXAMPLES_VIZ_DIR / "portfolio_drawdown.png")
    plt.close()


def simulate_portfolio(
    market_data, risk_manager, initial_capital=10000.0, max_position=0.2
):
    """
    Simule l'évolution d'un portefeuille avec le gestionnaire de risques avancé.

    Args:
        market_data (pd.DataFrame): Données de marché
        risk_manager (AdvancedRiskManager): Gestionnaire de risques
        initial_capital (float): Capital initial
        max_position (float): Position maximale (en % du capital)

    Returns:
        tuple: (historique du portefeuille, historique des positions)
    """
    portfolio_history = [initial_capital]
    position_history = [0]  # Commence sans position
    cash = initial_capital
    crypto = 0

    # Calculer les rendements pour l'analyse
    market_data["returns"] = market_data["close"].pct_change().fillna(0)

    for i in range(1, len(market_data)):
        current_price = market_data["close"].iloc[i]
        previous_price = market_data["close"].iloc[i - 1]

        # Calculer la valeur du portefeuille
        portfolio_value = cash + crypto * current_price
        portfolio_history.append(portfolio_value)

        # Décider de l'allocation avec le gestionnaire de risques
        data_slice = market_data.iloc[: i + 1]

        # Déterminer le type de position (long/short) en fonction de signaux simples
        position_type = (
            "long" if data_slice["returns"].iloc[-5:].mean() > 0 else "short"
        )

        # Calculer l'allocation optimale
        allocation = risk_manager.allocation_with_risk_limits(
            data_slice, position_type=position_type, portfolio_values=portfolio_history
        )

        # Convertir l'allocation en nombre d'unités
        target_position = allocation * portfolio_value / current_price

        # Calculer la différence avec la position actuelle
        position_diff = target_position - crypto

        # Exécuter le trade si nécessaire
        if abs(position_diff) > 0.01:  # Seuil minimal pour éviter les micro-trades
            # Acheter/vendre des crypto
            trade_amount = position_diff * current_price
            cash -= trade_amount
            crypto += position_diff

            logger.info(
                f"Jour {i}: {position_type.upper()} - Allocation: {allocation:.2%}, "
                f"Trade: {position_diff:.4f} units à {current_price:.2f}"
            )

        position_history.append(crypto)

    return portfolio_history, position_history


def main():
    """Fonction principale de l'exemple."""
    logger.info("Démarrage de l'exemple de gestion des risques avancée")

    # 1. Générer des données synthétiques ou charger des données réelles
    use_synthetic_data = True

    if use_synthetic_data:
        logger.info("Génération de données synthétiques")
        # Générer des données avec un crash à mi-parcours
        market_data = generate_synthetic_data(
            days=365,
            volatility=0.02,
            drift=0.0001,
            crash_period=(180, 200),  # Crash entre les jours 180 et 200
            crash_severity=0.25,  # Chute de 25%
        )
    else:
        logger.info("Chargement de données réelles")
        # Charger des données réelles (fonction à implémenter)
        market_data = load_crypto_data(symbol="BTC/USDT", timeframe="1d", limit=365)

    # 2. Initialiser le gestionnaire de risques avancé
    logger.info("Initialisation du gestionnaire de risques avancé")
    risk_manager = AdvancedRiskManager(
        config={
            "var_confidence_level": 0.95,
            "var_horizon": 1,
            "var_method": "historical",  # méthode plus robuste pour les cas pratiques
            "max_var_limit": 0.05,  # 5% VaR maximum
            "cvar_confidence_level": 0.95,
            "adaptive_capital_allocation": True,
            "kelly_fraction": 0.3,  # Kelly fractionnaire pour plus de prudence
            "max_drawdown_limit": 0.15,  # 15% de drawdown maximum
            "risk_parity_weights": True,
            "max_position_size": 0.5,  # Position maximale de 50% du capital
            "max_risk_per_trade": 0.02,  # 2% de risque maximum par trade
        }
    )

    # 3. Simuler l'évolution du portefeuille
    logger.info("Simulation de l'évolution du portefeuille")
    portfolio_history, position_history = simulate_portfolio(
        market_data, risk_manager, initial_capital=10000.0, max_position=0.5
    )

    # 4. Générer des visualisations
    logger.info("Génération des visualisations")

    # VaR et rendements
    plot_var_and_returns(
        market_data["returns"],
        risk_manager.var_history,
        title="VaR vs. Rendements quotidiens",
    )

    # Historique des allocations
    plot_allocation_history(
        risk_manager.allocation_history, title="Allocation du capital au fil du temps"
    )

    # Portfolio et drawdown
    plot_portfolio_drawdown(
        portfolio_history,
        risk_manager.drawdown_history,
        title="Évolution du portefeuille et drawdown",
    )

    # 5. Afficher les statistiques
    logger.info("Calcul des statistiques de performance")

    # Rendement total
    total_return = (portfolio_history[-1] / portfolio_history[0]) - 1
    logger.info(f"Rendement total: {total_return:.2%}")

    # Rendement annualisé
    days = len(portfolio_history)
    annualized_return = (1 + total_return) ** (365 / days) - 1
    logger.info(f"Rendement annualisé: {annualized_return:.2%}")

    # Calcul du drawdown maximum
    max_drawdown = risk_manager.calculate_maximum_drawdown(portfolio_history)
    logger.info(f"Drawdown maximum: {max_drawdown:.2%}")

    # Ratio de Sharpe approximatif (en supposant un taux sans risque de 0%)
    portfolio_returns = np.diff(portfolio_history) / portfolio_history[:-1]
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
    logger.info(f"Ratio de Sharpe: {sharpe_ratio:.2f}")

    # VaR moyenne et maximale
    if len(risk_manager.var_history) > 0:
        avg_var = np.mean([item["var"] for item in risk_manager.var_history])
        max_var = np.max([item["var"] for item in risk_manager.var_history])
        logger.info(f"VaR moyenne: {avg_var:.2%}, VaR maximale: {max_var:.2%}")

    logger.info("Exemple de gestion des risques avancée terminé")


if __name__ == "__main__":
    main()
