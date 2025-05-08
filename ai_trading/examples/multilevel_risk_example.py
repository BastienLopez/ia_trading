"""
Exemple d'utilisation de la gestion multi-niveaux des risques.

Ce script démontre comment utiliser la fonctionnalité de gestion multi-niveaux des risques
du gestionnaire de risques avancé pour adapter dynamiquement l'allocation de capital
en fonction des risques à différents niveaux.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ajouter le répertoire parent au chemin Python
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

from ai_trading.risk.advanced_risk_manager import AdvancedRiskManager


def generate_market_data(days=100, volatility=0.02, crash_intensity=0):
    """
    Génère des données de marché synthétiques.

    Args:
        days (int): Nombre de jours de données
        volatility (float): Niveau de volatilité des prix
        crash_intensity (float): Intensité d'un crash de marché (0 = pas de crash)

    Returns:
        pd.DataFrame: Données de marché synthétiques
    """
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")

    # Tendance haussière
    trend = np.linspace(100, 150, days)

    # Ajouter du bruit
    noise = np.random.normal(0, volatility * 100, days)

    # Simuler un crash si demandé
    crash = np.zeros(days)
    if crash_intensity > 0:
        crash_start = int(days * 0.6)
        crash_duration = int(days * 0.1)
        crash[crash_start : crash_start + crash_duration] = -np.linspace(
            0, crash_intensity * 100, crash_duration
        )

    prices = trend + noise + crash

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.random.uniform(0, 10, days),
            "low": prices - np.random.uniform(0, 10, days),
            "close": prices,
            "volume": np.random.uniform(1000, 5000, days),
        },
        index=dates,
    )

    return df


def generate_correlation_matrix(assets):
    """
    Génère une matrice de corrélation pour un ensemble d'actifs.

    Args:
        assets (list): Liste des symboles d'actifs

    Returns:
        pd.DataFrame: Matrice de corrélation
    """
    n = len(assets)

    # Générer une matrice de base avec des corrélations aléatoires
    np.random.seed(42)  # Pour reproductibilité
    base_matrix = np.random.uniform(0.3, 0.9, (n, n))

    # Assurer la symétrie
    correlation_matrix = (base_matrix + base_matrix.T) / 2

    # Mettre des 1 sur la diagonale
    np.fill_diagonal(correlation_matrix, 1.0)

    return pd.DataFrame(correlation_matrix, index=assets, columns=assets)


def create_portfolio_data(assets, balanced=True):
    """
    Crée des données de portefeuille synthétiques.

    Args:
        assets (list): Liste des symboles d'actifs
        balanced (bool): Si True, crée un portefeuille équilibré, sinon concentré

    Returns:
        dict: Données du portefeuille
    """
    n = len(assets)

    if balanced:
        # Portefeuille équilibré
        weights = np.ones(n) / n
    else:
        # Portefeuille concentré
        weights = np.array([0.6] + [0.4 / (n - 1)] * (n - 1))

    # Générer des rendements récents
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.03, 20)

    return {
        "weights": weights.tolist(),
        "assets": assets,
        "returns": returns.tolist(),
    }


def create_market_context(scenario="normal"):
    """
    Crée un contexte de marché basé sur différents scénarios.

    Args:
        scenario (str): Scénario de marché ('normal', 'bullish', 'bearish', 'crisis')

    Returns:
        dict: Données de contexte de marché
    """
    if scenario == "normal":
        return {
            "vix": 15.0,
            "fear_greed_index": 50,
            "credit_spread": 0.01,
            "market_liquidity": 0.3,
            "market_trend": 0.1,
        }
    elif scenario == "bullish":
        return {
            "vix": 12.0,
            "fear_greed_index": 75,
            "credit_spread": 0.008,
            "market_liquidity": 0.2,
            "market_trend": 0.7,
        }
    elif scenario == "bearish":
        return {
            "vix": 25.0,
            "fear_greed_index": 30,
            "credit_spread": 0.02,
            "market_liquidity": 0.5,
            "market_trend": -0.3,
        }
    elif scenario == "crisis":
        return {
            "vix": 40.0,
            "fear_greed_index": 10,
            "credit_spread": 0.04,
            "market_liquidity": 0.8,
            "market_trend": -0.9,
        }
    else:
        raise ValueError(f"Scénario de marché inconnu: {scenario}")


def run_multilevel_risk_scenario(
    asset_data, portfolio_data, market_data, scenario_name
):
    """
    Exécute un scénario de gestion multi-niveaux des risques.

    Args:
        asset_data (pd.DataFrame): Données de l'actif
        portfolio_data (dict): Données du portefeuille
        market_data (dict): Données du contexte de marché
        scenario_name (str): Nom du scénario

    Returns:
        dict: Résultats de l'allocation basée sur les risques
    """
    print(f"\n=== Scénario: {scenario_name} ===")

    risk_manager = AdvancedRiskManager(
        config={
            "var_confidence_level": 0.95,
            "max_var_limit": 0.05,
            "max_position_size": 0.8,  # 80% du capital max
            "use_multilevel_risk": True,
            "strategy_risk_weight": 0.4,
            "portfolio_risk_weight": 0.3,
            "market_risk_weight": 0.3,
            "max_correlation_exposure": 0.7,
        }
    )

    # Créer une matrice de corrélation
    correlation_matrix = generate_correlation_matrix(portfolio_data["assets"])

    # Calculer l'allocation multi-niveaux
    result = risk_manager.multilevel_risk_management(
        asset_data,
        market_data=market_data,
        portfolio_data=portfolio_data,
        correlation_matrix=correlation_matrix,
    )

    # Afficher les résultats
    print(f"Risque au niveau stratégie: {result['strategy_risk']:.4f}")
    print(f"Risque au niveau portefeuille: {result['portfolio_risk']:.4f}")
    print(f"Risque au niveau marché: {result['market_risk']:.4f}")
    print(f"Score de risque global: {result['risk_score']:.4f}")
    print(f"Allocation recommandée: {result['allocation']:.2%}")

    return result


def visualize_results(results):
    """
    Visualise les résultats des différents scénarios.

    Args:
        results (dict): Dictionnaire des résultats par scénario
    """
    scenarios = list(results.keys())

    # Données pour les graphiques
    risk_scores = [results[s]["risk_score"] for s in scenarios]
    allocations = [results[s]["allocation"] for s in scenarios]
    strategy_risks = [results[s]["strategy_risk"] for s in scenarios]
    portfolio_risks = [results[s]["portfolio_risk"] for s in scenarios]
    market_risks = [results[s]["market_risk"] for s in scenarios]

    # Créer la figure et les axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Graphique des scores de risque
    ax1 = axes[0]
    bar_width = 0.2
    index = np.arange(len(scenarios))

    ax1.bar(index - bar_width, strategy_risks, bar_width, label="Risque Stratégie")
    ax1.bar(index, portfolio_risks, bar_width, label="Risque Portefeuille")
    ax1.bar(index + bar_width, market_risks, bar_width, label="Risque Marché")
    ax1.plot(index, risk_scores, "ro-", label="Score Global")

    ax1.set_xlabel("Scénario")
    ax1.set_ylabel("Score de Risque")
    ax1.set_title("Scores de Risque par Niveau et par Scénario")
    ax1.set_xticks(index)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Graphique des allocations
    ax2 = axes[1]
    ax2.bar(scenarios, allocations, color="green", alpha=0.6)
    ax2.set_xlabel("Scénario")
    ax2.set_ylabel("Allocation (%)")
    ax2.set_title("Allocation Recommandée par Scénario")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Ajouter les valeurs sur les barres
    for i, v in enumerate(allocations):
        ax2.text(i, v + 0.02, f"{v:.1%}", ha="center")

    plt.tight_layout()
    plt.savefig("multilevel_risk_results.png")
    plt.show()


def main():
    """Fonction principale pour exécuter l'exemple."""
    print("=== Exemple de Gestion Multi-Niveaux des Risques ===")

    # Liste des actifs
    assets = ["BTC", "ETH", "SOL", "AVAX", "DOT"]

    # Générer des données pour un actif (BTC)
    normal_market = generate_market_data(days=100, volatility=0.02)
    volatile_market = generate_market_data(days=100, volatility=0.04)
    crash_market = generate_market_data(days=100, volatility=0.03, crash_intensity=0.3)

    # Créer des données de portefeuille
    balanced_portfolio = create_portfolio_data(assets, balanced=True)
    concentrated_portfolio = create_portfolio_data(assets, balanced=False)

    # Exécuter différents scénarios
    results = {}

    # Scénario 1: Marché normal avec portefeuille équilibré
    results["Normal + Équilibré"] = run_multilevel_risk_scenario(
        normal_market,
        balanced_portfolio,
        create_market_context("normal"),
        "Marché Normal + Portefeuille Équilibré",
    )

    # Scénario 2: Marché haussier avec portefeuille équilibré
    results["Haussier + Équilibré"] = run_multilevel_risk_scenario(
        normal_market,
        balanced_portfolio,
        create_market_context("bullish"),
        "Marché Haussier + Portefeuille Équilibré",
    )

    # Scénario 3: Marché baissier avec portefeuille concentré
    results["Baissier + Concentré"] = run_multilevel_risk_scenario(
        volatile_market,
        concentrated_portfolio,
        create_market_context("bearish"),
        "Marché Baissier + Portefeuille Concentré",
    )

    # Scénario 4: Crise de marché avec portefeuille concentré
    results["Crise + Concentré"] = run_multilevel_risk_scenario(
        crash_market,
        concentrated_portfolio,
        create_market_context("crisis"),
        "Crise de Marché + Portefeuille Concentré",
    )

    # Visualiser les résultats
    visualize_results(results)

    print(
        "\nL'allocation de capital est automatiquement ajustée en fonction des risques à tous les niveaux."
    )
    print(
        "En période de crise ou de marché baissier, l'allocation est réduite pour limiter les pertes."
    )
    print(
        "En marché haussier avec un portefeuille équilibré, l'allocation est maximisée pour optimiser les gains."
    )


if __name__ == "__main__":
    main()
