"""
Exemple d'utilisation du système d'ordres professionnels avec limites dynamiques.

Cet exemple démontre:
1. La création et configuration du gestionnaire d'ordres
2. La création et exécution de différents types d'ordres
3. L'utilisation des limites dynamiques basées sur la volatilité
4. L'intégration avec le gestionnaire de risques avancé
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_trading.orders.order_integration import (
    DynamicLimitOrderStrategy,
    OrderExecutionEnv,
)
from ai_trading.orders.order_manager import OrderManager
from ai_trading.risk.advanced_risk_manager import AdvancedRiskManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_data(symbol="BTC-USD", days=30):
    """
    Charge ou génère des données d'exemple pour les tests.

    Args:
        symbol (str): Symbole de l'actif
        days (int): Nombre de jours de données à générer

    Returns:
        pd.DataFrame: DataFrame avec les données OHLCV
    """
    try:
        # Essayer de charger des données réelles si disponibles
        data_dir = Path(__file__).parent.parent / "data"
        file_path = data_dir / f"{symbol.replace('-', '_').lower()}_daily.csv"

        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            return df.iloc[-days:]
    except Exception as e:
        logger.warning(f"Impossible de charger les données réelles: {e}")

    # Générer des données synthétiques si les données réelles ne sont pas disponibles
    logger.info("Génération de données synthétiques pour les tests")

    # Paramètres de base
    start_price = 50000.0  # Prix de départ
    volatility = 0.02  # Volatilité journalière

    # Générer les timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Générer les prix en utilisant un processus de marche aléatoire
    np.random.seed(42)  # Pour reproductibilité
    returns = np.random.normal(0, volatility, size=len(dates))
    log_returns = np.cumsum(returns)
    prices = start_price * np.exp(log_returns)

    # Créer les prix OHLCV
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        high = close * (1 + abs(np.random.normal(0, volatility * 0.5)))
        low = close * (1 - abs(np.random.normal(0, volatility * 0.5)))
        open_price = low + (high - low) * np.random.random()
        volume = np.random.gamma(shape=2.0, scale=1000) * (close / start_price)

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    # Créer le DataFrame
    df = pd.DataFrame(data)

    # Ajouter des colonnes supplémentaires utiles
    df["volatility"] = df["close"].pct_change().rolling(5).std()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["spread"] = (df["high"] - df["low"]) / df["close"]
    df["avg_volume"] = df["volume"].rolling(10).mean()

    return df


def compute_rsi(prices, window=14):
    """Calcule le RSI sur une série de prix."""
    # Calculer les différences
    delta = prices.diff()

    # Séparer les gains et les pertes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculer la moyenne mobile des gains et des pertes
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    # Première valeur de la moyenne mobile
    avg_gain.iloc[window] = gain.iloc[1 : window + 1].mean()
    avg_loss.iloc[window] = loss.iloc[1 : window + 1].mean()

    # Appliquer la formule de lissage exponentiel
    for i in range(window + 1, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) + loss.iloc[i]) / window

    # Calculer le ratio
    rs = avg_gain / avg_loss

    # Calculer le RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def run_order_example():
    """Exécute l'exemple d'utilisation du système d'ordres professionnels."""
    logger.info("Démarrage de l'exemple d'ordres professionnels")

    # Charger les données d'exemple
    df = load_sample_data(symbol="BTC-USD", days=30)
    logger.info(f"Données chargées: {len(df)} jours")

    # Configuration du gestionnaire de risques
    risk_config = {
        "var_confidence_level": 0.95,
        "var_method": "historical",
        "max_var_limit": 0.05,
        "adaptive_capital_allocation": True,
        "kelly_fraction": 0.5,
        "risk_per_trade": 0.02,
        "max_drawdown_limit": 0.20,
    }
    risk_manager = AdvancedRiskManager(risk_config)

    # Configuration du gestionnaire d'ordres
    order_config = {
        "max_orders_per_symbol": 10,
        "adaptive_limits": True,
        "position_sizing_method": "risk_based",
        "default_slippage": 0.001,
    }
    order_manager = OrderManager(risk_manager=risk_manager, config=order_config)

    # Configuration de l'environnement d'exécution
    exec_config = {
        "use_limit_orders": True,
        "use_stop_orders": True,
        "use_oco_orders": True,
        "dynamic_limit_orders": True,
        "limit_order_offset": 0.002,
    }
    order_env = OrderExecutionEnv(risk_manager=risk_manager, config=exec_config)

    # Configuration de la stratégie d'ordres limites dynamiques
    limit_strategy = DynamicLimitOrderStrategy(
        {"base_offset": 0.002, "volatility_factor": 5.0, "max_offset": 0.01}
    )

    # Simuler l'exécution d'ordres sur les données historiques
    results = []
    portfolio_value = 10000.0  # Valeur initiale du portefeuille
    position = 0.0  # Position initiale

    logger.info("Simulation de trading avec ordres professionnels...")

    # Parcourir les données (en sautant les premières lignes pour avoir des indicateurs valides)
    for i in range(20, len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]
        symbol = "BTC-USD"

        # Créer des données de marché pour le tick actuel
        market_data = {
            symbol: {
                "timestamp": current_row["timestamp"].isoformat(),
                "open": current_row["open"],
                "high": current_row["high"],
                "low": current_row["low"],
                "close": current_row["close"],
                "price": current_row["close"],
                "volume": current_row["volume"],
                "volatility": current_row["volatility"],
                "atr": current_row["atr"],
                "rsi": current_row["rsi"],
                "spread": current_row["spread"],
                "avg_volume": current_row["avg_volume"],
                "portfolio_value": portfolio_value,
            }
        }

        # Mettre à jour le gestionnaire d'ordres avec les données de marché
        order_manager.update_market_data(market_data)

        # Stratégie de trading simplifiée (RSI)
        rsi = current_row["rsi"]
        action_value = 0.0

        # Règles de trading basées sur le RSI
        if rsi < 30 and position <= 0:  # Survendu -> Acheter
            action_value = 0.7  # 70% du capital disponible
        elif rsi > 70 and position >= 0:  # Suracheté -> Vendre
            action_value = -0.8  # 80% de la position

        # Traiter l'action (création d'ordres)
        if abs(action_value) > 0.05:
            execution_result = order_env.process_action(
                symbol, action_value, market_data[symbol]
            )
            logger.info(
                f"Action: {action_value:.2f}, Ordres créés: {len(execution_result['orders'])}"
            )

        # Simuler l'avancement du marché et l'exécution des ordres
        next_market_data = {
            symbol: {
                "timestamp": next_row["timestamp"].isoformat(),
                "open": next_row["open"],
                "high": next_row["high"],
                "low": next_row["low"],
                "close": next_row["close"],
                "price": next_row["close"],
                "volume": next_row["volume"],
                "volatility": next_row["volatility"],
                "atr": next_row["atr"],
                "rsi": next_row["rsi"],
                "spread": next_row["spread"],
                "avg_volume": next_row["avg_volume"],
            }
        }

        # Exécuter les ordres
        executions = order_env.process_market_update(next_market_data)

        # Mettre à jour la position et la valeur du portefeuille
        position_data = order_env.get_position(symbol)
        position = position_data["quantity"]

        # Calculer la valeur du portefeuille (cash + position)
        portfolio_value = (
            10000.0  # Valeur initiale (cash)
            - sum(
                e["price"] * e["quantity"]
                for e in order_env.execution_history
                if e["side"] == "buy"
            )
            + sum(
                e["price"] * e["quantity"]
                for e in order_env.execution_history
                if e["side"] == "sell"
            )
            + position * next_row["close"]  # Valeur de la position actuelle
        )

        # Enregistrer les résultats
        results.append(
            {
                "date": next_row["timestamp"],
                "price": next_row["close"],
                "position": position,
                "portfolio_value": portfolio_value,
                "executions": len(executions),
                "rsi": next_row["rsi"],
            }
        )

        if len(executions) > 0:
            logger.info(
                f"Jour {i}: {len(executions)} ordres exécutés, position: {position}, "
                f"portefeuille: ${portfolio_value:.2f}"
            )

    # Afficher les résultats finaux
    results_df = pd.DataFrame(results)
    total_return = (results_df["portfolio_value"].iloc[-1] / 10000.0 - 1) * 100
    total_trades = len(order_env.execution_history)

    logger.info("=== Résultats de la simulation ===")
    logger.info(
        f"Valeur finale du portefeuille: ${results_df['portfolio_value'].iloc[-1]:.2f}"
    )
    logger.info(f"Rendement total: {total_return:.2f}%")
    logger.info(f"Nombre total de transactions: {total_trades}")

    # Visualiser les résultats
    visualize_results(results_df, order_env.execution_history)

    return results_df, order_env.execution_history


def visualize_results(results_df, executions):
    """
    Visualise les résultats de la simulation.

    Args:
        results_df (pd.DataFrame): DataFrame avec les résultats par jour
        executions (list): Liste des exécutions d'ordres
    """
    # Créer une figure avec plusieurs sous-graphiques
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Graphique 1: Prix et positions
    ax1 = axs[0]
    ax1.plot(results_df["date"], results_df["price"], label="Prix", color="blue")
    ax1.set_ylabel("Prix ($)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title("Simulation de trading avec ordres professionnels")

    # Ajouter un axe secondaire pour la position
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(
        results_df["date"],
        results_df["position"],
        0,
        where=results_df["position"] >= 0,
        color="green",
        alpha=0.3,
        label="Position Long",
    )
    ax1_twin.fill_between(
        results_df["date"],
        results_df["position"],
        0,
        where=results_df["position"] <= 0,
        color="red",
        alpha=0.3,
        label="Position Short",
    )
    ax1_twin.set_ylabel("Position", color="green")
    ax1_twin.tick_params(axis="y", labelcolor="green")

    # Ajouter les points d'achat et de vente
    buys = [
        (datetime.fromisoformat(e["timestamp"]), e["price"])
        for e in executions
        if e["side"] == "buy"
    ]
    sells = [
        (datetime.fromisoformat(e["timestamp"]), e["price"])
        for e in executions
        if e["side"] == "sell"
    ]

    if buys:
        buy_dates, buy_prices = zip(*buys)
        ax1.scatter(
            buy_dates, buy_prices, color="green", marker="^", s=100, label="Achat"
        )

    if sells:
        sell_dates, sell_prices = zip(*sells)
        ax1.scatter(
            sell_dates, sell_prices, color="red", marker="v", s=100, label="Vente"
        )

    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")

    # Graphique 2: Valeur du portefeuille
    ax2 = axs[1]
    ax2.plot(
        results_df["date"],
        results_df["portfolio_value"],
        label="Portefeuille",
        color="purple",
    )
    ax2.set_ylabel("Valeur ($)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.set_title("Évolution de la valeur du portefeuille")
    ax2.axhline(y=10000, color="gray", linestyle="--", label="Capital initial")
    ax2.legend()

    # Graphique 3: RSI
    ax3 = axs[2]
    ax3.plot(results_df["date"], results_df["rsi"], label="RSI", color="orange")
    ax3.set_ylabel("RSI", color="orange")
    ax3.tick_params(axis="y", labelcolor="orange")
    ax3.set_title("Indicateur RSI")
    ax3.axhline(y=70, color="red", linestyle="--", label="Suracheté")
    ax3.axhline(y=30, color="green", linestyle="--", label="Survendu")
    ax3.legend()

    # Paramètres communs
    ax3.set_xlabel("Date")
    plt.tight_layout()

    # Sauvegarder la figure
    fig.savefig("order_system_simulation.png")
    plt.close(fig)

    logger.info("Graphique des résultats sauvegardé dans 'order_system_simulation.png'")


if __name__ == "__main__":
    results_df, executions = run_order_example()
