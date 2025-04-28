import datetime
import logging
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ai_trading.config import VISUALIZATION_DIR

# Configuration du logger
logger = logging.getLogger("EvaluationRL")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Définir le chemin pour les visualisations
# VISUALIZATION_DIR = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "visualizations",
#     "evaluation",
# )
# Créer le répertoire s'il n'existe pas
save_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "info_retour/evaluation"
)
os.makedirs(save_dir, exist_ok=True)


class PerformanceMetrics:
    """
    Classe pour calculer les métriques de performance d'une stratégie de trading.
    """

    @staticmethod
    def calculate_returns(portfolio_values):
        """
        Calcule les rendements quotidiens.

        Args:
            portfolio_values (array): Historique des valeurs du portefeuille

        Returns:
            array: Rendements quotidiens
        """
        return np.diff(portfolio_values) / portfolio_values[:-1]

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        """
        Calcule le ratio de Sharpe.

        Args:
            returns (list): Liste des rendements
            risk_free_rate (float): Taux sans risque

        Returns:
            float: Ratio de Sharpe
        """
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate

        # Éviter la division par zéro
        std_dev = np.std(excess_returns)
        if std_dev == 0:
            return 0.0  # Retourner un float même si std_dev est 0

        sharpe_ratio = np.mean(excess_returns) / std_dev

        # Annualiser (supposons des rendements quotidiens)
        sharpe_ratio_annualized = sharpe_ratio * np.sqrt(252)

        return float(sharpe_ratio_annualized)  # Convertir explicitement en float

    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.0):
        """
        Calcule le ratio de Sortino.

        Args:
            returns (array): Rendements quotidiens
            risk_free_rate (float): Taux sans risque annualisé

        Returns:
            float: Ratio de Sortino
        """
        # Convertir le taux sans risque annuel en taux quotidien
        daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1

        excess_returns = returns - daily_risk_free

        # Calculer la semi-variance (uniquement les rendements négatifs)
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) == 0:
            return np.inf  # Pas de rendements négatifs

        downside_deviation = np.sqrt(np.mean(negative_returns**2))
        if downside_deviation == 0:
            return 0

        # Annualiser le ratio de Sortino
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)

    @staticmethod
    def calculate_max_drawdown(portfolio_values):
        """
        Calcule le drawdown maximum.

        Args:
            portfolio_values (array): Historique des valeurs du portefeuille

        Returns:
            tuple: (max_drawdown, start_idx, end_idx, recovery_idx)
        """
        # Calculer les pics cumulatifs
        peaks = np.maximum.accumulate(portfolio_values)

        # Calculer les drawdowns
        drawdowns = (portfolio_values - peaks) / peaks

        # Trouver le drawdown maximum
        max_drawdown = np.min(drawdowns)
        end_idx = np.argmin(drawdowns)

        # Trouver le pic précédent
        peak_value = peaks[end_idx]
        start_idx = np.where(portfolio_values[:end_idx] == peak_value)[0]
        start_idx = start_idx[-1] if len(start_idx) > 0 else 0

        # Trouver l'indice de récupération (ou None si pas de récupération)
        recovery_idx = None
        if end_idx < len(portfolio_values) - 1:
            recovery_indices = np.where(portfolio_values[end_idx:] >= peak_value)[0]
            if len(recovery_indices) > 0:
                recovery_idx = end_idx + recovery_indices[0]

        return max_drawdown, start_idx, end_idx, recovery_idx

    @staticmethod
    def calculate_win_rate(trades):
        """
        Calcule le taux de réussite des transactions.

        Args:
            trades (list): Liste des transactions avec leurs résultats

        Returns:
            float: Taux de réussite
        """
        if not trades:
            return 0

        winning_trades = sum(1 for trade in trades if trade["profit"] > 0)
        return winning_trades / len(trades)

    @staticmethod
    def calculate_profit_factor(trades):
        """
        Calcule le facteur de profit (profits bruts / pertes brutes).

        Args:
            trades (list): Liste des transactions avec leurs résultats

        Returns:
            float: Facteur de profit
        """
        gross_profit = sum(trade["profit"] for trade in trades if trade["profit"] > 0)
        gross_loss = sum(
            abs(trade["profit"]) for trade in trades if trade["profit"] < 0
        )

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0

        return gross_profit / gross_loss

    @staticmethod
    def calculate_average_trade(trades):
        """
        Calcule le profit moyen par transaction.

        Args:
            trades (list): Liste des transactions avec leurs résultats

        Returns:
            float: Profit moyen par transaction
        """
        if not trades:
            return 0

        total_profit = sum(trade["profit"] for trade in trades)
        return total_profit / len(trades)

    @staticmethod
    def calculate_expectancy(trades):
        """
        Calcule l'espérance mathématique par transaction.

        Args:
            trades (list): Liste des transactions avec leurs résultats

        Returns:
            float: Espérance mathématique par transaction
        """
        if not trades:
            return 0

        win_rate = PerformanceMetrics.calculate_win_rate(trades)

        avg_win = sum(trade["profit"] for trade in trades if trade["profit"] > 0)
        avg_win = (
            avg_win / sum(1 for trade in trades if trade["profit"] > 0)
            if sum(1 for trade in trades if trade["profit"] > 0) > 0
            else 0
        )

        avg_loss = sum(trade["profit"] for trade in trades if trade["profit"] < 0)
        avg_loss = (
            avg_loss / sum(1 for trade in trades if trade["profit"] < 0)
            if sum(1 for trade in trades if trade["profit"] < 0) > 0
            else 0
        )

        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    @staticmethod
    def calculate_all_metrics(
        portfolio_values, benchmark_values=None, risk_free_rate=0.0
    ):
        """
        Calcule toutes les métriques de performance.

        Args:
            portfolio_values (array): Historique des valeurs du portefeuille
            benchmark_values (array, optional): Historique des valeurs de référence
            risk_free_rate (float): Taux sans risque annualisé

        Returns:
            dict: Toutes les métriques de performance
        """
        # Vérifier que les données sont valides
        if len(portfolio_values) < 2:
            return {
                "total_return": 0,
                "annualized_return": 0,
                "volatility": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "calmar_ratio": 0,
                "omega_ratio": 0,
            }

        # Calculer les rendements
        returns = PerformanceMetrics.calculate_returns(portfolio_values)

        # Rendement total
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

        # Rendement annualisé (supposer des données quotidiennes)
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1

        # Volatilité annualisée
        volatility = np.std(returns) * np.sqrt(252)

        # Ratio de Sharpe
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(
            returns, risk_free_rate
        )

        # Ratio de Sortino
        sortino_ratio = PerformanceMetrics.calculate_sortino_ratio(
            returns, risk_free_rate
        )

        # Drawdown maximum
        max_drawdown, _, _, _ = PerformanceMetrics.calculate_max_drawdown(
            portfolio_values
        )

        # Ratio de Calmar
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Ratio Omega
        threshold = (1 + risk_free_rate) ** (1 / 252) - 1  # Seuil quotidien
        returns_above_threshold = returns[returns > threshold]
        returns_below_threshold = returns[returns <= threshold]

        if (
            len(returns_below_threshold) == 0
            or np.sum(threshold - returns_below_threshold) == 0
        ):
            omega_ratio = np.inf if len(returns_above_threshold) > 0 else 0
        else:
            omega_ratio = np.sum(returns_above_threshold - threshold) / np.sum(
                threshold - returns_below_threshold
            )

        # Métriques de base
        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "omega_ratio": omega_ratio,
        }

        # Ajouter les métriques relatives au benchmark si disponible
        if benchmark_values is not None and len(benchmark_values) == len(
            portfolio_values
        ):
            benchmark_returns = PerformanceMetrics.calculate_returns(benchmark_values)

            # Bêta
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            variance = np.var(benchmark_returns)
            beta = covariance / variance if variance != 0 else 0

            # Alpha
            benchmark_total_return = (benchmark_values[-1] / benchmark_values[0]) - 1
            benchmark_annualized_return = (1 + benchmark_total_return) ** (
                252 / n_periods
            ) - 1
            alpha = annualized_return - (
                risk_free_rate + beta * (benchmark_annualized_return - risk_free_rate)
            )

            # Erreur de suivi
            tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)

            # Ratio d'information
            information_ratio = (
                (annualized_return - benchmark_annualized_return) / tracking_error
                if tracking_error != 0
                else 0
            )

            # Ajouter les métriques au dictionnaire
            metrics.update(
                {
                    "beta": beta,
                    "alpha": alpha,
                    "tracking_error": tracking_error,
                    "information_ratio": information_ratio,
                }
            )

        return metrics


class PerformanceVisualizer:
    """
    Classe pour visualiser les performances d'une stratégie de trading.
    """

    def __init__(self, save_dir=None, figsize=(12, 6)):
        """
        Initialise le visualiseur de performances.

        Args:
            save_dir (str, optional): Répertoire pour sauvegarder les visualisations
            figsize (tuple): Taille des figures
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self.figsize = figsize

        # Configurer le style de seaborn
        sns.set_style("whitegrid")

        logger.info("Visualiseur de performances initialisé")

    def plot_portfolio_performance(
        self,
        portfolio_values,
        benchmark_values=None,
        dates=None,
        title="Performance du portefeuille",
    ):
        """
        Trace l'évolution de la valeur du portefeuille.

        Args:
            portfolio_values (array): Historique des valeurs du portefeuille
            benchmark_values (array, optional): Historique des valeurs de référence
            dates (array, optional): Dates correspondantes
            title (str): Titre du graphique
        """
        plt.figure(figsize=self.figsize)

        # Créer un axe des x basé sur les dates ou les indices
        x = dates if dates is not None else np.arange(len(portfolio_values))

        # Tracer la valeur du portefeuille
        plt.plot(x, portfolio_values, label="Stratégie", color="blue", linewidth=2)

        # Tracer la référence si disponible
        if benchmark_values is not None:
            plt.plot(
                x,
                benchmark_values,
                label="Référence",
                color="red",
                linestyle="--",
                linewidth=1.5,
            )

        # Formater l'axe des x si des dates sont fournies
        if dates is not None:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gcf().autofmt_xdate()

        plt.title(title)
        plt.xlabel("Date" if dates is not None else "Période")
        plt.ylabel("Valeur du portefeuille ($)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, "portfolio_performance.png"))
            plt.close()
        else:
            plt.show()

    def plot_returns_distribution(self, returns, title="Distribution des rendements"):
        """
        Trace la distribution des rendements.

        Args:
            returns (array): Rendements quotidiens
            title (str): Titre du graphique
        """
        plt.figure(figsize=self.figsize)

        # Tracer l'histogramme des rendements
        sns.histplot(returns, kde=True, bins=50, color="blue")

        # Ajouter une ligne verticale à zéro
        plt.axvline(x=0, color="red", linestyle="--", linewidth=1.5)

        # Ajouter une ligne verticale à la moyenne
        plt.axvline(
            x=np.mean(returns),
            color="green",
            linestyle="-",
            linewidth=1.5,
            label=f"Moyenne: {np.mean(returns):.4f}",
        )

        plt.title(title)
        plt.xlabel("Rendement quotidien")
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, "returns_distribution.png"))
            plt.close()
        else:
            plt.show()

    def plot_drawdown(self, portfolio_values, dates=None, title="Drawdown"):
        """
        Trace le drawdown du portefeuille.

        Args:
            portfolio_values (array): Historique des valeurs du portefeuille
            dates (array, optional): Dates correspondantes
            title (str): Titre du graphique
        """
        # Calculer les pics cumulatifs
        peaks = np.maximum.accumulate(portfolio_values)

        # Calculer les drawdowns
        drawdowns = (portfolio_values - peaks) / peaks

        # Trouver le drawdown maximum
        max_drawdown, start_idx, end_idx, recovery_idx = (
            PerformanceMetrics.calculate_max_drawdown(portfolio_values)
        )

        plt.figure(figsize=self.figsize)

        # Créer un axe des x basé sur les dates ou les indices
        x = dates if dates is not None else np.arange(len(portfolio_values))

        # Tracer le drawdown
        plt.plot(x, drawdowns, color="red", linewidth=2)

        # Mettre en évidence le drawdown maximum
        if max_drawdown < 0:
            plt.fill_between(
                x, 0, drawdowns, where=(drawdowns < 0), color="red", alpha=0.3
            )

            # Marquer le début, la fin et la récupération du drawdown maximum
            plt.scatter(x[start_idx], 0, color="green", s=100, marker="^", label="Pic")
            plt.scatter(
                x[end_idx], max_drawdown, color="red", s=100, marker="v", label="Creux"
            )

            if recovery_idx is not None:
                plt.scatter(
                    x[recovery_idx],
                    0,
                    color="blue",
                    s=100,
                    marker="o",
                    label="Récupération",
                )

        # Formater l'axe des x si des dates sont fournies
        if dates is not None:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.gcf().autofmt_xdate()

        plt.title(f"{title} (Max: {max_drawdown:.2%})")
        plt.xlabel("Date" if dates is not None else "Période")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, "drawdown.png"))
            plt.close()
        else:
            plt.show()

    def plot_trade_analysis(self, trades, title="Analyse des transactions"):
        """
        Trace l'analyse des transactions.

        Args:
            trades (list): Liste des transactions avec leurs résultats
            title (str): Titre du graphique
        """
        if not trades:
            logger.warning("Aucune transaction à analyser")
            return

        # Extraire les profits et les dates
        profits = [trade["profit"] for trade in trades]
        dates = [trade["exit_date"] for trade in trades if "exit_date" in trade]

        # Créer une figure avec plusieurs sous-graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16)

        # 1. Profits par transaction
        ax1 = axes[0, 0]
        ax1.bar(
            range(len(profits)),
            profits,
            color=["green" if p > 0 else "red" for p in profits],
        )
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.set_title("Profits par transaction")
        ax1.set_xlabel("Transaction #")
        ax1.set_ylabel("Profit ($)")
        ax1.grid(True)

        # 2. Distribution des profits
        ax2 = axes[0, 1]
        sns.histplot(profits, kde=True, bins=20, ax=ax2, color="blue")
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax2.axvline(
            x=np.mean(profits),
            color="green",
            linestyle="-",
            linewidth=1.5,
            label=f"Moyenne: {np.mean(profits):.2f}",
        )
        ax2.set_title("Distribution des profits")
        ax2.set_xlabel("Profit ($)")
        ax2.set_ylabel("Fréquence")
        ax2.legend()
        ax2.grid(True)

        # 3. Profits cumulés
        ax3 = axes[1, 0]
        cumulative_profits = np.cumsum(profits)
        ax3.plot(
            range(len(cumulative_profits)),
            cumulative_profits,
            color="blue",
            linewidth=2,
        )
        ax3.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
        ax3.set_title("Profits cumulés")
        ax3.set_xlabel("Transaction #")
        ax3.set_ylabel("Profit cumulé ($)")
        ax3.grid(True)

        # 4. Métriques de trading
        ax4 = axes[1, 1]
        ax4.axis("off")

        # Calculer les métriques
        win_rate = PerformanceMetrics.calculate_win_rate(trades)
        profit_factor = PerformanceMetrics.calculate_profit_factor(trades)
        avg_trade = PerformanceMetrics.calculate_average_trade(trades)
        expectancy = PerformanceMetrics.calculate_expectancy(trades)

        # Créer un tableau de métriques
        metrics_table = [
            ["Nombre de transactions", len(trades)],
            ["Transactions gagnantes", sum(1 for p in profits if p > 0)],
            ["Transactions perdantes", sum(1 for p in profits if p <= 0)],
            ["Taux de réussite", f"{win_rate:.2%}"],
            ["Facteur de profit", f"{profit_factor:.2f}"],
            ["Profit moyen", f"${avg_trade:.2f}"],
            ["Espérance", f"${expectancy:.2f}"],
            ["Profit total", f"${sum(profits):.2f}"],
        ]

        # Créer le tableau
        table = ax4.table(
            cellText=[[str(x) for x in row] for row in metrics_table],
            colLabels=["Métrique", "Valeur"],
            loc="center",
            cellLoc="center",
        )

        # Styliser le tableau
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, "trade_analysis.png"))
            plt.close()
        else:
            plt.show()

    def plot_performance_metrics(
        self, results, dates=None, title="Métriques de performance"
    ):
        """Trace les métriques de performance dans un tableau."""
        plt.figure(figsize=self.figsize)

        # Calcul des métriques
        portfolio_values = results["portfolio_history"]
        metrics = PerformanceMetrics.calculate_all_metrics(portfolio_values)
        metrics_df = pd.DataFrame([metrics])

        # Création du tableau
        cell_text = metrics_df.values.tolist()

        # Modification ici : utilisation d'un label personnalisé au lieu de l'index
        row_labels = ["Métriques"]  # Étiquette unique pour la seule ligne

        plt.table(
            cellText=cell_text,
            rowLabels=row_labels,  # Utilisation du label personnalisé
            colLabels=metrics_df.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],  # Ajustement de la position
        )

        plt.axis("off")
        plt.title(title)

        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, "performance_metrics.png"))
            plt.close()
        else:
            plt.show()

    def plot_actions_distribution(
        self, actions, action_labels=None, title="Distribution des actions"
    ):
        """
        Trace la distribution des actions prises par l'agent.

        Args:
            actions (array): Historique des actions
            action_labels (list, optional): Étiquettes des actions
            title (str): Titre du graphique
        """
        if action_labels is None:
            action_labels = ["Conserver", "Acheter", "Vendre"]

        # Compter les occurrences de chaque action
        unique_actions, counts = np.unique(actions, return_counts=True)

        # Créer un dictionnaire pour associer les actions à leurs étiquettes
        action_counts = {}
        for i, action in enumerate(unique_actions):
            if int(action) < len(action_labels):
                action_counts[action_labels[int(action)]] = counts[i]
            else:
                action_counts[f"Action {int(action)}"] = counts[i]

        plt.figure(figsize=self.figsize)

        # Tracer le camembert
        plt.pie(
            action_counts.values(),
            labels=action_counts.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

        plt.title(title)
        plt.tight_layout()

        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, "actions_distribution.png"))
            plt.close()
        else:
            plt.show()

    def create_performance_dashboard(
        self,
        results,
        benchmark_results=None,
        dates=None,
        trades=None,
        actions=None,
        risk_free_rate=0.0,
    ):
        """
        Crée un tableau de bord complet des performances.

        Args:
            results (dict): Résultats de l'évaluation
            benchmark_results (dict, optional): Résultats de référence
            dates (array, optional): Dates correspondantes
            trades (list, optional): Liste des transactions
            actions (array, optional): Historique des actions
            risk_free_rate (float): Taux sans risque annualisé
        """
        portfolio_values = results["portfolio_history"]
        benchmark_values = (
            benchmark_results["portfolio_history"] if benchmark_results else None
        )

        # 1. Performance du portefeuille
        self.plot_portfolio_performance(portfolio_values, benchmark_values, dates)

        # 2. Distribution des rendements
        returns = PerformanceMetrics.calculate_returns(portfolio_values)
        self.plot_returns_distribution(returns)

        # 3. Drawdown
        self.plot_drawdown(portfolio_values, dates)

        # 4. Analyse des transactions
        if trades:
            self.plot_trade_analysis(trades)

        # 5. Distribution des actions
        if actions is not None:
            self.plot_actions_distribution(actions)

        # 6. Tableau des métriques de performance
        self.plot_performance_metrics(results)

        logger.info(f"Tableau de bord de performance créé dans {self.save_dir}")


def evaluate_agent(agent, env, num_episodes=1, render=False):
    """
    Évalue un agent sur un environnement.

    Args:
        agent: L'agent à évaluer
        env: L'environnement de trading
        num_episodes (int): Nombre d'épisodes d'évaluation
        render (bool): Si True, affiche l'environnement pendant l'évaluation

    Returns:
        dict: Résultats de l'évaluation
    """
    logger.info(f"Évaluation de l'agent sur {num_episodes} épisodes...")

    all_portfolio_values = []
    all_actions = []
    all_rewards = []
    all_trades = []

    for e in tqdm(range(num_episodes), desc="Évaluation"):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        done = False
        episode_actions = []
        episode_rewards = []

        while not done:
            # Choisir une action (sans exploration)
            action = agent.act(state, use_epsilon=False)
            episode_actions.append(action)

            # Exécuter l'action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])

            # Enregistrer la récompense
            episode_rewards.append(reward)

            # Passer à l'état suivant
            state = next_state

            # Afficher l'environnement si demandé
            if render:
                env.render()

        # Enregistrer les résultats de l'épisode
        all_portfolio_values.append(env.get_portfolio_history())
        all_actions.append(episode_actions)
        all_rewards.append(episode_rewards)

        # Enregistrer les transactions si disponibles
        if hasattr(env, "trades"):
            all_trades.extend(env.trades)

    # Calculer les métriques moyennes
    avg_portfolio_values = np.mean([values[-1] for values in all_portfolio_values])
    avg_returns = (avg_portfolio_values / env.initial_balance) - 1

    # Calculer le ratio de Sharpe moyen
    all_returns = []
    for values in all_portfolio_values:
        returns = PerformanceMetrics.calculate_returns(values)
        all_returns.extend(returns)

    sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(np.array(all_returns))

    # Calculer le drawdown maximum moyen
    max_drawdowns = []
    for values in all_portfolio_values:
        max_dd, _, _, _ = PerformanceMetrics.calculate_max_drawdown(values)
        max_drawdowns.append(max_dd)

    avg_max_drawdown = np.mean(max_drawdowns)

    # Résultats
    results = {
        "final_value": avg_portfolio_values,
        "returns": avg_returns,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": avg_max_drawdown,
        "portfolio_history": (
            all_portfolio_values[0] if num_episodes == 1 else all_portfolio_values
        ),
        "actions": all_actions[0] if num_episodes == 1 else all_actions,
        "rewards": all_rewards[0] if num_episodes == 1 else all_rewards,
        "trades": all_trades if all_trades else None,
    }

    logger.info(
        f"Évaluation terminée. Rendement moyen: {avg_returns*100:.2f}%, Ratio de Sharpe: {sharpe_ratio:.4f}"
    )

    return results


def plot_portfolio_performance(
    df_portfolio, title="Portfolio Performance", show_trades=True
):
    """
    Trace la performance du portefeuille au fil du temps.

    Args:
        df_portfolio: DataFrame contenant la valeur du portefeuille, les prix et les actions
        title: Titre du graphique
        show_trades: Si True, affiche les points d'achat et de vente
    """
    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = "tab:blue"
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portefeuille ($)", color=color)
    ax1.plot(df_portfolio.index, df_portfolio["portfolio_value"], color=color, lw=2)
    ax1.tick_params(axis="y", labelcolor=color)

    # Formater l'axe des dates pour une meilleure lisibilité
    if isinstance(df_portfolio.index, pd.DatetimeIndex) and len(df_portfolio) > 20:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=max(1, len(df_portfolio) // 10))
        )
        plt.xticks(rotation=45)

    # Ajouter le prix sur un second axe Y
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Prix ($)", color=color)
    ax2.plot(df_portfolio.index, df_portfolio["close"], color=color, alpha=0.6)
    ax2.tick_params(axis="y", labelcolor=color)

    # Afficher les points d'achat et de vente si demandé
    if show_trades and "action" in df_portfolio.columns:
        buy_signals = df_portfolio[df_portfolio["action"] == 1]
        sell_signals = df_portfolio[df_portfolio["action"] == 2]

        if not buy_signals.empty:
            ax1.scatter(
                buy_signals.index,
                buy_signals["portfolio_value"],
                color="green",
                s=100,
                marker="^",
                alpha=0.7,
                label="Achat",
            )
        if not sell_signals.empty:
            ax1.scatter(
                sell_signals.index,
                sell_signals["portfolio_value"],
                color="red",
                s=100,
                marker="v",
                alpha=0.7,
                label="Vente",
            )
        ax1.legend()

    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Sauvegarder le graphique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"portfolio_performance_{timestamp}.png"
    output_path = os.path.join(VISUALIZATION_DIR, filename)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_performance_metrics(results, benchmark_results=None):
    """
    Trace les métriques de performance clés.

    Args:
        results (dict): Résultats de l'évaluation
        benchmark_results (dict): Résultats de la stratégie de référence
    """
    metrics = results.get("metrics", {})
    if not metrics:
        print("Aucune métrique disponible pour le tracé")
        return

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Métriques de performance", fontsize=16)

    # Rendement total
    ax[0, 0].bar(["Stratégie"], [metrics.get("total_return", 0) * 100], color="blue")
    if benchmark_results and "metrics" in benchmark_results:
        ax[0, 0].bar(
            ["Référence"],
            [benchmark_results["metrics"].get("total_return", 0) * 100],
            color="gray",
        )
    ax[0, 0].set_ylabel("Rendement total (%)")
    ax[0, 0].grid(axis="y", alpha=0.3)

    # Ratio de Sharpe
    ax[0, 1].bar(["Stratégie"], [metrics.get("sharpe_ratio", 0)], color="green")
    if benchmark_results and "metrics" in benchmark_results:
        ax[0, 1].bar(
            ["Référence"],
            [benchmark_results["metrics"].get("sharpe_ratio", 0)],
            color="gray",
        )
    ax[0, 1].set_ylabel("Ratio de Sharpe")
    ax[0, 1].grid(axis="y", alpha=0.3)

    # Drawdown maximum
    ax[1, 0].bar(["Stratégie"], [metrics.get("max_drawdown", 0) * 100], color="red")
    if benchmark_results and "metrics" in benchmark_results:
        ax[1, 0].bar(
            ["Référence"],
            [benchmark_results["metrics"].get("max_drawdown", 0) * 100],
            color="gray",
        )
    ax[1, 0].set_ylabel("Drawdown maximum (%)")
    ax[1, 0].grid(axis="y", alpha=0.3)

    # Volatilité
    ax[1, 1].bar(["Stratégie"], [metrics.get("volatility", 0) * 100], color="orange")
    if benchmark_results and "metrics" in benchmark_results:
        ax[1, 1].bar(
            ["Référence"],
            [benchmark_results["metrics"].get("volatility", 0) * 100],
            color="gray",
        )
    ax[1, 1].set_ylabel("Volatilité (%)")
    ax[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Sauvegarder le graphique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_metrics_{timestamp}.png"
    output_path = os.path.join(VISUALIZATION_DIR, filename)
    plt.savefig(output_path)
    plt.close()

    return output_path


def visualize_trading_signals(df_portfolio):
    """
    Visualise les signaux de trading sur un graphique des prix.

    Args:
        df_portfolio: DataFrame avec les prix, actions et valeurs de portefeuille
    """
    if "close" not in df_portfolio.columns or "action" not in df_portfolio.columns:
        print("Données manquantes pour visualiser les signaux de trading")
        return

    plt.figure(figsize=(14, 7))

    # Tracer le prix
    plt.plot(
        df_portfolio.index, df_portfolio["close"], color="blue", lw=2, label="Prix"
    )

    # Identifier les signaux d'achat et de vente
    buy_signals = df_portfolio[df_portfolio["action"] == 1]
    sell_signals = df_portfolio[df_portfolio["action"] == 2]

    # Ajouter les points d'achat et de vente
    if not buy_signals.empty:
        plt.scatter(
            buy_signals.index,
            buy_signals["close"],
            color="green",
            s=100,
            marker="^",
            alpha=0.7,
            label="Achat",
        )
    if not sell_signals.empty:
        plt.scatter(
            sell_signals.index,
            sell_signals["close"],
            color="red",
            s=100,
            marker="v",
            alpha=0.7,
            label="Vente",
        )

    # Formater l'axe des dates
    if isinstance(df_portfolio.index, pd.DatetimeIndex) and len(df_portfolio) > 20:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=max(1, len(df_portfolio) // 10))
        )
        plt.xticks(rotation=45)

    plt.title("Signaux de trading")
    plt.xlabel("Date")
    plt.ylabel("Prix ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Sauvegarder le graphique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trading_signals_{timestamp}.png"
    output_path = os.path.join(VISUALIZATION_DIR, filename)
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_training_history(history):
    """
    Trace l'historique d'entraînement d'un modèle.

    Args:
        history: L'historique d'entraînement
    """
    plt.figure(figsize=(12, 8))

    # Récompenses par épisode
    if "episode_rewards" in history:
        plt.subplot(2, 2, 1)
        plt.plot(history["episode_rewards"])
        plt.title("Récompenses par épisode")
        plt.xlabel("Épisode")
        plt.ylabel("Récompense totale")
        plt.grid(True, alpha=0.3)

    # Perte
    if "losses" in history:
        plt.subplot(2, 2, 2)
        plt.plot(history["losses"])
        plt.title("Perte par épisode")
        plt.xlabel("Épisode")
        plt.ylabel("Perte")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

    # Valeurs du portefeuille
    if "portfolio_values" in history:
        plt.subplot(2, 2, 3)
        plt.plot(history["portfolio_values"])
        plt.title("Valeur finale du portefeuille")
        plt.xlabel("Épisode")
        plt.ylabel("Valeur ($)")
        plt.grid(True, alpha=0.3)

    # Durée des épisodes
    if "episode_lengths" in history:
        plt.subplot(2, 2, 4)
        plt.plot(history["episode_lengths"])
        plt.title("Durée des épisodes")
        plt.xlabel("Épisode")
        plt.ylabel("Étapes")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Sauvegarder le graphique
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_history_{timestamp}.png"
    output_path = os.path.join(VISUALIZATION_DIR, filename)
    plt.savefig(output_path)
    plt.close()

    return output_path
