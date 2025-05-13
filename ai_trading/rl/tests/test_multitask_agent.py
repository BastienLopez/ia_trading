import logging
import sys
from pathlib import Path

# Configurer matplotlib pour utiliser un backend non-interactif (Agg)
import matplotlib

matplotlib.use("Agg")  # IMPORTANT: Doit être défini avant d'importer pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Ajuster les chemins d'importation
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import des modules
from ai_trading.rl.agents.multitask_agent import MultitaskTradingAgent
from ai_trading.rl.tests.test_multitask_learning import generate_synthetic_data

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = (
    BASE_DIR / "ai_trading" / "info_retour" / "visualisations" / "multitask_agent"
)
DATA_DIR = BASE_DIR / "ai_trading" / "info_retour" / "data" / "processed"

# Assurer que le répertoire d'output existe
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleTradingEnvironment:
    """
    Environnement de trading simplifié pour tester l'agent multi-tâches.
    """

    def __init__(self, data, window_size=50, transaction_cost=0.001):
        """
        Initialise l'environnement.

        Args:
            data: DataFrame avec les données de marché
            window_size: Taille de la fenêtre d'observation
            transaction_cost: Coût de transaction (proportionnel)
        """
        self.data = data
        self.window_size = window_size
        self.transaction_cost = transaction_cost

        # Extraire les prix de clôture
        self.close_prices = data["close"].values

        # Définir les dimensions d'état et d'action
        self.observation_space = SimpleSpace((window_size,))
        self.action_space = SimpleSpace((3,))  # [position_size, stop_loss, take_profit]

        # État de l'environnement
        self.current_step = window_size
        self.current_position = 0.0  # Position actuelle (-1 à 1)
        self.cash = 1000.0
        self.portfolio_value = self.cash
        self.portfolio_history = []

        # Paramètres de gestion des risques
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.entry_price = 0.0

    def reset(self):
        """
        Réinitialise l'environnement au début.

        Returns:
            État initial, info
        """
        self.current_step = self.window_size
        self.current_position = 0.0
        self.cash = 1000.0
        self.portfolio_value = self.cash
        self.portfolio_history = [self.portfolio_value]
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.entry_price = 0.0

        return self._get_observation(), {}

    def _get_observation(self):
        """
        Retourne l'observation actuelle (fenêtre des prix).

        Returns:
            Fenêtre de prix normalisée
        """
        # Extraire la fenêtre des prix
        window_start = self.current_step - self.window_size
        window_end = self.current_step
        window = self.close_prices[window_start:window_end]

        # Normaliser
        window_mean = np.mean(window)
        window_std = np.std(window) + 1e-8
        normalized_window = (window - window_mean) / window_std

        return normalized_window

    def step(self, action):
        """
        Exécute une action dans l'environnement.

        Args:
            action: Action à exécuter [position_size, stop_loss, take_profit]

        Returns:
            Tuple (nouvel état, récompense, terminé, info)
        """
        # Extraire les composantes de l'action
        new_position = action[0]  # Entre -1 (short max) et 1 (long max)
        stop_loss = abs(action[1]) * 0.1  # 0-10% de stop loss
        take_profit = abs(action[2]) * 0.2  # 0-20% de take profit

        # Prix actuel
        current_price = self.close_prices[self.current_step]

        # Calculer le P&L si nous avons une position
        pnl = 0.0
        if self.current_position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            pnl = price_change * self.current_position

            # Vérifier si stop loss ou take profit a été déclenché
            triggered = False
            if self.current_position > 0:  # Position longue
                if price_change <= -self.stop_loss:  # Stop loss
                    pnl = -self.stop_loss * self.current_position
                    triggered = True
                elif price_change >= self.take_profit:  # Take profit
                    pnl = self.take_profit * self.current_position
                    triggered = True
            else:  # Position courte
                if price_change >= self.stop_loss:  # Stop loss
                    pnl = -self.stop_loss * abs(self.current_position)
                    triggered = True
                elif price_change <= -self.take_profit:  # Take profit
                    pnl = self.take_profit * abs(self.current_position)
                    triggered = True

            # Si déclenché, fermer la position
            if triggered:
                # Appliquer le PnL au portefeuille
                self.portfolio_value += self.portfolio_value * pnl
                # Réinitialiser la position
                self.current_position = 0.0

        # Calculer le coût de transaction pour le changement de position
        position_change = abs(new_position - self.current_position)
        transaction_cost = (
            position_change * self.transaction_cost * self.portfolio_value
        )

        # Mettre à jour la position et appliquer les coûts
        self.current_position = new_position
        self.portfolio_value -= transaction_cost

        # Si nouvelle position, enregistrer le prix d'entrée et les paramètres de risque
        if new_position != 0 and self.current_position != 0:
            self.entry_price = current_price
            self.stop_loss = stop_loss
            self.take_profit = take_profit

        # Passer à l'étape suivante
        self.current_step += 1
        done = self.current_step >= len(self.close_prices) - 1

        # Enregistrer la valeur du portefeuille
        self.portfolio_history.append(self.portfolio_value)

        # Calculer la récompense (rendement du portefeuille à cette étape)
        if len(self.portfolio_history) > 1:
            reward = (self.portfolio_history[-1] / self.portfolio_history[-2]) - 1.0
        else:
            reward = 0.0

        # Informations supplémentaires
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.current_position,
            "transaction_cost": transaction_cost,
        }

        return self._get_observation(), reward, done, info

    def calculate_metrics(self):
        """
        Calcule les métriques de performance.

        Returns:
            Dictionnaire des métriques
        """
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Rendement total
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0

        # Rendement annualisé (supposant des données quotidiennes)
        n_days = len(portfolio_values)
        annualized_return = (
            (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        )

        # Volatilité
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # Ratio de Sharpe (supposant un taux sans risque de 0)
        sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0

        # Drawdown maximum
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns / running_max) - 1
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

    def render(self, title="Performance de l'agent multi-tâches"):
        """
        Visualise la performance de l'agent.

        Args:
            title: Titre du graphique
        """
        plt.figure(figsize=(14, 10))

        # Valeur du portefeuille
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_history)
        plt.title("Valeur du portefeuille")
        plt.xlabel("Pas de temps")
        plt.ylabel("Valeur")
        plt.grid(True)

        # Métriques
        metrics = self.calculate_metrics()
        metrics_text = "\n".join(
            [
                f"Rendement total: {metrics['total_return']:.2%}",
                f"Rendement annualisé: {metrics['annualized_return']:.2%}",
                f"Volatilité: {metrics['volatility']:.2%}",
                f"Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}",
                f"Drawdown maximum: {metrics['max_drawdown']:.2%}",
            ]
        )

        plt.figtext(
            0.02,
            0.02,
            metrics_text,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Rendements cumulatifs vs benchmark
        plt.subplot(2, 1, 2)

        # Rendements cumulatifs du portefeuille
        portfolio_values = np.array(self.portfolio_history)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        portfolio_cumulative = np.cumprod(1 + portfolio_returns)

        # Rendements cumulatifs du benchmark (buy & hold)
        benchmark_values = self.close_prices[
            self.window_size : self.window_size + len(portfolio_values)
        ]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        benchmark_cumulative = np.cumprod(1 + benchmark_returns)

        # Tracer les rendements cumulatifs
        plt.plot(portfolio_cumulative, label="Agent multi-tâches")
        plt.plot(benchmark_cumulative, label="Buy & Hold", linestyle="--")
        plt.title("Rendements cumulatifs vs Benchmark")
        plt.xlabel("Pas de temps")
        plt.ylabel("Rendement cumulatif")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "multitask_agent_performance.png")
        plt.close()


class SimpleSpace:
    """Espace simple pour remplacer gym.spaces."""

    def __init__(self, shape):
        self.shape = shape


def train_multitask_agent(env, agent, n_episodes=100, save_path=None):
    """
    Entraîne l'agent multi-tâches.

    Args:
        env: Environnement de trading
        agent: Agent multi-tâches
        n_episodes: Nombre d'épisodes d'entraînement
        save_path: Chemin pour sauvegarder le modèle

    Returns:
        Agent entraîné et historique des récompenses
    """
    reward_history = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        agent.reset_state_buffer()
        episode_rewards = []
        done = False

        # Batch pour l'entraînement
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        pbar = tqdm(
            desc=f"Épisode {episode+1}/{n_episodes}",
            total=len(env.close_prices) - env.window_size,
        )

        while not done:
            # Sélectionner une action
            action = agent.act(state)

            # Exécuter l'action
            next_state, reward, done, info = env.step(action)

            # Enregistrer pour l'entraînement
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            # Passer à l'état suivant
            state = next_state
            episode_rewards.append(reward)

            # Entraîner l'agent tous les N pas
            if len(states) >= 32:
                losses = agent.update(states, actions, rewards, next_states, dones)
                states, actions, rewards, next_states, dones = [], [], [], [], []

            pbar.update(1)

        pbar.close()

        # Entraîner avec les données restantes
        if states:
            agent.update(states, actions, rewards, next_states, dones)

        # Enregistrer et afficher les métriques
        total_reward = sum(episode_rewards)
        reward_history.append(total_reward)

        # Calculer les métriques
        metrics = env.calculate_metrics()

        logger.info(
            f"Épisode {episode+1}: Rendement total = {metrics['total_return']:.2%}, Sharpe = {metrics['sharpe_ratio']:.2f}"
        )

        # Réduire le taux d'exploration
        agent.exploration_rate *= 0.95

        # Sauvegarder le modèle périodiquement
        if save_path and (episode + 1) % 10 == 0:
            agent.save_model(save_path)

    # Sauvegarder le modèle final
    if save_path:
        agent.save_model(save_path)

    return agent, reward_history


def evaluate_agent(env, agent, visualize=True):
    """
    Évalue l'agent entraîné.

    Args:
        env: Environnement de trading
        agent: Agent entraîné
        visualize: Si True, visualise les résultats

    Returns:
        Métriques de performance
    """
    state, _ = env.reset()
    agent.reset_state_buffer()
    agent.exploration_rate = 0.0  # Désactiver l'exploration

    done = False
    while not done:
        action = agent.act(state, explore=False)
        state, reward, done, info = env.step(action)

    metrics = env.calculate_metrics()

    logger.info(
        f"Évaluation: Rendement total = {metrics['total_return']:.2%}, Sharpe = {metrics['sharpe_ratio']:.2f}"
    )

    if visualize:
        env.render(title="Performance de l'agent multi-tâches (Évaluation)")

    return metrics


def test_multitask_agent():
    """Fonction de test pour l'agent multi-tâches."""
    # Définir le périphérique
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du périphérique: {device}")

    # Générer des données synthétiques
    logger.info("Génération de données synthétiques...")
    data = generate_synthetic_data(n_samples=5000, num_assets=5)
    logger.info(f"Données générées: {data.shape}")

    # Créer l'environnement
    logger.info("Création de l'environnement de trading...")
    window_size = 50
    env = SimpleTradingEnvironment(data, window_size=window_size)

    # Déterminer les dimensions
    state_dim = window_size
    action_dim = 3  # [position_size, stop_loss, take_profit]

    # Créer l'agent
    logger.info("Création de l'agent multi-tâches...")
    agent = MultitaskTradingAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_assets=5,
        d_model=128,
        n_heads=4,
        num_layers=2,
        max_seq_len=window_size,
        device=device,
        risk_aversion=0.5,
        exploration_rate=0.3,
        lr=0.0005,
    )

    # Entraîner l'agent
    logger.info("Début de l'entraînement...")
    save_path = str(RESULTS_DIR / "multitask_agent_model.pt")
    agent, reward_history = train_multitask_agent(
        env=env,
        agent=agent,
        n_episodes=5,
        save_path=save_path,
    )

    # Visualiser l'historique des récompenses
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history)
    plt.title("Historique des récompenses")
    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale")
    plt.grid(True)
    plt.savefig(RESULTS_DIR / "multitask_agent_rewards.png")
    plt.close()

    # Évaluer l'agent
    logger.info("Évaluation de l'agent...")
    eval_metrics = evaluate_agent(env, agent, visualize=True)

    logger.info("Test de l'agent multi-tâches terminé avec succès!")

    return agent, env, reward_history, eval_metrics


if __name__ == "__main__":
    test_multitask_agent()
