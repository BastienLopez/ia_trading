"""
Exemple d'intégration du gestionnaire de risques avancé avec l'environnement de trading existant.
"""

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ajouter le répertoire parent au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ai_trading.config import VISUALIZATION_DIR
from ai_trading.risk.advanced_risk_manager import AdvancedRiskManager
from ai_trading.rl.environments.trading_environment import TradingEnvironment

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dossier pour les visualisations
VAR_EXAMPLES_DIR = VISUALIZATION_DIR / "var_examples"
VAR_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


class VaREnhancedTradingEnvironment(TradingEnvironment):
    """
    Environnement de trading amélioré avec la Value-at-Risk et l'allocation adaptative.
    """

    def __init__(self, df, risk_config=None, **kwargs):
        """
        Initialise l'environnement de trading avec VaR.

        Args:
            df (pd.DataFrame): DataFrame contenant les données du marché
            risk_config (dict, optional): Configuration du gestionnaire de risques
            **kwargs: Arguments supplémentaires pour TradingEnvironment
        """
        # Initialiser l'environnement de trading parent
        super().__init__(df, **kwargs)

        # Initialiser le gestionnaire de risques avancé
        self.risk_config = risk_config or {}
        self.advanced_risk_manager = AdvancedRiskManager(config=self.risk_config)

        # Historique du portefeuille pour le suivi des drawdowns
        self.portfolio_history = [self.initial_balance]

        logger.info("Environnement de trading avec VaR initialisé")

    def reset(self, seed=None):
        """Réinitialise l'environnement."""
        # Réinitialiser l'environnement parent
        state, info = super().reset(seed=seed)

        # Réinitialiser le gestionnaire de risques avancé
        self.advanced_risk_manager = AdvancedRiskManager(config=self.risk_config)

        # Réinitialiser l'historique du portefeuille
        self.portfolio_history = [self.initial_balance]

        return state, info

    def _update_position(self, action_value):
        """
        Met à jour la position en fonction de l'action, avec contraintes de risque avancées.
        """
        # Obtenir le slice des données jusqu'au point actuel
        data_slice = self.df.iloc[: self.current_step + 1].copy()

        # Déterminer le type de position basé sur l'action
        position_type = (
            "long" if action_value > 0 else "short" if action_value < 0 else "neutral"
        )

        # Calculer l'allocation optimale avec contraintes de risque
        risk_adjusted_allocation = (
            self.advanced_risk_manager.allocation_with_risk_limits(
                data_slice,
                position_type=position_type,
                portfolio_values=self.portfolio_history,
            )
        )

        # Si l'environnement suggère un arrêt du trading pour risque excessif
        if risk_adjusted_allocation == 0.0 and action_value != 0:
            logger.warning("Trading arrêté en raison de risques excessifs")
            return 0  # Pas de changement de position

        # Ajuster l'action en fonction de l'allocation calculée
        if position_type == "long":
            adjusted_action = min(action_value, risk_adjusted_allocation)
        elif position_type == "short":
            adjusted_action = max(action_value, risk_adjusted_allocation)
        else:
            adjusted_action = 0

        # Utiliser la méthode du parent avec l'action ajustée
        position_diff = super()._update_position(adjusted_action)

        # Mettre à jour l'historique du portefeuille
        portfolio_value = self.balance + self.current_position * self.current_price
        self.portfolio_history.append(portfolio_value)

        return position_diff

    def _calculate_reward(self, position_diff):
        """
        Calcule la récompense en tenant compte des métriques de risque.
        """
        # Calculer la récompense de base
        base_reward = super()._calculate_reward(position_diff)

        # Obtenir le drawdown actuel
        if len(self.portfolio_history) >= 2:
            current_drawdown = self.advanced_risk_manager.calculate_maximum_drawdown(
                self.portfolio_history[-30:]
                if len(self.portfolio_history) > 30
                else self.portfolio_history
            )

            # Pénaliser en fonction du drawdown (plus le drawdown est grand, plus la pénalité est forte)
            drawdown_penalty = (
                current_drawdown * 2
            )  # Facteur de 2 pour renforcer l'impact

            # Diminuer la récompense en fonction du drawdown
            risk_adjusted_reward = base_reward * (1 - drawdown_penalty)
        else:
            risk_adjusted_reward = base_reward

        return risk_adjusted_reward

    def render(self):
        """
        Rendu de l'environnement avec des informations sur le risque.
        """
        # Appeler le rendu de base si disponible
        if hasattr(super(), "render"):
            super().render()

        # Afficher les informations de risque
        if len(self.advanced_risk_manager.var_history) > 0:
            last_var = self.advanced_risk_manager.var_history[-1]["var"]
            logger.info(f"VaR actuelle: {last_var:.2%}")

        if len(self.portfolio_history) > 1:
            current_drawdown = self.advanced_risk_manager.calculate_maximum_drawdown(
                self.portfolio_history
            )
            logger.info(f"Drawdown actuel: {current_drawdown:.2%}")

        # Afficher l'allocation actuelle
        if len(self.advanced_risk_manager.allocation_history) > 0:
            last_allocation = self.advanced_risk_manager.allocation_history[-1][
                "allocation"
            ]
            logger.info(f"Allocation actuelle: {last_allocation:.2%}")


def run_demonstration():
    """
    Démonstration de l'environnement de trading avec VaR.
    """
    logger.info("Démarrage de la démonstration d'intégration VaR")

    # 1. Charger ou générer des données
    # Pour cet exemple, nous générons des données synthétiques
    from ai_trading.examples.advanced_risk_management_example import (
        generate_synthetic_data,
    )

    market_data = generate_synthetic_data(
        days=365,
        volatility=0.02,
        drift=0.0001,
        crash_period=(180, 200),  # Crash entre les jours 180 et 200
        crash_severity=0.25,  # Chute de 25%
    )

    # 2. Configurer le gestionnaire de risques avancé
    risk_config = {
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

    # 3. Créer l'environnement de trading amélioré avec VaR
    env = VaREnhancedTradingEnvironment(
        df=market_data,
        risk_config=risk_config,
        initial_balance=10000.0,
        max_position=1.0,
        execution_delay=1,
        slippage_model="fixed",
        slippage_value=0.001,
        position_penalty=0.0001,
        position_change_penalty=0.0001,
        reward_scaling=True,
        state_normalization=True,
        use_rsi=True,
        use_macd=True,
        use_bollinger=True,
    )

    # 4. Simuler quelques actions pour démontrer le fonctionnement
    logger.info("Simulation de trading avec contraintes VaR")

    state, _ = env.reset()
    done = False

    # Historiques pour les graphiques
    prices = []
    positions = []
    portfolio_values = []
    var_values = []

    while not done:
        # Générer une action simple pour la démonstration
        # Stratégie naïve: acheter quand le prix monte, vendre quand il baisse
        if env.current_step > 0:
            price_diff = (
                env.df["close"].iloc[env.current_step]
                - env.df["close"].iloc[env.current_step - 1]
            )
            action = (
                np.array([0.5])
                if price_diff > 0
                else np.array([-0.5]) if price_diff < 0 else np.array([0])
            )
        else:
            action = np.array([0])  # Pas d'action au premier pas

        # Exécuter l'action dans l'environnement
        next_state, reward, done, truncated, info = env.step(action)

        # Collecter les données pour les graphiques
        prices.append(env.current_price)
        positions.append(env.current_position)
        portfolio_values.append(env.portfolio_history[-1])

        # Collecter les valeurs VaR si disponibles
        if len(env.advanced_risk_manager.var_history) > 0:
            var_values.append(env.advanced_risk_manager.var_history[-1]["var"])
        else:
            var_values.append(0)

        # Afficher les informations périodiquement
        if env.current_step % 30 == 0:
            logger.info(
                f"Étape {env.current_step}: Prix={env.current_price:.2f}, "
                f"Position={env.current_position:.2f}, Reward={reward:.4f}"
            )
            env.render()

        state = next_state

    # 5. Générer des visualisations
    logger.info("Génération des visualisations de la démonstration")

    # Graphique des prix et positions
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(prices, label="Prix")
    plt.title("Évolution du prix")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(positions, label="Position", color="green")
    plt.title("Évolution de la position")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(portfolio_values, label="Portefeuille", color="blue")
    plt.plot(var_values, label="VaR (95%)", color="red", linestyle="--")
    plt.title("Évolution du portefeuille et VaR")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VAR_EXAMPLES_DIR / "var_trading_demonstration.png")
    plt.close()

    # 6. Calculer et afficher les statistiques finales
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value) - 1

    max_drawdown = env.advanced_risk_manager.calculate_maximum_drawdown(
        portfolio_values
    )

    # Calculer le ratio de Sharpe approximatif
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

    logger.info("\n--- Statistiques finales ---")
    logger.info(f"Capital initial: {initial_value:.2f}")
    logger.info(f"Capital final: {final_value:.2f}")
    logger.info(f"Rendement total: {total_return:.2%}")
    logger.info(f"Drawdown maximum: {max_drawdown:.2%}")
    logger.info(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
    logger.info(f"VaR moyenne: {np.mean(var_values):.2%}")
    logger.info(f"VaR maximale: {np.max(var_values):.2%}")

    logger.info("Démonstration terminée")


if __name__ == "__main__":
    run_demonstration()
