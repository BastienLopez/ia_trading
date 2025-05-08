"""
Environnement de trading avancé avec système d'ordres professionnels et gestion des risques.

Cet environnement étend l'environnement de trading de base avec:
- Un système d'ordres professionnels (market, limit, stop, etc.)
- Des contraintes de marché réalistes (slippage, délai d'exécution)
- Une gestion des risques avancée (VaR, allocation adaptative)
- Des limites dynamiques basées sur la volatilité et la liquidité
"""

import logging

import numpy as np

from ai_trading.orders.order_integration import (
    DynamicLimitOrderStrategy,
    OrderExecutionEnv,
)
from ai_trading.risk.advanced_risk_manager import AdvancedRiskManager
from ai_trading.rl.trading_environment import TradingEnvironment

logger = logging.getLogger(__name__)


class AdvancedTradingEnvironment(TradingEnvironment):
    """
    Environnement de trading avancé avec système d'ordres professionnels.

    Étend l'environnement de trading de base avec:
    - Un système d'ordres professionnels et des limites dynamiques
    - Une gestion plus précise du carnet d'ordres et de la liquidité
    - Des contraintes de marché plus réalistes (slippage, délai d'exécution)
    - Une meilleure simulation des frictions de marché
    """

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        transaction_fee=0.001,
        window_size=20,
        risk_management=True,
        risk_manager=None,
        use_advanced_orders=True,
        **kwargs,
    ):
        """
        Initialise l'environnement de trading avancé.

        Args:
            df (pd.DataFrame): DataFrame contenant les données historiques
            initial_balance (float): Solde initial
            transaction_fee (float): Frais de transaction (en %)
            window_size (int): Taille de la fenêtre d'observation
            risk_management (bool): Activer la gestion des risques
            risk_manager (AdvancedRiskManager, optional): Gestionnaire de risques externe
            use_advanced_orders (bool): Utiliser le système d'ordres professionnels
            **kwargs: Arguments supplémentaires pour l'environnement de base
        """
        # Initialiser avec la classe parent
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            transaction_fee=transaction_fee,
            window_size=window_size,
            risk_management=risk_management,
            **kwargs,
        )

        # Configuration du système d'ordres
        self.use_advanced_orders = use_advanced_orders

        # Gestionnaire de risques
        self.risk_manager = risk_manager
        if risk_management and risk_manager is None:
            # Créer un gestionnaire de risques par défaut
            risk_config = {
                "var_confidence_level": 0.95,
                "var_method": "historical",
                "max_var_limit": 0.05,
                "adaptive_capital_allocation": True,
                "kelly_fraction": 0.5,
                "risk_per_trade": 0.02,
                "max_drawdown_limit": 0.15,
                "stop_loss_atr_factor": 2.0,
                "trailing_stop_activation": 0.03,
            }
            self.risk_manager = AdvancedRiskManager(risk_config)

        # Configuration de l'environnement d'exécution des ordres
        order_config = {
            "use_limit_orders": kwargs.get("use_limit_orders", True),
            "use_stop_orders": kwargs.get("use_stop_orders", True),
            "use_oco_orders": kwargs.get("use_oco_orders", True),
            "use_trailing_stops": kwargs.get("use_trailing_stops", False),
            "dynamic_limit_orders": kwargs.get("dynamic_limit_orders", True),
            "limit_order_offset": kwargs.get("limit_order_offset", 0.002),
            "slippage_model": self.slippage_model,
            "max_slippage": kwargs.get("max_slippage", 0.01),
            "execution_delay": self.execution_delay,
        }

        if self.use_advanced_orders:
            self.order_execution = OrderExecutionEnv(
                risk_manager=self.risk_manager, config=order_config
            )

            # Stratégie pour les ordres limites dynamiques
            self.limit_strategy = DynamicLimitOrderStrategy(
                {
                    "base_offset": order_config["limit_order_offset"],
                    "volatility_factor": 5.0,
                    "max_offset": order_config["max_slippage"],
                }
            )

            logger.info(
                "Environnement de trading avancé initialisé avec système d'ordres professionnels"
            )
        else:
            logger.info(
                "Environnement de trading avancé initialisé sans système d'ordres professionnels"
            )

    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement pour un nouvel épisode.

        Args:
            seed (int, optional): Graine pour la génération de nombres aléatoires
            options (dict, optional): Options supplémentaires

        Returns:
            np.array: État initial
        """
        # Réinitialiser l'environnement de base
        observation = super().reset(seed=seed, options=options)

        # Réinitialiser le système d'ordres si utilisé
        if self.use_advanced_orders:
            # Fermer toutes les positions et annuler tous les ordres
            symbol = "CRYPTO"  # Symbole générique pour l'environnement à un seul actif
            self.order_execution.close_all_positions()
            self.order_execution.order_manager.cancel_all_orders()

            # Réinitialiser l'état du système d'ordres
            self.order_execution.position_status = {}
            self.order_execution.execution_history = []

        return observation

    def step(self, action):
        """
        Exécute une action dans l'environnement.

        Args:
            action: Action à exécuter (format dépendant de l'espace d'action)

        Returns:
            tuple: (état suivant, récompense, fin de l'épisode, tronqué, informations)
        """
        if not self.use_advanced_orders:
            # Utiliser l'implémentation de base si le système d'ordres avancé n'est pas activé
            return super().step(action)

        # Convertir l'action au format continu si nécessaire
        if self.action_type == "discrete":
            action_value = self._discrete_to_continuous(action)
        else:
            action_value = float(action)

        # Obtenir les informations de marché actuelles
        current_step = self.current_step
        current_price = self.df.iloc[current_step]["close"]

        # Créer les données de marché pour le symbole
        symbol = "CRYPTO"  # Symbole générique pour l'environnement à un seul actif
        market_data = {
            symbol: {
                "timestamp": (
                    self.df.index[current_step].isoformat()
                    if hasattr(self.df.index, "to_list")
                    else str(current_step)
                ),
                "open": self.df.iloc[current_step].get("open", current_price),
                "high": self.df.iloc[current_step].get("high", current_price),
                "low": self.df.iloc[current_step].get("low", current_price),
                "close": current_price,
                "price": current_price,
                "volume": self.df.iloc[current_step].get("volume", 0),
                "volatility": self.df.iloc[current_step].get("volatility", 0.02),
                "atr": self.df.iloc[current_step].get("atr", current_price * 0.02),
                "rsi": self.df.iloc[current_step].get("rsi", 50),
                "spread": self.df.iloc[current_step].get("spread", 0.002),
                "avg_volume": self.df.iloc[current_step].get(
                    "avg_volume",
                    (
                        self.df["volume"].rolling(10).mean().iloc[current_step]
                        if "volume" in self.df.columns
                        else 0
                    ),
                ),
                "portfolio_value": self.get_portfolio_value(),
            }
        }

        # Traiter l'action avec le système d'ordres
        execution_result = self.order_execution.process_action(
            symbol=symbol, action_value=action_value, data=market_data[symbol]
        )

        # Avancer d'un pas de temps
        self.current_step += 1

        # Obtenir les données du prochain pas de temps
        if self.current_step < len(self.df):
            next_price = self.df.iloc[self.current_step]["close"]

            # Mettre à jour les données de marché
            next_market_data = {
                symbol: {
                    "timestamp": (
                        self.df.index[self.current_step].isoformat()
                        if hasattr(self.df.index, "to_list")
                        else str(self.current_step)
                    ),
                    "open": self.df.iloc[self.current_step].get("open", next_price),
                    "high": self.df.iloc[self.current_step].get("high", next_price),
                    "low": self.df.iloc[self.current_step].get("low", next_price),
                    "close": next_price,
                    "price": next_price,
                    "volume": self.df.iloc[self.current_step].get("volume", 0),
                    "volatility": self.df.iloc[self.current_step].get(
                        "volatility", 0.02
                    ),
                    "atr": self.df.iloc[self.current_step].get(
                        "atr", next_price * 0.02
                    ),
                    "rsi": self.df.iloc[self.current_step].get("rsi", 50),
                    "spread": self.df.iloc[self.current_step].get("spread", 0.002),
                    "avg_volume": self.df.iloc[self.current_step].get(
                        "avg_volume",
                        (
                            self.df["volume"].rolling(10).mean().iloc[self.current_step]
                            if "volume" in self.df.columns
                            else 0
                        ),
                    ),
                }
            }

            # Traiter les ordres avec les nouvelles données de marché
            executions = self.order_execution.process_market_update(next_market_data)

        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.df) - 1

        # Récupérer la position et mettre à jour l'état de l'environnement
        position = self.order_execution.get_position(symbol)
        self.crypto_held = position["quantity"]

        # Mettre à jour la valeur du portefeuille
        portfolio_value = self.get_portfolio_value()

        # Calculer le rendement du portefeuille
        prev_portfolio_value = (
            self.portfolio_value_history[-1]
            if self.portfolio_value_history
            else self.initial_balance
        )
        portfolio_return = (portfolio_value / prev_portfolio_value) - 1

        # Ajouter la valeur du portefeuille à l'historique
        self.portfolio_value_history.append(portfolio_value)

        # Calculer la récompense
        reward = self._calculate_reward(portfolio_return)

        # Obtenir l'observation suivante
        next_observation = self._get_observation()

        # Informations supplémentaires
        info = {
            "portfolio_value": portfolio_value,
            "portfolio_return": portfolio_return,
            "current_price": (
                next_price if self.current_step < len(self.df) else current_price
            ),
            "crypto_held": self.crypto_held,
            "executions": len(self.order_execution.execution_history),
            "active_orders": len(self.order_execution.get_orders()),
        }

        return next_observation, reward, done, False, info

    def get_portfolio_value(self):
        """
        Calcule la valeur actuelle du portefeuille.

        Returns:
            float: Valeur du portefeuille
        """
        if self.use_advanced_orders:
            # Calculer la valeur à partir du système d'ordres
            crypto_value = 0
            symbol = "CRYPTO"  # Symbole générique

            if self.current_step < len(self.df):
                current_price = self.df.iloc[self.current_step]["close"]
                position = self.order_execution.get_position(symbol)
                crypto_value = position["quantity"] * current_price

                # Récupérer la valeur cash du gestionnaire d'ordres
                # Dans un environnement réel, cela viendrait du gestionnaire de compte
                balance = self.initial_balance
                balance -= sum(
                    e["price"] * e["quantity"]
                    for e in self.order_execution.execution_history
                    if e["side"] == "buy"
                )
                balance += sum(
                    e["price"] * e["quantity"]
                    for e in self.order_execution.execution_history
                    if e["side"] == "sell"
                )

                return balance + crypto_value

        # Utiliser le calcul de base si le système d'ordres n'est pas activé
        return super().get_portfolio_value()

    def close_position(self):
        """
        Ferme la position actuelle.

        Returns:
            bool: True si la position a été fermée, False sinon
        """
        if self.use_advanced_orders:
            # Fermer la position avec le système d'ordres
            symbol = "CRYPTO"  # Symbole générique

            if self.current_step < len(self.df):
                current_price = self.df.iloc[self.current_step]["close"]

                # Créer les données de marché pour le symbole
                market_data = {symbol: {"price": current_price, "close": current_price}}

                # Fermer toutes les positions
                executions = self.order_execution.close_all_positions(
                    market_data=market_data
                )
                return len(executions) > 0

        # Utiliser l'implémentation de base si le système d'ordres n'est pas activé
        if hasattr(self, "_close_position"):
            price = self.df.iloc[self.current_step]["close"]
            self._close_position(price)
            return True

        return False

    def _allocate_capital(self, action_value):
        """
        Alloue le capital en fonction de la valeur de l'action et du gestionnaire de risques.

        Args:
            action_value (float): Valeur de l'action (-1 à 1)

        Returns:
            float: Fraction du capital à allouer
        """
        # Utiliser le gestionnaire de risques avancé si disponible
        if (
            self.risk_management
            and self.risk_manager
            and isinstance(self.risk_manager, AdvancedRiskManager)
        ):
            # Créer un dataframe avec les données historiques récentes
            historical_window = min(100, self.current_step)
            if historical_window > 0:
                hist_data = self.df.iloc[
                    self.current_step - historical_window : self.current_step + 1
                ]

                # Déterminer le type de position
                position_type = "long" if action_value > 0 else "short"

                # Récupérer l'historique des valeurs du portefeuille
                portfolio_values = self.portfolio_value_history

                # Calculer l'allocation adaptative
                allocation = self.risk_manager.allocation_with_risk_limits(
                    hist_data, position_type, portfolio_values
                )

                # Ajuster l'allocation en fonction de la valeur de l'action
                allocation *= abs(action_value)

                return allocation

        # Utiliser l'allocation de base si le gestionnaire de risques n'est pas disponible
        direction = np.sign(action_value)
        allocation = abs(action_value) * self.max_position_size

        return allocation * direction

    def render(self, mode="human"):
        """
        Affiche l'état actuel de l'environnement.

        Args:
            mode (str): Mode d'affichage
        """
        # Utiliser l'implémentation de base
        super().render(mode=mode)

        # Ajouter des informations sur les ordres si le système avancé est activé
        if self.use_advanced_orders:
            symbol = "CRYPTO"  # Symbole générique

            print("\n--- Informations sur les ordres ---")
            print(
                f"Position actuelle: {self.order_execution.get_position(symbol)['quantity']}"
            )

            active_orders = self.order_execution.get_orders()
            print(f"Ordres actifs: {len(active_orders)}")

            for i, order in enumerate(
                active_orders[:5]
            ):  # Limiter à 5 ordres pour l'affichage
                print(
                    f"  Ordre {i+1}: {order.order_type.value} {order.side.value} "
                    f"{order.get_remaining_quantity()} @ "
                    f"{order.price if order.price else 'market'}"
                    f"{f' (stop: {order.stop_price})' if order.stop_price else ''}"
                )

            if len(active_orders) > 5:
                print(f"  ... et {len(active_orders) - 5} autres ordres")

            print(f"Exécutions: {len(self.order_execution.execution_history)}")
            print(f"Valeur du portefeuille: ${self.get_portfolio_value():.2f}")


# Fonction utilitaire pour créer l'environnement
def create_advanced_trading_env(df, config=None):
    """
    Crée un environnement de trading avancé avec les configurations spécifiées.

    Args:
        df (pd.DataFrame): DataFrame contenant les données historiques
        config (dict, optional): Configuration de l'environnement

    Returns:
        AdvancedTradingEnvironment: Instance de l'environnement
    """
    config = config or {}

    # Configuration du gestionnaire de risques
    risk_config = config.get(
        "risk_config",
        {
            "var_confidence_level": 0.95,
            "var_method": "historical",
            "max_var_limit": 0.05,
            "adaptive_capital_allocation": True,
            "kelly_fraction": 0.5,
            "risk_per_trade": 0.02,
            "max_drawdown_limit": 0.15,
        },
    )

    # Créer le gestionnaire de risques
    risk_manager = AdvancedRiskManager(risk_config)

    # Créer l'environnement
    env = AdvancedTradingEnvironment(
        df=df,
        initial_balance=config.get("initial_balance", 10000.0),
        transaction_fee=config.get("transaction_fee", 0.001),
        window_size=config.get("window_size", 20),
        risk_management=config.get("risk_management", True),
        risk_manager=risk_manager,
        use_advanced_orders=config.get("use_advanced_orders", True),
        include_position=config.get("include_position", True),
        include_balance=config.get("include_balance", True),
        include_technical_indicators=config.get("include_technical_indicators", True),
        normalize_observation=config.get("normalize_observation", True),
        reward_function=config.get("reward_function", "sharpe"),
        risk_aversion=config.get("risk_aversion", 0.1),
        transaction_penalty=config.get("transaction_penalty", 0.001),
        lookback_window=config.get("lookback_window", 20),
        action_type=config.get("action_type", "continuous"),
        n_discrete_actions=config.get("n_discrete_actions", 5),
        slippage_model=config.get("slippage_model", "dynamic"),
        slippage_value=config.get("slippage_value", 0.001),
        execution_delay=config.get("execution_delay", 1),
        use_limit_orders=config.get("use_limit_orders", True),
        use_stop_orders=config.get("use_stop_orders", True),
        use_oco_orders=config.get("use_oco_orders", True),
        dynamic_limit_orders=config.get("dynamic_limit_orders", True),
        limit_order_offset=config.get("limit_order_offset", 0.002),
        max_slippage=config.get("max_slippage", 0.01),
    )

    return env
