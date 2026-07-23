import datetime
import logging

import gymnasium as gym
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces

# Configuration du logger
logger = logging.getLogger("TradingEnvironment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Utiliser directement VISUALIZATION_DIR de config.py
from ai_trading.config import VISUALIZATION_DIR

# Ajouter l'import
from ai_trading.rl.adaptive_normalization import AdaptiveNormalizer

# Ajouter l'import
from ai_trading.rl.risk_manager import RiskManager
from ai_trading.rl.technical_indicators import TechnicalIndicators
from ai_trading.rl.market_regime import add_causal_regime_features
from ai_trading.rl.indicator_fusion import add_causal_indicator_fusion

# Ajouter l'import de la classe TechnicalIndicators


VISUALIZATION_DIR = VISUALIZATION_DIR / "trading_env"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


class TradingEnvironment(gym.Env):
    """
    Environnement de trading pour l'apprentissage par renforcement.

    Cet environnement simule un marché de trading avec des données réelles,
    où un agent peut acheter, vendre ou conserver des actifs.
    L'objectif est de maximiser la valeur du portefeuille.
    """

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        transaction_fee=0.001,
        window_size=20,
        include_position=True,
        include_balance=True,
        include_technical_indicators=True,
        risk_management=True,
        normalize_observation=True,
        reward_function="simple",  # Options: simple, sharpe, transaction_penalty, drawdown, excess_return
        risk_aversion=0.1,  # Paramètre pour le coefficient de risque dans la fonction de récompense
        transaction_penalty=0.001,  # Pénalité fixe pour chaque transaction
        invalid_action_penalty=0.0,
        benchmark_exposure=None,  # Exposition du benchmark utilisé par excess_return
        lookback_window=20,  # Fenêtre pour calculer le ratio de Sharpe
        action_type="discrete",  # Type d'action: "discrete" ou "continuous"
        n_discrete_actions=5,  # Nombre d'actions discrètes par catégorie (achat/vente)
        slippage_model="constant",  # Options: "constant", "proportional", "dynamic"
        slippage_value=0.001,  # Valeur de slippage pour le modèle constant
        execution_delay=0,  # Délai d'exécution en pas de temps
        allocation_strategy="equal",  # Stratégie d'allocation: "equal", "proportional", "risk_parity"
        **kwargs,
    ):
        """
        Initialise l'environnement de trading.

        Args:
            df (pd.DataFrame): DataFrame contenant les données du marché
            initial_balance (float): Solde initial du portefeuille
            transaction_fee (float): Frais de transaction en pourcentage
            window_size (int): Taille de la fenêtre d'observation
            include_position (bool): Inclure la position actuelle dans l'observation
            include_balance (bool): Inclure le solde dans l'observation
            include_technical_indicators (bool): Inclure les indicateurs techniques dans l'observation
            risk_management (bool): Activer la gestion des risques
            normalize_observation (bool): Normaliser les observations
            reward_function (str): Fonction de récompense à utiliser
            risk_aversion (float): Coefficient de risque pour la fonction de récompense
            transaction_penalty (float): Pénalité fixe pour chaque transaction
            lookback_window (int): Fenêtre pour calculer le ratio de Sharpe
            action_type (str): Type d'action ("discrete" ou "continuous")
            n_discrete_actions (int): Nombre d'actions discrètes par catégorie
            slippage_model (str): Modèle de slippage
            slippage_value (float): Valeur de slippage
            execution_delay (int): Délai d'exécution en pas de temps
            allocation_strategy (str): Stratégie d'allocation des actifs
        """
        super(TradingEnvironment, self).__init__()

        # Valider les paramètres
        if df is None or "close" not in df.columns or len(df) <= window_size:
            raise ValueError(
                f"Le DataFrame doit contenir une colonne close et plus de {window_size} points"
            )
        if initial_balance <= 0 or not 0 <= transaction_fee < 1:
            raise ValueError("Solde initial ou frais de transaction invalides")
        if reward_function not in {
            "simple", "sharpe", "transaction_penalty", "drawdown",
            "excess_return", "risk_adjusted_excess",
        }:
            raise ValueError("Fonction de récompense invalide")
        if action_type not in {"discrete", "continuous"}:
            raise ValueError("Type d'action invalide")
        if allocation_strategy not in {"equal", "proportional", "risk_parity"}:
            raise ValueError("Stratégie d'allocation invalide")
        if not np.isfinite(pd.to_numeric(df["close"], errors="coerce")).all() or (df["close"] <= 0).any():
            raise ValueError("Les prix de clôture doivent être finis et strictement positifs")

        # Stocker les paramètres
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.include_position = include_position
        self.include_balance = include_balance
        self.include_technical_indicators = include_technical_indicators
        self.risk_management = bool(kwargs.get("use_risk_manager", risk_management))
        self.risk_config = dict(kwargs.get("risk_config", {}))
        self.max_trade_fraction = float(kwargs.get("max_trade_fraction", 0.30))
        self.min_trade_fraction = float(kwargs.get("min_trade_fraction", 0.01))
        self.min_trade_interval = int(kwargs.get("min_trade_interval", 0))
        self.min_signal_confidence = float(kwargs.get("min_signal_confidence", 0.0))
        if not 0 < self.max_trade_fraction <= 1 or not 0 <= self.min_trade_fraction <= self.max_trade_fraction:
            raise ValueError("Fractions minimale ou maximale de trade invalides")
        if self.min_trade_interval < 0:
            raise ValueError("min_trade_interval doit être >= 0")
        if not 0.0 <= self.min_signal_confidence <= 1.0:
            raise ValueError("min_signal_confidence doit être dans [0, 1]")
        self.use_risk_manager = risk_management  # Alias pour compatibilité
        self.normalize_observation = normalize_observation
        self.use_adaptive_normalization = (
            normalize_observation  # Alias pour compatibilité
        )
        self.reward_function = reward_function
        self.risk_aversion = risk_aversion
        self.transaction_penalty = transaction_penalty
        self.invalid_action_penalty = max(0.0, float(invalid_action_penalty))
        if benchmark_exposure is None:
            benchmark_exposure = self.risk_config.get("max_position_size", 1.0) if self.risk_management else 1.0
        self.benchmark_exposure = float(benchmark_exposure)
        if not 0.0 <= self.benchmark_exposure <= 1.0:
            raise ValueError("benchmark_exposure doit être dans [0, 1]")
        self.lookback_window = lookback_window
        self.action_type = action_type  # Stocker le type d'action
        self.n_discrete_actions = (
            n_discrete_actions  # Stocker le nombre d'actions discrètes
        )

        # Les indicateurs sont calculés une seule fois, exclusivement avec des
        # opérations rétrospectives. Aucun backfill n'est autorisé, car il
        # injecterait des valeurs futures dans les premières observations.
        self.feature_columns = ["close"]
        if self.include_technical_indicators:
            required_ohlcv = {"open", "high", "low", "close", "volume"}
            missing = required_ohlcv.difference(self.df.columns)
            if missing:
                raise ValueError(
                    "Les indicateurs techniques requièrent OHLCV; colonnes manquantes: "
                    + ", ".join(sorted(missing))
                )
            self.df = TechnicalIndicators(self.df).add_all_indicators(self.df)
            self.df = add_causal_regime_features(self.df)
            self.df = add_causal_indicator_fusion(self.df)
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            # Pandas ne sait pas propager de manière fiable les float16 ; le
            # collecteur peut en produire, donc on stabilise en float32 avant
            # le nettoyage causal (forward-fill uniquement).
            self.df.loc[:, numeric_columns] = (
                self.df.loc[:, numeric_columns]
                .astype(np.float32)
                .replace([np.inf, -np.inf], np.nan)
                .ffill()
                .fillna(0.0)
            )
            base_features = ["close", "open", "high", "low", "volume"]
            sentiment_features = [
                column
                for column in ("polarity", "subjectivity", "compound_score")
                if column in self.df.columns
            ]
            technical_features = [
                column
                for column in self.df.columns
                if column not in base_features + sentiment_features
                and pd.api.types.is_numeric_dtype(self.df[column])
            ]
            self.feature_columns = base_features + technical_features + sentiment_features
            self.technical_indicators = technical_features

        # Ajouter les paramètres de marché réalistes
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        self.execution_delay = execution_delay
        self.pending_orders = []  # Liste des ordres en attente d'exécution
        self.reserved_cash = 0.0
        self.reserved_crypto = 0.0
        self.position_id = "long_position"
        self.entry_price = None
        self._benchmark_cash = 0.0
        self._benchmark_quantity = 0.0
        self.allocation_strategy = (
            allocation_strategy  # Stocker la stratégie d'allocation
        )
        self.n_assets = 1  # Par défaut, nous avons un seul actif
        self.allocation_history = []  # Historique des allocations

        # Initialiser les attributs manquants
        self._discrete_to_continuous = (
            self._discrete_to_continuous
        )  # Référence à la méthode
        self.allocation_strategy = (
            allocation_strategy  # Réinitialiser pour s'assurer qu'il est défini
        )

        # Variables supplémentaires pour le calcul des récompenses
        self.portfolio_value_history = []
        self.returns_history = []
        self.actions_history = []
        self.trade_events = []
        self.transaction_count = 0
        self.last_transaction_step = -1
        self.max_portfolio_value = 0
        self.max_drawdown = 0.05  # Drawdown maximum autorisé (5% par défaut)
        self.drawdown_penalty = 1.0  # Pénalité pour le dépassement du drawdown maximum
        self.reward_drawdown_weight = float(kwargs.get("reward_drawdown_weight", risk_aversion))
        self.reward_downside_weight = float(kwargs.get("reward_downside_weight", 0.0))
        if self.reward_drawdown_weight < 0.0 or self.reward_downside_weight < 0.0:
            raise ValueError("Les coefficients de récompense doivent être positifs")
        self.max_turnover = float(kwargs.get("max_turnover", 0.05))
        self.turnover_penalty = float(kwargs.get("turnover_penalty", 0.01))
        if not 0.0 <= self.max_turnover <= 1.0 or self.turnover_penalty < 0.0:
            raise ValueError("Paramètres de turnover invalides")
        self._agent_turnover = 0.0
        self._last_execution = {"executed": False, "blocked_by_risk": False, "invalid_action": False}
        self._agent_turnover = 0.0

        # Définir l'espace d'action selon le type
        if action_type == "discrete":
            # Actions: 0 (ne rien faire), 1-n (acheter x%), n+1-2n (vendre x%)
            # Exemple avec n_discrete_actions=5:
            # 0: ne rien faire
            # 1-5: acheter 20%, 40%, 60%, 80%, 100% du solde disponible
            # 6-10: vendre 20%, 40%, 60%, 80%, 100% des crypto détenues
            self.action_space = spaces.Discrete(1 + 2 * n_discrete_actions)
        elif action_type == "continuous":
            # Action continue entre -1 et 1
            # -1: vendre 100%, -0.5: vendre 50%, 0: ne rien faire, 0.5: acheter 50%, 1: acheter 100%
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            raise ValueError(f"Type d'action non supporté: {action_type}")

        # Réinitialiser l'environnement pour calculer la taille réelle de l'état
        temp_reset = self.reset()
        if isinstance(temp_reset, tuple):
            temp_state = temp_reset[
                0
            ]  # Pour la compatibilité avec les nouvelles versions de gym
        else:
            temp_state = temp_reset

        # Définir l'espace d'observation avec la taille réelle de l'état
        real_state_size = temp_state.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(real_state_size,), dtype=np.float32
        )

        logger.info(
            f"Environnement de trading initialisé avec {len(df)} points de données et espace d'action {action_type}"
        )

    def _build_observation_space(self):
        """Construit l'espace d'observation."""
        # Calcul correct du nombre de caractéristiques
        n_features = self.window_size + 1  # Historique des prix (close)
        n_features += 2  # Crypto détenue + solde

        if self.include_technical_indicators:
            n_features += self.window_size * 3  # RSI, MACD, BB

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement à l'état initial.

        Args:
            seed: Graine aléatoire pour la reproductibilité
            options: Options supplémentaires pour la réinitialisation

        Returns:
            observation: L'observation initiale
            info: Informations supplémentaires
        """
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.max_portfolio_value = self.initial_balance
        self.portfolio_value_history = [self.initial_balance]
        self.returns_history = []
        self.actions_history = []
        self.transaction_count = 0
        self.last_transaction_step = -1
        self.pending_orders = []
        self.reserved_cash = 0.0
        self.reserved_crypto = 0.0
        self.entry_price = None
        benchmark_entry_price = float(self.df.iloc[self.current_step]["close"])
        self._benchmark_cash = self.initial_balance * (1.0 - self.benchmark_exposure)
        self._benchmark_quantity = (
            self.initial_balance * self.benchmark_exposure
            / (benchmark_entry_price * (1.0 + self.transaction_fee))
        )
        self._last_execution = {"executed": False, "blocked_by_risk": False, "invalid_action": False}

        if self.risk_management:
            self.risk_manager = RiskManager(self.risk_config)
            self.risk_manager.indicators.set_data(
                self.df.iloc[: self.current_step + 1].copy()
            )

        if self.normalize_observation and self.include_technical_indicators:
            self.normalizer = AdaptiveNormalizer()

        observation = self._get_observation()
        info = {}

        return observation, info

    def get_action_mask(self):
        """Retourne les actions actuellement exécutables, sans modifier l'état.

        Les masques bloquent uniquement les ordres impossibles (pas de cash ou
        pas de position). Les plafonds de risque restent appliqués au moment de
        l'exécution pour conserver une gestion du risque déterministe.
        """
        price = float(self.df.iloc[self.current_step]["close"])
        portfolio_value = self.get_portfolio_value()
        min_trade_value = portfolio_value * self.min_trade_fraction
        available_cash = max(0.0, self.balance - self.reserved_cash)
        available_position_value = max(0.0, self.crypto_held - self.reserved_crypto) * price
        cooldown_active = self.current_step - self.last_transaction_step < self.min_trade_interval
        confidence = float(self.df.iloc[self.current_step].get("signal_confidence", 1.0))
        buys_allowed = not cooldown_active and np.isfinite(confidence) and confidence >= self.min_signal_confidence
        if self.action_type == "discrete":
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[0] = True
            if cooldown_active:
                return mask
            buy_capacity = self._max_buy_value(price, portfolio_value, available_cash)
            for action in range(1, self.n_discrete_actions + 1):
                requested_value = available_cash * action / self.n_discrete_actions
                mask[action] = buys_allowed and min(requested_value, buy_capacity) >= min_trade_value
            for action in range(self.n_discrete_actions + 1, 2 * self.n_discrete_actions + 1):
                sell_percentage = (action - self.n_discrete_actions) / self.n_discrete_actions
                mask[action] = available_position_value * sell_percentage >= min_trade_value
            return mask
        return {
            "low": np.asarray([-1.0 if not cooldown_active and available_position_value >= min_trade_value else 0.0], dtype=np.float32),
            "high": np.asarray([1.0 if buys_allowed and self._max_buy_value(price, portfolio_value, available_cash) >= min_trade_value else 0.0], dtype=np.float32),
        }

    def project_action(self, action):
        """Projette une action d'agent dans le domaine exécutable courant."""
        mask = self.get_action_mask()
        if self.action_type == "discrete":
            value = int(action)
            if value < 0 or value >= self.action_space.n or not mask[value]:
                return 0
            return value
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        return np.clip(values, mask["low"], mask["high"])

    def step(self, action):
        """Exécute une étape de trading."""
        info = {"action_adjusted": False}
        self._last_execution = {"executed": False, "blocked_by_risk": False, "invalid_action": False}
        trade_event_start = len(self.trade_events)
        balance_before_action, held_before_action = self.balance, self.crypto_held

        # Stocker l'action dans l'historique
        self.actions_history.append(action)

        # Appliquer l'action selon le type
        if self.action_type == "discrete":
            self._apply_discrete_action(action)
        else:
            self._apply_continuous_action(action)
        self._capture_fill(held_before_action, balance_before_action, "agent_order")

        previous_portfolio_value = self.portfolio_value_history[-1]
        self.current_step += 1
        balance_before_pending, held_before_pending = self.balance, self.crypto_held
        self._process_pending_orders()
        self._capture_fill(held_before_pending, balance_before_pending, "agent_order")

        # La valeur est marquée au prix du bar suivant : l'action ne bénéficie
        # jamais d'un prix futur dans son observation, mais elle subit bien les
        # frais, le slippage et le mouvement de marché suivant.
        current_price = float(self.df.iloc[self.current_step]["close"])
        previous_market_price = float(self.df.iloc[self.current_step - 1]["close"])
        benchmark_return = self._benchmark_step_return(previous_market_price, current_price)
        stop_event = {"stop_triggered": False, "stop_type": None, "stop_price": None}
        if self.risk_management:
            balance_before_stop, held_before_stop = self.balance, self.crypto_held
            stop_event = self._enforce_risk_stops(current_price)
            self._capture_fill(held_before_stop, balance_before_stop, stop_event["stop_type"] or "risk_exit")

        portfolio_value = self.get_portfolio_value()
        portfolio_return = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
        agent_fills = [
            fill for fill in self.trade_events[trade_event_start:]
            if fill.get("reason") == "agent_order"
        ]
        self._agent_turnover = sum(
            float(fill["quantity"]) * float(fill["price"]) for fill in agent_fills
        ) / max(previous_portfolio_value, 1e-12)
        self.portfolio_value_history.append(portfolio_value)
        self.returns_history.append(portfolio_return)

        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.df) - 1

        # Obtenir l'état suivant
        next_state = self._get_observation()

        # Calculer la récompense
        reward = self._calculate_reward(portfolio_return, benchmark_return)

        # Ajouter la valeur du portefeuille aux informations
        info["portfolio_value"] = portfolio_value
        info["agent_turnover"] = self._agent_turnover
        info["balance"] = self.balance
        info["crypto_held"] = self.crypto_held
        info["current_price"] = current_price
        info["portfolio_return"] = portfolio_return
        info["benchmark_return"] = benchmark_return
        info["trade_executed"] = self._last_execution["executed"]
        info["execution_reason"] = (
            stop_event["stop_type"]
            if stop_event["stop_triggered"]
            else ("agent_order" if self._last_execution["executed"] else None)
        )
        info["trade_events"] = list(self.trade_events)
        info["invalid_action"] = self._last_execution["invalid_action"]
        info["action_adjusted"] = self._last_execution["blocked_by_risk"]
        info["risk_info"] = {
            "max_position_fraction": self.risk_manager.max_position_size
            if self.risk_management
            else None,
            "position_value": self.crypto_held * current_price,
            "stop": stop_event,
            "reserved_cash": self.reserved_cash,
            "reserved_crypto": self.reserved_crypto,
            "market_regime": str(self.df.iloc[self.current_step].get("market_regime", "range")),
            "regime_exposure_multiplier": self._regime_exposure_multiplier(),
        }

        return next_state, reward, done, False, info

    def _capture_fill(self, held_before, balance_before, reason):
        """Enregistre un fill effectif avec prix/frais déduits du portefeuille."""
        quantity_delta = float(self.crypto_held - held_before)
        if abs(quantity_delta) <= 1e-12:
            return
        side = "buy" if quantity_delta > 0 else "sell"
        quantity = abs(quantity_delta)
        cash_delta = float(self.balance - balance_before)
        if side == "buy":
            price = -cash_delta / (quantity * (1 + self.transaction_fee))
        else:
            price = cash_delta / (quantity * (1 - self.transaction_fee))
        self.trade_events.append({
            "timestamp": self.df.index[self.current_step], "side": side,
            "quantity": quantity, "price": float(price),
            "fee": float(quantity * price * self.transaction_fee), "reason": reason,
        })
        if reason == "agent_order":
            self.last_transaction_step = self.current_step

    def _allowed_buy_value(self, requested_value, price):
        """Retourne le montant achetable après plafonds de risque et de liquidité."""
        portfolio_value = self.get_portfolio_value()
        available_cash = max(0.0, self.balance - self.reserved_cash)
        allowed_value = min(float(requested_value), self._max_buy_value(price, portfolio_value, available_cash))
        if allowed_value + 1e-12 < requested_value:
            self._last_execution["blocked_by_risk"] = True
        return max(0.0, allowed_value)

    def _max_buy_value(self, price, portfolio_value, available_cash):
        """Capacité d'achat pure utilisée à la fois par masque et exécution."""
        pending_exposure = sum(
            order["amount"] * price
            for order in self.pending_orders
            if order["action_value"] > 0
        )
        allowed_value = min(float(available_cash), portfolio_value * self.max_trade_fraction)
        if self.risk_management:
            current_exposure = self.crypto_held * price + pending_exposure
            regime_multiplier = self._regime_exposure_multiplier()
            exposure_remaining = max(
                0.0,
                portfolio_value * self.risk_manager.max_position_size * regime_multiplier - current_exposure,
            )
            allowed_value = min(allowed_value, exposure_remaining)
        return max(0.0, allowed_value)

    def _regime_exposure_multiplier(self):
        """Réduit causalement l'exposition en bear/range ou volatilité élevée."""
        regime = str(self.df.iloc[self.current_step].get("market_regime", "range"))
        multiplier = {"bull": 1.0, "range": 0.75, "bear": 0.50}.get(regime, 0.75)
        relative_volatility = float(self.df.iloc[self.current_step].get("regime_volatility", 1.0))
        if np.isfinite(relative_volatility) and relative_volatility > 1.5:
            multiplier *= 0.70
        return float(np.clip(multiplier, 0.20, 1.0))

    def _apply_slippage(self, price, action_value):
        """Applique le slippage au prix."""
        if self.slippage_model == "dynamic":
            # Calculer le slippage en fonction du volume et de la volatilité
            current_volume = float(self.df.iloc[self.current_step].get("volume", 1.0))
            avg_volume = self.df.iloc[
                max(0, self.current_step - 20) : self.current_step
            ]["volume"].mean() if "volume" in self.df.columns else 1.0
            historical_returns = self.df["close"].pct_change().iloc[
                max(0, self.current_step - 20) : self.current_step + 1
            ]
            volatility = float(historical_returns.std(ddof=0) or 0.0)

            # Calculer le facteur de slippage
            slippage_factor = (
                self.slippage_value * (1 + volatility) * (current_volume / max(avg_volume, 1e-12))
            )
            slippage_factor = float(np.clip(slippage_factor, 0.0, 0.05))

            # Appliquer le slippage
            if action_value > 0:  # Achat
                return price * (1 + slippage_factor)
            else:  # Vente
                return price * (1 - slippage_factor)
        elif self.slippage_model == "proportional":
            # Slippage proportionnel à la taille de l'action
            action_abs = abs(action_value)
            if action_value > 0:  # Achat
                return price * (1 + self.slippage_value * action_abs)
            else:  # Vente
                return price * (1 - self.slippage_value * action_abs)
        else:
            # Slippage constant (par défaut)
            if action_value > 0:  # Achat
                return price * (1 + self.slippage_value)
            else:  # Vente
                return price * (1 - self.slippage_value)

    def _process_pending_orders(self):
        """
        Traite les ordres en attente d'exécution.
        """
        current_price = self.df.iloc[self.current_step]["close"]
        executed_orders = []

        for order in self.pending_orders:
            order["delay"] -= 1
            if order["delay"] <= 0:
                # Exécuter l'ordre
                price_with_slippage = self._apply_slippage(
                    current_price, order["action_value"]
                )

                if order["action_value"] > 0:  # Achat
                    self.reserved_cash = max(
                        0.0, self.reserved_cash - order.get("reserved_cash", 0.0)
                    )
                    affordable_amount = self.balance / (
                        price_with_slippage * (1 + self.transaction_fee)
                    )
                    filled_amount = min(order["amount"], affordable_amount)
                    self.balance -= filled_amount * price_with_slippage * (1 + self.transaction_fee)
                    self.crypto_held += filled_amount
                    if filled_amount > 0:
                        self._register_open_position(price_with_slippage)
                else:  # Vente
                    self.reserved_crypto = max(
                        0.0, self.reserved_crypto - order.get("reserved_crypto", 0.0)
                    )
                    filled_amount = min(order["amount"], self.crypto_held)
                    self.balance += filled_amount * price_with_slippage * (1 - self.transaction_fee)
                    self.crypto_held -= filled_amount
                    if self.crypto_held <= 1e-12:
                        self._clear_open_position()

                self._last_execution["executed"] = True
                self.transaction_count += 1
                executed_orders.append(order)

        # Retirer les ordres exécutés
        self.pending_orders = [
            order for order in self.pending_orders if order not in executed_orders
        ]

    def _apply_discrete_action(self, action):
        """
        Applique une action discrète.

        Args:
            action (int): Indice de l'action à appliquer
        """
        # Obtenir le prix actuel
        current_price = self.df.iloc[self.current_step]["close"]

        if action == 0:  # Ne rien faire
            logger.debug("Action: HOLD")
            return

        # Calculer le pourcentage d'achat/vente
        if 1 <= action <= self.n_discrete_actions:  # Achat
            # Calculer le pourcentage d'achat (1/n, 2/n, ..., n/n)
            buy_percentage = action / self.n_discrete_actions

            buy_value = self._allowed_buy_value(
                self.balance * buy_percentage, current_price
            )
            if buy_value <= 0:
                return

            # Calculer la quantité de crypto à acheter
            max_crypto_to_buy = buy_value / (current_price * (1 + self.transaction_fee))

            # Si délai d'exécution > 0, ajouter à la liste des ordres en attente
            if self.execution_delay > 0:
                reserved_cash = max_crypto_to_buy * current_price * (1 + self.transaction_fee)
                self.reserved_cash += reserved_cash
                self.pending_orders.append(
                    {
                        "action_value": buy_percentage,
                        "amount": max_crypto_to_buy,
                        "delay": self.execution_delay,
                        "reserved_cash": reserved_cash,
                    }
                )
                logger.debug(
                    f"Ordre d'achat en attente: {max_crypto_to_buy:.6f} unités à ${current_price:.2f}, délai: {self.execution_delay}"
                )
            else:
                # Acheter la quantité calculée immédiatement
                self.crypto_held += max_crypto_to_buy
                self.balance -= (
                    max_crypto_to_buy * current_price * (1 + self.transaction_fee)
                )
                self._register_open_position(current_price)
                self._last_execution["executed"] = True
                self.transaction_count += 1
                logger.debug(
                    f"Achat: {max_crypto_to_buy:.6f} unités à ${current_price:.2f} (limité à 30% du portefeuille)"
                )

        elif self.n_discrete_actions < action <= 2 * self.n_discrete_actions:  # Vente
            if self.crypto_held > 0:
                # Calculer le pourcentage de vente (1/n, 2/n, ..., n/n)
                sell_percentage = (
                    action - self.n_discrete_actions
                ) / self.n_discrete_actions
                crypto_to_sell = max(0.0, (self.crypto_held - self.reserved_crypto) * sell_percentage)

                # Si délai d'exécution > 0, ajouter à la liste des ordres en attente
                if self.execution_delay > 0:
                    self.reserved_crypto += crypto_to_sell
                    self.pending_orders.append(
                        {
                            "action_value": -sell_percentage,  # Négatif pour indiquer une vente
                        "amount": crypto_to_sell,
                        "delay": self.execution_delay,
                        "reserved_crypto": crypto_to_sell,
                        }
                    )
                    logger.debug(
                        f"Ordre de vente en attente: {crypto_to_sell:.6f} unités à ${current_price:.2f}, délai: {self.execution_delay}"
                    )
                else:
                    # Vendre la quantité calculée immédiatement
                    self.balance += (
                        crypto_to_sell * current_price * (1 - self.transaction_fee)
                    )
                    self.crypto_held -= crypto_to_sell
                    if self.crypto_held <= 1e-12:
                        self._clear_open_position()
                    self._last_execution["executed"] = True
                    self.transaction_count += 1
                    logger.debug(
                        f"Vente: {crypto_to_sell:.6f} unités ({sell_percentage*100:.0f}%) à ${current_price:.2f}"
                    )
            else:
                self._last_execution["invalid_action"] = True
                logger.debug("Tentative de vente sans crypto détenue")

    def _apply_continuous_action(self, action):
        """
        Applique une action continue.

        Args:
            action (float): Valeur de l'action entre -1 et 1
        """
        # Obtenir le prix actuel
        current_price = self.df.iloc[self.current_step]["close"]

        # Extraire la valeur scalaire de l'action numpy
        action_value = (
            float(action[0]) if isinstance(action, np.ndarray) else float(action)
        )
        action_value = float(np.clip(action_value, -1.0, 1.0))

        # Zone neutre autour de 0 pour éviter des micro-transactions
        if -0.05 <= action_value <= 0.05:
            logger.debug("Action: HOLD (zone neutre)")
            return

        if action_value > 0:  # Achat
            buy_percentage = action_value

            buy_value = self._allowed_buy_value(
                self.balance * buy_percentage, current_price
            )
            if buy_value <= 0:
                return

            # Calculer la quantité de crypto à acheter
            max_crypto_to_buy = buy_value / (current_price * (1 + self.transaction_fee))

            # Si délai d'exécution > 0, ajouter à la liste des ordres en attente
            if self.execution_delay > 0:
                reserved_cash = max_crypto_to_buy * current_price * (1 + self.transaction_fee)
                self.reserved_cash += reserved_cash
                self.pending_orders.append(
                    {
                        "action_value": action_value,
                        "amount": max_crypto_to_buy,
                        "delay": self.execution_delay,
                        "reserved_cash": reserved_cash,
                    }
                )
                logger.debug(
                    f"Ordre d'achat en attente: {max_crypto_to_buy:.6f} unités ({buy_percentage*100:.0f}%) à ${current_price:.2f}, délai: {self.execution_delay}"
                )
            else:
                # Acheter la quantité calculée immédiatement
                self.crypto_held += max_crypto_to_buy
                self.balance -= (
                    max_crypto_to_buy * current_price * (1 + self.transaction_fee)
                )
                self._register_open_position(current_price)
                self._last_execution["executed"] = True
                self.transaction_count += 1
                logger.debug(
                    f"Achat: {max_crypto_to_buy:.6f} unités ({buy_percentage*100:.0f}%) à ${current_price:.2f} (limité à 30% du portefeuille)"
                )

        else:  # Vente (action_value < 0)
            if self.crypto_held > 0:
                sell_percentage = -action_value
                crypto_to_sell = max(0.0, (self.crypto_held - self.reserved_crypto) * sell_percentage)

                # Si délai d'exécution > 0, ajouter à la liste des ordres en attente
                if self.execution_delay > 0:
                    self.reserved_crypto += crypto_to_sell
                    self.pending_orders.append(
                        {
                            "action_value": action_value,
                        "amount": crypto_to_sell,
                        "delay": self.execution_delay,
                        "reserved_crypto": crypto_to_sell,
                        }
                    )
                    logger.debug(
                        f"Ordre de vente en attente: {crypto_to_sell:.6f} unités ({sell_percentage*100:.0f}%) à ${current_price:.2f}, délai: {self.execution_delay}"
                    )
                else:
                    # Vendre la quantité calculée immédiatement
                    self.balance += (
                        crypto_to_sell * current_price * (1 - self.transaction_fee)
                    )
                    self.crypto_held -= crypto_to_sell
                    if self.crypto_held <= 1e-12:
                        self._clear_open_position()
                    self._last_execution["executed"] = True
                    self.transaction_count += 1
                    logger.debug(
                        f"Vente: {crypto_to_sell:.6f} unités ({sell_percentage*100:.0f}%) à ${current_price:.2f}"
                    )
            else:
                self._last_execution["invalid_action"] = True
                logger.debug("Tentative de vente sans crypto détenue")

    def _get_observation(self):
        """
        Construit l'observation de l'état actuel.

        Returns:
            np.array: Un vecteur d'observation normalisé
        """
        # Obtenir l'indice actuel
        # La fenêtre se termine au bar actuellement connu. L'ancienne formule
        # ajoutait window_size et introduisait une fuite de données futures.
        current_idx = min(self.current_step + 1, len(self.df))

        window = self.df[self.feature_columns].iloc[
            current_idx - self.window_size : current_idx
        ]
        reference_price = max(float(window["close"].iloc[-1]), 1e-12)
        features = []
        bounded_features = {"rsi", "stoch_k", "stoch_d", "adx", "mfi"}
        sentiment_features = {"polarity", "subjectivity", "compound_score"}
        for column in self.feature_columns:
            values = window[column].to_numpy(dtype=np.float32, copy=True)
            if column in bounded_features:
                values /= 100.0
            elif column in sentiment_features:
                values = np.clip(values, -1.0, 1.0)
            elif column == "volume":
                values /= max(float(np.mean(np.abs(values))), 1e-12)
            elif column == "obv":
                values /= max(float(np.max(np.abs(values))), 1e-12)
            else:
                values /= reference_price
            features.extend(values)

        # Ajouter la position actuelle si activée
        if self.include_position:
            features.append(self.crypto_held / self.initial_balance)

        # Ajouter le solde si activé
        if self.include_balance:
            features.append(self.balance / self.initial_balance)

        # Convertir en array numpy et s'assurer que c'est en float16
        observation = np.array(features, dtype=np.float32)

        # Appliquer la normalisation adaptative si activée
        if self.normalize_observation:
            # Créer des noms de features génériques pour l'observation actuelle
            feature_names = [f"feature_{i}" for i in range(observation.shape[0])]

            if not hasattr(self, "normalizer"):
                # Initialiser le normalizer avec les noms de features
                self.normalizer = AdaptiveNormalizer(
                    window_size=1000,
                    method="minmax",
                    clip_values=True,
                    feature_names=feature_names,
                )
            elif len(self.normalizer.feature_names) != len(feature_names):
                # Mettre à jour le normalizer si la taille de l'observation a changé
                self.normalizer = AdaptiveNormalizer(
                    window_size=1000,
                    method="minmax",
                    clip_values=True,
                    feature_names=feature_names,
                )

            # Mettre à jour uniquement avec l'observation présente/past, puis
            # normaliser : aucune statistique future ne peut entrer ici.
            self.normalizer.update(
                {name: value for name, value in zip(feature_names, observation)}
            )
            observation = self.normalizer.normalize_array(observation, feature_names)

        return observation.astype(np.float32, copy=False)

    def render(self, mode="human"):
        """
        Affiche l'état actuel de l'environnement pour visualisation.
        """
        if self.current_step >= len(self.df):
            return

        fig = plt.figure(figsize=(16, 8))

        # Sous-graphique pour le prix et les actions
        price_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
        action_ax = plt.subplot2grid((4, 1), (2, 0), rowspan=1, sharex=price_ax)
        portfolio_ax = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=price_ax)

        # Tracer le prix
        price_subset = self.df.iloc[
            max(0, self.current_step - 30) : self.current_step + 1
        ]
        price_ax.plot(price_subset.index, price_subset["close"], "b-")
        price_ax.set_title(f"Prix {self.df.columns[0]} - Étape {self.current_step}")

        # Tracer les indicateurs techniques si activés
        if self.include_technical_indicators and hasattr(self, "technical_indicators"):
            for indicator in self.technical_indicators:
                if indicator in self.df.columns:
                    price_ax.plot(
                        price_subset.index,
                        price_subset[indicator],
                        alpha=0.7,
                        label=indicator,
                    )
            price_ax.legend(loc="upper left")

        # Tracer les actions
        action_colors = {0: "gray", 1: "green", 2: "red"}  # Hold, Buy, Sell
        actions = self.actions_history[-30:] if len(self.actions_history) > 0 else []
        if actions:
            action_indices = price_subset.index[-len(actions) :]
            for i, action in enumerate(actions):
                if i < len(
                    action_indices
                ):  # Assurer que nous avons un indice correspondant
                    action_ax.bar(
                        action_indices[i], 1, color=action_colors.get(action, "gray")
                    )
        action_ax.set_title("Actions (Gris=Hold, Vert=Achat, Rouge=Vente)")
        action_ax.set_yticks([])

        # Tracer la valeur du portefeuille
        portfolio_values = (
            self.portfolio_value_history[-30:]
            if len(self.portfolio_value_history) > 0
            else []
        )
        if portfolio_values:
            portfolio_indices = price_subset.index[-len(portfolio_values) :]
            portfolio_ax.plot(portfolio_indices, portfolio_values, "g-")
        portfolio_ax.set_title(
            f"Valeur du portefeuille: ${self.get_portfolio_value():.2f}"
        )

        # Formater les dates si l'index est un DatetimeIndex
        if isinstance(self.df.index, pd.DatetimeIndex):
            for ax in [price_ax, action_ax, portfolio_ax]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                ax.xaxis.set_major_locator(
                    mdates.WeekdayLocator(interval=max(1, len(price_subset) // 5))
                )

        plt.tight_layout()

        # Sauvegarder le graphique
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        step_str = f"step_{self.current_step:04d}"
        filename = f"trading_env_{step_str}_{timestamp}.png"
        output_path = VISUALIZATION_DIR / filename
        plt.savefig(output_path)

        if mode == "human":
            plt.pause(0.01)
        else:
            plt.close()

        return output_path

    def get_portfolio_value(self):
        """
        Retourne la valeur actuelle du portefeuille.

        Returns:
            float: Valeur du portefeuille
        """
        current_price = self.df.iloc[self.current_step]["close"]
        return self.balance + self.crypto_held * current_price

    def get_portfolio_history(self):
        """
        Retourne l'historique de la valeur du portefeuille.

        Returns:
            list: Historique des valeurs du portefeuille
        """
        return self.portfolio_value_history

    def get_portfolio_value_history(self):
        """
        Alias pour get_portfolio_history().
        Retourne l'historique de la valeur du portefeuille.

        Returns:
            list: Historique des valeurs du portefeuille
        """
        return self.portfolio_value_history

    def _calculate_reward(self, portfolio_return, benchmark_return=0.0):
        """Calcule la récompense basée sur le rendement du portefeuille."""
        if self.reward_function == "sharpe" and len(self.returns_history) >= 2:
            returns = np.asarray(self.returns_history[-self.lookback_window :])
            reward = float(returns.mean() / (returns.std(ddof=0) + 1e-8))
        elif self.reward_function == "transaction_penalty":
            reward = portfolio_return - (
                self.transaction_penalty if self._last_execution["executed"] else 0.0
            )
        elif self.reward_function == "drawdown":
            peak = max(self.portfolio_value_history)
            drawdown = max(0.0, (peak - self.portfolio_value_history[-1]) / peak)
            reward = portfolio_return - self.risk_aversion * drawdown
        elif self.reward_function in {"excess_return", "risk_adjusted_excess"}:
            # Un portefeuille en cash sous-performe un marché haussier : cette
            # différence rend HOLD et les ventes sans position apprenables.
            reward = portfolio_return - benchmark_return
            if self.reward_function == "risk_adjusted_excess":
                peak = max(self.portfolio_value_history)
                current_drawdown = max(
                    0.0, (peak - self.portfolio_value_history[-1]) / max(peak, 1e-12)
                )
                downside = max(0.0, -portfolio_return)
                reward -= self.reward_drawdown_weight * current_drawdown
                reward -= self.reward_downside_weight * downside
        else:
            reward = portfolio_return

        if self._last_execution["invalid_action"]:
            reward -= self.invalid_action_penalty

        # Pénalité fondée sur le notionnel réellement échangé, pas sur la
        # variation des rendements. Les sorties de sécurité restent exemptées.
        if self._agent_turnover > self.max_turnover:
            reward -= self.turnover_penalty * (self._agent_turnover - self.max_turnover)

        # Pénalité pour le drawdown
        if len(self.portfolio_value_history) > 1:
            current_drawdown = (
                max(self.portfolio_value_history) - self.portfolio_value_history[-1]
            ) / max(self.portfolio_value_history)
            if current_drawdown > self.max_drawdown:
                reward -= self.drawdown_penalty * (current_drawdown - self.max_drawdown)

        return reward

    def _benchmark_step_return(self, previous_price, current_price):
        """Rendement du Buy & Hold à l'exposition effectivement comparable.

        Le benchmark investit une seule fois la fraction configurée au reset,
        frais d'entrée inclus. Son calcul est ainsi identique au benchmark des
        rapports et ne demande jamais à un agent plafonné à 20 % de battre une
        position Buy & Hold à 100 %.
        """
        previous_value = self._benchmark_cash + self._benchmark_quantity * float(previous_price)
        current_value = self._benchmark_cash + self._benchmark_quantity * float(current_price)
        return current_value / max(previous_value, 1e-12) - 1.0

    def visualize_technical_indicators(self, window_size=100):
        """
        Visualise les indicateurs techniques utilisés dans l'environnement.

        Args:
            window_size: Nombre de périodes à afficher
        """
        if not self.include_technical_indicators:
            logger.warning(
                "Les indicateurs techniques ne sont pas activés dans cet environnement."
            )
            return

        start_idx = max(0, self.current_step - window_size)
        end_idx = min(self.current_step + 1, len(self.df))
        subset = self.df.iloc[start_idx:end_idx]

        # Organiser les indicateurs par type
        trend_indicators = ["sma", "ema", "wma", "macd", "macd_signal", "macd_hist"]
        oscillator_indicators = ["rsi", "stoch_k", "stoch_d", "cci", "williams_r"]
        volatility_indicators = [
            "atr",
            "bollinger_upper",
            "bollinger_middle",
            "bollinger_lower",
        ]
        volume_indicators = ["obv", "volume"]

        # Créer un graphique avec sous-graphiques pour chaque type d'indicateur
        fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True)

        # Prix (avec quelques indicateurs de tendance superposés)
        axs[0].plot(subset.index, subset["close"], "k-", label="Prix")
        for ind in trend_indicators:
            if ind in subset.columns:
                axs[0].plot(subset.index, subset[ind], alpha=0.7, label=ind)
        axs[0].set_title("Prix et indicateurs de tendance")
        axs[0].legend(loc="upper left")

        # Bandes de Bollinger (si disponibles)
        if "bollinger_upper" in subset.columns and "bollinger_lower" in subset.columns:
            axs[1].plot(subset.index, subset["close"], "k-", label="Prix")
            axs[1].plot(
                subset.index,
                subset["bollinger_upper"],
                "r-",
                alpha=0.5,
                label="BB Upper",
            )
            axs[1].plot(
                subset.index,
                subset["bollinger_middle"],
                "g--",
                alpha=0.5,
                label="BB Middle",
            )
            axs[1].plot(
                subset.index,
                subset["bollinger_lower"],
                "b-",
                alpha=0.5,
                label="BB Lower",
            )
            axs[1].fill_between(
                subset.index,
                subset["bollinger_upper"],
                subset["bollinger_lower"],
                color="gray",
                alpha=0.2,
            )
            axs[1].set_title("Bandes de Bollinger")
            axs[1].legend(loc="upper left")

        # Oscillateurs
        osc_plotted = False
        for ind in oscillator_indicators:
            if ind in subset.columns:
                axs[2].plot(subset.index, subset[ind], label=ind)
                osc_plotted = True
        if osc_plotted:
            axs[2].set_title("Oscillateurs")
            # Ajouter des lignes horizontales pour les niveaux courants
            if "rsi" in subset.columns:
                axs[2].axhline(y=70, color="r", linestyle="-", alpha=0.3)
                axs[2].axhline(y=30, color="g", linestyle="-", alpha=0.3)
            if "stoch_k" in subset.columns:
                axs[2].axhline(y=80, color="r", linestyle="--", alpha=0.3)
                axs[2].axhline(y=20, color="g", linestyle="--", alpha=0.3)
            axs[2].legend(loc="upper left")
        else:
            axs[2].set_visible(False)

        # Indicateurs de volatilité
        vol_plotted = False
        for ind in volatility_indicators:
            if ind == "atr" and ind in subset.columns:
                axs[3].plot(subset.index, subset[ind], label=ind)
                vol_plotted = True
        if vol_plotted:
            axs[3].set_title("Indicateurs de volatilité (ATR)")
            axs[3].legend(loc="upper left")
        else:
            axs[3].set_visible(False)

        # Volume
        if "volume" in subset.columns:
            axs[4].bar(
                subset.index, subset["volume"], color="b", alpha=0.5, label="Volume"
            )
            axs[4].set_title("Volume")
        else:
            axs[4].set_visible(False)

        plt.tight_layout()

        # Sauvegarder le graphique
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"technical_indicators_{timestamp}.png"
        output_path = VISUALIZATION_DIR / filename
        plt.savefig(output_path)
        plt.close()

        return output_path

    def _register_open_position(self, price):
        """Crée les protections de la première position longue ouverte."""
        if not self.risk_management or self.entry_price is not None:
            return
        self.entry_price = float(price)
        atr_value = self._current_atr_value()
        history = None if atr_value is not None else self.df.iloc[: self.current_step + 1].copy()
        stop_multiplier = 0.75 if str(self.df.iloc[self.current_step].get("market_regime", "range")) == "bear" else 1.0
        stop_loss = self.risk_manager.calculate_atr_stop_loss(
            history, current_price=price, position_id=self.position_id, atr_value=atr_value,
            factor_multiplier=stop_multiplier,
        )
        take_profit = self.risk_manager.calculate_atr_take_profit(
            history, current_price=price, position_id=self.position_id, atr_value=atr_value
        )
        stops = self.risk_manager.position_stops.setdefault(self.position_id, {})
        stops.setdefault("entry_price", float(price))
        stops["stop_loss"] = stop_loss if stop_loss is not None else price * (
            1 - self.risk_manager.max_risk_per_trade
        )
        stops["take_profit"] = take_profit if take_profit is not None else price * (
            1 + self.risk_manager.max_risk_per_trade * 1.5
        )
        stops.setdefault("trailing_stop", None)

    def _clear_open_position(self):
        self.crypto_held = max(0.0, self.crypto_held)
        self.entry_price = None
        if self.risk_management:
            self.risk_manager.clear_position(self.position_id)

    def _enforce_risk_stops(self, current_price):
        if self.crypto_held <= 0 or self.entry_price is None:
            return {"stop_triggered": False, "stop_type": None, "stop_price": None}
        self.risk_manager.update_trailing_stop(
            self.position_id, current_price, self.entry_price
        )
        self.risk_manager.update_atr_trailing_stop(
            None if self._current_atr_value() is not None else self.df.iloc[: self.current_step + 1],
            position_id=self.position_id,
            current_price=current_price,
            atr_value=self._current_atr_value(),
        )
        result = self.risk_manager.check_stop_conditions(self.position_id, current_price)
        if result["stop_triggered"]:
            self._close_position(current_price)
        return result

    def _current_atr_value(self):
        """Retourne l'ATR causal déjà calculé pour la bougie courante."""
        if "atr" not in self.df.columns:
            return None
        value = float(self.df.iloc[self.current_step]["atr"])
        return value if np.isfinite(value) and value > 0 else None

    def _close_position(self, price):
        """Liquidation forcée (stop/take-profit) avec frais réels."""
        amount = max(0.0, self.crypto_held - self.reserved_crypto)
        if amount <= 0:
            return
        execution_price = self._apply_slippage(float(price), -1.0)
        self.balance += amount * execution_price * (1 - self.transaction_fee)
        self.crypto_held -= amount
        self.transaction_count += 1
        self._last_execution["executed"] = True
        self._clear_open_position()

    def _discrete_to_continuous(self, action):
        """
        Convertit une action discrète en valeur continue.

        Args:
            action (int): Action discrète (0: hold, 1-n: buy x%, n+1-2n: sell x%)

        Returns:
            float: Valeur continue entre -1 et 1
        """
        if action == 0:  # Hold
            return 0.0
        elif 1 <= action <= self.n_discrete_actions:  # Buy
            return action / self.n_discrete_actions
        else:  # Sell
            sell_action = action - self.n_discrete_actions
            return -sell_action / self.n_discrete_actions

    def _allocate_assets(self, action):
        """Alloue les actifs selon la stratégie spécifiée."""
        if self.allocation_strategy == "equal":
            # Allocation égale entre tous les actifs
            allocation = np.ones(self.n_assets) / self.n_assets
        elif self.allocation_strategy == "proportional":
            # Allocation proportionnelle aux poids d'action
            allocation = np.abs(action) / np.sum(np.abs(action))
        elif self.allocation_strategy == "risk_parity":
            # Allocation basée sur la volatilité inverse
            volatilities = self.df.iloc[self.current_step][
                [f"volatility_{i}" for i in range(self.n_assets)]
            ].values
            allocation = 1 / (volatilities + 1e-6)
            allocation = allocation / np.sum(allocation)
        else:
            raise ValueError(
                f"Stratégie d'allocation inconnue: {self.allocation_strategy}"
            )

        # Mettre à jour l'historique d'allocation
        self.allocation_history.append(allocation)

        return allocation
