import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import gymnasium
import numpy as np
import pandas as pd
from gymnasium import spaces

from ai_trading.config import VISUALIZATION_DIR
from ai_trading.rl.indicator_fusion import add_causal_indicator_fusion
from ai_trading.rl.market_regime import add_causal_regime_features
from ai_trading.rl.portfolio_allocator import PortfolioAllocator

from .market_constraints import MarketConstraints

# Configuration du logger
logger = logging.getLogger("MultiAssetTradingEnvironment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Créer le répertoire s'il n'existe pas
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


class MultiAssetTradingEnvironment(gymnasium.Env):
    """
    Environnement de trading multi-actifs pour l'apprentissage par renforcement.

    Cet environnement permet à un agent de trader plusieurs actifs simultanément,
    avec une allocation dynamique du portefeuille.
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        initial_balance=10000.0,
        transaction_fee=0.001,
        window_size=20,
        include_position=True,
        include_balance=True,
        include_technical_indicators=True,
        risk_management=True,
        normalize_observation=True,
        reward_function="sharpe",
        allocation_method="equal",  # Options: "equal", "volatility", "momentum", "smart"
        rebalance_frequency=5,  # Fréquence de rééquilibrage (en pas de temps)
        min_trade_fraction=0.01,  # Notionnel minimal, fraction de la valeur du portefeuille
        min_action_magnitude=0.05,  # Signal minimal avant de créer un ordre
        atr_stop_multiplier=3.0,
        atr_trailing_multiplier=4.0,
        max_stop_loss_pct=0.10,
        max_trailing_drawdown_pct=0.12,
        max_asset_exposure=0.45,
        allow_short=False,
        max_short_exposure=0.25,
        max_total_short_exposure=0.45,
        short_initial_margin=0.50,
        short_maintenance_margin=0.35,
        short_borrow_fee_rate=0.0001,
        max_short_loss_pct=0.08,
        max_short_trailing_drawdown_pct=0.10,
        strict_short_entry=False,
        short_entry_min_confidence=0.25,
        signal_action_blend=0.0,
        regime_action_guard=False,
        regime_min_confidence=0.15,
        turnover_reward_penalty=0.002,
        bull_exposure_target=0.75,
        bull_underexposure_penalty=0.0,
        start_step: int | None = None,
        max_active_positions=3,  # Nombre maximum de positions actives simultanées
        action_type="continuous",  # Pour le multi-actifs, on utilise des actions continues
        slippage_model="dynamic",
        base_slippage=0.001,
        execution_delay=0,
        market_impact_factor=0.1,
        correlation_threshold=0.7,  # Seuil de corrélation pour la diversification
        volatility_threshold=0.05,  # Seuil de volatilité pour le filtrage des actifs
        **kwargs,
    ):
        """
        Initialise l'environnement de trading multi-actifs.

        Args:
            data_dict (Dict[str, pd.DataFrame]): Dictionnaire de DataFrames contenant les données du marché
                pour chaque actif, avec la structure {symbol: dataframe}
            initial_balance (float): Solde initial du portefeuille
            transaction_fee (float): Frais de transaction en pourcentage
            window_size (int): Taille de la fenêtre d'observation
            include_position (bool): Inclure la position actuelle dans l'observation
            include_balance (bool): Inclure le solde dans l'observation
            include_technical_indicators (bool): Inclure les indicateurs techniques dans l'observation
            risk_management (bool): Activer la gestion des risques
            normalize_observation (bool): Normaliser les observations
            reward_function (str): Fonction de récompense à utiliser
            allocation_method (str): Méthode d'allocation du portefeuille
            rebalance_frequency (int): Fréquence de rééquilibrage (en pas de temps)
            max_active_positions (int): Nombre maximum de positions actives simultanées
            action_type (str): Type d'action (doit être "continuous" pour multi-actifs)
            slippage_model (str): Modèle de slippage
            base_slippage (float): Slippage de base
            execution_delay (int): Délai d'exécution
            market_impact_factor (float): Facteur d'impact marché
            correlation_threshold (float): Seuil de corrélation pour la diversification
            volatility_threshold (float): Seuil de volatilité pour le filtrage des actifs
        """
        super(MultiAssetTradingEnvironment, self).__init__()

        # Validation des paramètres
        assert (
            isinstance(data_dict, dict) and len(data_dict) > 0
        ), "Le dictionnaire de données doit contenir au moins un actif"
        assert all(
            len(df) > window_size for df in data_dict.values()
        ), f"Tous les DataFrames doivent contenir plus de {window_size} points de données"
        assert initial_balance > 0, "Le solde initial doit être positif"
        assert (
            0 <= transaction_fee < 1
        ), "Les frais de transaction doivent être entre 0 et 1"
        assert reward_function in [
            "simple",
            "sharpe",
            "sortino",
            "transaction_penalty",
            "drawdown",
            "diversification",
            "risk_adjusted_excess",
        ], "Fonction de récompense invalide"
        assert allocation_method in [
            "equal",
            "volatility",
            "momentum",
            "smart",
        ], "Méthode d'allocation invalide"
        assert rebalance_frequency >= 1, "La fréquence de rééquilibrage doit être positive"
        assert 0 <= min_trade_fraction < 1, "La fraction minimale de trade doit être entre 0 et 1"
        assert 0 <= min_action_magnitude <= 1, "Le seuil de signal doit être entre 0 et 1"
        assert atr_stop_multiplier > 0 and atr_trailing_multiplier > 0, "Les multiplicateurs ATR doivent être positifs"
        assert 0 < max_stop_loss_pct < 1 and 0 < max_trailing_drawdown_pct < 1, "Les plafonds de perte doivent être entre 0 et 1"
        assert 0 < max_asset_exposure <= 1, "L'exposition maximale par actif doit être entre 0 et 1"
        assert 0 < max_short_exposure <= 1, "L'exposition short maximale doit être entre 0 et 1"
        assert 0 < max_total_short_exposure <= 1, "L'exposition short totale doit être entre 0 et 1"
        assert 0 < short_initial_margin <= 1 and 0 < short_maintenance_margin < 1, "Les marges short doivent être entre 0 et 1"
        assert short_borrow_fee_rate >= 0, "Le coût d'emprunt short doit être positif"
        assert 0 < max_short_loss_pct < 1 and 0 < max_short_trailing_drawdown_pct < 1, "Les plafonds short doivent être entre 0 et 1"
        assert 0 <= short_entry_min_confidence <= 1, "Le seuil de confiance short doit être entre 0 et 1"
        assert 0 <= signal_action_blend <= 1, "Le mélange de signal doit être entre 0 et 1"
        assert 0 <= regime_min_confidence <= 1, "La confiance minimale de régime doit être entre 0 et 1"
        assert turnover_reward_penalty >= 0, "La pénalité de turnover doit être positive"
        assert 0 <= bull_exposure_target <= 1, "La cible d'exposition bull doit être entre 0 et 1"
        assert bull_underexposure_penalty >= 0, "La pénalité de sous-exposition bull doit être positive"
        assert (
            action_type == "continuous"
        ), "Pour le trading multi-actifs, seul le type d'action 'continuous' est supporté"

        # Stockage des paramètres
        self.data_dict = {
            symbol: self._normalize_market_frame(frame, symbol)
            for symbol, frame in data_dict.items()
        }
        self.symbols = list(data_dict.keys())
        self.num_assets = len(self.symbols)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.include_position = include_position
        self.include_balance = include_balance
        self.include_technical_indicators = include_technical_indicators
        self.risk_management = risk_management
        self.normalize_observation = normalize_observation
        self.reward_function = reward_function
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency
        self.min_trade_fraction = min_trade_fraction
        self.min_action_magnitude = min_action_magnitude
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_trailing_multiplier = atr_trailing_multiplier
        self.max_stop_loss_pct = max_stop_loss_pct
        self.max_trailing_drawdown_pct = max_trailing_drawdown_pct
        self.max_asset_exposure = max_asset_exposure
        self.allow_short = allow_short
        self.max_short_exposure = max_short_exposure
        self.max_total_short_exposure = max_total_short_exposure
        self.short_initial_margin = short_initial_margin
        self.short_maintenance_margin = short_maintenance_margin
        self.short_borrow_fee_rate = short_borrow_fee_rate
        self.max_short_loss_pct = max_short_loss_pct
        self.max_short_trailing_drawdown_pct = max_short_trailing_drawdown_pct
        self.strict_short_entry = strict_short_entry
        self.short_entry_min_confidence = short_entry_min_confidence
        self.signal_action_blend = signal_action_blend
        self.regime_action_guard = regime_action_guard
        self.regime_min_confidence = regime_min_confidence
        self.turnover_reward_penalty = turnover_reward_penalty
        self.bull_exposure_target = bull_exposure_target
        self.bull_underexposure_penalty = bull_underexposure_penalty
        self.start_step = start_step
        self.max_active_positions = min(max_active_positions, self.num_assets)
        self.action_type = action_type
        self.slippage_model = slippage_model
        self.base_slippage = base_slippage
        self.slippage_value = base_slippage  # Initialisation du slippage_value
        self.execution_delay = execution_delay
        self.market_impact_factor = market_impact_factor
        self.correlation_threshold = correlation_threshold
        self.volatility_threshold = volatility_threshold

        # Aligner toutes les données sur les mêmes dates
        self._align_data()

        # Pré-calculer les indicateurs causaux une fois par actif. L'observation
        # multi-actifs utilisait auparavant seulement RSI/MACD, ce qui créait un
        # contrat différent de l'environnement mono-actif.
        self.technical_feature_data = {}
        if self.include_technical_indicators:
            from ai_trading.rl.technical_indicators import TechnicalIndicators

            for symbol in self.symbols:
                # ``get_all_indicators`` renvoie les indicateurs, pas OHLCV.
                # La fusion/régime causal a besoin du close réellement observé.
                features = self.data_dict[symbol][["close", "volume"]].join(
                    TechnicalIndicators(self.data_dict[symbol]).get_all_indicators()
                )
                features = add_causal_regime_features(features)
                features = add_causal_indicator_fusion(features)
                self.technical_feature_data[symbol] = (
                    features.select_dtypes(include=[np.number])
                    .replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
                )

        # Calculer les corrélations et volatilités avant l'initialisation du portfolio allocator
        self.asset_correlations = self._calculate_asset_correlations()
        self.asset_volatilities = self._calculate_asset_volatilities()

        # Initialiser le portfolio allocator
        self.portfolio_allocator = PortfolioAllocator(
            method=allocation_method, max_active_positions=self.max_active_positions
        )

        # Initialiser les variables de l'environnement
        self.balance = initial_balance
        self.crypto_holdings = {symbol: 0.0 for symbol in self.symbols}
        self.position_entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.position_peak_prices = {symbol: 0.0 for symbol in self.symbols}
        self.short_entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.short_trough_prices = {symbol: 0.0 for symbol in self.symbols}
        self.short_sale_proceeds = {symbol: 0.0 for symbol in self.symbols}
        self.short_margin_reserved = {symbol: 0.0 for symbol in self.symbols}
        self.last_prices = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value_history = []
        self.returns_history = []
        self.allocation_history = []
        self.current_step = self.window_size if self.start_step is None else self.start_step
        if not self.window_size <= self.current_step < len(self.data_dict[self.symbols[0]]) - 1:
            raise ValueError("start_step doit laisser au moins une bougie de trading après le warmup")
        # Le premier rééquilibrage reste autorisé immédiatement après reset.
        self.steps_since_rebalance = self.rebalance_frequency - 1
        self.active_assets = self.symbols[: self.max_active_positions]
        self.pending_orders = []
        self.reserved_cash = 0.0
        self.transaction_count = 0
        self.trade_events = []

        # Initialiser l'espace d'action: allocation pour chaque actif (-1 à 1 pour chaque actif)
        # -1: vendre 100%, 0: ne rien faire, 1: acheter 100%
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_assets,), dtype=np.float32
        )

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

        # Initialisation des contraintes de marché
        self.market_constraints = MarketConstraints(
            slippage_model=slippage_model,
            base_slippage=base_slippage,
            execution_delay=execution_delay,
            market_impact_factor=market_impact_factor,
        )

        # Historique des impacts marché
        self.market_impacts = {symbol: [] for symbol in self.symbols}

        logger.info(
            f"Environnement de trading multi-actifs initialisé avec {self.num_assets} actifs: {', '.join(self.symbols)}"
        )

    @staticmethod
    def _normalize_market_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalise une source partielle vers le contrat OHLCV minimal.

        Les collecteurs alternatifs peuvent fournir seulement ``close`` et
        ``volume``. Dans ce cas, réutiliser ``close`` pour open/high/low est un
        repli neutre et explicite : aucun range intrabougie n'est inventé, et
        les indicateurs qui en dépendent restent à zéro plutôt que d'échouer.
        """
        if "close" not in frame.columns:
            raise ValueError(f"Les données de {symbol} doivent contenir une colonne 'close'")

        normalized = frame.copy()
        close = pd.to_numeric(normalized["close"], errors="coerce")
        if close.isna().all():
            raise ValueError(f"La colonne 'close' de {symbol} ne contient aucune valeur numérique")
        normalized["close"] = close.ffill()
        if normalized["close"].isna().any():
            raise ValueError(
                f"La colonne 'close' de {symbol} commence par une valeur manquante; "
                "un remplissage futur est interdit"
            )

        for column in ("open", "high", "low"):
            if column not in normalized.columns:
                normalized[column] = normalized["close"]
        if "volume" not in normalized.columns:
            normalized["volume"] = 0.0
        return normalized

    def _align_data(self):
        """
        Aligne les données de tous les actifs sur les mêmes dates pour faciliter le trading simultané.
        Resample si nécessaire et assure que tous les DataFrames ont le même index.
        """
        # Trouver l'intersection des dates entre tous les actifs
        common_dates = None
        for symbol, df in self.data_dict.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))

        common_dates = sorted(list(common_dates))
        logger.info(f"Nombre de dates communes: {len(common_dates)}")

        # Filtrer chaque DataFrame pour ne garder que les dates communes
        for symbol in self.symbols:
            self.data_dict[symbol] = self.data_dict[symbol].loc[common_dates]

            # Vérifier qu'il reste suffisamment de données
            if len(self.data_dict[symbol]) <= self.window_size:
                raise ValueError(
                    f"Après alignement, les données pour {symbol} sont insuffisantes (il faut au moins {self.window_size+1} points)"
                )

        logger.info(f"Données alignées sur {len(common_dates)} dates communes")

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

        # Réinitialiser l'état
        self.current_step = self.window_size if self.start_step is None else self.start_step
        self.balance = self.initial_balance
        self.crypto_holdings = {symbol: 0.0 for symbol in self.symbols}
        self.position_entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.position_peak_prices = {symbol: 0.0 for symbol in self.symbols}
        self.short_entry_prices = {symbol: 0.0 for symbol in self.symbols}
        self.short_trough_prices = {symbol: 0.0 for symbol in self.symbols}
        self.short_sale_proceeds = {symbol: 0.0 for symbol in self.symbols}
        self.short_margin_reserved = {symbol: 0.0 for symbol in self.symbols}
        self.last_prices = {
            symbol: self.data_dict[symbol].iloc[self.current_step]["close"]
            for symbol in self.symbols
        }
        self.portfolio_value_history = [self.initial_balance]
        self.returns_history = []
        self.allocation_history = []
        self.steps_since_rebalance = self.rebalance_frequency - 1
        self.active_assets = self.symbols[: self.max_active_positions]
        self.pending_orders = []
        self.reserved_cash = 0.0
        self.transaction_count = 0
        self.trade_events = []

        # Obtenir l'observation initiale
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Exécute une étape de trading.

        Args:
            action: Action de trading pour chaque actif (-1 à 1)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Sauvegarder la valeur précédente du portefeuille
        previous_value = self.get_portfolio_value()
        transaction_count_before = self.transaction_count

        self._charge_short_borrow_fees()

        # Traitement des ordres en attente
        self._process_pending_orders()
        if self.risk_management:
            self._apply_atr_risk_exits()

        # Les positifs partagent le cash disponible et les négatifs vendent une
        # fraction de l'inventaire correspondant. Une vente n'est donc jamais
        # calculée à partir du cash (ancien comportement erroné).
        normalized_actions = self.project_action(action)
        buy_budget = self._available_cash()

        # Création des nouveaux ordres
        for i, symbol in enumerate(self.symbols):
            if (
                abs(normalized_actions[i]) > 1e-6
            ):  # Seulement si l'action n'est pas nulle
                current_price = self.data_dict[symbol].iloc[self.current_step]["close"]
                if normalized_actions[i] > 0:
                    quantity = (
                        abs(self.crypto_holdings[symbol]) * normalized_actions[i]
                        if self.crypto_holdings[symbol] < -1e-12
                        else buy_budget * normalized_actions[i] / current_price
                    )
                else:
                    quantity = (
                        self.crypto_holdings[symbol] * abs(normalized_actions[i])
                        if self.crypto_holdings[symbol] > 1e-12
                        else self._short_capacity(symbol, current_price, previous_value) * abs(normalized_actions[i]) / current_price
                    )
                trade_notional = quantity * current_price
                # Empêche les micro-achats récurrents qui gonflent le turnover
                # sans modifier matériellement le portefeuille.
                if (
                    quantity <= 1e-12
                    or trade_notional < self.min_trade_fraction * max(previous_value, 1e-12)
                ):
                    continue

                # Calculer l'impact marché
                impact, recovery_time = self.market_constraints.calculate_market_impact(
                    symbol=symbol,
                    action_value=normalized_actions[i],
                    volume=quantity * current_price,
                    price=current_price,
                    avg_volume=self._calculate_average_volume(symbol),
                )

                # Enregistrer l'impact marché
                if symbol not in self.market_impacts:
                    self.market_impacts[symbol] = []
                self.market_impacts[symbol].append(
                    {
                        "step": self.current_step,
                        "impact": impact,
                        "recovery_time": recovery_time,
                    }
                )

                # Calculer le slippage
                slippage = self._calculate_slippage(
                    symbol, normalized_actions[i], quantity
                )

                if self.execution_delay > 0:
                    reserved_cost = 0.0
                    if normalized_actions[i] > 0:
                        estimated_cost = quantity * current_price * (1 + slippage) * (1 + self.transaction_fee)
                        available_cash = max(0.0, self.balance - self.reserved_cash)
                        reserved_cost = min(estimated_cost, available_cash)
                        if reserved_cost <= 0:
                            continue
                        quantity = reserved_cost / (
                            current_price * (1 + slippage) * (1 + self.transaction_fee)
                        )
                        self.reserved_cash += reserved_cost
                    self.pending_orders.append(
                        {
                            "symbol": symbol,
                            "action_value": normalized_actions[i],
                            "volume": quantity,
                            "delay": self.execution_delay,
                            "reserved_cost": reserved_cost,
                        }
                    )
                else:
                    self._execute_trade(
                        symbol=symbol,
                        action_value=normalized_actions[i],
                        volume=quantity,
                        price=current_price,
                        slippage=slippage,
                    )

        # Mettre à jour l'étape courante et les données
        self.current_step += 1
        self._update_orderbook_data()

        # Calculer la nouvelle valeur du portefeuille et la récompense
        current_value = self.get_portfolio_value()
        reward = self._calculate_reward(
            previous_value, current_value, self.transaction_count - transaction_count_before
        )

        if self.transaction_count > transaction_count_before:
            self.steps_since_rebalance = 0
        else:
            self.steps_since_rebalance = min(
                self.rebalance_frequency - 1, self.steps_since_rebalance + 1
            )

        # Conserver les expositions nettes réelles. Normaliser uniquement les
        # positions ouvertes ferait paraitre un portefeuille largement cash
        # comme investi à 100 % dans son unique actif détenu.
        current_weights = {
            symbol: (
                self.crypto_holdings[symbol]
                * self.data_dict[symbol].iloc[self.current_step]["close"]
            )
            / max(current_value, 1e-6)
            for symbol in self.symbols
        }
        current_weights["CASH"] = (
            self.balance + sum(self.short_sale_proceeds.values())
        ) / max(current_value, 1e-6)

        self.allocation_history.append(current_weights)

        # Vérifier si l'épisode est terminé
        terminated = self.current_step >= len(self.data_dict[self.symbols[0]]) - 1
        truncated = False

        info = self._get_info()
        info["trade_executed"] = self.transaction_count > transaction_count_before
        info["trade_events"] = list(self.trade_events)
        return self._get_observation(), reward, terminated, truncated, info

    def _process_pending_orders(self):
        """
        Traite les ordres en attente.
        """
        remaining_orders = []
        for order in self.pending_orders:
            order["delay"] -= 1
            if order["delay"] <= 0:
                # Exécution de l'ordre
                slippage = self._calculate_slippage(
                    order["symbol"], order["action_value"], order["volume"]
                )
                self._execute_trade(
                    order["symbol"],
                    order["action_value"],
                    order["volume"],
                    self.data_dict[order["symbol"]].iloc[self.current_step]["close"],
                    slippage,
                )
                self.reserved_cash = max(
                    0.0, self.reserved_cash - order.get("reserved_cost", 0.0)
                )
            else:
                remaining_orders.append(order)
        self.pending_orders = remaining_orders

    def _execute_trade(
        self,
        symbol: str,
        action_value: float,
        volume: float,
        price: float,
        slippage: float,
        reason: str = "agent_order",
    ):
        """
        Exécute une transaction.

        Args:
            symbol (str): Symbole de l'actif
            action_value (float): Valeur de l'action (-1 à 1)
            volume (float): Volume à trader
            price (float): Prix de base
            slippage (float): Slippage calculé
        """
        execution_price = price * (1 + slippage if action_value > 0 else 1 - slippage)
        timestamp = self.data_dict[symbol].index[self.current_step]
        holding = self.crypto_holdings[symbol]

        if action_value > 0 and holding < -1e-12:
            covered = min(volume, abs(holding))
            if covered <= 1e-12:
                return
            entry_price = self.short_entry_prices[symbol]
            entry_proceeds = covered * entry_price
            fee = covered * execution_price * self.transaction_fee
            self.short_sale_proceeds[symbol] = max(0.0, self.short_sale_proceeds[symbol] - entry_proceeds)
            self.balance += entry_proceeds - covered * execution_price - fee
            self.crypto_holdings[symbol] += covered
            if self.crypto_holdings[symbol] >= -1e-12:
                self.crypto_holdings[symbol] = 0.0
                self.short_entry_prices[symbol] = 0.0
                self.short_trough_prices[symbol] = 0.0
                self.short_margin_reserved[symbol] = 0.0
            else:
                self.short_margin_reserved[symbol] = abs(self.crypto_holdings[symbol]) * entry_price * self.short_initial_margin
            self.transaction_count += 1
            self.trade_events.append({"timestamp": timestamp, "symbol": symbol, "side": "short_cover", "quantity": float(covered), "price": float(execution_price), "fee": float(fee), "reason": reason, "position_side": "short"})
            return

        if action_value > 0:  # Achat long
            cost = volume * execution_price * (1 + self.transaction_fee)
            available_cash = self._available_cash()
            if cost > available_cash:
                volume = available_cash / (execution_price * (1 + self.transaction_fee))
            if volume <= 1e-12:
                return
            cost = volume * execution_price * (1 + self.transaction_fee)
            previous_quantity = max(0.0, holding)
            self.balance -= cost
            self.crypto_holdings[symbol] += volume
            total_quantity = self.crypto_holdings[symbol]
            self.position_entry_prices[symbol] = (
                previous_quantity * self.position_entry_prices[symbol] + volume * execution_price
            ) / total_quantity
            self.position_peak_prices[symbol] = max(self.position_peak_prices[symbol], execution_price)
            self.transaction_count += 1
            self.trade_events.append({"timestamp": timestamp, "symbol": symbol, "side": "buy", "quantity": float(volume), "price": float(execution_price), "fee": float(volume * execution_price * self.transaction_fee), "reason": reason, "position_side": "long"})
            return

        if holding > 1e-12:  # Vente / clôture long
            sold = min(volume, holding)
            if sold <= 1e-12:
                return
            fee = sold * execution_price * self.transaction_fee
            self.balance += sold * execution_price - fee
            self.crypto_holdings[symbol] -= sold
            if self.crypto_holdings[symbol] <= 1e-12:
                self.crypto_holdings[symbol] = 0.0
                self.position_entry_prices[symbol] = 0.0
                self.position_peak_prices[symbol] = 0.0
            self.transaction_count += 1
            self.trade_events.append({"timestamp": timestamp, "symbol": symbol, "side": "sell", "quantity": float(sold), "price": float(execution_price), "fee": float(fee), "reason": reason, "position_side": "long"})
            return

        if not self.allow_short:
            return
        # La marge initiale est bloquée : un short ne peut pas utiliser du
        # collatéral déjà consommé par des longs, des ordres différés ou un
        # autre short. Cela évite un levier implicite dans le paper trading.
        margin_limited_volume = self._available_cash() / max(
            self.short_initial_margin * execution_price, 1e-12
        )
        opened = min(
            volume,
            self._short_capacity(symbol, price, self.get_portfolio_value()) / max(execution_price, 1e-12),
            margin_limited_volume,
        )
        if opened <= 1e-12:
            return
        previous_short = abs(holding)
        fee = opened * execution_price * self.transaction_fee
        self.balance -= fee
        self.crypto_holdings[symbol] -= opened
        total_short = abs(self.crypto_holdings[symbol])
        self.short_entry_prices[symbol] = (
            previous_short * self.short_entry_prices[symbol] + opened * execution_price
        ) / total_short
        self.short_trough_prices[symbol] = min(
            self.short_trough_prices[symbol] or execution_price, execution_price
        )
        self.short_sale_proceeds[symbol] += opened * execution_price
        self.short_margin_reserved[symbol] = total_short * self.short_entry_prices[symbol] * self.short_initial_margin
        self.transaction_count += 1
        self.trade_events.append({"timestamp": timestamp, "symbol": symbol, "side": "short_open", "quantity": float(opened), "price": float(execution_price), "fee": float(fee), "reason": reason, "position_side": "short"})

    def _available_cash(self) -> float:
        return max(0.0, self.balance - self.reserved_cash - sum(self.short_margin_reserved.values()))

    def close_all_positions(self, reason: str = "end_of_evaluation") -> None:
        """Réalise les positions au dernier prix pour un backtest auditable.

        Sans cette clôture de fin de fenêtre, la courbe d'equity inclut le
        mark-to-market mais le ledger FIFO ne contient pas le PnL correspondant.
        """
        for symbol in self.symbols:
            quantity = self.crypto_holdings[symbol]
            if abs(quantity) <= 1e-12:
                continue
            price = float(self.data_dict[symbol].iloc[self.current_step]["close"])
            action_value = -1.0 if quantity > 0 else 1.0
            self._execute_trade(
                symbol=symbol,
                action_value=action_value,
                volume=abs(quantity),
                price=price,
                slippage=self._calculate_slippage(symbol, action_value, abs(quantity)),
                reason=reason,
            )

    def _short_capacity(self, symbol: str, price: float, portfolio_value: float) -> float:
        current_notional = abs(min(0.0, self.crypto_holdings[symbol])) * price
        total_short_notional = sum(
            abs(min(0.0, self.crypto_holdings[name]))
            * float(self.data_dict[name].iloc[self.current_step]["close"])
            for name in self.symbols
        )
        per_asset_capacity = self.max_short_exposure * portfolio_value - current_notional
        total_capacity = self.max_total_short_exposure * portfolio_value - total_short_notional
        return max(0.0, min(per_asset_capacity, total_capacity))

    def _charge_short_borrow_fees(self) -> None:
        if not self.allow_short or self.short_borrow_fee_rate <= 0:
            return
        for symbol in self.symbols:
            quantity = abs(min(0.0, self.crypto_holdings[symbol]))
            if quantity <= 1e-12:
                continue
            price = float(self.data_dict[symbol].iloc[self.current_step]["close"])
            fee = quantity * price * self.short_borrow_fee_rate
            self.balance -= fee
            self.trade_events.append({"timestamp": self.data_dict[symbol].index[self.current_step], "symbol": symbol, "side": "borrow_fee", "quantity": 0.0, "price": price, "fee": float(fee), "reason": "short_borrow_fee", "position_side": "short"})

    def _apply_atr_risk_exits(self):
        """Ferme une position quand le stop ATR ou le trailing-stop est touché."""
        for symbol in self.symbols:
            quantity = self.crypto_holdings[symbol]
            if quantity < -1e-12:
                price = float(self.data_dict[symbol].iloc[self.current_step]["close"])
                features = self.technical_feature_data.get(symbol)
                atr = float(features.iloc[self.current_step].get("atr", 0.0)) if features is not None else 0.0
                atr = max(atr, price * 0.02)
                entry_price = self.short_entry_prices[symbol]
                self.short_trough_prices[symbol] = min(self.short_trough_prices[symbol] or price, price)
                stop_price = min(
                    entry_price + self.atr_stop_multiplier * atr,
                    entry_price * (1.0 + self.max_short_loss_pct),
                )
                trailing_price = min(
                    self.short_trough_prices[symbol] + self.atr_trailing_multiplier * atr,
                    self.short_trough_prices[symbol] * (1.0 + self.max_short_trailing_drawdown_pct),
                )
                liability = abs(quantity) * price
                margin_ratio = self.get_portfolio_value() / max(liability, 1e-12)
                reason = None
                if margin_ratio < self.short_maintenance_margin:
                    reason = "short_margin_liquidation"
                elif price >= stop_price:
                    reason = "short_atr_stop_loss"
                elif price >= trailing_price:
                    reason = "short_atr_trailing_stop"
                if reason:
                    self._execute_trade(symbol, 1.0, abs(quantity), price, self._calculate_slippage(symbol, 1.0, abs(quantity)), reason)
                continue
            if quantity <= 1e-12:
                continue
            price = float(self.data_dict[symbol].iloc[self.current_step]["close"])
            features = self.technical_feature_data.get(symbol)
            atr = float(features.iloc[self.current_step].get("atr", 0.0)) if features is not None else 0.0
            atr = max(atr, price * 0.02)
            self.position_peak_prices[symbol] = max(self.position_peak_prices[symbol], price)
            entry_price = self.position_entry_prices[symbol]
            stop_price = (
                max(
                    entry_price - self.atr_stop_multiplier * atr,
                    entry_price * (1.0 - self.max_stop_loss_pct),
                )
                if entry_price > 0 else -np.inf
            )
            trailing_price = max(
                self.position_peak_prices[symbol] - self.atr_trailing_multiplier * atr,
                self.position_peak_prices[symbol] * (1.0 - self.max_trailing_drawdown_pct),
            )
            reason = None
            if price <= stop_price:
                reason = "atr_stop_loss"
            elif price <= trailing_price:
                reason = "atr_trailing_stop"
            if reason:
                self._execute_trade(
                    symbol=symbol,
                    action_value=-1.0,
                    volume=quantity,
                    price=price,
                    slippage=self._calculate_slippage(symbol, -1.0, quantity),
                    reason=reason,
                )

    def _calculate_volatility(self, symbol: str) -> float:
        """Calcule la volatilité sur la fenêtre d'observation."""
        prices = self.data_dict[symbol].iloc[
            max(0, self.current_step - self.window_size) : self.current_step
        ]["close"]
        return np.std(np.log(prices / prices.shift(1)).dropna())

    def _calculate_average_volume(self, symbol: str) -> float:
        """Calcule le volume moyen sur la fenêtre d'observation."""
        volumes = self.data_dict[symbol].iloc[
            max(0, self.current_step - self.window_size) : self.current_step
        ]["volume"]
        return np.mean(volumes)

    def _update_orderbook_data(self):
        """Met à jour les données de profondeur du carnet d'ordres."""
        for symbol in self.active_assets:
            if "orderbook_depth" in self.data_dict[symbol].columns:
                depth_data = self.data_dict[symbol].iloc[self.current_step][
                    "orderbook_depth"
                ]
                self.market_constraints.update_orderbook_depth(symbol, depth_data)

    def _calculate_asset_correlations(self):
        """
        Calcule les corrélations entre les actifs.

        Returns:
            pd.DataFrame: DataFrame des corrélations entre les actifs
        """
        # Calculer les rendements pour chaque actif
        returns = {}
        for symbol in self.data_dict:
            prices = self.data_dict[symbol]["close"]
            returns[symbol] = prices.pct_change().dropna()

        # Créer un DataFrame de corrélations
        correlation_matrix = pd.DataFrame(index=self.symbols, columns=self.symbols)

        # Calculer les corrélations
        for symbol1 in self.symbols:
            correlation_matrix.loc[symbol1, symbol1] = 1.0  # Diagonale
            for symbol2 in self.symbols:
                if symbol1 != symbol2:
                    # Utiliser une fenêtre glissante pour les corrélations
                    correlation = (
                        returns[symbol1]
                        .rolling(window=30)
                        .corr(returns[symbol2])
                        .mean()
                    )
                    correlation_matrix.loc[symbol1, symbol2] = correlation
                    correlation_matrix.loc[symbol2, symbol1] = correlation  # Symétrie

        return correlation_matrix

    def _calculate_asset_volatilities(self) -> pd.Series:
        """
        Calcule la volatilité de chaque actif.

        Returns:
            Series: Volatilité de chaque actif
        """
        volatilities = {}
        for symbol, df in self.data_dict.items():
            returns = df["close"].pct_change()
            volatilities[symbol] = returns.std() * np.sqrt(252)  # Annualisée

        return pd.Series(volatilities)

    def _filter_assets(self) -> List[str]:
        """
        Filtre les actifs en fonction de la volatilité et de la corrélation.

        Returns:
            List[str]: Liste des actifs sélectionnés
        """
        # Filtrer par volatilité
        low_volatility_assets = [
            asset
            for asset in self.symbols
            if self.asset_volatilities[asset] <= self.volatility_threshold
        ]

        if not low_volatility_assets:
            return self.symbols[: self.max_active_positions]

        # Filtrer par corrélation
        selected_assets = [low_volatility_assets[0]]
        remaining_assets = low_volatility_assets[1:]

        while len(selected_assets) < self.max_active_positions and remaining_assets:
            # Trouver l'actif le moins corrélé avec les actifs sélectionnés
            min_correlation = float("inf")
            best_asset = None

            for asset in remaining_assets:
                max_corr = max(
                    abs(self.asset_correlations.loc[asset, selected])
                    for selected in selected_assets
                )
                if max_corr < min_correlation:
                    min_correlation = max_corr
                    best_asset = asset

            if min_correlation <= self.correlation_threshold:
                selected_assets.append(best_asset)
                remaining_assets.remove(best_asset)
            else:
                break

        return selected_assets

    def _get_observation(self):
        """
        Récupère l'observation actuelle (état) pour l'agent RL.

        Returns:
            np.ndarray: Vecteur d'observation
        """
        # Créer un vecteur d'observation composé de:
        # 1. Prix et indicateurs récents pour chaque actif
        # 2. Détention actuelle de chaque actif
        # 3. Valeur totale du portefeuille et balance

        obs_components = []

        # Pour chaque actif, ajouter le prix et d'autres features
        for symbol in self.symbols:
            # Prix récents (fenêtre)
            price_window = (
                self.data_dict[symbol]
                .iloc[self.current_step - self.window_size : self.current_step]["close"]
                .values
            )
            price_window = (
                price_window / price_window[0]
            )  # Normaliser par le premier prix
            obs_components.append(price_window)

            # Volume récent (si disponible)
            if "volume" in self.data_dict[symbol].columns:
                volume_window = (
                    self.data_dict[symbol]
                    .iloc[self.current_step - self.window_size : self.current_step][
                        "volume"
                    ]
                    .values
                )
                volume_window = volume_window / (
                    volume_window.max() if volume_window.max() > 0 else 1
                )  # Normaliser
                obs_components.append(volume_window)

            # Tous les indicateurs techniques, normalisés sur la fenêtre passée
            # de l'actif. Aucun prix futur ni backfill n'est utilisé.
            if self.include_technical_indicators:
                features = self.technical_feature_data[symbol]
                history = features.iloc[
                    self.current_step - self.window_size : self.current_step
                ]
                current = history.iloc[-1]
                std = history.std(ddof=0).replace(0.0, 1.0)
                normalized = ((current - history.mean()) / std).clip(-10.0, 10.0)
                obs_components.append(normalized.to_numpy(dtype=np.float32))

            # Position actuelle pour cet actif
            if self.include_position:
                position_value = (
                    self.crypto_holdings[symbol]
                    * self.data_dict[symbol].iloc[self.current_step]["close"]
                    / self.initial_balance
                )
                obs_components.append(np.array([position_value]))

        # Ajouter la balance et la valeur totale du portefeuille
        if self.include_balance:
            balance_normalized = self.balance / self.initial_balance
            portfolio_value_normalized = (
                self.get_portfolio_value() / self.initial_balance
            )
            obs_components.append(
                np.array([balance_normalized, portfolio_value_normalized])
            )

        # Concaténer tous les composants
        observation = np.concatenate([comp.flatten() for comp in obs_components])

        return observation

    def _get_info(self):
        """
        Récupère les informations supplémentaires sur l'état actuel.

        Returns:
            dict: Informations sur l'état actuel
        """
        # Calculer la valeur de chaque actif
        asset_values = {
            symbol: self.crypto_holdings[symbol]
            * self.data_dict[symbol].iloc[self.current_step]["close"]
            for symbol in self.symbols
        }

        # Calculer les poids actuels du portefeuille
        portfolio_value = self.get_portfolio_value()
        portfolio_weights = {
            symbol: value / portfolio_value if portfolio_value > 0 else 0
            for symbol, value in asset_values.items()
        }

        return {
            "current_step": self.current_step,
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "holdings": self.crypto_holdings.copy(),
            "asset_values": asset_values,
            "portfolio_weights": portfolio_weights,
            "current_prices": {
                symbol: self.data_dict[symbol].iloc[self.current_step]["close"]
                for symbol in self.symbols
            },
            "steps_since_rebalance": self.steps_since_rebalance,
            "trade_events": list(self.trade_events),
        }

    def get_action_mask(self):
        """Retourne les bornes exécutables par actif pour PPO/SAC.

        Sans short, une vente reste masquée sans inventaire. Avec short, une
        action négative peut ouvrir une position vendue sous plafond de marge.
        """
        if self.steps_since_rebalance < self.rebalance_frequency - 1:
            zeros = np.zeros(self.num_assets, dtype=np.float32)
            return {"low": zeros, "high": zeros}

        prices = np.asarray(
            [self.data_dict[symbol].iloc[self.current_step]["close"] for symbol in self.symbols],
            dtype=np.float32,
        )
        holdings = np.asarray([self.crypto_holdings[symbol] for symbol in self.symbols])
        low = np.where((holdings * prices > 1e-8) | self.allow_short, -1.0, 0.0).astype(np.float32)
        high = np.where(
            (self._available_cash() > 1e-8) | (holdings < -1e-8), 1.0, 0.0
        ).astype(np.float32)
        if self.regime_action_guard:
            for index, symbol in enumerate(self.symbols):
                direction = self._regime_direction(symbol)
                # En tendance haussière confirmée, l'agent peut alléger un
                # long existant mais ne peut pas ouvrir/augmenter un short.
                if direction > 0 and holdings[index] <= 1e-8:
                    low[index] = 0.0
                # Symétriquement en bear confirmé : aucun nouvel achat long,
                # mais la couverture d'un short reste toujours autorisée.
                elif direction < 0 and holdings[index] >= -1e-8:
                    # Si le short strict est lui-meme indisponible, bloquer
                    # aussi l'achat rendrait ce composant inexecutable alors
                    # que le cash et les limites de risque le permettent.
                    # Le garde-fou bear reste actif des qu'une exposition
                    # short conforme est reellement disponible.
                    strict_short_blocked = (
                        self.allow_short
                        and self.strict_short_entry
                        and not self._short_entry_allowed(symbol)
                    )
                    if not strict_short_blocked:
                        high[index] = 0.0
        if self.allow_short and self.strict_short_entry:
            for index, symbol in enumerate(self.symbols):
                # Une vente d'un long existant reste toujours possible. En
                # revanche, l'ouverture ou l'augmentation d'un short exige un
                # régime baissier causal et une confiance plus élevée que le
                # simple garde-fou de régime.
                if holdings[index] <= 1e-8 and not self._short_entry_allowed(symbol):
                    low[index] = 0.0
        return {"low": low, "high": high}

    def _short_entry_allowed(self, symbol: str) -> bool:
        features = self.technical_feature_data.get(symbol)
        if features is None:
            return False
        confidence = float(features.iloc[self.current_step].get("signal_confidence", 0.0))
        return (
            confidence >= self.short_entry_min_confidence
            and self._regime_direction(symbol) < 0
        )

    def _regime_direction(self, symbol: str) -> int:
        """Retourne -1/0/+1 uniquement pour un régime causal suffisamment confirmé."""
        features = self.technical_feature_data.get(symbol)
        if features is None:
            return 0
        row = features.iloc[self.current_step]
        confidence = float(row.get("signal_confidence", 0.0))
        trend = float(row.get("regime_trend", 0.0))
        direction = float(row.get("signal_direction", 0.0))
        if confidence < self.regime_min_confidence:
            return 0
        if trend >= 0.01 and direction > 0.0:
            return 1
        if trend <= -0.01 and direction < 0.0:
            return -1
        return 0

    def project_action(self, action):
        """Projette une action continue dans les contraintes cash/inventaire."""
        values = np.asarray(action, dtype=np.float32).reshape(self.num_assets)
        if self.signal_action_blend:
            # Prior causal borné : les indicateurs restent un contexte pour le
            # RL, mais évitent qu'une politique peu entraînée reste cash dans
            # une tendance unanimement confirmée. Aucun prix futur n'est lu.
            prior = []
            for symbol in self.symbols:
                row = self.technical_feature_data[symbol].iloc[self.current_step]
                direction = float(row.get("signal_direction", 0.0))
                confidence = float(row.get("signal_confidence", 0.0))
                prior.append(np.clip(direction * (0.5 + confidence), -1.0, 1.0))
            values = (
                (1.0 - self.signal_action_blend) * values
                + self.signal_action_blend * np.asarray(prior, dtype=np.float32)
            )
        mask = self.get_action_mask()
        values = np.clip(values, mask["low"], mask["high"])
        values[np.abs(values) < self.min_action_magnitude] = 0.0
        positive = np.clip(values, 0.0, None)
        # Un plafond d'exposition ne doit jamais empêcher une couverture
        # complète d'un short déjà ouvert : +1 signifie alors « couvrir 100 % ».
        holdings = np.asarray([self.crypto_holdings[symbol] for symbol in self.symbols])
        positive = np.where(
            holdings < -1e-12,
            positive,
            np.minimum(positive, self.max_asset_exposure),
        )
        positive_sum = float(positive.sum())
        if positive_sum > 1.0:
            positive /= positive_sum
        negative = np.clip(values, None, 0.0)
        return (positive + negative).astype(np.float32)

    def get_portfolio_value(self):
        """
        Calcule la valeur totale du portefeuille (balance + valeur des actifs détenus).

        Returns:
            float: Valeur totale du portefeuille
        """
        long_value = sum(
            max(0.0, self.crypto_holdings[symbol])
            * self.data_dict[symbol].iloc[self.current_step]["close"]
            for symbol in self.symbols
        )
        short_liability = sum(
            abs(min(0.0, self.crypto_holdings[symbol]))
            * self.data_dict[symbol].iloc[self.current_step]["close"]
            for symbol in self.symbols
        )
        return self.balance + sum(self.short_sale_proceeds.values()) + long_value - short_liability

    def _calculate_reward(self, previous_value, current_value, executed_trades: int = 0):
        """
        Calcule la récompense basée sur la fonction de récompense choisie.

        Args:
            previous_value: Valeur précédente du portefeuille
            current_value: Valeur actuelle du portefeuille

        Returns:
            float: La récompense calculée
        """
        # Calculer le changement en pourcentage
        pct_change = (
            (current_value - previous_value) / previous_value
            if previous_value > 0
            else 0
        )
        self.returns_history.append(pct_change)
        self.portfolio_value_history.append(current_value)

        # Choisir la fonction de récompense appropriée
        reward_functions = {
            "simple": lambda: pct_change,
            "sharpe": self._sharpe_reward,
            "sortino": self._sortino_reward,
            "diversification": lambda: self._diversification_reward(pct_change),
            "transaction_penalty": lambda: pct_change - 0.0001 * self.transaction_count,
            "drawdown": lambda: self._drawdown_reward(pct_change),
            "risk_adjusted_excess": lambda: self._risk_adjusted_excess_reward(pct_change, executed_trades),
        }

        # Obtenir et exécuter la fonction de récompense
        reward_func = reward_functions.get(self.reward_function, lambda: pct_change)
        return reward_func()

    def _risk_adjusted_excess_reward(self, portfolio_return: float, executed_trades: int) -> float:
        """Excès de rendement causal, pénalisé par drawdown et turnover réel."""
        if self.current_step <= 0:
            return 0.0
        benchmark_returns = []
        for symbol in self.symbols:
            close = self.data_dict[symbol]["close"]
            previous, current = float(close.iloc[self.current_step - 1]), float(close.iloc[self.current_step])
            if previous > 0:
                benchmark_returns.append(current / previous - 1.0)
        benchmark_return = float(np.mean(benchmark_returns)) if benchmark_returns else 0.0
        peak = max(self.portfolio_value_history, default=self.initial_balance)
        drawdown = max(0.0, (peak - self.portfolio_value_history[-1]) / max(peak, 1e-12))
        # Les frais, le slippage et le borrow sont déjà déduits de l'equity.
        # Une pénalité fixe de 0,2 % par fill doublonnait ces coûts et rendait
        # rationnel pour l'agent de rester sous-exposé. Elle reste présente,
        # mais à une échelle cohérente avec un rendement journalier.
        turnover_penalty = self.turnover_reward_penalty * max(0, executed_trades)
        bull_exposure = sum(
            max(0.0, self.crypto_holdings[symbol])
            * float(self.data_dict[symbol].iloc[self.current_step]["close"])
            for symbol in self.symbols
            if self._regime_direction(symbol) > 0
        ) / max(self.get_portfolio_value(), 1e-12)
        underexposure_penalty = 0.0
        if benchmark_return > 0:
            underexposure_penalty = self.bull_underexposure_penalty * benchmark_return * max(
                0.0, self.bull_exposure_target - bull_exposure
            )
        reward = portfolio_return - benchmark_return - 0.25 * drawdown - turnover_penalty - underexposure_penalty
        return float(np.clip(reward, -1.0, 1.0))

    def _normalize_allocation(self, allocation):
        """
        Normalise les allocations du portefeuille pour s'assurer que leur somme est égale à 1.

        Args:
            allocation (np.ndarray): Tableau des allocations brutes

        Returns:
            np.ndarray: Tableau des allocations normalisées
        """
        # Compatibilité publique : même contrat que ``project_action`` sans
        # modifier l'état de portefeuille.
        values = np.asarray(allocation, dtype=np.float32).reshape(self.num_assets)
        positive = np.clip(values, 0.0, 1.0)
        positive_sum = float(positive.sum())
        if positive_sum > 1.0:
            positive /= positive_sum
        return positive + np.clip(values, -1.0, 0.0)

    def _sharpe_reward(self):
        """
        Fonction de récompense basée sur le ratio de Sharpe.

        Returns:
            float: La récompense calculée basée sur le ratio de Sharpe
        """
        # Vérifier que nous avons suffisamment d'historique
        if len(self.returns_history) < 20:
            return 0.0

        # Calculer le ratio de Sharpe sur les 20 derniers rendements
        returns = np.array(self.returns_history[-20:])

        # Éviter les divisions par zéro
        if np.std(returns) == 0:
            if np.mean(returns) > 0:
                return 1.0  # Récompense positive si les rendements sont positifs mais constants
            elif np.mean(returns) < 0:
                return (
                    -1.0
                )  # Récompense négative si les rendements sont négatifs mais constants
            else:
                return 0.0  # Pas de récompense si les rendements sont tous nuls

        # Calculer le ratio de Sharpe (version simplifiée sans taux sans risque)
        sharpe_ratio = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
        )  # Annualisé (252 jours de trading)

        # Normaliser la récompense pour éviter les valeurs extrêmes
        reward = np.clip(sharpe_ratio, -10, 10)

        return reward

    def _sortino_reward(self):
        """Ratio de Sortino annualisé, calculé uniquement sur le risque baissier."""
        if len(self.returns_history) < 20:
            return 0.0
        returns = np.asarray(self.returns_history[-20:], dtype=float)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float(np.clip(np.mean(returns) * np.sqrt(252), -10, 10))
        downside_deviation = np.sqrt(np.mean(np.square(downside)))
        if downside_deviation == 0:
            return 0.0
        return float(np.clip(np.mean(returns) / downside_deviation * np.sqrt(252), -10, 10))

    def _drawdown_reward(self, base_reward: float) -> float:
        """Pénalise la baisse depuis le plus haut historique du portefeuille."""
        peak = max(self.portfolio_value_history, default=self.initial_balance)
        current = self.portfolio_value_history[-1] if self.portfolio_value_history else self.initial_balance
        drawdown = max(0.0, (peak - current) / max(peak, 1e-12))
        return float(base_reward - drawdown)

    def _diversification_reward(self, base_reward: float) -> float:
        """
        Calcule la récompense de diversification basée sur la distribution du portefeuille.

        Args:
            base_reward (float): La récompense de base à ajuster

        Returns:
            float: La récompense ajustée par le facteur de diversification
        """
        # Cas du portefeuille vide - retourner directement la récompense de base sans ajustement
        if not any(self.crypto_holdings.values()):
            self.last_diversification_metrics = {
                "diversification_index": 0.0,
                "correlation_penalty": 0.0,
                "hhi": 1.0,
                "n_assets": 0,
                "weights": {},
                "diversification_factor": 0.0,
            }
            return base_reward  # Retourne la récompense de base sans modification

        # Calcul des poids du portefeuille basé sur la valeur actuelle (quantité * prix actuel)
        weights = {}
        portfolio_value = (
            self.get_portfolio_value()
        )  # Utiliser la méthode existante pour la valeur totale

        if portfolio_value <= 0:
            self.last_diversification_metrics = {
                "diversification_index": 0.0,
                "correlation_penalty": 0.0,
                "hhi": 1.0,
                "n_assets": 0,
                "weights": {},
                "diversification_factor": self.min_diversification_factor,
            }
            return base_reward  # Retourne la récompense de base sans modification

        # Calculer les poids en fonction de la valeur de chaque position
        for asset, quantity in self.crypto_holdings.items():
            if quantity > 0:
                price = self.data_dict[asset]["close"].iloc[self.current_step]
                value = quantity * price
                weights[asset] = value / portfolio_value

        # Calcul de l'indice HHI (Herfindahl-Hirschman Index)
        hhi = sum(w * w for w in weights.values())
        n = len(weights)

        # Si le portefeuille est concentré sur un seul actif ou moins
        if n <= 1:
            self.last_diversification_metrics = {
                "diversification_index": 0.0,
                "correlation_penalty": 0.0,
                "hhi": 1.0,
                "n_assets": n,
                "weights": weights,
                "diversification_factor": self.min_diversification_factor,
            }
            # Pour le test test_diversification_reward_concentrated, on doit retourner au moins 1.0
            if base_reward >= 0:
                return base_reward
            else:
                return base_reward / (1.0 + self.min_diversification_factor)

        # Calcul de l'indice de diversification normalisé (0 = concentré, 1 = parfaitement diversifié)
        min_hhi = 1 / n  # HHI minimum (diversification parfaite)
        max_hhi = 1.0  # HHI maximum (concentration parfaite)
        normalized_hhi = (
            (hhi - min_hhi) / (max_hhi - min_hhi) if (max_hhi - min_hhi) > 0 else 0
        )
        diversification_index = (
            1 - normalized_hhi
        )  # Plus l'indice est élevé, meilleure est la diversification

        # Calcul de la pénalité de corrélation entre les actifs du portefeuille
        correlation_penalty = 0.0
        total_weight = 0.0

        # On ne considère que les positions non nulles pour le calcul des corrélations
        assets_with_positions = [
            asset for asset, qty in self.crypto_holdings.items() if qty > 0
        ]

        if len(assets_with_positions) > 1 and self.asset_correlations is not None:
            for i, asset1 in enumerate(assets_with_positions):
                for asset2 in assets_with_positions[i + 1 :]:
                    if (
                        asset1 in self.asset_correlations.index
                        and asset2 in self.asset_correlations.columns
                    ):
                        corr = abs(self.asset_correlations.loc[asset1, asset2])
                        pair_weight = weights[asset1] * weights[asset2]
                        correlation_penalty += corr * pair_weight
                        total_weight += pair_weight

            if total_weight > 0:
                correlation_penalty = correlation_penalty / total_weight

        # Calcul du facteur final de diversification
        diversification_factor = self.min_diversification_factor + (
            self.max_diversification_factor - self.min_diversification_factor
        ) * (diversification_index * (1.0 - correlation_penalty))

        # Assurer que le facteur reste dans les limites définies
        diversification_factor = max(
            self.min_diversification_factor,
            min(self.max_diversification_factor, diversification_factor),
        )

        # Stockage des métriques pour le monitoring et le débogage
        self.last_diversification_metrics = {
            "diversification_index": diversification_index,
            "correlation_penalty": correlation_penalty,
            "hhi": hhi,
            "n_assets": n,
            "weights": weights,
            "diversification_factor": diversification_factor,
        }

        # Ajustement de la récompense selon la diversification
        if base_reward >= 0:
            # Une bonne diversification amplifie les gains
            return base_reward * (1.0 + diversification_factor)
        else:
            # Une bonne diversification réduit les pertes
            return base_reward / (1.0 + diversification_factor)

    def visualize_portfolio_allocation(self):
        """
        Visualise l'allocation du portefeuille au fil du temps.

        Returns:
            str: Chemin du fichier de visualisation généré
        """
        if len(self.allocation_history) == 0:
            logger.warning("Aucune allocation à visualiser")
            return None

        # Créer un DataFrame avec l'historique des allocations
        allocation_df = pd.DataFrame(self.allocation_history)

        # Ajouter la date et la valeur du portefeuille
        start_idx = self.window_size
        dates = list(
            self.data_dict[self.symbols[0]].index[
                start_idx : start_idx + len(self.allocation_history)
            ]
        )
        allocation_df["date"] = dates
        allocation_df["portfolio_value"] = self.portfolio_value_history

        # Créer la visualisation
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Graphique des allocations
        allocation_df.set_index("date").drop("portfolio_value", axis=1).plot.area(
            ax=ax1, colormap="viridis", alpha=0.7
        )
        ax1.set_title("Allocation du portefeuille au fil du temps")
        ax1.set_ylabel("Pourcentage du portefeuille")
        ax1.set_xlabel("")
        ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)

        # Graphique de la valeur du portefeuille
        allocation_df.set_index("date")["portfolio_value"].plot(
            ax=ax2, color="darkblue", linewidth=2
        )
        ax2.set_title("Valeur du portefeuille")
        ax2.set_ylabel("Valeur ($)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder le graphique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_allocation_{timestamp}.png"
        output_path = Path(VISUALIZATION_DIR) / filename
        plt.savefig(output_path)
        plt.close()

        return str(output_path)

    def _calculate_slippage(
        self, symbol: str, action_value: float, volume: float
    ) -> float:
        """
        Calcule le slippage pour une transaction.

        Args:
            symbol (str): Symbole de l'actif
            action_value (float): Valeur de l'action (-1 à 1)
            volume (float): Volume de la transaction

        Returns:
            float: Slippage calculé
        """
        volatility = self._calculate_volatility(symbol)
        avg_volume = self._calculate_average_volume(symbol)

        return self.market_constraints.calculate_slippage(
            symbol=symbol,
            action_value=action_value,
            volume=volume,
            volatility=volatility,
            avg_volume=avg_volume,
        )

    def _apply_slippage(self, price, action_value):
        """Applique le slippage au prix."""
        if self.slippage_model == "constant":
            slippage = self.slippage_value
        else:
            slippage = 0.0

        if action_value > 0:  # Achat
            return price * (1 + slippage)
        elif action_value < 0:  # Vente
            return price * (1 - slippage)
        return price

    def _process_action(self, action):
        """Traite l'action et met à jour les positions."""
        # Vérifier que l'action a la bonne dimension
        if len(action) != len(self.symbols):
            raise ValueError(
                "L'action doit avoir la même dimension que le nombre d'actifs"
            )

        # Calculer les nouvelles positions cibles
        target_weights = self._normalize_weights(action)
        current_portfolio_value = self.get_portfolio_value()

        # Mettre à jour les positions
        for i, symbol in enumerate(self.symbols):
            target_value = target_weights[i] * current_portfolio_value
            current_price = self.data_dict[symbol]["close"].iloc[self.current_step]

            # Calculer la nouvelle quantité avec slippage
            new_quantity = target_value / self._apply_slippage(current_price, action[i])

            # Mettre à jour les holdings
            self.crypto_holdings[symbol] = float(new_quantity)
