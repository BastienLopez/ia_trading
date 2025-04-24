import logging
import os
from typing import Dict, List
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from datetime import datetime

from ai_trading.rl.portfolio_allocator import PortfolioAllocator
from ai_trading.config import VISUALIZATION_DIR
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
            "transaction_penalty",
            "drawdown",
            "diversification"
        ], "Fonction de récompense invalide"
        assert allocation_method in [
            "equal",
            "volatility",
            "momentum",
            "smart",
        ], "Méthode d'allocation invalide"
        assert (
            action_type == "continuous"
        ), "Pour le trading multi-actifs, seul le type d'action 'continuous' est supporté"

        # Stockage des paramètres
        self.data_dict = data_dict
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
        self.max_active_positions = min(max_active_positions, self.num_assets)
        self.action_type = action_type
        self.slippage_model = slippage_model
        self.base_slippage = base_slippage
        self.execution_delay = execution_delay
        self.market_impact_factor = market_impact_factor
        self.correlation_threshold = correlation_threshold
        self.volatility_threshold = volatility_threshold

        # Aligner toutes les données sur les mêmes dates
        self._align_data()

        # Initialiser le portfolio allocator
        self.portfolio_allocator = PortfolioAllocator(
            method=allocation_method, max_active_positions=self.max_active_positions
        )

        # Initialiser les variables de l'environnement
        self.balance = initial_balance
        self.crypto_holdings = {symbol: 0.0 for symbol in self.symbols}
        self.last_prices = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value_history = []
        self.returns_history = []
        self.allocation_history = []
        self.current_step = self.window_size
        self.steps_since_rebalance = 0
        self.active_assets = self.symbols[:self.max_active_positions]
        self.pending_orders = []

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

        # Ajouter les paramètres de gestion de portefeuille
        self.asset_correlations = self._calculate_asset_correlations()
        self.asset_volatilities = self._calculate_asset_volatilities()

        # Initialisation des contraintes de marché
        self.market_constraints = MarketConstraints(
            slippage_model=slippage_model,
            base_slippage=base_slippage,
            execution_delay=execution_delay,
            market_impact_factor=market_impact_factor
        )
        
        # Historique des impacts marché
        self.market_impacts = {symbol: [] for symbol in self.symbols}

        logger.info(
            f"Environnement de trading multi-actifs initialisé avec {self.num_assets} actifs: {', '.join(self.symbols)}"
        )

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
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_holdings = {symbol: 0.0 for symbol in self.symbols}
        self.last_prices = {
            symbol: self.data_dict[symbol].iloc[self.current_step]["close"]
            for symbol in self.symbols
        }
        self.portfolio_value_history = [self.initial_balance]
        self.returns_history = []
        self.allocation_history = []
        self.steps_since_rebalance = 0
        self.active_assets = self.symbols[:self.max_active_positions]
        self.pending_orders = []

        # Obtenir l'observation initiale
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Exécute une étape de trading.
        
        Args:
            action: Action de trading pour chaque actif
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Sauvegarder la valeur précédente du portefeuille
        previous_value = self.get_portfolio_value()
        
        # Traitement des ordres en attente
        self._process_pending_orders()
        
        # Création des nouveaux ordres
        for i, symbol in enumerate(self.symbols):
            if abs(action[i]) > 1e-6:  # Seulement si l'action n'est pas nulle
                volume = abs(action[i]) * self.balance / self.last_prices[symbol]
                
                # Calculer l'impact marché
                impact, recovery_time = self.market_constraints.calculate_market_impact(
                    symbol=symbol,
                    action_value=action[i],
                    volume=volume,
                    price=self.last_prices[symbol],
                    avg_volume=self._calculate_average_volume(symbol)
                )
                
                # Enregistrer l'impact marché
                if symbol not in self.market_impacts:
                    self.market_impacts[symbol] = []
                self.market_impacts[symbol].append({
                    'step': self.current_step,
                    'impact': impact,
                    'recovery_time': recovery_time
                })
                
                delay = self.market_constraints.calculate_execution_delay(
                    symbol, action[i], volume, self._calculate_average_volume(symbol)
                )
                if delay > 0:
                    self.pending_orders.append({
                        'symbol': symbol,
                        'action_value': action[i],
                        'volume': volume,
                        'price': self.last_prices[symbol],
                        'delay': delay
                    })
                else:
                    # Exécution immédiate si pas de délai
                    slippage = self._calculate_slippage(symbol, action[i], volume)
                    self._execute_trade(symbol, action[i], volume, 
                                     self.last_prices[symbol], 
                                     slippage)
        
        # Mettre à jour l'étape courante et les données
        self.current_step += 1
        self._update_orderbook_data()
        
        # Calculer la nouvelle valeur du portefeuille et la récompense
        current_value = self.get_portfolio_value()
        reward = self._calculate_reward(previous_value, current_value)
        
        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.data_dict[self.symbols[0]]) - 1
        
        return self._get_observation(), reward, done, self._get_info()

    def _process_pending_orders(self):
        """
        Traite les ordres en attente.
        """
        remaining_orders = []
        for order in self.pending_orders:
            order['delay'] -= 1
            if order['delay'] <= 0:
                # Exécution de l'ordre
                slippage = self._calculate_slippage(
                    order['symbol'], 
                    order['action_value'], 
                    order['volume']
                )
                self._execute_trade(
                    order['symbol'],
                    order['action_value'],
                    order['volume'],
                    self.last_prices[order['symbol']],
                    slippage
                )
            else:
                remaining_orders.append(order)
        self.pending_orders = remaining_orders

    def _execute_trade(self, symbol: str, action_value: float, volume: float, price: float, slippage: float):
        """
        Exécute une transaction.
        
        Args:
            symbol (str): Symbole de l'actif
            action_value (float): Valeur de l'action (-1 à 1)
            volume (float): Volume à trader
            price (float): Prix de base
            slippage (float): Slippage calculé
        """
        # Appliquer le slippage au prix
        execution_price = price * (1 + slippage if action_value > 0 else 1 - slippage)
        
        if action_value > 0:  # Achat
            cost = volume * execution_price * (1 + self.transaction_fee)
            if cost <= self.balance:
                self.balance -= cost
                self.crypto_holdings[symbol] += volume
        else:  # Vente
            revenue = volume * execution_price * (1 - self.transaction_fee)
            if volume <= self.crypto_holdings[symbol]:
                self.balance += revenue
                self.crypto_holdings[symbol] -= volume

    def _calculate_volatility(self, symbol: str) -> float:
        """Calcule la volatilité sur la fenêtre d'observation."""
        prices = self.data_dict[symbol].iloc[max(0, self.current_step - self.window_size):self.current_step]['close']
        return np.std(np.log(prices / prices.shift(1)).dropna())

    def _calculate_average_volume(self, symbol: str) -> float:
        """Calcule le volume moyen sur la fenêtre d'observation."""
        volumes = self.data_dict[symbol].iloc[max(0, self.current_step - self.window_size):self.current_step]['volume']
        return np.mean(volumes)

    def _update_orderbook_data(self):
        """Met à jour les données de profondeur du carnet d'ordres."""
        for symbol in self.active_assets:
            if 'orderbook_depth' in self.data_dict[symbol].columns:
                depth_data = self.data_dict[symbol].iloc[self.current_step]['orderbook_depth']
                self.market_constraints.update_orderbook_depth(symbol, depth_data)

    def _calculate_asset_correlations(self):
        """
        Calcule la matrice de corrélation entre les actifs sur une fenêtre glissante.
        
        Returns:
            pd.DataFrame: Matrice de corrélation entre les actifs
        """
        # Créer un DataFrame avec les prix de clôture des derniers jours
        window_size = min(30, self.current_step)  # Utiliser au maximum 30 jours d'historique
        prices_data = {}
        
        for symbol in self.symbols:
            prices = self.data_dict[symbol].iloc[max(0, self.current_step - window_size):self.current_step + 1]["close"]
            prices_data[symbol] = prices
            
        prices_df = pd.DataFrame(prices_data)
        
        # Calculer les rendements journaliers
        returns_df = prices_df.pct_change().dropna()
        
        # Calculer la matrice de corrélation
        correlation_matrix = returns_df.corr()
        
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
            asset for asset in self.symbols
            if self.asset_volatilities[asset] <= self.volatility_threshold
        ]
        
        if not low_volatility_assets:
            return self.symbols[:self.max_active_positions]
        
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

            # Indicateurs techniques (si activés)
            if self.include_technical_indicators:
                # Ajouter les principaux indicateurs techniques pour cet actif
                from ai_trading.rl.technical_indicators import TechnicalIndicators

                indicators = TechnicalIndicators(
                    self.data_dict[symbol].iloc[: self.current_step]
                )
                rsi = indicators.calculate_rsi()
                if rsi is not None and len(rsi) > 0:
                    rsi_value = rsi.iloc[-1] / 100.0  # Normaliser entre 0 et 1
                    obs_components.append(np.array([rsi_value]))

                macd, signal, hist = indicators.calculate_macd()
                if macd is not None and len(macd) > 0:
                    # Normaliser MACD et signal par la plage typique
                    macd_range = 20.0  # Valeur typique pour la plage du MACD
                    macd_value = (macd.iloc[-1] + macd_range) / (
                        2 * macd_range
                    )  # Normaliser entre 0 et 1
                    signal_value = (signal.iloc[-1] + macd_range) / (
                        2 * macd_range
                    )  # Normaliser entre 0 et 1
                    obs_components.append(np.array([macd_value, signal_value]))

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
        }

    def get_portfolio_value(self):
        """
        Calcule la valeur totale du portefeuille (balance + valeur des actifs détenus).

        Returns:
            float: Valeur totale du portefeuille
        """
        assets_value = sum(
            self.crypto_holdings[symbol]
            * self.data_dict[symbol].iloc[self.current_step]["close"]
            for symbol in self.symbols
        )
        return self.balance + assets_value

    def _calculate_reward(self, previous_value, current_value):
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

        # Choisir la fonction de récompense appropriée
        reward_functions = {
            "simple": lambda: pct_change,
            "sharpe": self._sharpe_reward,
            "diversification": lambda: self._diversification_reward(pct_change),
            "transaction_penalty": lambda: pct_change,  # TODO: Implement
            "drawdown": lambda: pct_change,  # TODO: Implement
        }

        # Obtenir et exécuter la fonction de récompense
        reward_func = reward_functions.get(self.reward_function, lambda: pct_change)
        return reward_func()

    def _normalize_allocation(self, allocation):
        """
        Normalise les allocations du portefeuille pour s'assurer que leur somme est égale à 1.

        Args:
            allocation (np.ndarray): Tableau des allocations brutes

        Returns:
            np.ndarray: Tableau des allocations normalisées
        """
        # Gérer le cas où toutes les allocations sont nulles
        if np.sum(np.abs(allocation)) == 0:
            return np.zeros_like(allocation)
        
        # Normaliser les allocations positives et négatives séparément
        positive_mask = allocation > 0
        negative_mask = allocation < 0
        
        positive_sum = np.sum(allocation[positive_mask])
        negative_sum = np.abs(np.sum(allocation[negative_mask]))
        
        normalized_allocation = np.zeros_like(allocation)
        
        if positive_sum > 0:
            normalized_allocation[positive_mask] = allocation[positive_mask] / positive_sum
        if negative_sum > 0:
            normalized_allocation[negative_mask] = allocation[negative_mask] / negative_sum
        
        return normalized_allocation

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

    def _diversification_reward(self, base_reward):
        """
        Calcule une récompense qui encourage la diversification du portefeuille.
        
        Args:
            base_reward (float): La récompense de base (variation du portefeuille)
            
        Returns:
            float: La récompense ajustée selon la diversification
        """
        # Paramètres de configuration pour la diversification
        diversification_weight = getattr(self, 'diversification_weight', 0.5)  # Poids relatif de la diversification
        min_diversification_factor = getattr(self, 'min_diversification_factor', 0.2)  # Facteur minimum
        max_diversification_factor = getattr(self, 'max_diversification_factor', 2.0)  # Facteur maximum
        correlation_penalty_weight = getattr(self, 'correlation_penalty_weight', 1.0)  # Poids de la pénalité de corrélation
        
        # Obtenir les allocations actuelles (en excluant le cash)
        current_allocations = np.array([self.crypto_holdings[symbol] * self.data_dict[symbol].iloc[self.current_step]["close"] 
                                      for symbol in self.symbols])
        total_value = np.sum(current_allocations)
        
        if total_value == 0:
            return base_reward
            
        # Calculer les poids normalisés
        weights = current_allocations / total_value
        
        # Calculer l'indice de diversification (1 - somme des carrés des poids)
        # Plus l'indice est proche de 1, plus le portefeuille est diversifié
        diversification_index = 1 - np.sum(weights ** 2)
        
        # Calculer les corrélations entre les actifs
        correlations = self._calculate_asset_correlations()
        
        # Calculer la moyenne des corrélations absolues (excluant la diagonale)
        mask = ~np.eye(correlations.shape[0], dtype=bool)
        mean_correlation = np.abs(correlations.values[mask]).mean()
        
        # Pénalité de corrélation (plus forte pour les corrélations élevées)
        correlation_penalty = mean_correlation * correlation_penalty_weight
        
        # Facteur de diversification avec pénalité de corrélation
        diversification_factor = (diversification_index * (1 - correlation_penalty)) * diversification_weight
        
        # Appliquer les seuils min/max
        diversification_factor = np.clip(diversification_factor, min_diversification_factor, max_diversification_factor)
        
        # Ajuster la récompense
        # Si le portefeuille est bien diversifié (facteur proche de max_diversification_factor), 
        # la récompense est amplifiée
        adjusted_reward = base_reward * (1 + diversification_factor)
        
        # Ajouter les métriques aux informations de l'environnement
        self.last_diversification_metrics = {
            'diversification_index': diversification_index,
            'mean_correlation': mean_correlation,
            'correlation_penalty': correlation_penalty,
            'diversification_factor': diversification_factor,
            'adjusted_reward': adjusted_reward
        }
        
        return adjusted_reward

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

    def _calculate_slippage(self, symbol: str, action_value: float, volume: float) -> float:
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
            avg_volume=avg_volume
        )
