import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from ai_trading.rl.trading_environment import TradingEnvironment
from ai_trading.rl.portfolio_allocator import PortfolioAllocator

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

# Définir le chemin pour les visualisations
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualizations', 'multi_asset_env')
# Créer le répertoire s'il n'existe pas
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

class MultiAssetTradingEnvironment(gym.Env):
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
        rebalance_frequency=5,      # Fréquence de rééquilibrage (en pas de temps)
        max_active_positions=3,     # Nombre maximum de positions actives simultanées
        action_type="continuous",   # Pour le multi-actifs, on utilise des actions continues
        **kwargs
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
        """
        super(MultiAssetTradingEnvironment, self).__init__()
        
        # Validation des paramètres
        assert isinstance(data_dict, dict) and len(data_dict) > 0, "Le dictionnaire de données doit contenir au moins un actif"
        assert all(len(df) > window_size for df in data_dict.values()), f"Tous les DataFrames doivent contenir plus de {window_size} points de données"
        assert initial_balance > 0, "Le solde initial doit être positif"
        assert 0 <= transaction_fee < 1, "Les frais de transaction doivent être entre 0 et 1"
        assert reward_function in ["simple", "sharpe", "transaction_penalty", "drawdown"], "Fonction de récompense invalide"
        assert allocation_method in ["equal", "volatility", "momentum", "smart"], "Méthode d'allocation invalide"
        assert action_type == "continuous", "Pour le trading multi-actifs, seul le type d'action 'continuous' est supporté"
        
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
        
        # Aligner toutes les données sur les mêmes dates
        self._align_data()
        
        # Initialiser le portfolio allocator
        self.portfolio_allocator = PortfolioAllocator(
            method=allocation_method,
            max_active_positions=self.max_active_positions
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
        
        # Initialiser l'espace d'action: allocation pour chaque actif (-1 à 1 pour chaque actif)
        # -1: vendre 100%, 0: ne rien faire, 1: acheter 100%
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(self.num_assets,), 
            dtype=np.float32
        )
        
        # Réinitialiser l'environnement pour calculer la taille réelle de l'état
        temp_reset = self.reset()
        if isinstance(temp_reset, tuple):
            temp_state = temp_reset[0]  # Pour la compatibilité avec les nouvelles versions de gym
        else:
            temp_state = temp_reset
        
        # Définir l'espace d'observation avec la taille réelle de l'état
        real_state_size = temp_state.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(real_state_size,), 
            dtype=np.float32
        )
        
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
                raise ValueError(f"Après alignement, les données pour {symbol} sont insuffisantes (il faut au moins {self.window_size+1} points)")
        
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
        self.last_prices = {symbol: self.data_dict[symbol].iloc[self.current_step]["close"] for symbol in self.symbols}
        self.portfolio_value_history = [self.initial_balance]
        self.returns_history = []
        self.allocation_history = []
        self.steps_since_rebalance = 0
        
        # Obtenir l'observation initiale
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: Vecteur d'actions pour chaque actif (entre -1 et 1)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Vérifier que l'action est valide
        if not self.action_space.contains(action):
            action = np.clip(action, -1, 1)
            logger.warning(f"Action hors limites, clippée à {action}")
        
        # Sauvegarder l'état précédent pour calculer la récompense
        previous_portfolio_value = self.get_portfolio_value()
        
        # Incrémenter le compteur de pas depuis le dernier rééquilibrage
        self.steps_since_rebalance += 1
        
        # Appliquer l'action (allocation du portefeuille)
        self._apply_allocation(action)
        
        # Passer à l'étape suivante
        self.current_step += 1
        
        # Mettre à jour les derniers prix connus
        for symbol in self.symbols:
            self.last_prices[symbol] = self.data_dict[symbol].iloc[self.current_step]["close"]
        
        # Vérifier si l'épisode est terminé
        done = self.current_step >= min(len(df) for df in self.data_dict.values()) - 1
        
        # Calculer la valeur actuelle du portefeuille
        current_portfolio_value = self.get_portfolio_value()
        
        # Calculer la récompense
        reward = self._calculate_reward(previous_portfolio_value, current_portfolio_value)
        
        # Enregistrer la valeur du portefeuille
        self.portfolio_value_history.append(current_portfolio_value)
        
        # Calculer le rendement
        pct_change = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value if previous_portfolio_value > 0 else 0
        self.returns_history.append(pct_change)
        
        # Obtenir l'observation et les informations
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info
    
    def _apply_allocation(self, action):
        """
        Applique l'allocation du portefeuille basée sur l'action.
        
        Args:
            action: Vecteur d'actions pour chaque actif (entre -1 et 1)
        """
        # Si c'est le moment de rééquilibrer ou si c'est la première action
        if self.steps_since_rebalance >= self.rebalance_frequency or len(self.allocation_history) == 0:
            # Réinitialiser le compteur
            self.steps_since_rebalance = 0
            
            # Obtenir la valeur totale du portefeuille
            portfolio_value = self.get_portfolio_value()
            
            # Normaliser l'action pour obtenir des poids d'allocation (entre 0 et 1)
            # Convertir l'action de [-1, 1] à [0, 1]
            normalized_action = (action + 1) / 2.0
            
            # Utiliser le portfolio allocator pour ajuster les poids si nécessaire
            allocation_weights = self.portfolio_allocator.allocate(
                action_weights=normalized_action,
                symbols=self.symbols,
                prices={s: self.data_dict[s].iloc[self.current_step]["close"] for s in self.symbols},
                volatilities={s: self.data_dict[s].iloc[self.current_step-20:self.current_step]["close"].pct_change().std() * np.sqrt(252) for s in self.symbols},
                returns={s: self.data_dict[s].iloc[self.current_step-5:self.current_step]["close"].pct_change().mean() * 252 for s in self.symbols}
            )
            
            # Calculer la valeur cible pour chaque actif
            target_values = {symbol: portfolio_value * weight for symbol, weight in allocation_weights.items()}
            
            # Calculer les différences avec les allocations actuelles
            current_values = {
                symbol: self.crypto_holdings[symbol] * self.data_dict[symbol].iloc[self.current_step]["close"]
                for symbol in self.symbols
            }
            
            # Exécuter les trades nécessaires pour atteindre l'allocation cible
            for symbol in self.symbols:
                current_price = self.data_dict[symbol].iloc[self.current_step]["close"]
                target_value = target_values[symbol]
                current_value = current_values[symbol]
                
                # Calculer la différence à trader
                value_difference = target_value - current_value
                
                # Si la différence est significative (>1% du portefeuille)
                if abs(value_difference) > portfolio_value * 0.01:
                    if value_difference > 0:  # Acheter
                        # Calculer la quantité à acheter
                        amount_to_buy = value_difference / (current_price * (1 + self.transaction_fee))
                        # Vérifier si nous avons assez de balance
                        if amount_to_buy * current_price * (1 + self.transaction_fee) <= self.balance:
                            self.crypto_holdings[symbol] += amount_to_buy
                            self.balance -= amount_to_buy * current_price * (1 + self.transaction_fee)
                            logger.debug(f"Achat: {amount_to_buy:.6f} unités de {symbol} à {current_price:.2f}")
                        else:
                            # Acheter autant que possible avec la balance disponible
                            max_amount = self.balance / (current_price * (1 + self.transaction_fee))
                            self.crypto_holdings[symbol] += max_amount
                            self.balance = 0
                            logger.debug(f"Achat partiel: {max_amount:.6f} unités de {symbol} à {current_price:.2f} (balance insuffisante)")
                    else:  # Vendre
                        # Calculer la quantité à vendre
                        amount_to_sell = abs(value_difference) / current_price
                        # Vérifier si nous avons assez de crypto
                        if amount_to_sell <= self.crypto_holdings[symbol]:
                            self.crypto_holdings[symbol] -= amount_to_sell
                            self.balance += amount_to_sell * current_price * (1 - self.transaction_fee)
                            logger.debug(f"Vente: {amount_to_sell:.6f} unités de {symbol} à {current_price:.2f}")
                        else:
                            # Vendre tout ce que nous avons
                            amount_to_sell = self.crypto_holdings[symbol]
                            self.balance += amount_to_sell * current_price * (1 - self.transaction_fee)
                            self.crypto_holdings[symbol] = 0
                            logger.debug(f"Vente complète: {amount_to_sell:.6f} unités de {symbol} à {current_price:.2f}")
            
            # Enregistrer l'allocation
            self.allocation_history.append(allocation_weights)
    
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
            price_window = self.data_dict[symbol].iloc[self.current_step-self.window_size:self.current_step]["close"].values
            price_window = price_window / price_window[0]  # Normaliser par le premier prix
            obs_components.append(price_window)
            
            # Volume récent (si disponible)
            if "volume" in self.data_dict[symbol].columns:
                volume_window = self.data_dict[symbol].iloc[self.current_step-self.window_size:self.current_step]["volume"].values
                volume_window = volume_window / (volume_window.max() if volume_window.max() > 0 else 1)  # Normaliser
                obs_components.append(volume_window)
            
            # Indicateurs techniques (si activés)
            if self.include_technical_indicators:
                # Ajouter les principaux indicateurs techniques pour cet actif
                from ai_trading.rl.technical_indicators import TechnicalIndicators
                indicators = TechnicalIndicators(self.data_dict[symbol].iloc[:self.current_step])
                rsi = indicators.calculate_rsi()
                if rsi is not None and len(rsi) > 0:
                    rsi_value = rsi.iloc[-1] / 100.0  # Normaliser entre 0 et 1
                    obs_components.append(np.array([rsi_value]))
                
                macd, signal, hist = indicators.calculate_macd()
                if macd is not None and len(macd) > 0:
                    # Normaliser MACD et signal par la plage typique
                    macd_range = 20.0  # Valeur typique pour la plage du MACD
                    macd_value = (macd.iloc[-1] + macd_range) / (2 * macd_range)  # Normaliser entre 0 et 1
                    signal_value = (signal.iloc[-1] + macd_range) / (2 * macd_range)  # Normaliser entre 0 et 1
                    obs_components.append(np.array([macd_value, signal_value]))
            
            # Position actuelle pour cet actif
            if self.include_position:
                position_value = self.crypto_holdings[symbol] * self.data_dict[symbol].iloc[self.current_step]["close"] / self.initial_balance
                obs_components.append(np.array([position_value]))
        
        # Ajouter la balance et la valeur totale du portefeuille
        if self.include_balance:
            balance_normalized = self.balance / self.initial_balance
            portfolio_value_normalized = self.get_portfolio_value() / self.initial_balance
            obs_components.append(np.array([balance_normalized, portfolio_value_normalized]))
        
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
            symbol: self.crypto_holdings[symbol] * self.data_dict[symbol].iloc[self.current_step]["close"]
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
            "current_prices": {symbol: self.data_dict[symbol].iloc[self.current_step]["close"] for symbol in self.symbols},
            "steps_since_rebalance": self.steps_since_rebalance
        }
    
    def get_portfolio_value(self):
        """
        Calcule la valeur totale du portefeuille (balance + valeur des actifs détenus).
        
        Returns:
            float: Valeur totale du portefeuille
        """
        assets_value = sum(
            self.crypto_holdings[symbol] * self.data_dict[symbol].iloc[self.current_step]["close"]
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
        pct_change = (current_value - previous_value) / previous_value if previous_value > 0 else 0
        
        # Choisir la fonction de récompense appropriée
        if self.reward_function == "simple":
            return pct_change
        elif self.reward_function == "sharpe":
            return self._sharpe_reward()
        elif self.reward_function == "transaction_penalty":
            # TODO: Implement transaction penalty reward for multi-asset
            return pct_change
        elif self.reward_function == "drawdown":
            # TODO: Implement drawdown reward for multi-asset
            return pct_change
        else:
            return pct_change
    
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
                return -1.0  # Récompense négative si les rendements sont négatifs mais constants
            else:
                return 0.0  # Pas de récompense si les rendements sont tous nuls
        
        # Calculer le ratio de Sharpe (version simplifiée sans taux sans risque)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualisé (252 jours de trading)
        
        # Normaliser la récompense pour éviter les valeurs extrêmes
        reward = np.clip(sharpe_ratio, -10, 10)
        
        return reward
    
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
        dates = list(self.data_dict[self.symbols[0]].index[start_idx:start_idx+len(self.allocation_history)])
        allocation_df['date'] = dates
        allocation_df['portfolio_value'] = self.portfolio_value_history
        
        # Créer la visualisation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Graphique des allocations
        allocation_df.set_index('date').drop('portfolio_value', axis=1).plot.area(ax=ax1, colormap='viridis', alpha=0.7)
        ax1.set_title('Allocation du portefeuille au fil du temps')
        ax1.set_ylabel('Pourcentage du portefeuille')
        ax1.set_xlabel('')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)
        
        # Graphique de la valeur du portefeuille
        allocation_df.set_index('date')['portfolio_value'].plot(ax=ax2, color='darkblue', linewidth=2)
        ax2.set_title('Valeur du portefeuille')
        ax2.set_ylabel('Valeur ($)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"portfolio_allocation_{timestamp}.png"
        output_path = os.path.join(VISUALIZATION_DIR, filename)
        plt.savefig(output_path)
        plt.close()
        
        return output_path 