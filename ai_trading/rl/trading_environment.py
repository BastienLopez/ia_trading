import logging

import gymnasium as gym
import numpy as np
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

# Ajouter l'import de la classe TechnicalIndicators
from .technical_indicators import TechnicalIndicators

# Ajouter l'import
from ai_trading.rl.risk_manager import RiskManager

# Ajouter l'import
from ai_trading.rl.adaptive_normalization import AdaptiveNormalizer

class TradingEnvironment(gym.Env):
    """
    Environnement de trading pour l'apprentissage par renforcement.
    Version améliorée avec actions d'achat/vente partielles.
    """

    def __init__(
        self,
        df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=50,
        include_technical_indicators=True,
        action_type="discrete",
        n_discrete_actions=3,
        max_crypto_purchase_percent=0.3,
        use_risk_manager=True,
        use_adaptive_normalization=True,
        risk_config=None,
    ):
        """
        Initialise l'environnement de trading.

        Args:
            df (DataFrame): Données historiques avec au moins une colonne 'close' pour les prix
            initial_balance (float): Solde initial en USD
            transaction_fee (float): Frais de transaction (pourcentage)
            window_size (int): Nombre de périodes précédentes à inclure dans l'observation
            include_technical_indicators (bool): Inclure des indicateurs techniques dans l'observation
            action_type (str): Type d'espace d'action ('discrete' ou 'continuous')
            n_discrete_actions (int): Nombre d'actions discrètes pour chaque direction (achat/vente)
            max_crypto_purchase_percent (float): Pourcentage maximum du portefeuille à investir en une seule transaction
            use_risk_manager (bool): Utiliser le gestionnaire de risque
            use_adaptive_normalization (bool): Utiliser le normalisateur adaptatif
            risk_config (dict, optional): Configuration du gestionnaire de risques
        """
        super(TradingEnvironment, self).__init__()

        # Validation des paramètres
        if window_size < 1:
            raise ValueError("window_size doit être >= 1")
        if not 0 <= transaction_fee <= 1:
            raise ValueError("transaction_fee doit être entre 0 et 1")

        # Vérifier que le DataFrame contient les colonnes nécessaires
        required_columns = ["close"]
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Le DataFrame doit contenir une colonne '{column}'")

        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.include_technical_indicators = include_technical_indicators
        self.action_type = action_type
        self.n_discrete_actions = n_discrete_actions
        self.max_crypto_purchase_percent = max_crypto_purchase_percent

        # Initialiser le gestionnaire de risque
        self.use_risk_manager = use_risk_manager
        self.risk_manager = RiskManager(config=risk_config) if use_risk_manager else None

        # Initialiser le normalisateur adaptatif
        self.use_adaptive_normalization = use_adaptive_normalization
        if use_adaptive_normalization:
            # Créer une liste de noms de features
            feature_names = ['price', 'volume']
            
            # Ajouter les noms des indicateurs techniques
            indicator_names = [
                'ema9', 'ema21', 'ema50', 'ema200', 
                'macd_line', 'macd_signal', 'macd_histogram',
                'momentum', 'adx', 'plus_di', 'minus_di',
                'upper_bb', 'middle_bb', 'lower_bb', 'atr',
                'stoch_k', 'stoch_d', 'obv', 'volume_avg',
                'mfi', 'rsi', 'cci'
            ]
            feature_names.extend(indicator_names)
            
            # Ajouter les noms des features de sentiment
            sentiment_names = [
                'compound_score', 'positive_score', 'negative_score', 
                'neutral_score', 'sentiment_volume', 'sentiment_change'
            ]
            feature_names.extend(sentiment_names)
            
            # Ajouter les noms des features de portefeuille
            portfolio_names = ['balance', 'crypto_value', 'portfolio_value']
            feature_names.extend(portfolio_names)
            
            self.normalizer = AdaptiveNormalizer(
                window_size=1000,
                method='minmax',
                clip_values=True,
                feature_names=feature_names
            )

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
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            raise ValueError(f"Type d'action non supporté: {action_type}")

        # Mettre à jour la taille de l'espace d'observation
        # window_size (prix) + 18 indicateurs techniques + 3 infos portefeuille
        observation_size = window_size + 18 + 3
        
        # Définir l'espace d'observation
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(observation_size,), dtype=np.float32
        )

        # Réinitialiser l'environnement
        self.reset()

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

    def reset(self, seed=None):
        """
        Réinitialise l'environnement au début d'un épisode.

        Returns:
            observation (np.array): L'état initial
            info (dict): Informations supplémentaires
        """
        # Réinitialiser le générateur aléatoire si un seed est fourni
        if seed is not None:
            super().reset(seed=seed)
            
        # Réinitialiser l'indice de temps
        self.current_step = self.window_size

        # Réinitialiser le portefeuille
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.portfolio_value_history = [self.initial_balance]
        self.action_history = []

        # Obtenir l'observation initiale
        observation = self._get_observation()

        logger.debug(f"Environnement réinitialisé. Observation initiale: {observation}")

        return observation, {}  # Retourner l'observation et un dict info vide

    def step(self, action):
        """
        Exécute une action dans l'environnement.
        
        Args:
            action: Action à exécuter (discrète ou continue selon action_type)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Vérifier que l'action est valide
        if self.action_type == "continuous":
            if isinstance(action, np.ndarray):
                # Convertir l'action en float pour l'espace continu
                action_value = float(action[0])
            else:
                action_value = float(action)
            
            # Créer un tableau numpy pour la vérification de l'espace d'action
            action_for_check = np.array([action_value], dtype=np.float32)
            
            if not self.action_space.contains(action_for_check):
                raise ValueError(f"Action invalide: {action_value}, doit être entre -1 et 1")
        else:
            if not self.action_space.contains(action):
                raise ValueError(f"Action invalide: {action}")
        
        # Initialiser le dictionnaire d'informations dès le début
        info = {
            "action_adjusted": False  # Par défaut, l'action n'est pas ajustée
        }
        
        # Sauvegarder l'état précédent pour calculer la récompense
        previous_portfolio_value = self.get_portfolio_value()
        
        # Appliquer le gestionnaire de risque si activé
        if self.use_risk_manager:
            adjusted_action = self.risk_manager.adjust_action(
                action,
                self.portfolio_value_history[-1],
                self.crypto_held,
                current_price=self.df.iloc[self.current_step]["close"] if not self.df.empty else 0
            )
            if adjusted_action != action:
                action = adjusted_action
                info["action_adjusted"] = True
        
        # Appliquer l'action
        if self.action_type == "discrete":
            self._apply_discrete_action(action)
        else:  # continuous
            if isinstance(action, np.ndarray):
                self._apply_continuous_action(float(action[0]))
            else:
                self._apply_continuous_action(float(action))

        # Passer à l'étape suivante
        self.current_step += 1

        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.df) - 1

        # Calculer la récompense
        current_portfolio_value = self.get_portfolio_value()
        reward = self._calculate_reward(previous_portfolio_value, current_portfolio_value)

        # Enregistrer la valeur du portefeuille
        self.portfolio_value_history.append(current_portfolio_value)

        # Construire l'observation
        observation = self._get_observation()

        # Mettre à jour les informations à la fin
        current_price = self.df.iloc[self.current_step]["close"]
        info.update({
            "portfolio_value": self.get_portfolio_value(),
            "balance": self.balance,
            "crypto_held": self.crypto_held,
            "current_price": current_price
        })
        
        # Vérifier les conditions de stop-loss pour les positions ouvertes
        if self.use_risk_manager and self.crypto_held > 0:
            position_id = f"position_{self.current_step}"
            
            # Mettre à jour le trailing stop si nécessaire
            if self.crypto_held > 0:
                self.risk_manager.update_trailing_stop(
                    position_id, current_price, self.df.iloc[self.current_step]["close"], 'long')
            else:
                self.risk_manager.update_trailing_stop(
                    position_id, current_price, self.df.iloc[self.current_step]["close"], 'short')
            
            # Vérifier si un stop est déclenché
            stop_result = self.risk_manager.check_stop_conditions(
                position_id, current_price, 'long')
            
            if stop_result['stop_triggered']:
                # Fermer la position au prix du stop
                stop_price = stop_result['stop_price']
                self._close_position(stop_price)
                logger.info(f"{stop_result['stop_type']} déclenché à {stop_price}")
        
        return observation, reward, done, False, info

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
            
            # Limiter l'achat à 30% du portefeuille total
            portfolio_value = self.get_portfolio_value()
            max_buy_value = portfolio_value * 0.3
            
            # Calculer la valeur d'achat basée sur le pourcentage
            buy_value = self.balance * buy_percentage
            
            # Appliquer la limite de 30%
            buy_value = min(buy_value, max_buy_value)
            
            # Calculer la quantité de crypto à acheter
            max_crypto_to_buy = buy_value / (current_price * (1 + self.transaction_fee))
            
            # Acheter la quantité calculée
            self.crypto_held += max_crypto_to_buy
            self.balance -= (max_crypto_to_buy * current_price * (1 + self.transaction_fee))
            
            logger.debug(
                f"Achat: {max_crypto_to_buy:.6f} unités à ${current_price:.2f} (limité à 30% du portefeuille)"
            )
            
        elif self.n_discrete_actions < action <= 2 * self.n_discrete_actions:  # Vente
            if self.crypto_held > 0:
                # Calculer le pourcentage de vente (1/n, 2/n, ..., n/n)
                sell_percentage = (action - self.n_discrete_actions) / self.n_discrete_actions
                crypto_to_sell = self.crypto_held * sell_percentage
                
                # Vendre la quantité calculée
                self.balance += (
                    crypto_to_sell * current_price * (1 - self.transaction_fee)
                )
                
                logger.debug(
                    f"Vente: {crypto_to_sell:.6f} unités ({sell_percentage*100:.0f}%) à ${current_price:.2f}"
                )
                
                self.crypto_held -= crypto_to_sell
            else:
                logger.debug("Tentative de vente sans crypto détenue")

    def _apply_continuous_action(self, action):
        """
        Applique une action continue.
        
        Args:
            action (float): Valeur de l'action entre -1 et 1
        """
        # Obtenir le prix actuel
        current_price = self.df.iloc[self.current_step]["close"]
        
        # Zone neutre autour de 0 pour éviter des micro-transactions
        if -0.05 <= action <= 0.05:
            logger.debug("Action: HOLD (zone neutre)")
            return
            
        if action > 0:  # Achat
            buy_percentage = action
            
            # Limiter l'achat à 30% du portefeuille total
            portfolio_value = self.get_portfolio_value()
            max_buy_value = portfolio_value * 0.3
            
            # Calculer la valeur d'achat basée sur le pourcentage
            buy_value = self.balance * buy_percentage
            
            # Appliquer la limite de 30%
            buy_value = min(buy_value, max_buy_value)
            
            # Calculer la quantité de crypto à acheter
            max_crypto_to_buy = buy_value / (current_price * (1 + self.transaction_fee))
            
            # Acheter la quantité calculée
            self.crypto_held += max_crypto_to_buy
            self.balance -= (max_crypto_to_buy * current_price * (1 + self.transaction_fee))

            logger.debug(
                f"Achat: {max_crypto_to_buy:.6f} unités ({buy_percentage*100:.0f}%) à ${current_price:.2f} (limité à 30% du portefeuille)"
            )
            
        else:  # Vente (action_value < 0)
            if self.crypto_held > 0:
                sell_percentage = -action
                crypto_to_sell = self.crypto_held * sell_percentage
                
                # Vendre la quantité calculée
                self.balance += (
                    crypto_to_sell * current_price * (1 - self.transaction_fee)
                )

                logger.debug(
                    f"Vente: {crypto_to_sell:.6f} unités ({sell_percentage*100:.0f}%) à ${current_price:.2f}"
                )

                self.crypto_held -= crypto_to_sell
            else:
                logger.debug("Tentative de vente sans crypto détenue")

    def _get_observation(self):
        """
        Récupère l'observation actuelle (état) pour l'agent RL.
        Inclut tous les indicateurs techniques et les données de sentiment pour une décision plus précise.
        """
        # Fenêtre de prix et volumes
        price_window = self.df.iloc[self.current_step-self.window_size:self.current_step]
        
        # Calculer tous les indicateurs techniques sur les données complètes
        indicators = TechnicalIndicators(self.df.iloc[:self.current_step])
        all_indicators = indicators.get_all_indicators(normalize=True)
        
        # Récupérer uniquement les indicateurs pour le pas de temps actuel
        current_indicators = all_indicators.iloc[-1].values if not all_indicators.empty else np.zeros(22)
        
        # Extraire les données de sentiment si disponibles
        sentiment_features = []
        sentiment_columns = ['compound_score', 'positive_score', 'negative_score', 'neutral_score', 
                             'sentiment_volume', 'sentiment_change']
        
        for col in sentiment_columns:
            if col in self.df.columns:
                # Normaliser la valeur de sentiment
                value = self.df.iloc[self.current_step][col]
                # Pour les scores déjà entre -1 et 1, normaliser entre 0 et 1
                if col in ['compound_score', 'positive_score', 'negative_score', 'neutral_score']:
                    value = (value + 1) / 2
                sentiment_features.append(value)
        
        # Si aucune donnée de sentiment n'est disponible, utiliser des zéros
        if not sentiment_features:
            sentiment_features = np.zeros(len(sentiment_columns))
        
        # Informations sur le portefeuille
        portfolio_info = np.array([
            self.balance / self.initial_balance,  # Solde normalisé
            self.crypto_held * self.df.iloc[self.current_step]["close"] / self.initial_balance,  # Valeur des cryptos détenues
            self.get_portfolio_value() / self.initial_balance,  # Valeur totale du portefeuille
        ])
        
        # Concaténer toutes les informations
        observation = np.concatenate([
            price_window.values.flatten(),  # Historique des prix
            current_indicators,             # Tous les indicateurs techniques
            sentiment_features,             # Données de sentiment
            portfolio_info                  # État du portefeuille
        ])
        
        # Appliquer la normalisation adaptative si activée
        if self.use_adaptive_normalization:
            # Créer un dictionnaire de features pour la mise à jour du normalisateur
            feature_dict = {}
            
            # Ajouter les prix et volumes actuels
            feature_dict['price'] = self.df.iloc[self.current_step]['close']
            if 'volume' in self.df.columns:
                feature_dict['volume'] = self.df.iloc[self.current_step]['volume']
            
            # Ajouter les indicateurs techniques
            if not all_indicators.empty:
                for col in all_indicators.columns:
                    feature_dict[col] = all_indicators.iloc[-1][col]
            
            # Ajouter les données de sentiment
            for i, col in enumerate(sentiment_columns):
                if col in self.df.columns:
                    feature_dict[col] = self.df.iloc[self.current_step][col]
            
            # Ajouter les informations de portefeuille
            feature_dict['balance'] = self.balance / self.initial_balance
            feature_dict['crypto_value'] = self.crypto_held * self.df.iloc[self.current_step]["close"] / self.initial_balance
            feature_dict['portfolio_value'] = self.get_portfolio_value() / self.initial_balance
            
            # Mettre à jour le normalisateur
            self.normalizer.update(feature_dict)
            
            # Normaliser l'observation
            # Nous ne pouvons pas utiliser directement normalize_array car l'observation
            # contient des séquences (price_window). Nous normalisons donc chaque composant séparément.
            
            # Normaliser la fenêtre de prix
            price_window_flat = price_window.values.flatten()
            for i in range(len(price_window_flat)):
                price_window_flat[i] = self.normalizer.normalize({'price': price_window_flat[i]})['price']
            
            # Normaliser les indicateurs techniques
            for i in range(len(current_indicators)):
                indicator_name = all_indicators.columns[i % len(all_indicators.columns)]
                current_indicators[i] = self.normalizer.normalize({indicator_name: current_indicators[i]})[indicator_name]
            
            # Normaliser les features de sentiment
            for i in range(len(sentiment_features)):
                col = sentiment_columns[i]
                if col in feature_dict:
                    sentiment_features[i] = self.normalizer.normalize({col: sentiment_features[i]})[col]
            
            # Reconstruire l'observation normalisée
            observation = np.concatenate([
                price_window_flat,
                current_indicators,
                sentiment_features,
                portfolio_info  # Déjà normalisé
            ])
        
        return observation

    def render(self, mode="human"):
        """
        Affiche l'état actuel de l'environnement.

        Args:
            mode (str): Mode d'affichage
        """
        if mode == "human":
            current_price = self.df.iloc[self.current_step]["close"]
            portfolio_value = self.balance + self.crypto_held * current_price

            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Crypto held: {self.crypto_held:.6f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Portfolio value: ${portfolio_value:.2f}")
            print(
                f"Profit/Loss: {((portfolio_value / self.initial_balance) - 1) * 100:.2f}%"
            )
            print("-" * 50)

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

    def _calculate_reward(self, previous_portfolio_value, current_portfolio_value):
        """
        Calcule la récompense en fonction de la variation de la valeur du portefeuille.
        
        Args:
            previous_portfolio_value (float): Valeur précédente du portefeuille
            current_portfolio_value (float): Valeur actuelle du portefeuille
            
        Returns:
            float: Récompense
        """
        # Calculer le rendement
        if previous_portfolio_value == 0:
            return 0
        
        # Rendement simple
        return_pct = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # Récompense de base
        reward = return_pct * 100  # Multiplier par 100 pour avoir une échelle plus grande
        
        # Ajouter des bonus/malus en fonction de la tendance récente
        if len(self.portfolio_value_history) >= 21:
            # S'assurer que les tableaux ont la même taille
            recent_values = self.portfolio_value_history[-21:]
            recent_returns = np.diff(recent_values) / recent_values[:-1]
            
            # Bonus pour une tendance positive constante
            if np.all(recent_returns[-5:] > 0):
                reward += 0.5
            
            # Malus pour une tendance négative constante
            if np.all(recent_returns[-5:] < 0):
                reward -= 0.5
        
        return reward

    def visualize_indicators(self, start_step=None, end_step=None):
        """
        Visualise les indicateurs techniques utilisés dans l'environnement.
        
        Args:
            start_step (int, optional): Pas de temps de début pour la visualisation.
            end_step (int, optional): Pas de temps de fin pour la visualisation.
        """
        if start_step is None:
            start_step = self.window_size
        if end_step is None:
            end_step = len(self.df) - 1
        
        # Extraire les données pour la visualisation
        data = self.df.iloc[start_step:end_step+1].copy()
        
        # Initialiser la classe d'indicateurs
        indicators = TechnicalIndicators(data)
        
        # Importer la fonction de visualisation
        from ai_trading.examples.visualize_indicators import plot_indicators
        
        # Tracer les indicateurs
        plot_indicators(data, indicators)

    def _close_position(self, price):
        # ...
        
        # Supprimer les stop-loss et take-profit
        if self.use_risk_manager:
            position_id = f"position_{self.current_step}"
            self.risk_manager.clear_position(position_id)
        
        # ...
