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
        window_size=5,
        include_technical_indicators=False,
        action_type="discrete",
        n_discrete_actions=5,
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

        # Définir l'espace d'observation (état)
        self._build_observation_space()

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

        # Sauvegarder l'état précédent pour calculer la récompense
        previous_portfolio_value = self.get_portfolio_value()

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

        # Informations supplémentaires
        info = {
            "portfolio_value": current_portfolio_value,
            "balance": self.balance,
            "crypto_held": self.crypto_held,
            "current_price": self.df.iloc[self.current_step]["close"],
        }

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
        Construit l'observation (état) actuelle.

        Returns:
            np.array: L'observation
        """
        # Extraire les prix des window_size dernières périodes
        prices = self.df.iloc[
            self.current_step - self.window_size : self.current_step + 1
        ]["close"].values

        # Normaliser les prix (variation en pourcentage par rapport au prix actuel)
        current_price = prices[-1]
        normalized_prices = prices / current_price - 1

        # Ajouter la position actuelle (crypto détenue et solde)
        crypto_held_normalized = self.crypto_held * current_price / self.initial_balance
        balance_normalized = self.balance / self.initial_balance

        # Construire l'observation
        observation = np.concatenate(
            [normalized_prices, [crypto_held_normalized, balance_normalized]]
        )

        # Ajouter des indicateurs techniques si demandé
        if self.include_technical_indicators:
            # Extraire les indicateurs techniques
            tech_indicators = self._get_technical_indicators()
            observation = np.concatenate([observation, tech_indicators])

        return observation.astype(np.float32)

    def _get_technical_indicators(self):
        """
        Calcule les indicateurs techniques pour l'observation.

        Returns:
            np.array: Indicateurs techniques
        """
        # Implémentation simplifiée - à développer selon les besoins
        # Ici, on pourrait calculer RSI, MACD, Bollinger Bands, etc.
        # Pour l'instant, on retourne un tableau vide
        return np.array([])

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

    def _calculate_reward(self, previous_value, current_value):
        """
        Calcule la récompense à partir des valeurs précédente et actuelle du portefeuille.

        Args:
            previous_value (float): Valeur du portefeuille à l'étape précédente
            current_value (float): Valeur du portefeuille à l'étape actuelle

        Returns:
            float: Récompense calculée
        """
        return (current_value - previous_value) / previous_value
