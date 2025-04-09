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
    Version simple avec actions d'achat, vente et maintien.
    """

    def __init__(
        self,
        df,
        initial_balance=10000,
        transaction_fee=0.001,
        window_size=5,
        include_technical_indicators=False,
    ):
        """
        Initialise l'environnement de trading.

        Args:
            df (DataFrame): Données historiques avec au moins une colonne 'close' pour les prix
            initial_balance (float): Solde initial en USD
            transaction_fee (float): Frais de transaction (pourcentage)
            window_size (int): Nombre de périodes précédentes à inclure dans l'observation
            include_technical_indicators (bool): Inclure des indicateurs techniques dans l'observation
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

        # Définir l'espace d'action: 0 (ne rien faire), 1 (acheter), 2 (vendre)
        self.action_space = spaces.Discrete(3)

        # Définir l'espace d'observation (état)
        # Pour la version simple, on utilise:
        # - window_size derniers prix normalisés
        # - Quantité de crypto détenue
        # - Solde en USD
        self._build_observation_space()

        # Réinitialiser l'environnement
        self.reset()

        logger.info(
            f"Environnement de trading initialisé avec {len(df)} points de données"
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

    def reset(self):
        """
        Réinitialise l'environnement au début d'un épisode.

        Returns:
            observation (np.array): L'état initial
        """
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

        return observation

    def step(self, action):
        """
        Exécute une action dans l'environnement.

        Args:
            action (int): 0 (ne rien faire), 1 (acheter), 2 (vendre)

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Vérifier que l'action est valide
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide: {action}. Doit être 0, 1 ou 2.")

        # Enregistrer l'action
        self.action_history.append(action)

        # Obtenir le prix actuel
        current_price = self.df.iloc[self.current_step]["close"]

        # Exécuter l'action
        if action == 1:  # Acheter
            # Calculer le montant maximum que nous pouvons acheter
            max_crypto_to_buy = self.balance / (
                current_price * (1 + self.transaction_fee)
            )
            # Acheter tout ce que nous pouvons
            self.crypto_held += max_crypto_to_buy
            self.balance -= (
                max_crypto_to_buy * current_price * (1 + self.transaction_fee)
            )

            logger.debug(
                f"Achat: {max_crypto_to_buy:.6f} unités à ${current_price:.2f}"
            )

        elif action == 2:  # Vendre
            if self.crypto_held > 0:
                # Vendre toute la crypto détenue
                self.balance += (
                    self.crypto_held * current_price * (1 - self.transaction_fee)
                )

                logger.debug(
                    f"Vente: {self.crypto_held:.6f} unités à ${current_price:.2f}"
                )

                self.crypto_held = 0
            else:
                logger.debug("Tentative de vente sans crypto détenue")

        # Passer à l'étape suivante
        self.current_step += 1

        # Calculer la valeur du portefeuille
        portfolio_value = self.balance + self.crypto_held * current_price
        self.portfolio_value_history.append(portfolio_value)

        # Vérifier si l'épisode est terminé
        done = self.current_step >= len(self.df) - 1

        # Calculer la récompense (changement de valeur du portefeuille)
        reward = (
            portfolio_value - self.portfolio_value_history[-2]
        ) / self.portfolio_value_history[-2]

        # Obtenir la nouvelle observation
        obs = self._get_observation()

        # Informations supplémentaires pour le débogage
        info = {
            "portfolio_value": portfolio_value,
            "crypto_held": self.crypto_held,
            "balance": self.balance,
            "current_price": current_price,
            "return": (portfolio_value / self.initial_balance) - 1,
        }

        if done:
            logger.info(
                f"Épisode terminé. Valeur finale du portefeuille: ${portfolio_value:.2f}, "
                f"Rendement: {((portfolio_value / self.initial_balance) - 1) * 100:.2f}%"
            )

        return obs, reward, done, info

    def _get_observation(self):
        """
        Construit l'observation (état) actuelle.

        Returns:
            np.array: L'état actuel
        """
        # Obtenir les window_size derniers prix
        price_history = self.df.iloc[
            self.current_step - self.window_size : self.current_step + 1
        ]["close"].values

        # Normaliser les prix (diviser par le prix actuel)
        current_price = price_history[-1]
        normalized_prices = price_history / current_price

        # Normaliser la quantité de crypto détenue et le solde
        normalized_crypto_held = self.crypto_held * current_price / self.initial_balance
        normalized_balance = self.balance / self.initial_balance

        # Construire l'observation de base
        features = [*normalized_prices, normalized_crypto_held, normalized_balance]

        # Ajouter des indicateurs techniques si demandé
        if self.include_technical_indicators:
            # Exemple d'indicateurs techniques (à adapter selon les besoins)
            # RSI
            rsi_values = (
                np.random.random(self.window_size) * 100
            )  # Simulé pour l'exemple
            # MACD
            macd_values = (
                np.random.random(self.window_size) * 2 - 1
            )  # Simulé pour l'exemple
            # Bandes de Bollinger
            bb_values = (
                np.random.random(self.window_size) * 2 - 1
            )  # Simulé pour l'exemple

            # Ajouter les indicateurs à l'observation
            features.extend(rsi_values)
            features.extend(macd_values)
            features.extend(bb_values)

        # Convertir en tableau numpy
        obs = np.array(features, dtype=np.float32)

        return obs

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
