import logging

import numpy as np
import pandas as pd

from ai_trading.rl.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Gère les risques liés au trading, notamment les stop-loss et take-profit.
    """

    def __init__(self, config=None):
        """
        Initialise le gestionnaire de risques.

        Args:
            config (dict, optional): Configuration du gestionnaire de risques
                - max_position_size (float): Taille maximale d'une position en % du capital
                - max_risk_per_trade (float): Risque maximal par trade en % du capital
                - stop_loss_atr_factor (float): Facteur multiplicateur de l'ATR pour le stop-loss
                - take_profit_atr_factor (float): Facteur multiplicateur de l'ATR pour le take-profit
                - trailing_stop_activation (float): % de profit pour activer le trailing stop
                - trailing_stop_distance (float): Distance du trailing stop en % du prix
                - volatility_lookback (int): Période pour calculer la volatilité
                - volatility_threshold (float): Seuil de volatilité (15%)
                - position_adjustment_factor (float): Réduction de position en cas de haute volatilité
        """
        self.config = config or {}

        # Paramètres par défaut
        self.max_position_size = self.config.get(
            "max_position_size", 0.2
        )  # 20% du capital max
        self.max_risk_per_trade = self.config.get(
            "max_risk_per_trade", 0.02
        )  # 2% du capital max
        self.stop_loss_atr_factor = self.config.get("stop_loss_atr_factor", 2.0)
        self.take_profit_atr_factor = self.config.get("take_profit_atr_factor", 3.0)
        self.trailing_stop_activation = self.config.get(
            "trailing_stop_activation", 0.02
        )  # 2% de profit
        self.trailing_stop_distance = self.config.get(
            "trailing_stop_distance", 0.01
        )  # 1% du prix
        self.volatility_lookback = self.config.get(
            "volatility_lookback", 14
        )  # Période pour calculer la volatilité
        self.volatility_threshold = self.config.get(
            "volatility_threshold", 0.15
        )  # Seuil de volatilité (15%)
        self.position_adjustment_factor = self.config.get(
            "position_adjustment_factor", 0.5
        )  # Réduction de position en cas de haute volatilité

        # Initialiser les indicateurs techniques avec un DataFrame minimal
        self.indicators = TechnicalIndicators(df=pd.DataFrame(columns=["close"]))

        # Dictionnaire pour stocker les stop-loss et take-profit pour chaque position
        self.position_stops = {}

        logger.info("Gestionnaire de risques initialisé")

    def calculate_position_size(self, capital, entry_price, stop_loss_price):
        """
        Calcule la taille optimale d'une position en fonction du risque.

        Args:
            capital (float): Capital disponible
            entry_price (float): Prix d'entrée
            stop_loss_price (float): Prix du stop-loss

        Returns:
            float: Taille de la position en unités
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            logger.warning("Prix d'entrée ou de stop-loss invalide")
            return 0

        # Calculer le risque par unité
        risk_per_unit = abs(entry_price - stop_loss_price)
        if risk_per_unit <= 0:
            logger.warning("Risque par unité nul ou négatif")
            return 0

        # Calculer le montant à risquer
        risk_amount = capital * self.max_risk_per_trade

        # Calculer la taille de la position
        position_size = risk_amount / risk_per_unit

        # Limiter la taille de la position au maximum autorisé
        max_position_size = capital * self.max_position_size / entry_price
        position_size = min(position_size, max_position_size)

        # Ajustement dynamique basé sur la volatilité
        atr = self.indicators.calculate_atr(14)
        if atr is not None and len(atr) > 0:
            risk_adjustment = 1 - (
                atr.iloc[-1] / entry_price
            )  # Plus l'ATR est élevé, plus on réduit la position
            position_size *= max(
                risk_adjustment, 0.2
            )  # Ne pas descendre en dessous de 20%

        # Ajustement basé sur la volatilité historique
        volatility = self.calculate_historical_volatility(self.volatility_lookback)
        if volatility > self.volatility_threshold:
            adjustment = (
                1
                - (volatility - self.volatility_threshold)
                * self.position_adjustment_factor
            )
            position_size *= max(adjustment, 0.2)  # Ne pas descendre en dessous de 20%
            logger.info(
                f"Ajustement de position basé sur la volatilité ({volatility:.2%} > {self.volatility_threshold:.2%})"
            )

        logger.info(f"Taille de position calculée: {position_size} unités")
        return position_size

    def calculate_atr_stop_loss(
        self, data, period=14, direction="long", current_price=None, position_id=None
    ):
        """
        Calcule un stop-loss dynamique basé sur l'ATR.

        Args:
            data (pd.DataFrame): Données de marché avec colonnes high, low, close
            period (int): Période pour le calcul de l'ATR
            direction (str): Direction de la position ('long' ou 'short')
            current_price (float, optional): Prix actuel, si différent du dernier prix dans data
            position_id (str, optional): Identifiant unique de la position

        Returns:
            float: Prix du stop-loss
        """
        if data is None or len(data) < period:
            logger.warning(
                f"Données insuffisantes pour calculer l'ATR (besoin de {period} points)"
            )
            return None

        # Initialiser les indicateurs avec les données
        self.indicators = TechnicalIndicators(data)

        # Calculer l'ATR
        atr = self.indicators.calculate_atr(period=period)

        # Vérifier la validité de l'ATR
        if atr is None or len(atr) == 0 or np.isnan(atr.iloc[-1]):
            logger.warning("Impossible de calculer l'ATR")
            return None

        # Utiliser le dernier ATR calculé
        last_atr = atr.iloc[-1]

        # Utiliser le prix actuel ou le dernier prix de clôture
        price = current_price if current_price is not None else data["close"].iloc[-1]

        # Calculer le stop-loss en fonction de la direction
        if direction == "long":
            stop_loss = price - (last_atr * self.stop_loss_atr_factor)
        else:  # short
            stop_loss = price + (last_atr * self.stop_loss_atr_factor)

        # Enregistrer le stop-loss si un ID de position est fourni
        if position_id is not None:
            if position_id not in self.position_stops:
                self.position_stops[position_id] = {
                    "stop_loss": stop_loss,
                    "take_profit": None,
                    "trailing_stop": None,
                    "entry_price": (
                        current_price
                        if current_price is not None
                        else data["close"].iloc[-1]
                    ),
                }
            else:
                self.position_stops[position_id]["stop_loss"] = stop_loss
                self.position_stops[position_id]["entry_price"] = (
                    current_price
                    if current_price is not None
                    else data["close"].iloc[-1]
                )

        logger.info(f"Stop-loss ATR calculé: {stop_loss} pour position {direction}")
        return stop_loss

    def calculate_atr_take_profit(
        self, data, period=14, direction="long", current_price=None, position_id=None
    ):
        """
        Calcule un take-profit dynamique basé sur l'ATR.

        Args:
            data (pd.DataFrame): Données de marché avec colonnes high, low, close
            period (int): Période pour le calcul de l'ATR
            direction (str): Direction de la position ('long' ou 'short')
            current_price (float, optional): Prix actuel, si différent du dernier prix dans data
            position_id (str, optional): Identifiant unique de la position

        Returns:
            float: Prix du take-profit
        """
        if data is None or len(data) < period:
            logger.warning(
                f"Données insuffisantes pour calculer l'ATR (besoin de {period} points)"
            )
            return None

        # Initialiser les indicateurs avec les données
        self.indicators = TechnicalIndicators(data)

        # Calculer l'ATR
        atr = self.indicators.calculate_atr(period=period)

        # Vérifier la validité de l'ATR
        if atr is None or len(atr) == 0 or np.isnan(atr.iloc[-1]):
            logger.warning("Impossible de calculer l'ATR")
            return None

        # Utiliser le dernier ATR calculé
        last_atr = atr.iloc[-1]

        # Utiliser le prix actuel ou le dernier prix de clôture
        price = current_price if current_price is not None else data["close"].iloc[-1]

        # Calculer le take-profit en fonction de la direction
        if direction == "long":
            take_profit = price + (last_atr * self.take_profit_atr_factor)
        else:  # short
            take_profit = price - (last_atr * self.take_profit_atr_factor)

        # Enregistrer le take-profit si un ID de position est fourni
        if position_id is not None:
            if position_id not in self.position_stops:
                self.position_stops[position_id] = {
                    "stop_loss": None,
                    "take_profit": take_profit,
                    "trailing_stop": None,
                }
            else:
                self.position_stops[position_id]["take_profit"] = take_profit

        logger.info(f"Take-profit ATR calculé: {take_profit} pour position {direction}")
        return take_profit

    def update_trailing_stop(
        self, position_id, current_price, entry_price, direction="long"
    ):
        """
        Met à jour le trailing stop pour une position.

        Args:
            position_id (str): Identifiant unique de la position
            current_price (float): Prix actuel
            entry_price (float): Prix d'entrée de la position
            direction (str): Direction de la position ('long' ou 'short')

        Returns:
            float: Nouveau prix du trailing stop, ou None si pas activé
        """
        if position_id not in self.position_stops:
            logger.warning(
                f"Position {position_id} non trouvée dans le gestionnaire de risques"
            )
            return None

        # Vérifier si le trailing stop doit être activé
        profit_pct = 0
        if direction == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price

        # Si le profit est suffisant, activer ou mettre à jour le trailing stop
        if profit_pct >= self.trailing_stop_activation:
            if direction == "long":
                new_stop = current_price * (1 - self.trailing_stop_distance)
                # Ne mettre à jour que si le nouveau stop est plus élevé que l'ancien
                if (
                    self.position_stops[position_id]["trailing_stop"] is None
                    or new_stop > self.position_stops[position_id]["trailing_stop"]
                ):
                    self.position_stops[position_id]["trailing_stop"] = new_stop
                    logger.info(
                        f"Trailing stop mis à jour: {new_stop} pour position {position_id}"
                    )
            else:  # short
                new_stop = current_price * (1 + self.trailing_stop_distance)
                # Ne mettre à jour que si le nouveau stop est plus bas que l'ancien
                if (
                    self.position_stops[position_id]["trailing_stop"] is None
                    or new_stop < self.position_stops[position_id]["trailing_stop"]
                ):
                    self.position_stops[position_id]["trailing_stop"] = new_stop
                    logger.info(
                        f"Trailing stop mis à jour: {new_stop} pour position {position_id}"
                    )

        return self.position_stops[position_id]["trailing_stop"]

    def check_stop_conditions(self, position_id, current_price, direction="long"):
        """
        Vérifie si les conditions de stop-loss ou take-profit sont atteintes.

        Args:
            position_id (str): Identifiant unique de la position
            current_price (float): Prix actuel
            direction (str): Direction de la position ('long' ou 'short')

        Returns:
            dict: Résultat de la vérification avec les clés:
                - 'stop_triggered': True si un stop est déclenché
                - 'stop_type': Type de stop déclenché ('stop_loss', 'take_profit', 'trailing_stop')
                - 'stop_price': Prix du stop déclenché
        """
        if position_id not in self.position_stops:
            logger.warning(
                f"Position {position_id} non trouvée dans le gestionnaire de risques"
            )
            return {"stop_triggered": False, "stop_type": None, "stop_price": None}

        stops = self.position_stops[position_id]
        result = {"stop_triggered": False, "stop_type": None, "stop_price": None}

        # Vérifier le stop-loss
        if stops["stop_loss"] is not None:
            if (direction == "long" and current_price <= stops["stop_loss"]) or (
                direction == "short" and current_price >= stops["stop_loss"]
            ):
                result = {
                    "stop_triggered": True,
                    "stop_type": "stop_loss",
                    "stop_price": stops["stop_loss"],
                }
                logger.info(
                    f"Stop-loss déclenché pour position {position_id} à {stops['stop_loss']}"
                )
                return result

        # Vérifier le take-profit
        if stops["take_profit"] is not None:
            if (direction == "long" and current_price >= stops["take_profit"]) or (
                direction == "short" and current_price <= stops["take_profit"]
            ):
                result = {
                    "stop_triggered": True,
                    "stop_type": "take_profit",
                    "stop_price": stops["take_profit"],
                }
                logger.info(
                    f"Take-profit déclenché pour position {position_id} à {stops['take_profit']}"
                )
                return result

        # Vérifier le trailing stop
        if stops["trailing_stop"] is not None:
            if (direction == "long" and current_price <= stops["trailing_stop"]) or (
                direction == "short" and current_price >= stops["trailing_stop"]
            ):
                result = {
                    "stop_triggered": True,
                    "stop_type": "trailing_stop",
                    "stop_price": stops["trailing_stop"],
                }
                logger.info(
                    f"Trailing stop déclenché pour position {position_id} à {stops['trailing_stop']}"
                )
                return result

        return result

    def clear_position(self, position_id):
        """
        Supprime les informations de stop pour une position.

        Args:
            position_id (str): Identifiant unique de la position
        """
        if position_id in self.position_stops:
            del self.position_stops[position_id]
            logger.info(f"Position {position_id} supprimée du gestionnaire de risques")

    def update_atr_trailing_stop(
        self, data, period=14, position_id=None, current_price=None, direction="long"
    ):
        """
        Met à jour le trailing stop dynamique basé sur l'ATR.

        Args:
            data (pd.DataFrame): Données historiques pour calculer l'ATR
            period (int): Période de calcul de l'ATR
            position_id (str): ID de la position
            current_price (float): Prix actuel
            direction (str): Direction de la position ('long' ou 'short')

        Returns:
            float: Nouveau prix du trailing stop
        """
        if position_id not in self.position_stops:
            logger.warning(f"Position {position_id} non trouvée")
            return None

        # Calculer l'ATR
        self.indicators = TechnicalIndicators(data)
        atr = self.indicators.calculate_atr(period=period)

        if atr is None or len(atr) == 0 or np.isnan(atr.iloc[-1]):
            return self.position_stops[position_id]["trailing_stop"]

        last_atr = atr.iloc[-1]
        price = current_price if current_price else data["close"].iloc[-1]

        # Calculer la distance du trailing stop
        trailing_distance = last_atr * self.config.get("trailing_stop_atr_factor", 1.5)

        # Vérifier le breakout
        breakout_threshold = self.config.get("breakout_threshold", 0.5)  # 0.5%
        price_change = (
            (current_price - self.position_stops[position_id]["entry_price"])
            / self.position_stops[position_id]["entry_price"]
            * 100
        )

        if price_change > breakout_threshold:
            # Augmenter l'agressivité du trailing stop
            trailing_distance *= self.config.get("breakout_multiplier", 0.8)
            logger.info(
                f"Breakout détecté ({price_change:.2f}%), ajustement du trailing stop"
            )

        # Mettre à jour le trailing stop
        if direction == "long":
            new_stop = price - trailing_distance
            current_stop = self.position_stops[position_id]["trailing_stop"]
            if current_stop is None or new_stop > current_stop:
                self.position_stops[position_id]["trailing_stop"] = new_stop
                logger.info(f"Trailing stop ATR mis à jour: {new_stop}")
        else:  # short
            new_stop = price + trailing_distance
            current_stop = self.position_stops[position_id]["trailing_stop"]
            if current_stop is None or new_stop < current_stop:
                self.position_stops[position_id]["trailing_stop"] = new_stop
                logger.info(f"Trailing stop ATR mis à jour: {new_stop}")

        return self.position_stops[position_id]["trailing_stop"]

    def update_ma_trailing_stop(
        self, data, ma_period=20, position_id=None, current_price=None, direction="long"
    ):
        """
        Trailing stop basé sur une moyenne mobile.
        """
        if position_id not in self.position_stops:
            return None

        # Calculer la moyenne mobile
        ma = TechnicalIndicators(data).calculate_sma(ma_period)
        if ma is None or len(ma) < ma_period:
            return None

        current_ma = ma.iloc[-1]

        # Calculer la distance
        distance = self.config.get("ma_trailing_distance", 0.02)  # 2%

        if direction == "long":
            new_stop = current_ma * (1 - distance)
            current_stop = self.position_stops[position_id]["trailing_stop"]
            if new_stop > current_stop:
                self.position_stops[position_id]["trailing_stop"] = new_stop
        else:
            new_stop = current_ma * (1 + distance)
            current_stop = self.position_stops[position_id]["trailing_stop"]
            if new_stop < current_stop:
                self.position_stops[position_id]["trailing_stop"] = new_stop

        return new_stop

    def calculate_historical_volatility(self, lookback=14):
        """
        Calcule la volatilité historique sur la période donnée.
        """
        if self.indicators.df is None or len(self.indicators.df) < lookback:
            return 0.0

        returns = np.log(
            self.indicators.df["close"] / self.indicators.df["close"].shift(1)
        )
        volatility = returns.rolling(window=lookback).std() * np.sqrt(252)  # Annualisé
        return volatility.iloc[-1] if not np.isnan(volatility.iloc[-1]) else 0.0

    def update_volatility_adjusted_stops(self, data, position_id):
        """
        Met à jour les stops en fonction de la volatilité actuelle.
        """
        if position_id not in self.position_stops:
            return

        current_volatility = self.calculate_historical_volatility(
            self.volatility_lookback
        )
        original_stop = self.position_stops[position_id]["stop_loss"]

        # Ajuster le stop-loss si la volatilité augmente
        if current_volatility > self.volatility_threshold:
            adjustment = (
                1 + (current_volatility - self.volatility_threshold) * 0.5 + 0.5
            )
            new_stop = original_stop * adjustment
            self.position_stops[position_id]["stop_loss"] = new_stop
            logger.info(
                f"Stop-loss ajusté à {new_stop:.2f} (volatilité {current_volatility:.2%})"
            )

    def should_limit_position(self, portfolio_history, current_crypto):
        """
        Détermine si la position actuelle dépasse les limites de risque.

        Args:
            portfolio_history (list): Historique des valeurs du portefeuille
            current_crypto (float): Quantité de crypto détenue

        Returns:
            bool: True si la position doit être limitée
        """
        if len(portfolio_history) < 2:
            return False

        # Vérifier que les données sont disponibles et valides
        if (
            not hasattr(self, "indicators")
            or not hasattr(self.indicators, "df")
            or self.indicators.df is None
            or self.indicators.df.empty
            or "close" not in self.indicators.df.columns
        ):
            # Au lieu d'afficher un avertissement à chaque fois, on utilise debug
            # et on retourne False (pas de limitation de position)
            if hasattr(
                logger, "debug"
            ):  # Vérifier si le niveau de journalisation debug est disponible
                logger.debug("Données de prix non disponibles pour le calcul de risque")
            return False

        # Calculer la valeur actuelle de la position
        current_price = self.indicators.df["close"].iloc[-1]
        position_value = current_crypto * current_price

        # Calculer la valeur maximale autorisée
        max_position = portfolio_history[-1] * self.max_position_size

        # Vérifier la volatilité
        volatility = self.calculate_historical_volatility(self.volatility_lookback)
        if volatility > self.volatility_threshold:
            risk_adjustment = 1 - (volatility - self.volatility_threshold)
            max_position *= risk_adjustment

        return position_value > max_position

    def adjust_action(self, action, portfolio_value, current_crypto, current_price):
        """
        Ajuste l'action en fonction des limites de risque.

        Args:
            action (float): Action originale (-1 à 1)
            portfolio_value (float): Valeur totale du portefeuille
            current_crypto (float): Quantité de crypto détenue
            current_price (float): Prix actuel de la crypto

        Returns:
            float: Action ajustée
        """
        if self.should_limit_position([portfolio_value], current_crypto):
            # Calculer la réduction nécessaire
            position_value = current_crypto * current_price
            max_allowed = portfolio_value * self.max_position_size
            adjustment_factor = (
                max_allowed / position_value if position_value > 0 else 0
            )

            # Appliquer l'ajustement progressif
            adjusted_action = (
                action
                * adjustment_factor
                * self.config.get("risk_adjustment_factor", 0.7)
            )
            logger.info(
                f"Ajustement de l'action: {action:.2f} -> {adjusted_action:.2f}"
            )
            return adjusted_action
        return action

    def reset(self):
        """
        Réinitialise l'état du gestionnaire de risques.
        """
        # Réinitialisation du tracking des performances
        self.peak_value = 0
        self.current_drawdown = 0

        # Réinitialisation du suivi des positions
        self.positions = {}
